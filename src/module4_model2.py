"""
src/module4_model2.py
=====================
Module 4 — Model 2: ResNet-34 Severity Classifier — Spinal Canal Stenosis (SCS)

Input  : 3-slice stacked Sagittal T2 patches (3, 128, 128) from .npy files
Output : 3-class severity (Normal/Mild=0, Moderate=1, Severe=2)
Loss   : CrossEntropyLoss with class weights + label smoothing (eps=0.1)
Series : Sagittal T2/STIR  (condition = spinal_canal_stenosis, 5 levels)

Two-phase training
------------------
  Phase 1 (epochs 1–5)  : backbone frozen, head only, LR warmup 1e-6 → 1e-4
  Phase 2 (epoch 6+)    : unfreeze layer4, LR → 1e-5, CosineAnnealingLR

Val metric : weighted log-loss (competition metric — NOT CrossEntropy)
             weight = {Normal/Mild: 1, Moderate: 2, Severe: 4}

Checkpoints  (saved to CFG.CKPT_DIR)
-----------
  model2_best.pt     — best val weighted log-loss
  model2_latest.pt   — overwritten every epoch  (resume target)
  model2_epoch{N}.pt — every 10 epochs

Run
---
  python -m src.module4_model2
  python -m src.module4_model2 --smoke-only
  python -m src.module4_model2 --run-id my_run_001
"""

import argparse
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary
from src.module3 import RSNADataset, build_train_transforms, build_val_transforms, resolve_patch_paths


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "model2"
SCRIPT_NAME = "module4_model2"
CONDITION   = "spinal_canal_stenosis"
N_CLASSES   = 3

# Training hyper-parameters — same regime as model1
LR_HEAD          = 1e-4
LR_FINETUNE      = 1e-5
WD               = 1e-4
WARMUP_EPOCHS    = 5
MAX_EPOCHS       = 200   # safety cap only — early stopping governs termination
EARLY_STOP_PAT   = 20    # increased from 15 — SCS best was at epoch 16 with
                          # val still fluctuating; extra patience allows more
                          # stable convergence for the Moderate class
EARLY_STOP_DELTA = 1e-4
CKPT_EVERY       = 10
LABEL_SMOOTHING  = 0.1    # softens hard targets (0/1 → 0.9/0.1), forcing the
                          # model to retain uncertainty on ambiguous boundary cases
                          # between Normal/Mild and Moderate (SCS-specific)

# Competition severity weights for log-loss
SEV_WEIGHTS = torch.tensor(
    [CFG.SEVERITY_WEIGHTS[i] for i in range(3)], dtype=torch.float32
)


# ─────────────────────────────────────────────────────────────────────────────
# Seed helpers  (identical to model1)
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = CFG.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_seeds() -> dict:
    return {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }


def restore_seeds(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    ResNet-34 with pretrained ImageNet weights.
    fc replaced with Linear(512 → 3).
    """
    model    = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_feat  = model.fc.in_features    # 512
    model.fc = nn.Linear(in_feat, N_CLASSES)
    return model


def freeze_backbone(model: nn.Module):
    """Freeze everything except fc."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_layer4(model: nn.Module):
    """Unfreeze layer4 + fc for Phase 2."""
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("fc.") or name.startswith("layer4.")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_ds: RSNADataset, logger) -> torch.Tensor:
    """
    weight[c] = total_samples / (N_CLASSES * count[c])
    Computed from train split only.
    """
    counts = train_ds.label_counts()
    total  = sum(counts.values())
    weights = []
    for c in range(N_CLASSES):
        n = counts.get(c, 0)
        w = total / (N_CLASSES * n) if n > 0 else 1.0
        weights.append(w)
        sev = CFG.SEVERITY_INV[c]
        logger.info(f"  Class {c} ({sev:12s}): {n:,} samples  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def weighted_log_loss(
    probs:   np.ndarray,   # (N, 3) softmax probabilities
    labels:  np.ndarray,   # (N,)   integer labels 0/1/2
    weights: np.ndarray,   # (3,)   per-class competition weights
    eps:     float = 1e-7,
) -> float:
    """
    Competition metric: weighted multi-class log-loss.
    loss = -sum_i( w[y_i] * log(p[i, y_i]) ) / sum_i( w[y_i] )
    """
    probs   = np.clip(probs, eps, 1 - eps)
    n       = len(labels)
    w_sum   = 0.0
    loss    = 0.0
    for i in range(n):
        c      = int(labels[i])
        w      = weights[c]
        loss  += w * np.log(probs[i, c])
        w_sum += w
    return float(-loss / w_sum) if w_sum > 0 else float("inf")


def val_metrics(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> Tuple[float, Dict[int, float]]:
    """
    Returns (weighted_log_loss, {class: accuracy}).
    Computed on val split with no augmentation.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    sev_weights = np.array([CFG.SEVERITY_WEIGHTS[i] for i in range(N_CLASSES)])
    val_loss    = weighted_log_loss(all_probs, all_labels, sev_weights)

    preds       = np.argmax(all_probs, axis=1)
    per_class_acc = {}
    for c in range(N_CLASSES):
        mask = (all_labels == c)
        if mask.sum() > 0:
            per_class_acc[c] = float((preds[mask] == c).mean() * 100)
        else:
            per_class_acc[c] = float("nan")

    return val_loss, per_class_acc


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (identical pattern to model1)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path, model: nn.Module, optimiser, scheduler,
    epoch: int, best_val_loss: float, patience_ctr: int, run_id: str,
):
    torch.save({
        "model_state":     model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch":           epoch,
        "best_val_loss":   best_val_loss,
        "patience_ctr":    patience_ctr,
        "run_id":          run_id,
        "seeds":           get_seeds(),
    }, path)


def load_checkpoint(
    path: Path, model: nn.Module, optimiser, scheduler,
    logger, run_id: str,
) -> Tuple[int, float, int]:
    ckpt          = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    patience_ctr  = ckpt["patience_ctr"]
    restore_seeds(ckpt["seeds"])
    logger.info(
        f"  Resumed from {path.name}  "
        f"(epoch {ckpt['epoch']}, best_val={best_val_loss:.6f})"
    )
    log_event(run_id, SCRIPT_NAME, "checkpoint_resumed",
              path=str(path), resumed_epoch=ckpt["epoch"])
    return start_epoch, best_val_loss, patience_ctr


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    logger,
    run_id:    str,
) -> bool:
    logger.info("── Smoke test ──────────────────────────────────────────────")
    issues = []

    model.train()
    opt_smoke = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WD,
    )

    # Single forward + backward
    try:
        imgs, labels = next(iter(loader))
        imgs, labels = imgs.to(device), labels.to(device)

        opt_smoke.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()

        loss_val = float(loss.item())
        logger.info(f"  Forward pass loss : {loss_val:.6f}")

        if np.isnan(loss_val):
            issues.append("Loss is NaN")
        if loss_val == 0.0:
            issues.append("Loss is exactly zero")

        head_grad_norm = float(
            model.fc.weight.grad.norm().item()
        ) if model.fc.weight.grad is not None else 0.0
        logger.info(f"  Head gradient norm: {head_grad_norm:.6f}")
        if head_grad_norm == 0.0:
            issues.append("Head gradient norm is zero — gradients not flowing")

    except Exception as exc:
        issues.append(f"Forward/backward failed: {exc}")
        logger.error(f"  ✗ {exc}")

    # Loss decreases over 5 steps
    logger.info("  Checking loss trend over 15 steps …")
    try:
        losses = []
        for _ in range(15):
            imgs, labels = next(iter(loader))
            imgs, labels = imgs.to(device), labels.to(device)
            opt_smoke.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt_smoke.step()
            losses.append(float(loss.item()))
        logger.info(f"  15-step losses: {[f'{l:.4f}' for l in losses]}")
        first_half  = float(np.mean(losses[:7]))
        second_half = float(np.mean(losses[8:]))
        if second_half >= first_half:
            issues.append(
                f"Loss trend not decreasing: "
                f"first-half mean={first_half:.4f} → second-half mean={second_half:.4f}"
            )
        else:
            logger.info(
                f"  Loss trend decreasing ✓ "
                f"(first-half={first_half:.4f} → second-half={second_half:.4f})"
            )
    except Exception as exc:
        issues.append(f"5-step check failed: {exc}")
        logger.error(f"  ✗ {exc}")

    if issues:
        for iss in issues:
            logger.error(f"  SMOKE ISSUE: {iss}")
        log_event(run_id, SCRIPT_NAME, "smoke_test_failed", issues=issues)
        return False

    logger.info("  Smoke test PASSED ✓")
    log_event(run_id, SCRIPT_NAME, "smoke_test_passed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    run_id:       str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    class_weights: torch.Tensor,
    logger,
    device:       torch.device,
) -> Tuple[float, int, int]:

    ensure_dirs()
    ckpt_dir = CFG.CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    freeze_backbone(model)
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WD,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=1e-6 / LR_HEAD,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )

    start_epoch   = 1
    best_val_loss = float("inf")
    patience_ctr  = 0
    phase         = 1
    best_epoch    = -1

    # Resume if checkpoint exists for this run_id
    latest_path = ckpt_dir / f"{MODEL_NAME}_latest.pt"
    if latest_path.exists():
        try:
            saved_run = torch.load(
                latest_path, map_location="cpu", weights_only=False
            ).get("run_id", "")
            if saved_run == run_id:
                start_epoch, best_val_loss, patience_ctr = load_checkpoint(
                    latest_path, model, optimiser, scheduler, logger, run_id
                )
                if start_epoch > WARMUP_EPOCHS:
                    unfreeze_layer4(model)
                    phase = 2
            else:
                logger.info(
                    f"  latest.pt belongs to run {saved_run}, not {run_id} "
                    f"— starting fresh."
                )
        except Exception as exc:
            logger.warning(f"  Could not load checkpoint: {exc}. Starting fresh.")

    logger.info(f"\n  Starting training from epoch {start_epoch}")
    logger.info(f"  Device       : {device}")
    logger.info(f"  Phase        : {phase}")
    logger.info(f"  Early stop   : patience={EARLY_STOP_PAT}, delta={EARLY_STOP_DELTA} (no hard epoch cap)")
    logger.info(f"  Safety cap   : {MAX_EPOCHS} epochs")

    log_event(run_id, SCRIPT_NAME, "training_start",
              start_epoch=start_epoch, device=str(device),
              safety_cap=MAX_EPOCHS)

    train_losses = []
    val_losses   = []
    lr_history   = []
    train_start  = time.time()

    epoch = start_epoch
    while True:
        epoch_start = time.time()

        # Phase 2 transition
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            logger.info(f"\n  ── Phase 2: unfreezing layer4, lr → {LR_FINETUNE} ──")
            unfreeze_layer4(model)
            optimiser = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE, weight_decay=WD,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=200, eta_min=1e-7,
            )
            phase = 2
            log_event(run_id, SCRIPT_NAME, "phase2_start", epoch=epoch)

        # Train one epoch
        model.train()
        epoch_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # Gradient norm
        grad_norm = float(sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5)

        # Validate — weighted log-loss + per-class accuracy
        val_loss, per_class_acc = val_metrics(model, val_loader, device)
        val_losses.append(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]
        lr_history.append(current_lr)

        # Timing and ETA
        epoch_secs    = time.time() - epoch_start
        elapsed_secs  = time.time() - train_start
        epochs_done   = epoch - start_epoch + 1
        avg_secs      = elapsed_secs / epochs_done
        patience_left = EARLY_STOP_PAT - patience_ctr
        eta_secs      = avg_secs * patience_left
        epoch_mins, epoch_s = divmod(int(epoch_secs), 60)
        eta_hrs,  eta_m     = divmod(int(eta_secs), 3600)
        eta_mins, _         = divmod(eta_m, 60)

        logger.info(
            f"  Epoch {epoch:3d} | "
            f"train_ce={train_loss:.4f} | "
            f"val_wlogloss={val_loss:.4f} | "
            f"lr={current_lr:.2e} | "
            f"grad_norm={grad_norm:.3f} | "
            f"time={epoch_mins}m{epoch_s:02d}s | "
            f"patience={patience_ctr}/{EARLY_STOP_PAT} | "
            f"ETA≈{eta_hrs}h{eta_mins:02d}m"
        )
        acc_str = "  ".join(
            f"{CFG.SEVERITY_INV[c]}={per_class_acc[c]:.1f}%"
            for c in range(N_CLASSES)
        )
        logger.info(f"    per-class acc: {acc_str}")

        log_event(run_id, SCRIPT_NAME, "epoch_end",
                  epoch=epoch, train_ce=train_loss,
                  val_wlogloss=val_loss,
                  per_class_acc={CFG.SEVERITY_INV[c]: per_class_acc[c]
                                 for c in range(N_CLASSES)},
                  lr=current_lr, grad_norm=grad_norm,
                  epoch_secs=round(epoch_secs, 1), phase=phase)

        # Checkpointing
        save_checkpoint(
            latest_path, model, optimiser, scheduler,
            epoch, best_val_loss, patience_ctr, run_id,
        )

        if epoch % CKPT_EVERY == 0:
            periodic = ckpt_dir / f"{MODEL_NAME}_epoch{epoch}.pt"
            save_checkpoint(
                periodic, model, optimiser, scheduler,
                epoch, best_val_loss, patience_ctr, run_id,
            )
            logger.info(f"  Periodic checkpoint saved → {periodic.name}")
            log_event(run_id, SCRIPT_NAME, "checkpoint_periodic",
                      epoch=epoch, path=str(periodic))

        # Best checkpoint — lower val log-loss is better
        if val_loss < best_val_loss - EARLY_STOP_DELTA:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            best_path     = ckpt_dir / f"{MODEL_NAME}_best.pt"
            save_checkpoint(
                best_path, model, optimiser, scheduler,
                epoch, best_val_loss, patience_ctr, run_id,
            )
            logger.info(
                f"  ✓ New best val_wlogloss={best_val_loss:.4f} — "
                f"saved {best_path.name}"
            )
            log_event(run_id, SCRIPT_NAME, "checkpoint_best",
                      epoch=epoch, best_val_wlogloss=best_val_loss)
        else:
            patience_ctr += 1
            logger.info(
                f"  No improvement. Patience {patience_ctr}/{EARLY_STOP_PAT}"
            )

        # Early stopping
        if patience_ctr >= EARLY_STOP_PAT:
            logger.info(
                f"\n  Early stopping triggered at epoch {epoch}. "
                f"Best val_wlogloss={best_val_loss:.4f}  "
                f"(best epoch={best_epoch})"
            )
            log_event(run_id, SCRIPT_NAME, "early_stop",
                      epoch=epoch, best_val_wlogloss=best_val_loss,
                      best_epoch=best_epoch)
            break

        # Safety cap
        if epoch >= MAX_EPOCHS:
            logger.info(f"\n  Safety cap reached at epoch {epoch}.")
            log_event(run_id, SCRIPT_NAME, "safety_cap_reached",
                      epoch=epoch, best_val_wlogloss=best_val_loss)
            break

        epoch += 1

    # Post-training
    total_mins = int((time.time() - train_start) / 60)
    logger.info(f"\n  Total training time : {total_mins // 60}h {total_mins % 60}m")
    logger.info(f"  Best checkpoint     : epoch {best_epoch}  "
                f"(val_wlogloss={best_val_loss:.4f})")

    _plot_curves(train_losses, val_losses, lr_history, run_id, start_epoch)

    return best_val_loss, len(train_losses), best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_curves(
    train_losses: List[float],
    val_losses:   List[float],
    lr_history:   List[float],
    run_id:       str,
    start_epoch:  int,
):
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model 2 (SCS) training curves — {run_id}", fontsize=11)

    ax1.plot(epochs, train_losses, color="#3498DB", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train CrossEntropy loss")
    ax1.set_title("Train CE loss"); ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, val_losses, color="#E74C3C", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val weighted log-loss")
    ax2.set_title("Val weighted log-loss (competition metric)")
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    if lr_history:
        ax3.plot(epochs, lr_history, color="#2ECC71", linewidth=1.5)
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning rate")
        ax3.set_title("Learning rate schedule")
        ax3.set_yscale("log"); ax3.grid(alpha=0.3)
        ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_{MODEL_NAME}_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, smoke_only: bool = False):
    run_id     = run_id or make_run_id("m4m2")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, SCRIPT_NAME)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem    = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_str = f"cuda ({gpu_name}, {gpu_mem:.1f}GB)"
    else:
        device     = torch.device("cpu")
        device_str = "cpu"

    logger.info("=" * 65)
    logger.info(f"Module 4 — Model 2 (SCS Classifier) | run_id={run_id}")
    logger.info(f"  Condition  : {CONDITION}")
    logger.info(f"  CKPT_DIR   : {CFG.CKPT_DIR}")
    logger.info(f"  Device     : {device_str}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, SCRIPT_NAME, "script_start",
              device=device_str, condition=CONDITION)

    ensure_dirs()
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds()

    try:
        # Load split CSVs and resolve patch paths
        logger.info("── Loading data ─────────────────────────────────────────")
        train_df = resolve_patch_paths(pd.read_csv(CFG.SPLIT_TRAIN), "train")
        val_df   = resolve_patch_paths(pd.read_csv(CFG.SPLIT_VAL),   "val")

        # Build condition-filtered datasets
        train_ds = RSNADataset(
            train_df, transform=build_train_transforms(),
            condition_filter=CONDITION,
        )
        val_ds = RSNADataset(
            val_df, transform=build_val_transforms(),
            condition_filter=CONDITION,
        )

        logger.info(f"  Train samples ({CONDITION}): {len(train_ds):,}")
        logger.info(f"  Val   samples ({CONDITION}): {len(val_ds):,}")

        if len(train_ds) == 0:
            raise RuntimeError(f"No train patches for condition: {CONDITION}")

        # Class weights — train only
        logger.info("\n── Class weights (from train split) ────────────────────")
        class_weights = compute_class_weights(train_ds, logger)
        log_event(run_id, SCRIPT_NAME, "class_weights",
                  weights=class_weights.tolist())

        # DataLoaders
        train_loader = DataLoader(
            train_ds, batch_size=CFG.BATCH_SIZE,
            shuffle=True, num_workers=0,
            pin_memory=(device.type == "cuda"), drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG.BATCH_SIZE,
            shuffle=False, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        # Build model
        logger.info("\n── Building model ──────────────────────────────────────")
        model = build_model().to(device)
        freeze_backbone(model)
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        logger.info(f"  Architecture  : ResNet-34 → Linear(512, {N_CLASSES})")
        logger.info(f"  Pretrained    : ImageNet")
        logger.info(f"  Phase 1       : backbone frozen, head only")
        logger.info(f"  Total params  : {total_params:,}")
        logger.info(f"  Trainable now : {trainable_params:,}")

        log_event(run_id, SCRIPT_NAME, "model_built",
                  total_params=total_params,
                  trainable_params=trainable_params)

        # Criterion with class weights + label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=LABEL_SMOOTHING,
        )

        # Smoke test
        smoke_ok = smoke_test(
            model, train_loader, criterion, device, logger, run_id
        )
        if not smoke_ok:
            raise RuntimeError("Smoke test failed — fix before full training.")

        if smoke_only:
            logger.info("--smoke-only: stopping after smoke test.")
            return

        # Reset model after smoke test
        model = build_model().to(device)
        freeze_backbone(model)
        set_seeds()
        logger.info("  Model reset after smoke test. Seeds restored.\n")

        # Full training
        logger.info("── Training ────────────────────────────────────────────")
        best_val, n_epochs, best_epoch = train(
            run_id, train_loader, val_loader,
            class_weights, logger, device,
        )

        logger.info(f"\n  Training complete.")
        logger.info(f"  Best val weighted log-loss : {best_val:.4f}")
        logger.info(f"  Best epoch                 : {best_epoch}")
        logger.info(f"  Epochs trained             : {n_epochs}")

        log_event(run_id, SCRIPT_NAME, "training_complete",
                  best_val_wlogloss=best_val,
                  n_epochs=n_epochs, best_epoch=best_epoch)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, SCRIPT_NAME, "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name=MODEL_NAME,
        start_time=start_time,
        end_time=end_time,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        notes=f"SCS classifier, label_smoothing={LABEL_SMOOTHING}, patience={EARLY_STOP_PAT}. Val weighted log-loss={best_val:.4f}",
    )
    log_event(run_id, SCRIPT_NAME, "script_complete", end_time=end_time)
    logger.info(f"\nModule 4 Model 2 complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Module 4 — Model 2: SCS Classifier"
    )
    parser.add_argument("--run-id",     default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run(run_id=args.run_id, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()
