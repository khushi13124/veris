"""
src/module4_model1.py
=====================
Module 4 — Model 1: ResNet-18 Keypoint Regression

Predicts (x, y) coordinates for 5 disc levels (L1/L2 → L5/S1) on
Sagittal T2 slices.  Output: 10 values — normalised (x/W, y/H) per level.

This model MUST be trained and validated before Models 2/3/4.
The predicted keypoints drive every downstream crop in inference.

Architecture
------------
  ResNet-18 (pretrained ImageNet)
  Replace fc  →  Linear(512, 10)
  Loss        →  MSE on normalised coordinates
  Val metric  →  mean pixel error (px) on val studies

Two-phase training
------------------
  Phase 1 (epochs 1–5)  : backbone frozen, head only, LR warmup 1e-6 → 1e-4
  Phase 2 (epoch 6+)    : all layers unfrozen, LR → 1e-5, CosineAnnealingLR

Checkpoints  (saved to CFG.CKPT_DIR)
-----------
  model1_best.pt    — best val pixel error (never overwritten with worse)
  model1_latest.pt  — overwritten every epoch  (resume target)
  model1_epoch{N}.pt — every 10 epochs

Run
---
  python -m src.module4_model1
  python -m src.module4_model1 --smoke-only
  python -m src.module4_model1 --run-id my_run_001
"""

import argparse
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary
from src.module2 import (
    build_instance_map,
    load_sorted_series,
    rescale_pixel_array,
    window_to_uint8,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "model1"
SCRIPT_NAME  = "module4_model1"
INPUT_SIZE   = 512          # resize full slice to INPUT_SIZE × INPUT_SIZE
N_KEYPOINTS  = 5            # L1/L2, L2/L3, L3/L4, L4/L5, L5/S1
N_OUTPUTS    = N_KEYPOINTS * 2   # (x, y) per level → 10 values

# Training hyper-parameters
LR_HEAD      = 1e-4
LR_FINETUNE  = 1e-5
WD           = 1e-4
WARMUP_EPOCHS    = 5
MAX_EPOCHS       = 200   # safety cap only — early stopping governs termination
EARLY_STOP_PAT   = 15
EARLY_STOP_DELTA = 1e-4
CKPT_EVERY       = 10       # save periodic checkpoint every N epochs

LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

# ImageNet normalisation (same as modules 2/3)
_MEAN = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)
_STD  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Seed helpers
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
# Keypoint table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_keypoint_table(
    master: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each study in split_df, find its Sagittal T2 series and extract
    the annotation (x, y) for each of the 5 SCS disc levels.

    SCS annotations on Sagittal T2 sit at disc centres — exactly the
    keypoint locations Model 1 needs to predict.

    Returns a DataFrame with columns:
        study_id, series_id,
        x_l1_l2, y_l1_l2, x_l2_l3, y_l2_l3, ... x_l5_s1, y_l5_s1
    One row per study.  Studies missing any level are dropped and logged.
    """
    study_ids = split_df["study_id"].unique()

    # Filter master to SCS on Sagittal T2 only
    scs = master[
        (master["condition"] == "spinal_canal_stenosis") &
        (master["series_type"].str.contains("Sagittal T2", na=False))
    ].copy()

    rows = []
    dropped = []

    for sid in study_ids:
        sub = scs[scs["study_id"] == sid]
        if sub.empty:
            dropped.append((sid, "no SCS Sagittal T2 rows"))
            continue

        # Use the first (most common) series_id for this study
        series_id = int(sub["series_id"].iloc[0])
        row = {"study_id": int(sid), "series_id": series_id}

        ok = True
        for lvl in LEVELS:
            lvl_rows = sub[sub["level"] == lvl]
            if lvl_rows.empty:
                dropped.append((sid, f"missing level {lvl}"))
                ok = False
                break
            # Average if multiple annotators
            row[f"x_{lvl}"] = float(lvl_rows["x"].mean())
            row[f"y_{lvl}"] = float(lvl_rows["y"].mean())

        if ok:
            rows.append(row)

    return pd.DataFrame(rows), dropped


# ─────────────────────────────────────────────────────────────────────────────
# DICOM slice loader
# ─────────────────────────────────────────────────────────────────────────────

def load_middle_slice(series_dir: Path) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    Load the middle slice of a DICOM series.
    Returns (uint8_array, H, W) or None on failure.
    """
    datasets = load_sorted_series(series_dir)
    if not datasets:
        return None
    mid_ds   = datasets[len(datasets) // 2]
    rescaled = rescale_pixel_array(mid_ds)
    windowed = window_to_uint8(rescaled, mid_ds)
    H, W     = windowed.shape
    return windowed, H, W


def prepare_slice(windowed: np.ndarray) -> np.ndarray:
    """
    Resize to INPUT_SIZE × INPUT_SIZE, replicate to 3 channels,
    divide by 255, apply ImageNet normalisation.
    Returns (3, INPUT_SIZE, INPUT_SIZE) float32.
    """
    resized = cv2.resize(
        windowed, (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0

    # (H, W) → (3, H, W)
    img3 = np.stack([resized, resized, resized], axis=0)

    # ImageNet normalise
    mean = _MEAN[:, None, None]
    std  = _STD[:, None, None]
    return (img3 - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    """
    One sample per study.
    Input : (3, INPUT_SIZE, INPUT_SIZE) float32 — full Sagittal T2 middle slice
    Target: (10,) float32 — normalised (x/W, y/H) for L1/L2 … L5/S1
    """

    def __init__(
        self,
        kp_df:     pd.DataFrame,
        augment:   bool = False,
    ):
        self.df      = kp_df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row       = self.df.iloc[idx]
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])

        series_dir = CFG.DICOM_ROOT / str(study_id) / str(series_id)
        result     = load_middle_slice(series_dir)

        if result is None:
            # Return zeros — will not happen after smoke test validation
            img    = np.zeros((3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            target = np.zeros(N_OUTPUTS, dtype=np.float32)
            return torch.from_numpy(img), torch.from_numpy(target)

        windowed, H, W = result

        if self.augment:
            # Light augmentation for training — brightness/contrast only.
            # No flips (would invert L/R), no rotation > 5° (spine is vertical).
            import albumentations as A
            aug = A.Compose([
                A.RandomBrightnessContrast(p=0.4),
                A.RandomGamma(p=0.3),
                A.GaussNoise(std_range=(0.005, 0.02), p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=0,
                    p=0.4,
                ),
            ])
            augmented = aug(image=windowed)
            windowed  = augmented["image"]

        img_tensor = torch.from_numpy(prepare_slice(windowed))

        # Build normalised keypoint target
        kp = []
        for lvl in LEVELS:
            kp.append(float(row[f"x_{lvl}"]) / W)
            kp.append(float(row[f"y_{lvl}"]) / H)
        target_tensor = torch.tensor(kp, dtype=torch.float32)

        return img_tensor, target_tensor

    def get_image_size(self, idx: int) -> Tuple[int, int]:
        """Return (H, W) of the original DICOM slice for a given index."""
        row       = self.df.iloc[idx]
        series_dir = (
            CFG.DICOM_ROOT
            / str(int(row["study_id"]))
            / str(int(row["series_id"]))
        )
        result = load_middle_slice(series_dir)
        if result is None:
            return INPUT_SIZE, INPUT_SIZE
        _, H, W = result
        return H, W


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """
    ResNet-18 with pretrained ImageNet weights.
    fc replaced with Linear(512 → N_OUTPUTS).
    """
    model   = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features          # 512
    model.fc = nn.Linear(in_feat, N_OUTPUTS)
    return model


def freeze_backbone(model: nn.Module):
    """Freeze everything except the final fc layer."""
    for name, param in model.named_parameters():
        param.requires_grad = (name.startswith("fc."))


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimiser:    torch.optim.Optimizer,
    current_epoch: int,
    total_epochs:  int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    If still in warmup phase, return LinearLR.
    Otherwise return CosineAnnealingLR for remaining epochs.
    """
    remaining = max(1, total_epochs - WARMUP_EPOCHS)
    if current_epoch < WARMUP_EPOCHS:
        return torch.optim.lr_scheduler.LinearLR(
            optimiser,
            start_factor=1e-6 / LR_HEAD,
            end_factor=1.0,
            total_iters=WARMUP_EPOCHS,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=remaining, eta_min=1e-7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:          Path,
    model:         nn.Module,
    optimiser:     torch.optim.Optimizer,
    scheduler,
    epoch:         int,
    best_val_loss: float,
    patience_ctr:  int,
    run_id:        str,
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
    path:      Path,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    logger,
    run_id:    str,
) -> Tuple[int, float, int]:
    """
    Load checkpoint. Returns (start_epoch, best_val_loss, patience_ctr).
    """
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
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def val_pixel_error(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    kp_df:   pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mean pixel error on the val set.
    Predictions are in normalised [0,1] space; we convert back to pixels
    using each study's original DICOM slice dimensions.

    Returns (overall_mean_px_error, {level: mean_px_error}).
    """
    model.eval()
    all_errors = {lvl: [] for lvl in LEVELS}

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs    = imgs.to(device)
            preds   = model(imgs).cpu().numpy()   # (B, 10)
            targets = targets.numpy()              # (B, 10)

            for i in range(len(preds)):
                # Get original image size for this sample
                global_idx = batch_idx * loader.batch_size + i
                if global_idx >= len(kp_df):
                    continue
                row = kp_df.iloc[global_idx]
                series_dir = (
                    CFG.DICOM_ROOT
                    / str(int(row["study_id"]))
                    / str(int(row["series_id"]))
                )
                result = load_middle_slice(series_dir)
                H, W   = (result[1], result[2]) if result else (INPUT_SIZE, INPUT_SIZE)

                for j, lvl in enumerate(LEVELS):
                    px = preds[i,   j*2]   * W
                    py = preds[i,   j*2+1] * H
                    gx = targets[i, j*2]   * W
                    gy = targets[i, j*2+1] * H
                    err = float(np.sqrt((px - gx)**2 + (py - gy)**2))
                    all_errors[lvl].append(err)

    per_level = {lvl: float(np.mean(v)) if v else float("inf")
                 for lvl, v in all_errors.items()}
    overall   = float(np.mean([e for v in all_errors.values() for e in v]))
    return overall, per_level


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    logger,
    run_id:    str,
) -> bool:
    """
    1 forward + 1 backward pass.
    Checks: loss not NaN, not zero, head gradients flowing.
    Loss decreases over 5 consecutive SGD steps.
    """
    logger.info("── Smoke test ──────────────────────────────────────────────")
    issues = []

    model.train()
    criterion  = nn.MSELoss()
    opt_smoke  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WD,
    )

    # ── Single forward + backward ────────────────────────────────────────────
    try:
        imgs, targets = next(iter(loader))
        imgs, targets = imgs.to(device), targets.to(device)

        opt_smoke.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, targets)
        loss.backward()

        loss_val = float(loss.item())
        logger.info(f"  Forward pass loss : {loss_val:.6f}")

        if np.isnan(loss_val):
            issues.append("Loss is NaN after first forward pass")
        if loss_val == 0.0:
            issues.append("Loss is exactly zero — something is wrong")

        # Check head gradients
        head_grad_norm = float(
            model.fc.weight.grad.norm().item()
        ) if model.fc.weight.grad is not None else 0.0
        logger.info(f"  Head gradient norm: {head_grad_norm:.6f}")
        if head_grad_norm == 0.0:
            issues.append("Head gradient norm is zero — gradients not flowing")

    except Exception as exc:
        issues.append(f"Forward/backward pass failed: {exc}")
        logger.error(f"  ✗ {exc}")

    # ── Loss decreases over 5 steps ──────────────────────────────────────────
    logger.info("  Checking loss decreases over 5 steps …")
    try:
        losses = []
        for _ in range(5):
            imgs, targets = next(iter(loader))
            imgs, targets = imgs.to(device), targets.to(device)
            opt_smoke.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            opt_smoke.step()
            losses.append(float(loss.item()))

        logger.info(f"  5-step losses: {[f'{l:.6f}' for l in losses]}")
        if losses[-1] >= losses[0]:
            issues.append(
                f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
            )
        else:
            logger.info("  Loss decreased over 5 steps ✓")

    except Exception as exc:
        issues.append(f"5-step loss check failed: {exc}")
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
    run_id:      str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    val_kp_df:    pd.DataFrame,
    logger,
    device:       torch.device,
):
    ensure_dirs()
    ckpt_dir = CFG.CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model     = build_model().to(device)
    criterion = nn.MSELoss()

    # ── Phase 1 setup: freeze backbone ───────────────────────────────────────
    freeze_backbone(model)
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WD,
    )
    scheduler = build_scheduler(optimiser, 0, MAX_EPOCHS)

    start_epoch   = 1
    best_val_loss = float("inf")
    patience_ctr  = 0
    phase         = 1

    # ── Resume if latest checkpoint exists ───────────────────────────────────
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
                # Restore correct phase
                if start_epoch > WARMUP_EPOCHS:
                    unfreeze_all(model)
                    phase = 2
            else:
                logger.info(
                    f"  latest.pt belongs to run {saved_run}, not {run_id} "
                    f"— starting fresh."
                )
        except Exception as exc:
            logger.warning(f"  Could not load latest checkpoint: {exc}. Starting fresh.")

    logger.info(f"\n  Starting training from epoch {start_epoch}")
    logger.info(f"  Device  : {device}")
    logger.info(f"  Phase   : {phase}")
    logger.info(f"  Early stop   : patience={EARLY_STOP_PAT}, delta={EARLY_STOP_DELTA} (no hard epoch cap)")
    logger.info(f"  Safety cap   : {MAX_EPOCHS} epochs")

    log_event(run_id, SCRIPT_NAME, "training_start",
              start_epoch=start_epoch, device=str(device),
              safety_cap=MAX_EPOCHS)

    train_losses = []
    val_losses   = []
    lr_history   = []
    best_epoch   = -1
    train_start  = time.time()

    epoch = start_epoch
    while True:
        epoch_start = time.time()

        # ── Phase transition at epoch 6 ───────────────────────────────────────
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            logger.info(
                f"\n  ── Phase 2: unfreezing all layers, lr → {LR_FINETUNE} ──"
            )
            unfreeze_all(model)
            optimiser = torch.optim.AdamW(
                model.parameters(), lr=LR_FINETUNE, weight_decay=WD
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=200,   # large value — LR decays slowly, early stop governs end
                eta_min=1e-7,
            )
            phase = 2
            log_event(run_id, SCRIPT_NAME, "phase2_start", epoch=epoch)

        # ── Train one epoch ───────────────────────────────────────────────────
        model.train()
        epoch_losses = []

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimiser.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, targets)
            loss.backward()
            optimiser.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # ── Gradient norm (all params with grad) ──────────────────────────────
        grad_norm = float(sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5)

        # ── Validate ──────────────────────────────────────────────────────────
        val_px_err, per_level = val_pixel_error(
            model, val_loader, device, val_kp_df
        )
        val_losses.append(val_px_err)

        current_lr = optimiser.param_groups[0]["lr"]
        lr_history.append(current_lr)

        # ── Timing and ETA ────────────────────────────────────────────────────
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
            f"train_mse={train_loss:.6f} | "
            f"val_px_err={val_px_err:.2f}px | "
            f"lr={current_lr:.2e} | "
            f"grad_norm={grad_norm:.3f} | "
            f"time={epoch_mins}m{epoch_s:02d}s | "
            f"patience={patience_ctr}/{EARLY_STOP_PAT} | "
            f"ETA≈{eta_hrs}h{eta_mins:02d}m"
        )
        logger.info(
            "    per-level: " +
            "  ".join(f"{lvl}={per_level[lvl]:.1f}px" for lvl in LEVELS)
        )

        log_event(run_id, SCRIPT_NAME, "epoch_end",
                  epoch=epoch, train_mse=train_loss,
                  val_px_err=val_px_err, per_level=per_level,
                  lr=current_lr, grad_norm=grad_norm,
                  epoch_secs=round(epoch_secs, 1), phase=phase)

        # ── Checkpointing ─────────────────────────────────────────────────────
        # Always save latest
        save_checkpoint(
            latest_path, model, optimiser, scheduler,
            epoch, best_val_loss, patience_ctr, run_id,
        )

        # Periodic checkpoint every CKPT_EVERY epochs
        if epoch % CKPT_EVERY == 0:
            periodic = ckpt_dir / f"{MODEL_NAME}_epoch{epoch}.pt"
            save_checkpoint(
                periodic, model, optimiser, scheduler,
                epoch, best_val_loss, patience_ctr, run_id,
            )
            logger.info(f"  Periodic checkpoint saved → {periodic.name}")
            log_event(run_id, SCRIPT_NAME, "checkpoint_periodic",
                      epoch=epoch, path=str(periodic))

        # Best checkpoint — lower val_px_err is better
        if val_px_err < best_val_loss - EARLY_STOP_DELTA:
            best_val_loss = val_px_err
            best_epoch    = epoch
            patience_ctr  = 0
            best_path     = ckpt_dir / f"{MODEL_NAME}_best.pt"
            save_checkpoint(
                best_path, model, optimiser, scheduler,
                epoch, best_val_loss, patience_ctr, run_id,
            )
            logger.info(
                f"  ✓ New best val_px_err={best_val_loss:.2f}px — "
                f"saved {best_path.name}"
            )
            log_event(run_id, SCRIPT_NAME, "checkpoint_best",
                      epoch=epoch, best_val_px_err=best_val_loss)
        else:
            patience_ctr += 1
            logger.info(
                f"  No improvement. Patience {patience_ctr}/{EARLY_STOP_PAT}"
            )

        # Early stopping
        if patience_ctr >= EARLY_STOP_PAT:
            logger.info(
                f"\n  Early stopping triggered at epoch {epoch}. "
                f"Best val_px_err={best_val_loss:.2f}px  "
                f"(best epoch={best_epoch})"
            )
            log_event(run_id, SCRIPT_NAME, "early_stop",
                      epoch=epoch, best_val_px_err=best_val_loss,
                      best_epoch=best_epoch)
            break

        # Safety cap — should never trigger in normal training
        if epoch >= MAX_EPOCHS:
            logger.info(
                f"\n  Safety cap reached at epoch {epoch}. "
                f"Best val_px_err={best_val_loss:.2f}px"
            )
            log_event(run_id, SCRIPT_NAME, "safety_cap_reached",
                      epoch=epoch, best_val_px_err=best_val_loss)
            break

        epoch += 1

    # ── Post-training visualisations ─────────────────────────────────────────
    total_mins = int((time.time() - train_start) / 60)
    logger.info(
        f"\n  Total training time : {total_mins // 60}h {total_mins % 60}m"
    )
    logger.info(f"  Best checkpoint     : epoch {best_epoch}  "
                f"(val_px_err={best_val_loss:.2f}px)")

    _plot_curves(train_losses, val_losses, lr_history, run_id, start_epoch)

    return best_val_loss, len(train_losses), best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_curves(
    train_losses: List[float],
    val_losses:   List[float],
    lr_history:   List[float],
    run_id:       str,
    start_epoch:  int,
):
    """Plot train MSE, val pixel error, and LR decay curves."""
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model 1 training curves — {run_id}", fontsize=11)

    ax1.plot(epochs, train_losses, color="#3498DB", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train MSE loss")
    ax1.set_title("Train MSE loss"); ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(epochs, val_losses, color="#E74C3C", linewidth=1.5)
    ax2.axhline(y=20, color="gray", linewidth=0.8, linestyle="--",
                label="20px threshold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val mean pixel error (px)")
    ax2.set_title("Val mean pixel error"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    if lr_history:
        ax3.plot(epochs, lr_history, color="#2ECC71", linewidth=1.5)
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning rate")
        ax3.set_title("Learning rate schedule")
        ax3.set_yscale("log"); ax3.grid(alpha=0.3)
        ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_model1_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, smoke_only: bool = False):
    run_id     = run_id or make_run_id("m4m1")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, SCRIPT_NAME)

    # ── Device setup ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # faster for fixed input sizes
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_str = f"cuda ({gpu_name}, {gpu_mem:.1f}GB)"
    else:
        device = torch.device("cpu")
        device_str = "cpu"

    logger.info("=" * 65)
    logger.info(f"Module 4 — Model 1 (Keypoint Regression) | run_id={run_id}")
    logger.info(f"  DICOM_ROOT : {CFG.DICOM_ROOT}")
    logger.info(f"  CKPT_DIR   : {CFG.CKPT_DIR}")
    logger.info(f"  Device     : {device_str}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, SCRIPT_NAME, "script_start",
              device=device_str,
              input_size=INPUT_SIZE,
              safety_cap=MAX_EPOCHS)

    ensure_dirs()
    CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds()

    try:
        # ── Load master index and split CSVs ──────────────────────────────────
        master     = pd.read_csv(CFG.MASTER_INDEX_PATH)
        train_df   = pd.read_csv(CFG.SPLIT_TRAIN)
        val_df     = pd.read_csv(CFG.SPLIT_VAL)

        logger.info(f"Loaded master index: {len(master):,} rows")
        logger.info(f"Train studies: {train_df['study_id'].nunique():,}")
        logger.info(f"Val   studies: {val_df['study_id'].nunique():,}\n")

        # ── Build keypoint tables ─────────────────────────────────────────────
        logger.info("── Building keypoint tables ────────────────────────────")
        train_kp, train_dropped = build_keypoint_table(master, train_df)
        val_kp,   val_dropped   = build_keypoint_table(master, val_df)

        logger.info(
            f"  Train keypoint table: {len(train_kp):,} studies "
            f"({len(train_dropped)} dropped)"
        )
        logger.info(
            f"  Val   keypoint table: {len(val_kp):,} studies "
            f"({len(val_dropped)} dropped)"
        )

        if train_dropped:
            for sid, reason in train_dropped[:5]:
                logger.warning(f"    Dropped train study {sid}: {reason}")
            if len(train_dropped) > 5:
                logger.warning(f"    … and {len(train_dropped)-5} more")

        log_event(run_id, SCRIPT_NAME, "keypoint_tables_built",
                  n_train=len(train_kp), n_val=len(val_kp),
                  n_train_dropped=len(train_dropped),
                  n_val_dropped=len(val_dropped))

        if len(train_kp) == 0:
            raise RuntimeError("Train keypoint table is empty — cannot train.")

        # ── Build datasets and loaders ────────────────────────────────────────
        logger.info("\n── Building datasets ───────────────────────────────────")
        train_ds = KeypointDataset(train_kp, augment=True)
        val_ds   = KeypointDataset(val_kp,   augment=False)

        # num_workers=0 on Windows to avoid multiprocessing issues with DICOM
        train_loader = DataLoader(
            train_ds, batch_size=CFG.BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG.BATCH_SIZE,
            shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
        )

        logger.info(f"  Train samples : {len(train_ds):,}")
        logger.info(f"  Val   samples : {len(val_ds):,}")
        logger.info(f"  Batch size    : {CFG.BATCH_SIZE}")

        # ── Build model ───────────────────────────────────────────────────────
        logger.info("\n── Building model ──────────────────────────────────────")
        model = build_model().to(device)
        freeze_backbone(model)
        logger.info(f"  Architecture  : ResNet-18 → Linear(512, {N_OUTPUTS})")
        logger.info(f"  Pretrained    : ImageNet")
        logger.info(f"  Phase 1       : backbone frozen, head only")

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total params  : {total_params:,}")
        logger.info(f"  Trainable now : {trainable_params:,}")

        log_event(run_id, SCRIPT_NAME, "model_built",
                  total_params=total_params,
                  trainable_params=trainable_params)

        # ── Smoke test ────────────────────────────────────────────────────────
        smoke_ok = smoke_test(model, train_loader, device, logger, run_id)
        if not smoke_ok:
            raise RuntimeError(
                "Smoke test failed — fix issues before full training."
            )

        if smoke_only:
            logger.info("--smoke-only: stopping after smoke test.")
            return

        # Reset model after smoke test steps
        model = build_model().to(device)
        freeze_backbone(model)
        set_seeds()
        logger.info("  Model reset after smoke test. Seeds restored.\n")

        # ── Full training ─────────────────────────────────────────────────────
        logger.info("── Training ────────────────────────────────────────────")
        best_val, n_epochs, best_epoch = train(
            run_id, train_loader, val_loader, val_kp,
            logger, device,
        )

        logger.info(f"\n  Training complete.")
        logger.info(f"  Best val pixel error : {best_val:.2f}px")
        logger.info(f"  Best epoch           : {best_epoch}")
        logger.info(f"  Epochs trained       : {n_epochs}")

        # Go/no-go guidance
        if best_val < 20.0:
            verdict = "GOOD — proceed to eval script, then Models 2/3/4"
        elif best_val < 35.0:
            verdict = "BORDERLINE — review keypoint visualisations before proceeding"
        else:
            verdict = "POOR — retrain before proceeding to Models 2/3/4"
        logger.info(f"  Go/no-go verdict     : {verdict}")

        log_event(run_id, SCRIPT_NAME, "training_complete",
                  best_val_px_err=best_val, n_epochs=n_epochs,
                  best_epoch=best_epoch, verdict=verdict)

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
        notes=f"Keypoint regression. {verdict}",
    )
    log_event(run_id, SCRIPT_NAME, "script_complete", end_time=end_time)
    logger.info(f"\nModule 4 Model 1 complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Module 4 — Model 1: ResNet-18 Keypoint Regression"
    )
    parser.add_argument("--run-id",     default=None)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run smoke test then exit")
    args = parser.parse_args()
    run(run_id=args.run_id, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()