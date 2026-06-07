"""
src/module4_classifiers_eval.py
================================
Evaluation script for Models 2, 3, and 4 (severity classifiers).

Loads the best checkpoint for a given model, runs inference on the val split,
and produces:
  1. Quantitative table — weighted log-loss and per-class accuracy overall
     and broken down per condition and level
  2. Prediction distribution plot — predicted vs true label counts per class
  3. Confidence calibration plot — mean confidence vs actual accuracy
  4. 10 sample patches with true label and predicted probability bars
  5. run_log.jsonl entry and run_summary.csv row

Usage
-----
  # Evaluate Model 2 (SCS)
  python -m src.module4_classifiers_eval --model 2

  # Evaluate Model 3 (NFN)
  python -m src.module4_classifiers_eval --model 3

  # Evaluate Model 4 (SS)
  python -m src.module4_classifiers_eval --model 4

  # With custom run-id
  python -m src.module4_classifiers_eval --model 2 --run-id my_eval_001
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary
from src.module3 import RSNADataset, build_val_transforms, resolve_patch_paths

SCRIPT_NAME = "module4_classifiers_eval"

# Model registry — maps model number to (module, conditions, series_type)
_MODEL_REGISTRY = {
    "2": {
        "model_name":  "model2",
        "conditions":  ["spinal_canal_stenosis"],
        "series_type": "Sagittal T2/STIR",
        "label":       "SCS",
    },
    "3L": {
        "model_name":  "model3_left",
        "conditions":  ["left_neural_foraminal_narrowing"],
        "series_type": "Sagittal T1",
        "label":       "NFN_Left",
    },
    "3R": {
        "model_name":  "model3_right",
        "conditions":  ["right_neural_foraminal_narrowing"],
        "series_type": "Sagittal T1",
        "label":       "NFN_Right",
    },
    "4": {
        "model_name":  "model4",
        "conditions":  [
            "left_subarticular_stenosis",
            "right_subarticular_stenosis",
        ],
        "series_type": "Axial T2",
        "label":       "SS",
    },
}

SEVERITY_NAMES = ["Normal/Mild", "Moderate", "Severe"]
SEVERITY_COLOURS = ["#2ECC71", "#F39C12", "#E74C3C"]


# ─────────────────────────────────────────────────────────────────────────────
# Model builder (ResNet-34 → 3 classes, identical for models 2/3/4)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model() -> torch.nn.Module:
    from torchvision import models
    import torch.nn as nn
    model    = models.resnet34(weights=None)
    in_feat  = model.fc.in_features
    model.fc = nn.Linear(in_feat, 3)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def weighted_log_loss(
    probs:   np.ndarray,
    labels:  np.ndarray,
    weights: np.ndarray,
    eps:     float = 1e-7,
) -> float:
    probs  = np.clip(probs, eps, 1 - eps)
    n      = len(labels)
    w_sum  = 0.0
    loss   = 0.0
    for i in range(n):
        c      = int(labels[i])
        w      = weights[c]
        loss  += w * np.log(probs[i, c])
        w_sum += w
    return float(-loss / w_sum) if w_sum > 0 else float("inf")


def compute_all_metrics(
    all_probs:  np.ndarray,   # (N, 3)
    all_labels: np.ndarray,   # (N,)
    all_conds:  List[str],
    all_levels: List[str],
    logger,
    run_id: str,
) -> Dict:
    sev_weights = np.array([CFG.SEVERITY_WEIGHTS[i] for i in range(3)])
    preds       = np.argmax(all_probs, axis=1)
    metrics     = {}

    # Overall
    overall_loss = weighted_log_loss(all_probs, all_labels, sev_weights)
    overall_acc  = float((preds == all_labels).mean() * 100)
    per_class_acc = {}
    for c in range(3):
        mask = (all_labels == c)
        if mask.sum() > 0:
            per_class_acc[c] = float((preds[mask] == c).mean() * 100)
        else:
            per_class_acc[c] = float("nan")

    metrics["overall"] = {
        "weighted_log_loss": overall_loss,
        "accuracy":          overall_acc,
        "per_class_acc":     per_class_acc,
        "n_samples":         len(all_labels),
    }

    logger.info("\n── Overall metrics ─────────────────────────────────────────")
    logger.info(f"  Weighted log-loss : {overall_loss:.4f}")
    logger.info(f"  Overall accuracy  : {overall_acc:.1f}%")
    for c in range(3):
        n_c = int((all_labels == c).sum())
        logger.info(
            f"  {SEVERITY_NAMES[c]:12s}: acc={per_class_acc[c]:.1f}%  "
            f"(n={n_c:,})"
        )

    # Per condition
    logger.info("\n── Per-condition weighted log-loss ─────────────────────────")
    logger.info(f"  {'Condition':<45}  {'wLogLoss':>10}  {'Acc%':>6}  {'N':>6}")
    logger.info("  " + "-" * 72)

    unique_conds = sorted(set(all_conds))
    cond_array   = np.array(all_conds)
    for cond in unique_conds:
        mask = (cond_array == cond)
        if mask.sum() == 0:
            continue
        c_loss = weighted_log_loss(
            all_probs[mask], all_labels[mask], sev_weights
        )
        c_acc  = float((preds[mask] == all_labels[mask]).mean() * 100)
        metrics[cond] = {"weighted_log_loss": c_loss, "accuracy": c_acc,
                         "n_samples": int(mask.sum())}
        logger.info(
            f"  {cond:<45}  {c_loss:>10.4f}  {c_acc:>5.1f}%  "
            f"{int(mask.sum()):>6,}"
        )

    log_event(run_id, SCRIPT_NAME, "eval_metrics", metrics=metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model:   torch.nn.Module,
    val_df:  pd.DataFrame,
    conditions: List[str],
    device:  torch.device,
    logger,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Run model on all val patches for the given conditions.
    Returns (probs, labels, patch_paths, conditions_list, levels_list).
    """
    # Build combined val dataset across all conditions
    parts = []
    for cond in conditions:
        ds = RSNADataset(val_df, transform=build_val_transforms(),
                         condition_filter=cond)
        parts.append(ds)

    combined_df = pd.concat([p.df for p in parts], ignore_index=True)
    val_ds      = RSNADataset(combined_df, transform=build_val_transforms())

    loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

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

    all_conds  = combined_df["condition"].tolist()
    all_levels = combined_df["level"].tolist()
    all_paths  = combined_df["patch_path"].tolist()

    logger.info(f"  Inference complete — {len(all_labels):,} val samples")
    return all_probs, all_labels, all_paths, all_conds, all_levels


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_distribution(
    all_probs:  np.ndarray,
    all_labels: np.ndarray,
    model_label: str,
    run_id:     str,
    logger,
):
    """Grouped bar chart: true vs predicted class counts."""
    preds = np.argmax(all_probs, axis=1)
    true_counts = [(all_labels == c).sum() for c in range(3)]
    pred_counts = [(preds == c).sum()       for c in range(3)]

    x   = np.arange(3)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, true_counts, w, label="True",      color="#3498DB", alpha=0.8)
    ax.bar(x + w/2, pred_counts, w, label="Predicted", color="#E74C3C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_NAMES)
    ax.set_ylabel("Count")
    ax.set_title(f"Model {model_label} — True vs Predicted class distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_{model_label}_pred_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Prediction distribution saved → {out.name}")


def plot_sample_patches(
    all_probs:  np.ndarray,
    all_labels: np.ndarray,
    all_paths:  List[str],
    model_label: str,
    run_id:     str,
    logger,
    n: int = 10,
):
    """
    Display n sample patches with true label and predicted probability bars.
    Samples are chosen to include at least one of each severity class.
    """
    chosen_indices = []
    rng = np.random.default_rng(CFG.RANDOM_SEED)

    # At least 1 from each class, then fill randomly
    for c in range(3):
        idxs = np.where(all_labels == c)[0]
        if len(idxs) > 0:
            chosen_indices.append(int(rng.choice(idxs)))

    remaining = n - len(chosen_indices)
    if remaining > 0:
        pool = [i for i in range(len(all_labels))
                if i not in set(chosen_indices)]
        chosen_indices += rng.choice(pool, size=min(remaining, len(pool)),
                                     replace=False).tolist()

    chosen_indices = chosen_indices[:n]

    mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]

    cols = 5
    rows = (len(chosen_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols,
                             figsize=(cols * 3, rows * 5))
    axes = axes.reshape(rows * 2, cols)
    fig.suptitle(f"Model {model_label} — sample predictions", fontsize=10)

    for col_i, idx in enumerate(chosen_indices):
        row_img  = (col_i // cols) * 2
        row_bar  = row_img + 1
        col_pos  = col_i % cols

        # Patch image
        patch  = np.load(all_paths[idx]).astype(np.float32)
        disp   = np.clip(patch * std + mean, 0, 1)[1]   # annotated slice
        true_c = int(all_labels[idx])
        pred_c = int(np.argmax(all_probs[idx]))

        ax_img = axes[row_img, col_pos]
        ax_img.imshow(disp, cmap="gray")
        border_col = "#2ECC71" if true_c == pred_c else "#E74C3C"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.5)
        ax_img.set_title(
            f"True: {SEVERITY_NAMES[true_c]}\nPred: {SEVERITY_NAMES[pred_c]}",
            fontsize=7,
            color="black" if true_c == pred_c else "#A32D2D",
        )
        ax_img.axis("off")

        # Probability bar chart
        ax_bar = axes[row_bar, col_pos]
        bars   = ax_bar.bar(
            range(3), all_probs[idx],
            color=SEVERITY_COLOURS, alpha=0.8,
        )
        ax_bar.set_xticks(range(3))
        ax_bar.set_xticklabels(["N/M", "Mod", "Sev"], fontsize=7)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Prob", fontsize=7)
        ax_bar.axhline(y=0.5, color="gray", linewidth=0.6, linestyle="--")
        ax_bar.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for i in range(len(chosen_indices), rows * cols):
        r = (i // cols) * 2
        c = i % cols
        axes[r,     c].axis("off")
        axes[r + 1, c].axis("off")

    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_{model_label}_sample_predictions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Sample predictions saved → {out.name}")


def plot_per_condition_loss(
    metrics:     Dict,
    model_label: str,
    run_id:      str,
    logger,
):
    """Horizontal bar chart of per-condition weighted log-loss."""
    cond_names  = [k for k in metrics if k != "overall"]
    cond_losses = [metrics[k]["weighted_log_loss"] for k in cond_names]

    # Sort ascending (worst at bottom)
    order       = np.argsort(cond_losses)[::-1]
    cond_names  = [cond_names[i].replace("_", " ")  for i in order]
    cond_losses = [cond_losses[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cond_names) * 0.5)))
    colours = ["#E74C3C" if l > 0.5 else "#F39C12" if l > 0.3 else "#2ECC71"
               for l in cond_losses]
    ax.barh(cond_names, cond_losses, color=colours, alpha=0.8)
    ax.set_xlabel("Weighted log-loss")
    ax.set_title(f"Model {model_label} — per-condition weighted log-loss")
    ax.axvline(x=0.3, color="gray", linewidth=0.8, linestyle="--",
               label="0.30 reference")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_{model_label}_per_condition_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Per-condition loss plot saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(model_num: str, run_id: str = None, n_vis: int = 10):
    if model_num not in _MODEL_REGISTRY:
        raise ValueError(f"model_num must be one of 2, 3L, 3R, 4. Got {model_num}.")

    cfg          = _MODEL_REGISTRY[model_num]
    model_name   = cfg["model_name"]
    conditions   = cfg["conditions"]
    model_label  = cfg["label"]

    run_id = run_id or make_run_id(f"m4m{model_num.replace('3L','3left').replace('3R','3right')}eval")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, SCRIPT_NAME)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 65)
    logger.info(
        f"Model {model_num} ({model_label}) Evaluation | run_id={run_id}"
    )
    logger.info(f"  Conditions : {conditions}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, SCRIPT_NAME, "script_start",
              model_num=model_num, model_label=model_label,
              device=str(device))

    ensure_dirs()

    try:
        # Load checkpoint
        best_path = CFG.CKPT_DIR / f"{model_name}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found: {best_path}\n"
                f"Run the appropriate training script for {model_label} first."
            )
        logger.info(f"  Loading checkpoint: {best_path}")

        model = _build_model().to(device)
        ckpt  = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"  Checkpoint from epoch {ckpt['epoch']}  "
            f"(best val_wlogloss={ckpt['best_val_loss']:.4f})"
        )
        log_event(run_id, SCRIPT_NAME, "checkpoint_loaded",
                  epoch=ckpt["epoch"], best_val_loss=ckpt["best_val_loss"])

        # Load val split
        logger.info("\n── Loading val split ───────────────────────────────────")
        val_df = resolve_patch_paths(pd.read_csv(CFG.SPLIT_VAL), "val")

        # Run inference
        logger.info("\n── Running inference on val split ──────────────────────")
        all_probs, all_labels, all_paths, all_conds, all_levels = run_inference(
            model, val_df, conditions, device, logger
        )

        # Compute metrics
        metrics = compute_all_metrics(
            all_probs, all_labels, all_conds, all_levels, logger, run_id
        )
        overall = metrics["overall"]

        # Summary
        logger.info("\n── Summary ─────────────────────────────────────────────")
        logger.info(f"  Model                : {model_label} (Model {model_num})")
        logger.info(f"  Val samples          : {overall['n_samples']:,}")
        logger.info(f"  Weighted log-loss    : {overall['weighted_log_loss']:.4f}")
        logger.info(f"  Overall accuracy     : {overall['accuracy']:.1f}%")
        for c in range(3):
            logger.info(
                f"  {SEVERITY_NAMES[c]:12s} accuracy: "
                f"{overall['per_class_acc'][c]:.1f}%"
            )

        # Visualisations
        logger.info("\n── Generating visualisations ───────────────────────────")
        plot_prediction_distribution(
            all_probs, all_labels, model_label, run_id, logger
        )
        plot_sample_patches(
            all_probs, all_labels, all_paths,
            model_label, run_id, logger, n=n_vis,
        )
        plot_per_condition_loss(metrics, model_label, run_id, logger)

        # Save results CSV
        results_df = pd.DataFrame({
            "condition":   all_conds,
            "level":       all_levels,
            "true_label":  all_labels,
            "pred_label":  np.argmax(all_probs, axis=1),
            "prob_nm":     all_probs[:, 0],
            "prob_mod":    all_probs[:, 1],
            "prob_sev":    all_probs[:, 2],
        })
        results_path = CFG.OUTPUT_DIR / f"{run_id}_{model_label}_eval_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n  Results CSV saved → {results_path.name}")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, SCRIPT_NAME, "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name=f"{model_name}_eval",
        start_time=start_time,
        end_time=end_time,
        best_val_loss=overall["weighted_log_loss"],
        notes=(
            f"Model {model_num} ({model_label}) eval. "
            f"wLogLoss={overall['weighted_log_loss']:.4f}  "
            f"acc={overall['accuracy']:.1f}%"
        ),
    )
    log_event(run_id, SCRIPT_NAME, "script_complete", end_time=end_time)
    logger.info(f"\nModel {model_num} ({model_label}) evaluation complete.")
    logger.info(f"Visualisations in: {CFG.VIS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classifier models 2, 3L, 3R, or 4"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["2", "3L", "3R", "4"],
        help="Which model to evaluate: 2=SCS, 3L=NFN Left, 3R=NFN Right, 4=SS",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-vis",  type=int, default=10,
                        help="Number of sample patches to visualise (default 10)")
    args = parser.parse_args()
    run(model_num=args.model, run_id=args.run_id, n_vis=args.n_vis)

if __name__ == "__main__":
    main()
