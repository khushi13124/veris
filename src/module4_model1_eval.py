"""
src/module4_model1_eval.py
==========================
Model 1 Evaluation — Keypoint Regression

Loads the best Model 1 checkpoint and runs inference on the val split.
(Test split stays locked until final evaluation in Module 5.)

Outputs
-------
  1. Per-study keypoint visualisations (predicted vs ground truth)
     overlaid on middle Sagittal T2 slice — saved to CFG.VIS_DIR
  2. Quantitative summary:
       - Mean pixel error overall and per level
       - % predictions within 10 / 20 / 30 px of ground truth
       - Per-level breakdown to identify problem levels
  3. run_log.jsonl entry and run_summary.csv row

Go / No-Go thresholds
---------------------
  Mean pixel error < 20 px  → GOOD   — proceed to Models 2/3/4
  Mean pixel error 20–35 px → BORDERLINE — review visuals, decide
  Mean pixel error > 35 px  → POOR   — retrain Model 1

Run
---
  python -m src.module4_model1_eval
  python -m src.module4_model1_eval --run-id my_run_001
  python -m src.module4_model1_eval --n-vis 20
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary
from src.module4_model1 import (
    INPUT_SIZE,
    LEVELS,
    MODEL_NAME,
    N_OUTPUTS,
    KeypointDataset,
    build_keypoint_table,
    build_model,
    load_middle_slice,
    prepare_slice,
    set_seeds,
)

SCRIPT_NAME = "module4_model1_eval"

# Colour per disc level for visualisations
_LEVEL_COLOURS = {
    "l1_l2": "#E74C3C",
    "l2_l3": "#F39C12",
    "l3_l4": "#2ECC71",
    "l4_l5": "#3498DB",
    "l5_s1": "#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model:   torch.nn.Module,
    kp_df:   pd.DataFrame,
    device:  torch.device,
) -> pd.DataFrame:
    """
    Run model on every study in kp_df.
    Returns kp_df with additional columns:
        pred_x_l1_l2, pred_y_l1_l2, ... pred_x_l5_s1, pred_y_l5_s1
    and original image dims:
        img_H, img_W
    """
    model.eval()
    ds      = KeypointDataset(kp_df, augment=False)
    results = []

    with torch.no_grad():
        for idx in range(len(ds)):
            img_tensor, _ = ds[idx]
            img_tensor    = img_tensor.unsqueeze(0).to(device)
            pred_norm     = model(img_tensor).cpu().numpy()[0]  # (10,)

            row        = kp_df.iloc[idx]
            series_dir = (
                CFG.DICOM_ROOT
                / str(int(row["study_id"]))
                / str(int(row["series_id"]))
            )
            result = load_middle_slice(series_dir)
            H, W   = (result[1], result[2]) if result else (INPUT_SIZE, INPUT_SIZE)

            entry = {
                "study_id":  int(row["study_id"]),
                "series_id": int(row["series_id"]),
                "img_H":     H,
                "img_W":     W,
            }
            for j, lvl in enumerate(LEVELS):
                entry[f"pred_x_{lvl}"] = pred_norm[j*2]   * W
                entry[f"pred_y_{lvl}"] = pred_norm[j*2+1] * H
                entry[f"gt_x_{lvl}"]   = float(row[f"x_{lvl}"])
                entry[f"gt_y_{lvl}"]   = float(row[f"y_{lvl}"])

            results.append(entry)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    results_df: pd.DataFrame,
    logger,
    run_id: str,
) -> Dict:
    """
    Compute per-level and overall pixel errors.
    Also compute % within 10 / 20 / 30 px thresholds.
    """
    metrics = {}
    all_errors = []

    logger.info("\n── Quantitative metrics ────────────────────────────────────")
    logger.info(f"  {'Level':<12}  {'Mean (px)':>10}  {'Median (px)':>12}  "
                f"{'≤10px %':>8}  {'≤20px %':>8}  {'≤30px %':>8}")
    logger.info("  " + "-" * 68)

    for lvl in LEVELS:
        errors = np.sqrt(
            (results_df[f"pred_x_{lvl}"] - results_df[f"gt_x_{lvl}"])**2 +
            (results_df[f"pred_y_{lvl}"] - results_df[f"gt_y_{lvl}"])**2
        ).values

        all_errors.extend(errors.tolist())
        mean_err   = float(np.mean(errors))
        median_err = float(np.median(errors))
        pct_10     = float(np.mean(errors <= 10) * 100)
        pct_20     = float(np.mean(errors <= 20) * 100)
        pct_30     = float(np.mean(errors <= 30) * 100)

        metrics[lvl] = {
            "mean_px":   mean_err,
            "median_px": median_err,
            "pct_10px":  pct_10,
            "pct_20px":  pct_20,
            "pct_30px":  pct_30,
        }

        logger.info(
            f"  {lvl:<12}  {mean_err:>10.2f}  {median_err:>12.2f}  "
            f"{pct_10:>7.1f}%  {pct_20:>7.1f}%  {pct_30:>7.1f}%"
        )

    # Overall
    all_errors = np.array(all_errors)
    overall = {
        "mean_px":   float(np.mean(all_errors)),
        "median_px": float(np.median(all_errors)),
        "pct_10px":  float(np.mean(all_errors <= 10) * 100),
        "pct_20px":  float(np.mean(all_errors <= 20) * 100),
        "pct_30px":  float(np.mean(all_errors <= 30) * 100),
    }
    metrics["overall"] = overall

    logger.info("  " + "-" * 68)
    logger.info(
        f"  {'OVERALL':<12}  {overall['mean_px']:>10.2f}  "
        f"{overall['median_px']:>12.2f}  "
        f"{overall['pct_10px']:>7.1f}%  "
        f"{overall['pct_20px']:>7.1f}%  "
        f"{overall['pct_30px']:>7.1f}%"
    )

    log_event(run_id, SCRIPT_NAME, "eval_metrics", metrics=metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def visualise_predictions(
    results_df: pd.DataFrame,
    n:          int,
    run_id:     str,
    logger,
):
    """
    For n val studies, display:
      - Middle Sagittal T2 slice (greyscale)
      - Predicted keypoints (coloured dots + horizontal lines)
      - Ground truth keypoints (coloured crosses)
      - Legend
    Saved as a single multi-panel PNG per study.
    """
    sample = results_df.head(n)
    logger.info(f"\n── Keypoint visualisations ({n} studies) ───────────────────")

    for _, row in sample.iterrows():
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])

        series_dir = CFG.DICOM_ROOT / str(study_id) / str(series_id)
        result     = load_middle_slice(series_dir)
        if result is None:
            logger.warning(f"  Could not load slice for study {study_id}")
            continue

        windowed, H, W = result

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(windowed, cmap="gray", origin="upper",
                  aspect="auto", vmin=0, vmax=255)
        ax.set_title(
            f"Study {study_id} — keypoint predictions vs ground truth",
            fontsize=9, pad=6,
        )
        ax.axis("off")

        legend_handles = []
        for lvl in LEVELS:
            col   = _LEVEL_COLOURS[lvl]
            px    = float(row[f"pred_x_{lvl}"])
            py    = float(row[f"pred_y_{lvl}"])
            gx    = float(row[f"gt_x_{lvl}"])
            gy    = float(row[f"gt_y_{lvl}"])
            err   = float(np.sqrt((px-gx)**2 + (py-gy)**2))

            # Horizontal line at predicted y — full width
            ax.axhline(y=py, color=col, linewidth=0.8, alpha=0.6, linestyle="--")

            # Predicted keypoint — filled circle
            ax.plot(px, py, "o", color=col, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8)

            # Ground truth — cross
            ax.plot(gx, gy, "+", color=col, markersize=10,
                    markeredgewidth=2)

            # Error label
            ax.text(px + 6, py - 4, f"{err:.0f}px",
                    color=col, fontsize=6.5,
                    bbox=dict(facecolor="black", alpha=0.35,
                              pad=1, edgecolor="none"))

            legend_handles.append(mpatches.Patch(
                color=col,
                label=f"{lvl}  pred=● gt=✚  err={err:.0f}px",
            ))

        ax.legend(
            handles=legend_handles,
            loc="lower right", fontsize=6.5,
            framealpha=0.65, edgecolor="none",
        )

        out = CFG.VIS_DIR / f"{run_id}_model1_kp_{study_id}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"  Saved {n} keypoint visualisation(s) to {CFG.VIS_DIR}")


def plot_error_distribution(
    results_df: pd.DataFrame,
    run_id:     str,
    logger,
):
    """
    Box plot of pixel error distribution per disc level.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    data   = []
    labels = []
    for lvl in LEVELS:
        errors = np.sqrt(
            (results_df[f"pred_x_{lvl}"] - results_df[f"gt_x_{lvl}"])**2 +
            (results_df[f"pred_y_{lvl}"] - results_df[f"gt_y_{lvl}"])**2
        ).values
        data.append(errors)
        labels.append(lvl)

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    colours = list(_LEVEL_COLOURS.values())
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax.set_ylabel("Pixel error (px)")
    ax.set_title(f"Keypoint error distribution per level — {run_id}")
    ax.axhline(y=20, color="gray", linewidth=0.8, linestyle="--",
               label="20 px threshold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    out = CFG.VIS_DIR / f"{run_id}_model1_error_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Error distribution plot saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, n_vis: int = 20):
    run_id     = run_id or make_run_id("m4m1eval")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, SCRIPT_NAME)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 65)
    logger.info(f"Model 1 Evaluation — Keypoint Regression | run_id={run_id}")
    logger.info(f"  Device   : {device}")
    logger.info(f"  n_vis    : {n_vis}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, SCRIPT_NAME, "script_start",
              device=str(device), n_vis=n_vis)

    ensure_dirs()
    set_seeds()

    try:
        # ── Check best checkpoint exists ──────────────────────────────────────
        best_path = CFG.CKPT_DIR / f"{MODEL_NAME}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found: {best_path}\n"
                "Run module4_model1.py first."
            )
        logger.info(f"  Loading checkpoint: {best_path}")

        # ── Load model ────────────────────────────────────────────────────────
        model = build_model().to(device)
        ckpt  = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"  Checkpoint from epoch {ckpt['epoch']}  "
            f"(best val_px_err={ckpt['best_val_loss']:.2f}px)"
        )
        log_event(run_id, SCRIPT_NAME, "checkpoint_loaded",
                  epoch=ckpt["epoch"],
                  best_val_loss=ckpt["best_val_loss"])

        # ── Build val keypoint table ──────────────────────────────────────────
        logger.info("\n── Building val keypoint table ─────────────────────────")
        master = pd.read_csv(CFG.MASTER_INDEX_PATH)
        val_df = pd.read_csv(CFG.SPLIT_VAL)

        val_kp, val_dropped = build_keypoint_table(master, val_df)
        logger.info(
            f"  Val studies: {len(val_kp):,} "
            f"({len(val_dropped)} dropped — no SCS Sagittal T2)"
        )

        if len(val_kp) == 0:
            raise RuntimeError("Val keypoint table is empty — cannot evaluate.")

        # ── Run inference ─────────────────────────────────────────────────────
        logger.info("\n── Running inference on val split ──────────────────────")
        logger.info(f"  Processing {len(val_kp):,} studies …")
        results_df = run_inference(model, val_kp, device)
        logger.info(f"  Inference complete.")

        # ── Compute metrics ───────────────────────────────────────────────────
        metrics = compute_metrics(results_df, logger, run_id)
        overall = metrics["overall"]

        # ── Go / No-go verdict ────────────────────────────────────────────────
        logger.info("\n── Go / No-Go Assessment ───────────────────────────────")
        mean_err = overall["mean_px"]

        if mean_err < 20.0:
            verdict = "GO"
            detail  = (
                f"Mean pixel error {mean_err:.2f}px < 20px threshold. "
                "Proceed to Models 2/3/4."
            )
        elif mean_err < 35.0:
            verdict = "BORDERLINE"
            detail  = (
                f"Mean pixel error {mean_err:.2f}px is between 20–35px. "
                "Review visualisations carefully before proceeding."
            )
        else:
            verdict = "NO-GO"
            detail  = (
                f"Mean pixel error {mean_err:.2f}px > 35px. "
                "Retrain Model 1 before proceeding."
            )

        logger.info(f"  Verdict  : {verdict}")
        logger.info(f"  Detail   : {detail}")
        logger.info(f"  ≤20px    : {overall['pct_20px']:.1f}% of predictions")
        logger.info(f"  ≤10px    : {overall['pct_10px']:.1f}% of predictions")

        log_event(run_id, SCRIPT_NAME, "eval_verdict",
                  verdict=verdict, mean_px_err=mean_err,
                  pct_10px=overall["pct_10px"],
                  pct_20px=overall["pct_20px"])

        # ── Visualisations ────────────────────────────────────────────────────
        logger.info("\n── Generating visualisations ───────────────────────────")
        visualise_predictions(results_df, n_vis, run_id, logger)
        plot_error_distribution(results_df, run_id, logger)

        # ── Save results CSV ──────────────────────────────────────────────────
        results_path = CFG.OUTPUT_DIR / f"{run_id}_model1_eval_results.csv"
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
        model_name=f"{MODEL_NAME}_eval",
        start_time=start_time,
        end_time=end_time,
        best_val_loss=metrics["overall"]["mean_px"],
        notes=f"Eval verdict: {verdict}. {detail}",
    )
    log_event(run_id, SCRIPT_NAME, "script_complete", end_time=end_time)
    logger.info(f"\nModel 1 evaluation complete. Verdict: {verdict}")
    logger.info(f"Visualisations in: {CFG.VIS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Model 1 Evaluation — Keypoint Regression"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-vis",  type=int, default=20,
                        help="Number of val studies to visualise (default 20)")
    args = parser.parse_args()
    run(run_id=args.run_id, n_vis=args.n_vis)


if __name__ == "__main__":
    main()
