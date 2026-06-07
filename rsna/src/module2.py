"""
src/module2.py
==============
Module 2: Image Processing, Normalisation and Binding DCM to Format

Steps (exactly as specified in the strategy document):
  1. Load and sort DICOM slices by InstanceNumber tag
  2. Rescaling  — RescaleSlope × pixel + RescaleIntercept (default 1, 0)
  3. Windowing  — WindowCenter/Width → uint8; fallback to 1st–99th percentile
  4. 3-slice stacking  — [inst-1, inst, inst+1], duplicate at boundaries
  5. ROI cropping  — 160×160 centred on (x,y) → resize to 128×128 bilinear
  6. ImageNet normalisation  — /255 → float32; mean/std per channel
  7. Save as .npy; update master index with patch_path column
  8. Smoke test on 2 studies with full per-stage value-range logging
  9. Visualisation  — pipeline panel + show_crops for all annotations on a slice

Run:
    python -m src.module2
    python -m src.module2 --smoke-only
    python -m src.module2 --run-id my_run_001
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import pydicom
from pydicom.multival import MultiValue

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary


# ─────────────────────────────────────────────────────────────────────────────
# Low-level DICOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sorted_series(series_dir: Path) -> List[pydicom.Dataset]:
    """
    Read all .dcm files in series_dir, sort ascending by InstanceNumber tag.
    Never relies on filename ordering.
    Returns a list of pydicom Dataset objects.
    """
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return []

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            datasets.append(ds)
        except Exception:
            continue

    # Sort by InstanceNumber tag; fall back to 0 if tag missing
    datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    return datasets


def _scalar(val):
    """Extract scalar from a DICOM MultiValue or return the value directly."""
    if isinstance(val, MultiValue):
        return float(val[0])
    return float(val)


def rescale_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """
    Apply: real = pixel_array × RescaleSlope + RescaleIntercept.
    Defaults to slope=1, intercept=0 if tags absent.
    Returns float32 array.
    """
    arr   = ds.pixel_array.astype(np.float32)
    slope = _scalar(ds.RescaleSlope)      if hasattr(ds, "RescaleSlope")      else 1.0
    inter = _scalar(ds.RescaleIntercept)  if hasattr(ds, "RescaleIntercept")  else 0.0
    return arr * slope + inter


def window_to_uint8(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Apply DICOM windowing to produce a uint8 image.

    Primary:  WindowCenter / WindowWidth tags (take first value if MultiValue).
    Fallback: clip to 1st–99th percentile of pixel values, scale to 0–255.
    Returns uint8 numpy array, same shape as arr.
    """
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = _scalar(ds.WindowCenter)
        ww = _scalar(ds.WindowWidth)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        # Fallback: percentile windowing (common in MRI where window tags absent)
        lo = float(np.percentile(arr, CFG.PERCENTILE_LOW))
        hi = float(np.percentile(arr, CFG.PERCENTILE_HIGH))

    # Clamp + scale to [0, 255]
    arr_clipped = np.clip(arr, lo, hi)
    if hi == lo:                        # degenerate case — all pixels identical
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr_clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Series-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_instance_map(
    datasets: List[pydicom.Dataset],
) -> Dict[int, pydicom.Dataset]:
    """Return {InstanceNumber: dataset} for fast O(1) lookup."""
    return {int(getattr(ds, "InstanceNumber", i)): ds
            for i, ds in enumerate(datasets)}


def get_three_slices(
    instance_map: Dict[int, pydicom.Dataset],
    target_inst:  int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """
    Return (ds_prev, ds_curr, ds_next) as windowed uint8 arrays.
    Boundary cases: duplicate the available neighbour.

    Returns
    -------
    (ch0, ch1, ch2, used_default_slope, used_fallback_window)
      ch0 = inst-1, ch1 = inst, ch2 = inst+1
    """
    sorted_keys = sorted(instance_map.keys())

    def _get_ds(inst_num: int) -> pydicom.Dataset:
        if inst_num in instance_map:
            return instance_map[inst_num]
        # Boundary: clamp to nearest available
        if inst_num < sorted_keys[0]:
            return instance_map[sorted_keys[0]]
        return instance_map[sorted_keys[-1]]

    used_default_slope    = False
    used_fallback_window  = False

    def _process(ds: pydicom.Dataset):
        nonlocal used_default_slope, used_fallback_window
        if not (hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept")):
            used_default_slope = True
        if not (hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth")):
            used_fallback_window = True
        rescaled = rescale_pixel_array(ds)
        windowed = window_to_uint8(rescaled, ds)
        return windowed

    ch0 = _process(_get_ds(target_inst - 1))
    ch1 = _process(_get_ds(target_inst))
    ch2 = _process(_get_ds(target_inst + 1))

    return ch0, ch1, ch2, used_default_slope, used_fallback_window


# ─────────────────────────────────────────────────────────────────────────────
# ROI cropping
# ─────────────────────────────────────────────────────────────────────────────

def crop_and_resize(
    channel: np.ndarray,
    cx: float,
    cy: float,
    crop_size: int = None,
    patch_size: int = None,
) -> np.ndarray:
    """
    Crop a square of crop_size × crop_size centred on (cx, cy).
    Pads with zeros if crop extends outside image boundaries.
    Resizes to patch_size × patch_size using bilinear interpolation.
    Returns uint8 array of shape (patch_size, patch_size).
    """
    crop_size  = crop_size  or CFG.CROP_SIZE_PX
    patch_size = patch_size or CFG.PATCH_SIZE_PX
    H, W       = channel.shape
    half       = crop_size // 2

    # Compute source and destination regions
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    # Allocate zero-padded canvas
    canvas = np.zeros((crop_size, crop_size), dtype=np.uint8)

    # Clamp to image boundaries
    src_x0 = max(0, x0);  src_y0 = max(0, y0)
    src_x1 = min(W, x1);  src_y1 = min(H, y1)

    dst_x0 = src_x0 - x0;  dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = channel[src_y0:src_y1, src_x0:src_x1]

    # Bilinear resize
    resized = cv2.resize(canvas, (patch_size, patch_size),
                         interpolation=cv2.INTER_LINEAR)
    return resized


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_patch(patch_uint8: np.ndarray) -> np.ndarray:
    """
    (3, H, W) uint8  →  (3, H, W) float32 with ImageNet normalisation.
    Step 1: / 255  → [0, 1]
    Step 2: (x - mean) / std  per channel
    """
    patch = patch_uint8.astype(np.float32) / 255.0
    mean  = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std   = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
    return (patch - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Per-annotation processing
# ─────────────────────────────────────────────────────────────────────────────

def process_annotation(
    row:          pd.Series,
    instance_map: Dict[int, pydicom.Dataset],
    logger,
    run_id:       str,
    verbose:      bool = False,
) -> Optional[Tuple[np.ndarray, List[np.ndarray], np.ndarray]]:
    """
    Process a single annotation row from the master index.

    Returns
    -------
    (final_patch, crops_uint8, ch1_windowed_full)
      final_patch    : (3, 128, 128) float32 normalised
      crops_uint8    : [ch0, ch1, ch2] each (160, 160) uint8, before resize
      ch1_windowed   : full windowed slice for the annotated channel (for vis)
    Returns None on failure.
    """
    try:
        inst = int(row["instance_number"])
        cx   = float(row["x"])
        cy   = float(row["y"])

        ch0, ch1, ch2, used_default_slope, used_fallback_window = \
            get_three_slices(instance_map, inst)

        if used_default_slope:
            log_event(run_id, "module2", "default_rescale_used",
                      study_id=int(row["study_id"]),
                      series_id=int(row["series_id"]),
                      instance=inst)
        if used_fallback_window:
            log_event(run_id, "module2", "fallback_windowing_used",
                      study_id=int(row["study_id"]),
                      series_id=int(row["series_id"]),
                      instance=inst)

        # Crop all three channels with same centre
        cr0 = crop_and_resize(ch0, cx, cy)
        cr1 = crop_and_resize(ch1, cx, cy)
        cr2 = crop_and_resize(ch2, cx, cy)

        # Stack → (3, 128, 128) uint8
        stack_uint8 = np.stack([cr0, cr1, cr2], axis=0)

        # Normalise → (3, 128, 128) float32
        final_patch = normalise_patch(stack_uint8)

        if verbose:
            ds_curr = instance_map.get(inst, next(iter(instance_map.values())))
            raw_arr = ds_curr.pixel_array.astype(np.float32)
            rescaled_arr = rescale_pixel_array(ds_curr)
            logger.info(
                f"    study={row['study_id']} cond={row['condition']} "
                f"lvl={row['level']} inst={inst}"
            )
            logger.info(f"      raw         min={raw_arr.min():.1f}  max={raw_arr.max():.1f}")
            logger.info(f"      rescaled    min={rescaled_arr.min():.1f}  max={rescaled_arr.max():.1f}")
            logger.info(f"      windowed    min={ch1.min()}  max={ch1.max()}")
            logger.info(f"      crop ch1    min={cr1.min()}  max={cr1.max()}")
            logger.info(f"      normalised  min={final_patch.min():.4f}  max={final_patch.max():.4f}")
            logger.info(f"      patch shape : {final_patch.shape}  dtype={final_patch.dtype}")

        return final_patch, [cr0, cr1, cr2], ch1

    except Exception as exc:
        log_event(run_id, "module2", "annotation_error",
                  study_id=int(row.get("study_id", -1)),
                  condition=str(row.get("condition", "")),
                  level=str(row.get("level", "")),
                  error=str(exc))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Patch filename
# ─────────────────────────────────────────────────────────────────────────────

def patch_filename(row: pd.Series) -> str:
    """
    {study_id}_{series_id}_{instance_number}_{condition}_{level}.npy
    Matches the naming convention specified in the strategy document.
    """
    return (
        f"{int(row['study_id'])}"
        f"_{int(row['series_id'])}"
        f"_{int(row['instance_number'])}"
        f"_{row['condition']}"
        f"_{row['level']}"
        f".npy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(master: pd.DataFrame, logger, run_id: str) -> bool:
    """
    Process exactly CFG.SMOKE_N_STUDIES studies end to end.
    Prints patch shapes and per-stage value ranges.
    Saves one pipeline visualisation and one show_crops image per study.
    """
    from src.visualise import show_crops, show_patch_pipeline  # training-only vis
    logger.info("── Smoke test ──────────────────────────────────────────────")
    n = CFG.SMOKE_N_STUDIES

    # Pick smoke studies from train split first — this guarantees that when
    # module 3 smoke test runs, it finds patches in the train split (which it
    # validates). Fall back to val only if train has fewer than n studies.
    # Never pick from test.
    if CFG.SPLIT_TRAIN.exists():
        train_ids = pd.read_csv(CFG.SPLIT_TRAIN)["study_id"].unique()
        smoke_ids = list(train_ids[:n])
        # Top up from val if needed
        if len(smoke_ids) < n and CFG.SPLIT_VAL.exists():
            val_ids = pd.read_csv(CFG.SPLIT_VAL)["study_id"].unique()
            extra   = [i for i in val_ids if i not in set(smoke_ids)]
            smoke_ids = smoke_ids + extra[: n - len(smoke_ids)]
        logger.info(f"  Smoke studies: {smoke_ids}  (from train split)")
    else:
        smoke_ids = list(master["study_id"].unique()[:n])
        logger.warning("  split_train.csv not found — picking from full master. "
                       "Run module1 first to guarantee test exclusion.")

    smoke_df = master[master["study_id"].isin(smoke_ids)]

    ok               = True
    smoke_patch_paths = {}          # populated as patches are saved
    for study_id, study_df in smoke_df.groupby("study_id"):
        logger.info(f"\n  Study {study_id} — {len(study_df)} annotations")

        for series_id, series_df in study_df.groupby("series_id"):
            series_dir = CFG.DICOM_ROOT / str(int(study_id)) / str(int(series_id))
            if not series_dir.exists():
                logger.warning(f"    Series dir missing: {series_dir}")
                ok = False
                continue

            datasets     = load_sorted_series(series_dir)
            instance_map = build_instance_map(datasets)
            logger.info(
                f"    series={int(series_id)}  "
                f"type={series_df['series_type'].iloc[0]}  "
                f"slices={len(datasets)}"
            )

            # Visualise all annotations on the first slice of this series
            if datasets:
                first_ds     = datasets[len(datasets) // 2]   # mid-slice for vis
                first_rescaled = rescale_pixel_array(first_ds)
                first_windowed = window_to_uint8(first_rescaled, first_ds)
                inst_num     = int(getattr(first_ds, "InstanceNumber", 0))

                # Gather annotations that fall on this instance
                vis_anns = []
                for _, r in series_df[
                    series_df["instance_number"] == inst_num
                ].iterrows():
                    vis_anns.append((
                        float(r["x"]), float(r["y"]),
                        r["condition"], r["level"],
                        int(r["severity_label"]),
                    ))

                if vis_anns:
                    crops_path = CFG.VIS_DIR / f"{run_id}_smoke_{int(study_id)}_{int(series_id)}_crops.png"
                    show_crops(first_windowed, vis_anns, crops_path,
                              title=f"Study {int(study_id)} | series {int(series_id)}")
                    logger.info(f"    show_crops saved → {crops_path.name}")

            # Process ALL annotations in this series, save .npy for each
            first_result = None
            for _, ann_row in series_df.iterrows():
                ann_result = process_annotation(
                    ann_row, instance_map, logger, run_id,
                    verbose=(ann_row.name == series_df.index[0])   # verbose only for first
                )
                if ann_result is None:
                    ok = False
                    continue

                ann_patch, _, _ = ann_result
                # Capture first row result here — reuse for pipeline vis, no second call
                if first_result is None:
                    first_result = ann_result
                    first_row    = ann_row
                fname = patch_filename(ann_row)
                fpath = CFG.PATCHES_DIR / fname
                np.save(str(fpath), ann_patch)
                smoke_patch_paths[
                    (int(ann_row["study_id"]), int(ann_row["series_id"]),
                     int(ann_row["instance_number"]),
                     ann_row["condition"], ann_row["level"])
                ] = str(fpath)

            if first_result is None:
                continue

            final_patch, crops_uint8, ch1_full = first_result

            # Save pipeline visualisation
            ds_curr = instance_map.get(
                int(first_row["instance_number"]),
                datasets[0] if datasets else None,
            )
            if ds_curr is not None:
                raw_arr      = ds_curr.pixel_array.astype(np.float32)
                rescaled_arr = rescale_pixel_array(ds_curr)
                windowed_arr = window_to_uint8(rescaled_arr, ds_curr)
                vis_path     = (
                    CFG.VIS_DIR
                    / f"{run_id}_smoke_{int(study_id)}_{int(series_id)}_pipeline.png"
                )
                show_patch_pipeline(
                    raw=raw_arr,
                    windowed=windowed_arr,
                    crops=crops_uint8,
                    final_patch=final_patch,
                    save_path=vis_path,
                    title=(
                        f"Study {study_id} | {first_row['condition']} "
                        f"{first_row['level']}"
                    ),
                )
                logger.info(f"    pipeline vis saved → {vis_path.name}")

    logger.info(f"  Smoke patches saved: {len(smoke_patch_paths)}")
    log_event(run_id, "module2", "smoke_patches_saved",
              n_patches=len(smoke_patch_paths))

    status = "PASSED" if ok else "FAILED — check warnings above"
    logger.info(f"\n  Smoke test {status}")
    log_event(run_id, "module2", "smoke_test_result",
              passed=ok, n_studies=int(n))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Full processing run
# ─────────────────────────────────────────────────────────────────────────────

def process_all(master: pd.DataFrame, logger, run_id: str) -> None:
    """
    Process every annotation in master for train and val splits only.
    Test split is intentionally excluded — patches for test studies are built
    at inference time in module 5, not here.
    Saves .npy patches to CFG.PATCHES_DIR.
    Does NOT write back to master_index.csv — that file is read-only after module 1.
    patch_path is resolved at load time in module 3 by matching filenames.
    """
    logger.info("── Processing all studies (train + val only) ───────────────")
    ensure_dirs()

    # Load test study IDs so we can skip them
    if CFG.SPLIT_TEST.exists():
        test_study_ids = set(pd.read_csv(CFG.SPLIT_TEST)["study_id"].unique())
        logger.info(f"  Skipping {len(test_study_ids):,} test studies "
                    f"(patches built at inference time)")
    else:
        test_study_ids = set()
        logger.warning("  split_test.csv not found — processing all studies")

    patch_paths   = {}     # (study_id, series_id, instance, condition, level) → path
    n_ok = n_err = n_skipped_test = 0
    default_slope_studies  = set()
    fallback_window_studies = set()

    # Group by (study_id, series_id) to load each series once
    groups = master.groupby(["study_id", "series_id"])
    total  = len(groups)

    for idx, ((study_id, series_id), group_df) in enumerate(groups, 1):

        # Skip test studies — patches built at inference time
        if study_id in test_study_ids:
            n_skipped_test += len(group_df)
            continue

        series_dir = CFG.DICOM_ROOT / str(int(study_id)) / str(int(series_id))

        if not series_dir.exists():
            logger.warning(f"  [{idx}/{total}] Missing series dir: {series_dir}")
            n_err += len(group_df)
            continue

        datasets     = load_sorted_series(series_dir)
        instance_map = build_instance_map(datasets)

        if not datasets:
            logger.warning(f"  [{idx}/{total}] No DICOMs in {series_dir}")
            n_err += len(group_df)
            continue

        if idx % 50 == 0 or idx == 1:
            logger.info(
                f"  [{idx}/{total}]  study={int(study_id)}  series={int(series_id)}  "
                f"slices={len(datasets)}  annotations={len(group_df)}"
            )

        for _, row in group_df.iterrows():
            result = process_annotation(row, instance_map, logger, run_id,
                                        verbose=False)
            if result is None:
                n_err += 1
                continue

            final_patch, _, _ = result

            # Save .npy — skip if file already exists (e.g. saved by smoke test)
            fname = patch_filename(row)
            fpath = CFG.PATCHES_DIR / fname
            if fpath.exists():
                n_ok += 1   # count it as done
                key = (
                    int(row["study_id"]), int(row["series_id"]),
                    int(row["instance_number"]),
                    row["condition"], row["level"],
                )
                patch_paths[key] = str(fpath)
                continue
            np.save(str(fpath), final_patch)

            key = (
                int(row["study_id"]),
                int(row["series_id"]),
                int(row["instance_number"]),
                row["condition"],
                row["level"],
            )
            patch_paths[key] = str(fpath)
            n_ok += 1

    logger.info(f"\n  Patches saved    : {n_ok:,}")
    logger.info(f"  Errors           : {n_err:,}")
    logger.info(f"  Skipped (test)   : {n_skipped_test:,}  (no patches by design)")
    logger.info(f"  Patches dir      : {CFG.PATCHES_DIR}")
    log_event(run_id, "module2", "processing_complete",
              n_ok=n_ok, n_err=n_err, n_skipped_test=n_skipped_test)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, smoke_only: bool = False):
    run_id     = run_id or make_run_id("m2")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, "module2")

    logger.info("=" * 65)
    logger.info(f"Module 2 — Image Processing & Normalisation | run_id={run_id}")
    logger.info(f"  DICOM_ROOT  : {CFG.DICOM_ROOT}")
    logger.info(f"  PATCHES_DIR : {CFG.PATCHES_DIR}")
    logger.info(f"  CROP_SIZE   : {CFG.CROP_SIZE_PX}px → {CFG.PATCH_SIZE_PX}px")
    logger.info("=" * 65 + "\n")

    log_event(run_id, "module2", "script_start",
              dicom_root=str(CFG.DICOM_ROOT),
              patches_dir=str(CFG.PATCHES_DIR),
              crop_size=CFG.CROP_SIZE_PX,
              patch_size=CFG.PATCH_SIZE_PX)

    ensure_dirs()

    # Load master index produced by module1
    if not CFG.MASTER_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Master index not found at {CFG.MASTER_INDEX_PATH}. "
            "Run module1 first."
        )
    master = pd.read_csv(CFG.MASTER_INDEX_PATH)
    logger.info(f"Loaded master index: {len(master):,} rows, "
                f"{master['study_id'].nunique():,} studies\n")

    try:
        # Smoke test (always run first)
        smoke_ok = smoke_test(master, logger, run_id)

        if smoke_only:
            logger.info("--smoke-only: stopping after smoke test.")
            return

        if not smoke_ok:
            logger.warning(
                "Smoke test had issues — continuing with full run. "
                "Investigate warnings above."
            )

        # Full processing — saves .npy files, does not modify master index
        process_all(master, logger, run_id)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, "module2", "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name="module2_image_processing",
        start_time=start_time,
        end_time=end_time,
        notes=f"Patches saved to {CFG.PATCHES_DIR}",
    )
    log_event(run_id, "module2", "script_complete", end_time=end_time)
    logger.info(f"\nModule 2 complete. Patches in: {CFG.PATCHES_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Module 2 — Image Processing & Normalisation"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run smoke test on 2 studies then exit")
    args = parser.parse_args()
    run(run_id=args.run_id, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()