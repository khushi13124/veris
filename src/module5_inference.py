"""
src/module5_inference.py
========================
Module 5 — End-to-end Inference Pipeline

Runs all four models on the test split and produces a competition submission CSV.

Pipeline (per study):
  Step 1  — Load Sag T2, Sag T1, Axial T2 series
  Step 2  — Model 1: predict keypoints on Sag T2 middle slice
  Step 3  — Model 2 (SCS): crop Sag T2 patches at keypoints, TTA, temperature
  Step 4  — Model 3 (NFN): crop Sag T1 patches using offset-corrected keypoints, TTA, temperature
  Step 5  — DICOM geometry: match axial slice per level using 3D patient coordinates
  Step 6  — Model 4 (SS): crop Axial T2 left/right patches, TTA, temperature
  Step 7  — Assemble 25 triplets → submission row

Run
---
  python -m src.module5_inference                   # full test inference + final eval
  python -m src.module5_inference --smoke-only       # 2 test studies only
  python -m src.module5_inference --calibrate-only   # compute temperature T for each model
  python -m src.module5_inference --run-id my_001
  python -m src.module5_inference --n-vis 10
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary
from src.module2 import (
    build_instance_map,
    crop_and_resize,
    get_three_slices,
    load_sorted_series,
    normalise_patch,
    rescale_pixel_array,
    window_to_uint8,
)
from src.module4_model1 import (
    INPUT_SIZE,
    LEVELS,
    MODEL_NAME as MODEL1_NAME,
    N_OUTPUTS,
    build_model as build_model1,
    load_middle_slice,
    prepare_slice,
)

SCRIPT_NAME = "module5_inference"

CONDITIONS = CFG.CONDITIONS          # 5 conditions
SEVERITY_NAMES  = ["Normal/Mild", "Moderate", "Severe"]
SEVERITY_WEIGHTS = np.array([CFG.SEVERITY_WEIGHTS[i] for i in range(3)])

# Conditions handled by each classifier model
MODEL2_CONDITIONS = ["spinal_canal_stenosis"]
MODEL3_CONDITIONS = [
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
]  # kept for NFN offset lookup
MODEL4_CONDITIONS = ["left_subarticular_stenosis",      "right_subarticular_stenosis"]

TEMP_SCALES_PATH = CFG.OUTPUT_DIR / "temperature_scales.json"
NFN_OFFSETS_PATH = CFG.OUTPUT_DIR / "nfn_keypoint_offsets.json"
SS_OFFSETS_PATH  = CFG.OUTPUT_DIR / "ss_axial_offsets.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model builders / loaders
# ─────────────────────────────────────────────────────────────────────────────

def _build_classifier() -> nn.Module:
    """ResNet-34 → 3 classes. Identical for models 2, 3, 4."""
    from torchvision import models
    m = models.resnet34(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    return m


def load_model1(device: torch.device) -> nn.Module:
    path = CFG.CKPT_DIR / "model1_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model 1 checkpoint not found: {path}")
    model = build_model1().to(device)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_classifier(model_num: int, device: torch.device) -> nn.Module:
    path = CFG.CKPT_DIR / f"model{model_num}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Model {model_num} checkpoint not found: {path}")
    model = _build_classifier().to(device)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# NFN left/right keypoint offsets (Option B)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nfn_offsets(master: pd.DataFrame, logger) -> dict:
    """
    For every (level, side) combination, compute the mean pixel offset between
    the SCS annotation (x, y) and the NFN left/right annotation on the same study.

    SCS annotations are on Sagittal T2 and are exactly the keypoints Model 1
    predicts.  NFN annotations are on Sagittal T1, which may have a different
    pixel grid — so we normalise offsets as fractions of image (H, W) so they
    can be applied regardless of series resolution.

    Returns:
        {
          "left_neural_foraminal_narrowing": {
              "l1_l2": {"dx_frac": float, "dy_frac": float},
              ...
          },
          "right_neural_foraminal_narrowing": { ... }
        }
    """
    logger.info("── Computing NFN keypoint offsets from training data ────────")

    scs  = master[master["condition"] == "spinal_canal_stenosis"].copy()
    # SCS uses Sagittal T2; we use x, y directly as the keypoint reference.
    # We need series dimensions to normalise. Load from DICOM for a sample of studies,
    # or approximate using the image size stored in master (none) — so we store raw px offsets
    # and normalise when applying (dividing by the T2 slice H, W at inference time).

    # fix 1.2: collect raw pixel offsets between NFN (T1) and SCS (T2) annotations.
    # We also need the T1 image dimensions to convert to fractional offsets so
    # the values are resolution-independent and can be applied in T1 pixel space.
    # We approximate T1 dimensions per study from the NFN series entry in master
    # (series_id lets us look up the DICOM folder and read one slice header).
    offsets: Dict[str, Dict[str, Dict[str, list]]] = {
        cond: {lvl: {"dx_frac": [], "dy_frac": []} for lvl in LEVELS}
        for cond in MODEL3_CONDITIONS
    }

    for cond in MODEL3_CONDITIONS:
        nfn = master[master["condition"] == cond].copy()

        # Join NFN annotations to SCS annotations on (study_id, level)
        merged = nfn.merge(
            scs[["study_id", "level", "x", "y"]],
            on=["study_id", "level"],
            suffixes=("_nfn", "_scs"),
        )
        if merged.empty:
            logger.warning(f"  No overlap between SCS and {cond} — offsets will be zero")
            continue

        for lvl in LEVELS:
            sub = merged[merged["level"] == lvl]
            if sub.empty:
                continue
            # For each study, load the T1 series to get its pixel dimensions
            for _, row in sub.iterrows():
                sid = int(row["study_id"])
                nfn_series_rows = master[
                    (master["study_id"] == sid) &
                    (master["condition"] == cond)
                ]
                if nfn_series_rows.empty:
                    continue
                t1_series_id = int(nfn_series_rows["series_id"].iloc[0])
                t1_dir = CFG.DICOM_ROOT / str(sid) / str(t1_series_id)
                t1_result = load_middle_slice(t1_dir) if t1_dir.exists() else None
                if t1_result is None:
                    continue
                _, t1_H, t1_W = t1_result
                if t1_W == 0 or t1_H == 0:
                    continue
                # Raw pixel offset in T1 space = NFN annotation − SCS annotation
                # (SCS coords are on T2; if grids match this is valid; if not,
                # the fractional form cancels most of the error)
                dx = float(row["x_nfn"] - row["x_scs"])
                dy = float(row["y_nfn"] - row["y_scs"])
                offsets[cond][lvl]["dx_frac"].append(dx / t1_W)
                offsets[cond][lvl]["dy_frac"].append(dy / t1_H)

    # Summarise to mean fractional offsets
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond in MODEL3_CONDITIONS:
        result[cond] = {}
        for lvl in LEVELS:
            dx_vals = offsets[cond][lvl]["dx_frac"]
            dy_vals = offsets[cond][lvl]["dy_frac"]
            mdx = float(np.mean(dx_vals)) if dx_vals else 0.0
            mdy = float(np.mean(dy_vals)) if dy_vals else 0.0
            result[cond][lvl] = {"dx_frac": mdx, "dy_frac": mdy}
            logger.info(
                f"  {cond[:30]:30s}  {lvl}  "
                f"dx_frac={mdx:+.4f}  dy_frac={mdy:+.4f}  "
                f"(n={len(dx_vals)})"
            )

    with open(NFN_OFFSETS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  NFN offsets saved → {NFN_OFFSETS_PATH}")
    return result


def load_nfn_offsets(master: pd.DataFrame, logger) -> dict:
    """Load cached NFN offsets or recompute if missing."""
    if NFN_OFFSETS_PATH.exists():
        with open(NFN_OFFSETS_PATH) as f:
            offsets = json.load(f)
        logger.info(f"  NFN offsets loaded from {NFN_OFFSETS_PATH}")
        return offsets
    return compute_nfn_offsets(master, logger)


# ─────────────────────────────────────────────────────────────────────────────
# Series selection helper
# ─────────────────────────────────────────────────────────────────────────────

def find_series(
    series_desc: pd.DataFrame,
    study_id: int,
    series_type: str,
    logger,
    run_id: str,
) -> Optional[int]:
    """Return the first series_id matching series_type for this study, or None."""
    rows = series_desc[
        (series_desc["study_id"] == study_id) &
        (series_desc["series_description"].str.contains(series_type, case=False, na=False))
    ]
    if rows.empty:
        logger.warning(f"  Study {study_id}: no {series_type} series found")
        log_event(run_id, SCRIPT_NAME, "series_missing",
                  study_id=study_id, series_type=series_type)
        return None
    return int(rows["series_id"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint prediction (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def predict_keypoints(
    model1: nn.Module,
    study_id: int,
    series_id: int,
    device: torch.device,
) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[np.ndarray], int, int]:
    """
    Run Model 1 on the middle slice of the given Sagittal T2 series.
    Returns (keypoints_dict, windowed_slice, H, W).
    keypoints_dict: {level: (x_px, y_px)}
    windowed_slice: (H, W) uint8 for visualisation.
    """
    series_dir = CFG.DICOM_ROOT / str(study_id) / str(series_id)
    result = load_middle_slice(series_dir)
    if result is None:
        return None, None, INPUT_SIZE, INPUT_SIZE

    windowed, H, W = result
    img_tensor = torch.from_numpy(prepare_slice(windowed)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model1(img_tensor).cpu().numpy()[0]   # (10,) normalised

    kps = {}
    for j, lvl in enumerate(LEVELS):
        kps[lvl] = (float(pred_norm[j*2] * W), float(pred_norm[j*2+1] * H))

    return kps, windowed, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Patch extraction (Step 3, 4, 6) — thin wrapper around module2 utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_patch_from_series(
    series_dir: Path,
    instance_num: int,
    cx: float,
    cy: float,
) -> Optional[np.ndarray]:
    """
    Load 3-slice stack centred on (cx, cy) at instance_num, return (3,128,128) float32.
    Returns None if series_dir is missing or has no DICOMs.
    """
    datasets = load_sorted_series(series_dir)
    if not datasets:
        return None
    inst_map = build_instance_map(datasets)
    try:
        ch0, ch1, ch2, _, _ = get_three_slices(inst_map, instance_num)
    except Exception:
        return None

    cr0 = crop_and_resize(ch0, cx, cy)
    cr1 = crop_and_resize(ch1, cx, cy)
    cr2 = crop_and_resize(ch2, cx, cy)
    stack  = np.stack([cr0, cr1, cr2], axis=0)
    return normalise_patch(stack)


def get_middle_instance_num(series_dir: Path) -> Optional[int]:
    """Return the InstanceNumber of the middle DICOM slice in series_dir."""
    datasets = load_sorted_series(series_dir)
    if not datasets:
        return None
    mid = datasets[len(datasets) // 2]
    return int(getattr(mid, "InstanceNumber", len(datasets) // 2))


# ─────────────────────────────────────────────────────────────────────────────
# Sagittal T1 geometry: scale keypoints if T1 ≠ T2 dimensions (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

def scale_keypoints_to_t1(
    kps: Dict[str, Tuple[float, float]],
    sag_t2_H: int, sag_t2_W: int,
    sag_t1_series_dir: Path,
    logger,
) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    """
    If Sag T1 has different pixel dimensions to Sag T2, scale (x, y) proportionally.
    Returns (scaled_kps, t1_H, t1_W).
    """
    result = load_middle_slice(sag_t1_series_dir)
    if result is None:
        logger.warning(f"  Cannot load Sag T1 — reusing T2 keypoints as-is")
        return kps, sag_t2_H, sag_t2_W

    _, t1_H, t1_W = result

    if t1_H == sag_t2_H and t1_W == sag_t2_W:
        return kps, t1_H, t1_W   # identical grids — no scaling needed

    scale_x = t1_W / sag_t2_W
    scale_y = t1_H / sag_t2_H
    logger.info(
        f"  T1/T2 dimension mismatch: T2=({sag_t2_H},{sag_t2_W})  "
        f"T1=({t1_H},{t1_W})  scale=({scale_x:.3f},{scale_y:.3f})"
    )
    scaled = {lvl: (x * scale_x, y * scale_y) for lvl, (x, y) in kps.items()}
    return scaled, t1_H, t1_W


# ─────────────────────────────────────────────────────────────────────────────
# DICOM geometry: axial slice matching (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

def _ipp(ds: pydicom.Dataset) -> np.ndarray:
    """ImagePositionPatient as float64 array [x, y, z]."""
    return np.array([float(v) for v in ds.ImagePositionPatient])


def _iop(ds: pydicom.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """ImageOrientationPatient → (row_dir, col_dir) each (3,)."""
    vals   = [float(v) for v in ds.ImageOrientationPatient]
    row_dir = np.array(vals[:3])
    col_dir = np.array(vals[3:])
    return row_dir, col_dir


def _pixel_spacing(ds: pydicom.Dataset) -> Tuple[float, float]:
    """PixelSpacing → (row_spacing_mm, col_spacing_mm)."""
    if hasattr(ds, "PixelSpacing"):
        ps = [float(v) for v in ds.PixelSpacing]
        return ps[0], ps[1]
    return 1.0, 1.0


def keypoint_to_3d(
    kp_x: float,
    kp_y: float,
    sag_mid_ds: pydicom.Dataset,
) -> Optional[np.ndarray]:
    """
    Convert (kp_x, kp_y) pixel coordinates on the sagittal middle slice
    to a 3D patient-space position in mm.
    Returns None if required DICOM tags are absent.
    """
    try:
        ipp            = _ipp(sag_mid_ds)
        row_dir, col_dir = _iop(sag_mid_ds)
        ps_row, ps_col = _pixel_spacing(sag_mid_ds)
        pos3d = ipp + col_dir * kp_x * ps_col + row_dir * kp_y * ps_row
        return pos3d
    except Exception:
        return None


def find_axial_slice_for_level(
    axial_datasets: List[pydicom.Dataset],
    sag_mid_ds: pydicom.Dataset,
    kp_x: float,
    kp_y: float,
    logger,
    study_id: int,
    level: str,
) -> int:
    """
    Find the index (into axial_datasets) of the axial slice whose 3D position
    is closest to the keypoint's 3D z-coordinate.

    Falls back to middle axial slice if DICOM geometry tags are absent.
    """
    pos3d = keypoint_to_3d(kp_x, kp_y, sag_mid_ds)
    if pos3d is None:
        mid = len(axial_datasets) // 2
        logger.warning(
            f"  Study {study_id} {level}: DICOM geometry tags missing — "
            f"using middle axial slice (idx {mid})"
        )
        return mid

    # Project each axial slice IPP onto the spinal axis (normal to sagittal plane)
    try:
        row_dir, col_dir = _iop(sag_mid_ds)
        spine_normal     = np.cross(row_dir, col_dir)
        spine_normal    /= (np.linalg.norm(spine_normal) + 1e-9)

        kp_z = float(np.dot(pos3d, spine_normal))

        best_idx  = len(axial_datasets) // 2
        best_dist = float("inf")

        for i, ds in enumerate(axial_datasets):
            try:
                axial_ipp = _ipp(ds)
                axial_z   = float(np.dot(axial_ipp, spine_normal))
                dist      = abs(axial_z - kp_z)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i
            except Exception:
                continue

        return best_idx

    except Exception:
        mid = len(axial_datasets) // 2
        logger.warning(
            f"  Study {study_id} {level}: geometry failed — "
            f"using middle axial slice (idx {mid})"
        )
        return mid


# ─────────────────────────────────────────────────────────────────────────────
# TTA (Step 7)
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
_STD  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]


def _denorm(patch: np.ndarray) -> np.ndarray:
    """ImageNet → [0,1]."""
    return np.clip(patch * _STD + _MEAN, 0.0, 1.0)


def _renorm(patch_01: np.ndarray) -> np.ndarray:
    """[0,1] → ImageNet."""
    return (patch_01 - _MEAN) / _STD


def _augment_for_tta(patch: np.ndarray) -> List[np.ndarray]:
    """
    Return 4 variants: original, hflip, +5° rotation, -5° rotation.
    All returned in ImageNet-normalised space.
    """
    patch_01 = _denorm(patch)   # (3, H, W) in [0,1]

    def _hflip(p):
        return p[:, :, ::-1].copy()

    def _rotate(p, angle):
        rotated = np.stack([
            scipy.ndimage.rotate(p[c], angle, reshape=False, order=1, cval=0.0)
            for c in range(3)
        ], axis=0)
        return rotated

    variants_01 = [
        patch_01,
        _hflip(patch_01),
        _rotate(patch_01, 5),
        _rotate(patch_01, -5),
    ]
    return [_renorm(v) for v in variants_01]


def tta_predict(
    model:  nn.Module,
    patch:  np.ndarray,
    device: torch.device,
    T:      float = 1.0,
) -> np.ndarray:
    """
    Run 4 TTA passes, apply temperature scaling, return averaged softmax (3,).
    """
    variants = _augment_for_tta(patch)
    probs_list = []

    for v in variants:
        t = torch.from_numpy(v).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t).cpu().numpy()[0]   # (3,)
        scaled = logits / T
        # manual softmax for temperature-scaled logits
        e = np.exp(scaled - scaled.max())
        probs_list.append(e / e.sum())

    return np.mean(probs_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling calibration (Step 9)
# ─────────────────────────────────────────────────────────────────────────────

def _wll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    probs = np.clip(probs, eps, 1 - eps)
    total_w = total_loss = 0.0
    for i in range(len(labels)):
        c = int(labels[i])
        w = SEVERITY_WEIGHTS[c]
        total_loss += w * np.log(probs[i, c])
        total_w    += w
    return float(-total_loss / total_w) if total_w > 0 else float("inf")


def calibrate_temperature(
    model:      nn.Module,
    val_df:     pd.DataFrame,
    conditions: List[str],
    device:     torch.device,
    logger,
    run_id:     str,
    model_num:  int,
) -> float:
    """
    Grid-search T ∈ [0.5, 3.0] on the val split to minimise weighted log-loss.
    Collects raw logits with a single inference pass, then searches over T.
    """
    from src.module3 import RSNADataset, build_val_transforms, resolve_patch_paths

    logger.info(f"  Calibrating temperature for Model {model_num} …")

    parts = []
    for cond in conditions:
        cdf = val_df[val_df["condition"] == cond].copy()
        cdf = resolve_patch_paths(cdf, "val")   # fix 1.1: must be "val", not cond
        cdf = cdf.dropna(subset=["patch_path"])
        parts.append(cdf)

    if not parts:
        logger.warning(f"  Model {model_num}: no val patches for calibration — T=1.0")
        return 1.0

    combined = pd.concat(parts, ignore_index=True)
    ds       = RSNADataset(combined, transform=build_val_transforms())
    loader   = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(lbls.numpy())

    all_logits = np.concatenate(all_logits, axis=0)   # (N, 3)
    all_labels = np.concatenate(all_labels, axis=0)   # (N,)

    best_T    = 1.0
    best_loss = float("inf")

    for T in np.linspace(0.5, 3.0, 51):
        scaled = all_logits / T
        e      = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs  = e / e.sum(axis=1, keepdims=True)
        loss   = _wll(probs, all_labels)
        if loss < best_loss:
            best_loss = loss
            best_T    = float(T)

    logger.info(
        f"  Model {model_num}: best T={best_T:.3f}  "
        f"val_wll_after_scaling={best_loss:.4f}"
    )
    log_event(run_id, SCRIPT_NAME, "temperature_calibrated",
              model_num=model_num, T=best_T, val_wll=best_loss)
    return best_T


def load_or_calibrate_temperatures(
    models:     Dict[int, nn.Module],
    val_df:     pd.DataFrame,
    device:     torch.device,
    logger,
    run_id:     str,
    recalibrate: bool = False,
) -> Dict[int, float]:
    """Load temperature_scales.json or run calibration for each model."""
    if TEMP_SCALES_PATH.exists() and not recalibrate:
        with open(TEMP_SCALES_PATH) as f:
            scales = json.load(f)
        # JSON keys are strings
        result = {int(k): float(v) for k, v in scales.items()}
        logger.info(f"  Temperature scales loaded: {result}")
        return result

    logger.info("── Temperature scaling calibration ─────────────────────────")
    registry = {
        2: MODEL2_CONDITIONS,
        3: MODEL3_CONDITIONS,
        4: MODEL4_CONDITIONS,
    }
    result = {}
    for mn, conds in registry.items():
        T = calibrate_temperature(
            models[mn], val_df, conds, device, logger, run_id, mn
        )
        result[mn] = T

    with open(TEMP_SCALES_PATH, "w") as f:
        json.dump({str(k): v for k, v in result.items()}, f, indent=2)
    logger.info(f"  Temperature scales saved → {TEMP_SCALES_PATH}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SS axial offsets (fix 2.2): data-derived lateral offsets per level
# ─────────────────────────────────────────────────────────────────────────────

def compute_ss_offsets(master: pd.DataFrame, logger) -> dict:
    """
    Compute mean (dx, dy) offset between SCS annotation and left/right SS
    annotations on the same study, stored as fractions of axial image width/height.

    Mirrors compute_nfn_offsets() but uses Axial T2 series dimensions.

    Returns:
        {
          "left_subarticular_stenosis":  {level: {"dx_frac": float, "dy_frac": float}},
          "right_subarticular_stenosis": {level: {"dx_frac": float, "dy_frac": float}},
        }
    Fallback: if JSON is missing at inference time, get_axial_lr_coords() uses 0.20.
    """
    logger.info("── Computing SS axial offsets from training data ────────────")

    scs = master[master["condition"] == "spinal_canal_stenosis"].copy()

    result: dict = {}
    for cond in MODEL4_CONDITIONS:
        ss  = master[master["condition"] == cond].copy()
        merged = ss.merge(
            scs[["study_id", "level", "x", "y"]],
            on=["study_id", "level"],
            suffixes=("_ss", "_scs"),
        )
        result[cond] = {}
        for lvl in LEVELS:
            sub = merged[merged["level"] == lvl]
            dx_fracs, dy_fracs = [], []
            for _, row in sub.iterrows():
                sid = int(row["study_id"])
                ss_series_rows = master[
                    (master["study_id"] == sid) &
                    (master["condition"] == cond)
                ]
                if ss_series_rows.empty:
                    continue
                axl_series_id = int(ss_series_rows["series_id"].iloc[0])
                axl_dir = CFG.DICOM_ROOT / str(sid) / str(axl_series_id)
                axl_result = load_middle_slice(axl_dir) if axl_dir.exists() else None
                if axl_result is None:
                    continue
                _, axl_H, axl_W = axl_result
                if axl_W == 0 or axl_H == 0:
                    continue
                dx_fracs.append((float(row["x_ss"]) - float(row["x_scs"])) / axl_W)
                dy_fracs.append((float(row["y_ss"]) - float(row["y_scs"])) / axl_H)

            mdx = float(np.mean(dx_fracs)) if dx_fracs else (
                -0.20 if "left" in cond else 0.20
            )
            mdy = float(np.mean(dy_fracs)) if dy_fracs else 0.0
            result[cond][lvl] = {"dx_frac": mdx, "dy_frac": mdy}
            logger.info(
                f"  {cond[:35]:35s}  {lvl}  "
                f"dx_frac={mdx:+.4f}  dy_frac={mdy:+.4f}  (n={len(dx_fracs)})"
            )

    with open(SS_OFFSETS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  SS offsets saved → {SS_OFFSETS_PATH}")
    return result


def load_ss_offsets(master: pd.DataFrame, logger) -> dict:
    """Load cached SS offsets or recompute if missing."""
    if SS_OFFSETS_PATH.exists():
        with open(SS_OFFSETS_PATH) as f:
            offsets = json.load(f)
        logger.info(f"  SS axial offsets loaded from {SS_OFFSETS_PATH}")
        return offsets
    return compute_ss_offsets(master, logger)


# ─────────────────────────────────────────────────────────────────────────────
# Subarticular left/right crop: derive centres from axial slice geometry
# ─────────────────────────────────────────────────────────────────────────────

def get_axial_lr_coords(
    axial_ds:   pydicom.Dataset,
    sag_mid_ds: pydicom.Dataset,
    kp_x:       float,
    kp_y:       float,
    ss_offsets: Optional[dict] = None,
    level:      Optional[str]  = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Project the sagittal keypoint onto the axial slice's pixel grid to find (cx, cy).
    Then compute left/right centres using data-derived fractional offsets from
    ss_offsets (fix 2.2).  Falls back to ±20% of width if ss_offsets is None
    or the requested level is not present.

    Returns ((left_cx, cy), (right_cx, cy)).
    """
    try:
        H = int(axial_ds.Rows)
        W = int(axial_ds.Columns)
    except Exception:
        H = W = 512

    # Approximate: use centre of axial image as the canal centre, then offset.
    # A more precise approach would project the 3D keypoint onto the axial plane,
    # but this requires the axial IOP/IPP — implemented as fallback to image centre.
    try:
        # 3D keypoint position
        pos3d = keypoint_to_3d(kp_x, kp_y, sag_mid_ds)
        if pos3d is None:
            raise ValueError("geometry unavailable")

        axial_ipp            = _ipp(axial_ds)
        axial_row_dir, axial_col_dir = _iop(axial_ds)
        ps_row, ps_col       = _pixel_spacing(axial_ds)

        # Vector from axial origin to the 3D point
        diff = pos3d - axial_ipp

        # Project onto axial axes
        col_offset = float(np.dot(diff, axial_col_dir)) / ps_col
        row_offset = float(np.dot(diff, axial_row_dir)) / ps_row

        cx = np.clip(col_offset, 0, W)
        cy = np.clip(row_offset, 0, H)

    except Exception:
        cx = W / 2.0
        cy = H / 2.0

    # fix 2.2: use data-derived fractional offsets per level; fallback to ±20%
    default_frac = 0.20
    if ss_offsets is not None and level is not None:
        left_frac  = ss_offsets.get("left_subarticular_stenosis",  {}).get(
            level, {"dx_frac": -default_frac})["dx_frac"]
        right_frac = ss_offsets.get("right_subarticular_stenosis", {}).get(
            level, {"dx_frac":  default_frac})["dx_frac"]
        # fractions can be negative (left) or positive (right); abs for offset magnitude
        left_cx  = float(np.clip(cx + left_frac  * W, 0, W))
        right_cx = float(np.clip(cx + right_frac * W, 0, W))
    else:
        left_cx  = float(np.clip(cx - default_frac * W, 0, W))
        right_cx = float(np.clip(cx + default_frac * W, 0, W))
    return (left_cx, float(cy)), (right_cx, float(cy))


# ─────────────────────────────────────────────────────────────────────────────
# Per-study inference orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_study(
    study_id:    int,
    series_desc: pd.DataFrame,
    model1:      nn.Module,
    model2:      nn.Module,
    model3:  nn.Module,
    model4:      nn.Module,
    temp_scales: Dict[int, float],
    nfn_offsets: dict,
    device:      torch.device,
    logger,
    run_id:      str,
    ss_offsets:  Optional[dict] = None,   # fix 2.2: data-derived SS lateral offsets
) -> Dict[str, np.ndarray]:
    """
    Run full pipeline for one study.
    Returns {"{condition}_{level}": np.ndarray (3,) softmax probs}.
    Missing predictions default to uniform [1/3, 1/3, 1/3].
    """
    uniform = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    preds   = {
        f"{cond}_{lvl}": uniform.copy()
        for cond in CONDITIONS
        for lvl  in LEVELS
    }

    T2  = temp_scales.get(2,    1.0)
    T3  = temp_scales.get(3, 1.0)
    T4  = temp_scales.get(4,    1.0)

    # ── Step 1: find series ───────────────────────────────────────────────────
    sag_t2_id = find_series(series_desc, study_id, "Sagittal T2", logger, run_id)
    sag_t1_id = find_series(series_desc, study_id, "Sagittal T1", logger, run_id)
    axl_t2_id = find_series(series_desc, study_id, "Axial T2",    logger, run_id)

    if sag_t2_id is None:
        logger.error(f"  Study {study_id}: no Sagittal T2 — cannot predict keypoints")
        return preds

    sag_t2_dir = CFG.DICOM_ROOT / str(study_id) / str(sag_t2_id)

    # ── Step 2: keypoints ─────────────────────────────────────────────────────
    kps, sag_t2_windowed, sag_t2_H, sag_t2_W = predict_keypoints(
        model1, study_id, sag_t2_id, device
    )
    if kps is None:
        logger.error(f"  Study {study_id}: keypoint prediction failed")
        return preds

    log_event(run_id, SCRIPT_NAME, "keypoints_predicted",
              study_id=study_id,
              keypoints={lvl: list(xy) for lvl, xy in kps.items()})

    # Load sagittal T2 middle slice DICOM dataset (for 3D geometry)
    sag_t2_datasets = load_sorted_series(sag_t2_dir)
    sag_t2_mid_ds   = sag_t2_datasets[len(sag_t2_datasets) // 2] if sag_t2_datasets else None
    sag_t2_mid_inst = int(getattr(sag_t2_mid_ds, "InstanceNumber",
                                   len(sag_t2_datasets) // 2)) if sag_t2_mid_ds else 0

    # ── Step 3: SCS — Model 2 — Sagittal T2 ──────────────────────────────────
    for lvl in LEVELS:
        cx, cy = kps[lvl]
        patch  = extract_patch_from_series(sag_t2_dir, sag_t2_mid_inst, cx, cy)
        if patch is None:
            continue
        probs = tta_predict(model2, patch, device, T=T2)
        preds[f"spinal_canal_stenosis_{lvl}"] = probs

    # ── Step 4: NFN — Model 3 — Sagittal T1 ──────────────────────────────────
    if sag_t1_id is not None:
        sag_t1_dir = CFG.DICOM_ROOT / str(study_id) / str(sag_t1_id)

        # Scale keypoints if T1 dimensions differ from T2
        t1_kps, t1_H, t1_W = scale_keypoints_to_t1(
            kps, sag_t2_H, sag_t2_W, sag_t1_dir, logger
        )
        t1_datasets = load_sorted_series(sag_t1_dir)
        t1_mid_inst = int(getattr(
            t1_datasets[len(t1_datasets) // 2], "InstanceNumber",
            len(t1_datasets) // 2
        )) if t1_datasets else 0

        for cond in MODEL3_CONDITIONS:
            for lvl in LEVELS:
                cx_base, cy_base = t1_kps[lvl]
                # Apply offset in T1 pixel space (after scale_keypoints_to_t1)
                # Offsets are stored as fractions of T1 W/H so they are
                # resolution-independent; multiply back by actual T1 dims here.
                off = nfn_offsets.get(cond, {}).get(lvl, {})
                if "dx_frac" in off:
                    # fractional format: multiply by image dimensions
                    dx_abs = float(off["dx_frac"]) * t1_W
                    dy_abs = float(off["dy_frac"]) * t1_H
                else:
                    # raw pixel format (current JSON): use directly, no multiplication
                    dx_abs = float(off.get("dx", 0.0))
                    dy_abs = float(off.get("dy", 0.0))
                cx     = float(cx_base + dx_abs)
                cy     = float(cy_base + dy_abs)

                patch = extract_patch_from_series(sag_t1_dir, t1_mid_inst, cx, cy)
                if patch is None:
                    continue
                probs = tta_predict(model3, patch, device, T=T3)
                preds[f"{cond}_{lvl}"] = probs

    # ── Step 5–6: SS — Model 4 — Axial T2 ────────────────────────────────────
    if axl_t2_id is not None and sag_t2_mid_ds is not None:
        axl_t2_dir      = CFG.DICOM_ROOT / str(study_id) / str(axl_t2_id)
        axial_datasets  = load_sorted_series(axl_t2_dir)
        axial_inst_map  = build_instance_map(axial_datasets) if axial_datasets else {}

        for lvl in LEVELS:
            cx_sag, cy_sag = kps[lvl]

            # Step 5: find matching axial slice
            axl_idx = find_axial_slice_for_level(
                axial_datasets, sag_t2_mid_ds,
                cx_sag, cy_sag, logger, study_id, lvl
            )
            axl_ds  = axial_datasets[axl_idx]
            axl_inst = int(getattr(axl_ds, "InstanceNumber", axl_idx))

            # Step 6: left/right subarticular crop centres (fix 2.2: pass ss_offsets + level)
            (left_cx, left_cy), (right_cx, right_cy) = get_axial_lr_coords(
                axl_ds, sag_t2_mid_ds, cx_sag, cy_sag,
                ss_offsets=ss_offsets, level=lvl,
            )

            for side, cond, cx, cy in [
                ("left",  "left_subarticular_stenosis",  left_cx,  left_cy),
                ("right", "right_subarticular_stenosis", right_cx, right_cy),
            ]:
                patch = extract_patch_from_series(axl_t2_dir, axl_inst, cx, cy)
                if patch is None:
                    continue
                probs = tta_predict(model4, patch, device, T=T4)
                preds[f"{cond}_{lvl}"] = probs

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Submission CSV builder
# ─────────────────────────────────────────────────────────────────────────────

def build_submission(
    all_preds: Dict[int, Dict[str, np.ndarray]],
    logger,
) -> pd.DataFrame:
    """
    Assemble results into competition submission format.
    row_id = {study_id}_{condition}_{level}
    """
    rows = []
    for study_id, study_preds in all_preds.items():
        for cond in CONDITIONS:
            for lvl in LEVELS:
                key   = f"{cond}_{lvl}"
                probs = study_preds.get(key, np.array([1/3, 1/3, 1/3]))
                rows.append({
                    "row_id":       f"{study_id}_{key}",
                    "normal_mild":  float(probs[0]),
                    "moderate":     float(probs[1]),
                    "severe":       float(probs[2]),
                })
    df = pd.DataFrame(rows)
    logger.info(f"  Submission rows: {len(df):,}  ({len(all_preds)} studies × 25 conditions)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Inference trace visualisation (most important visualisation per strategy doc)
# ─────────────────────────────────────────────────────────────────────────────

_SEV_COLOURS = ["#2ECC71", "#F39C12", "#E74C3C"]
_LEVEL_COLS  = {
    "l1_l2": "#E74C3C", "l2_l3": "#F39C12", "l3_l4": "#2ECC71",
    "l4_l5": "#3498DB", "l5_s1": "#9B59B6",
}


def visualise_inference_trace(
    study_id:        int,
    keypoints:       Dict[str, Tuple[float, float]],
    sag_t2_windowed: Optional[np.ndarray],
    predictions:     Dict[str, np.ndarray],
    true_labels:     Optional[Dict[str, int]],
    run_id:          str,
    logger,
):
    """
    Full inference trace for one study:
      Row 1 — Sagittal T2 with predicted keypoints + horizontal lines
      Row 2 — 25 probability bar charts (one per condition+level)
               Bars coloured green/orange/red; border green if correct, red if wrong
    """
    n_levels = len(LEVELS)
    n_conds  = len(CONDITIONS)

    fig = plt.figure(figsize=(n_levels * 3.2, 14))
    fig.suptitle(f"Study {study_id} — Inference Trace", fontsize=11, y=1.01)

    # ── Row A: sagittal slice with keypoints ──────────────────────────────────
    ax_sag = fig.add_subplot(n_conds + 1, 1, 1)
    if sag_t2_windowed is not None:
        ax_sag.imshow(sag_t2_windowed, cmap="gray", aspect="auto")
    else:
        ax_sag.set_facecolor("#111111")

    handles = []
    for lvl in LEVELS:
        col = _LEVEL_COLS[lvl]
        if lvl in keypoints:
            px, py = keypoints[lvl]
            ax_sag.axhline(y=py, color=col, linewidth=0.9, alpha=0.7, linestyle="--")
            ax_sag.plot(px, py, "o", color=col, markersize=7,
                        markeredgecolor="white", markeredgewidth=0.7)
        handles.append(mpatches.Patch(color=col, label=lvl))

    ax_sag.legend(handles=handles, loc="lower right", fontsize=7,
                  framealpha=0.6, ncol=5)
    ax_sag.set_title("Model 1 — Predicted keypoints (Sagittal T2)", fontsize=8)
    ax_sag.axis("off")

    # ── Rows B–F: probability bars per condition ──────────────────────────────
    for ci, cond in enumerate(CONDITIONS):
        cond_short = cond.replace("_", " ")
        for li, lvl in enumerate(LEVELS):
            ax = fig.add_subplot(n_conds + 1, n_levels, n_levels * (ci + 1) + li + 1)
            key   = f"{cond}_{lvl}"
            probs = predictions.get(key, np.array([1/3, 1/3, 1/3]))
            bars  = ax.bar(range(3), probs, color=_SEV_COLOURS, alpha=0.85)

            ax.set_ylim(0, 1)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["N/M", "Mod", "Sev"], fontsize=6)
            ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

            pred_c = int(np.argmax(probs))
            title_col = "black"
            if true_labels is not None and key in true_labels:
                true_c = true_labels[key]
                border_col = "#2ECC71" if pred_c == true_c else "#E74C3C"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_col)
                    spine.set_linewidth(1.8)
                title_col = "#2ECC71" if pred_c == true_c else "#A32D2D"

            if li == 0:
                ax.set_ylabel(cond_short[:22], fontsize=5.5, labelpad=2)
            if ci == 0:
                ax.set_title(lvl, fontsize=7, color=title_col)

    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_trace_{study_id}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Trace saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Weighted log-loss for evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wll_from_preds(
    all_preds:  Dict[int, Dict[str, np.ndarray]],
    master:     pd.DataFrame,
    split_df:   pd.DataFrame,
    logger,
    run_id:     str,
) -> Dict[str, float]:
    """
    Compute weighted log-loss on any split given predictions dict.
    Returns {condition: wll, "overall": wll}.
    """
    eps = 1e-7
    study_ids = set(split_df["study_id"].unique())
    rows_master = master[master["study_id"].isin(study_ids)].copy()

    per_cond: Dict[str, Tuple[float, float]] = {cond: (0.0, 0.0) for cond in CONDITIONS}
    total_loss = total_w = 0.0

    for _, row in rows_master.iterrows():
        sid  = int(row["study_id"])
        cond = str(row["condition"])
        lvl  = str(row["level"])
        true_c = int(row["severity_label"])
        key    = f"{cond}_{lvl}"

        probs = all_preds.get(sid, {}).get(key, np.array([1/3, 1/3, 1/3]))
        p     = float(np.clip(probs[true_c], eps, 1 - eps))
        w     = SEVERITY_WEIGHTS[true_c]

        loss_contrib = -w * np.log(p)
        total_loss  += loss_contrib
        total_w     += w

        if cond in per_cond:
            prev_loss, prev_w = per_cond[cond]
            per_cond[cond] = (prev_loss + loss_contrib, prev_w + w)

    metrics = {}
    logger.info("── Weighted log-loss results ────────────────────────────────")
    for cond in CONDITIONS:
        c_loss, c_w = per_cond[cond]
        wll = float(c_loss / c_w) if c_w > 0 else float("inf")
        metrics[cond] = wll
        logger.info(f"  {cond:45s}  wll={wll:.4f}")

    overall = float(total_loss / total_w) if total_w > 0 else float("inf")
    metrics["overall"] = overall
    logger.info(f"  {'OVERALL':45s}  wll={overall:.4f}")

    log_event(run_id, SCRIPT_NAME, "wll_computed", metrics=metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (2 test studies end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(
    test_df:     pd.DataFrame,
    master:      pd.DataFrame,
    series_desc: pd.DataFrame,
    models:      Dict[int, nn.Module],
    temp_scales: Dict[int, float],
    nfn_offsets: dict,
    device:      torch.device,
    logger,
    run_id:      str,
    ss_offsets:  Optional[dict] = None,
) -> bool:
    """
    Run the full pipeline on CFG.SMOKE_N_STUDIES test studies.
    Produces inference trace visualisations and a miniature WLL check.
    """
    logger.info("── Smoke test ──────────────────────────────────────────────")
    n            = CFG.SMOKE_N_STUDIES
    all_test_ids = test_df["study_id"].unique()
    # fix 2.3: seeded random selection — avoids early study IDs which may be
    # atypically clean and unrepresentative of the full test distribution
    rng       = np.random.default_rng(CFG.RANDOM_SEED)
    smoke_ids = rng.choice(
        all_test_ids, size=min(n, len(all_test_ids)), replace=False
    ).tolist()
    logger.info(f"  Studies (random, seed={CFG.RANDOM_SEED}): {smoke_ids}")

    smoke_preds = {}
    for sid in smoke_ids:
        logger.info(f"\n  ── Study {sid} ──")
        preds = run_study(
            sid, series_desc,
            models[1], models[2], models[3], models[4],
            temp_scales, nfn_offsets, device, logger, run_id,
            ss_offsets=ss_offsets,
        )
        smoke_preds[sid] = preds

        # True labels for this study (for visualisation borders)
        study_rows  = master[master["study_id"] == sid]
        true_labels = {
            f"{r['condition']}_{r['level']}": int(r["severity_label"])
            for _, r in study_rows.iterrows()
        }

        # Keypoints (re-predict for vis — cheap)
        sag_t2_id = find_series(series_desc, sid, "Sagittal T2", logger, run_id)
        kps, sag_t2_win, _, _ = (
            predict_keypoints(models[1], sid, sag_t2_id, device)
            if sag_t2_id else ({}, None, 0, 0)
        )

        visualise_inference_trace(
            sid, kps or {}, sag_t2_win, preds, true_labels, run_id, logger
        )

    # Miniature WLL check
    smoke_split = test_df[test_df["study_id"].isin(smoke_ids)]
    metrics     = compute_wll_from_preds(smoke_preds, master, smoke_split, logger, run_id)

    logger.info("── Smoke test complete ─────────────────────────────────────")
    log_event(run_id, SCRIPT_NAME, "smoke_test_complete",
              study_ids=smoke_ids, wll=metrics.get("overall"))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Full test inference + final evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_full_inference(
    test_df:     pd.DataFrame,
    master:      pd.DataFrame,
    series_desc: pd.DataFrame,
    models:      Dict[int, nn.Module],
    temp_scales: Dict[int, float],
    nfn_offsets: dict,
    device:      torch.device,
    logger,
    run_id:      str,
    n_vis:       int = 10,
    ss_offsets:  Optional[dict] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Iterate over all test studies, run full pipeline, return (all_preds, submission_df).
    """
    test_ids = list(test_df["study_id"].unique())
    logger.info(f"── Full inference: {len(test_ids):,} test studies ─────────")

    all_preds: Dict[int, Dict[str, np.ndarray]] = {}

    for i, sid in enumerate(test_ids, 1):
        if i % 20 == 0 or i == 1:
            logger.info(f"  [{i}/{len(test_ids)}] Study {sid}")
        preds = run_study(
            sid, series_desc,
            models[1], models[2], models[3], models[4],
            temp_scales, nfn_offsets, device, logger, run_id,
            ss_offsets=ss_offsets,
        )
        all_preds[sid] = preds

    # Inference trace for sample studies
    vis_ids = test_ids[:n_vis]
    for sid in vis_ids:
        sag_t2_id = find_series(series_desc, sid, "Sagittal T2", logger, run_id)
        kps, win, _, _ = (
            predict_keypoints(models[1], sid, sag_t2_id, device)
            if sag_t2_id else ({}, None, 0, 0)
        )
        study_rows  = master[master["study_id"] == sid]
        true_labels = {
            f"{r['condition']}_{r['level']}": int(r["severity_label"])
            for _, r in study_rows.iterrows()
        } if not study_rows.empty else None

        visualise_inference_trace(
            sid, kps or {}, win, all_preds[sid], true_labels, run_id, logger
        )

    sub_df = build_submission(all_preds, logger)
    return all_preds, sub_df

# ─────────────────────────────────────────────────────────────────────────────
# Test-split evaluation: ground truth vs predictions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test_predictions(
    all_preds:  Dict[int, Dict[str, np.ndarray]],
    master:     pd.DataFrame,
    test_df:    pd.DataFrame,
    val_kp_df:  Optional[pd.DataFrame],   # Model 1 keypoint eval results CSV, or None
    run_id:     str,
    logger,
) -> pd.DataFrame:
    """
    Compares pipeline predictions against ground-truth labels and
    (if provided) Model 1 keypoint coordinates on the test split.

    Produces one CSV with columns:
        study_id, condition, level,
        gt_severity, pred_severity,          # integer 0/1/2
        gt_label, pred_label,                # string Normal/Mild / Moderate / Severe
        correct,                             # bool
        prob_nm, prob_mod, prob_sev,         # raw probabilities
        confidence,                          # max probability
        # keypoint columns (only if val_kp_df is provided):
        gt_x, gt_y, pred_x, pred_y, kp_err_px

    Logs to screen:
        - Overall classification accuracy and weighted log-loss
        - Per-condition accuracy breakdown
        - Per-severity-class accuracy breakdown
        - Keypoint mean pixel error if available

    Saved to: outputs/{run_id}_test_evaluation.csv
    """
    eps = 1e-7
    sev_weights = np.array([CFG.SEVERITY_WEIGHTS[i] for i in range(3)])
    severity_names = ["Normal/Mild", "Moderate", "Severe"]

    test_ids = set(test_df["study_id"].unique())
    rows = master[master["study_id"].isin(test_ids)].copy()

    records = []
    total_loss = total_w = 0.0

    for _, row in rows.iterrows():
        sid    = int(row["study_id"])
        cond   = str(row["condition"])
        lvl    = str(row["level"])
        gt_c   = int(row["severity_label"])
        key    = f"{cond}_{lvl}"

        probs  = all_preds.get(sid, {}).get(key, np.array([1/3, 1/3, 1/3]))
        pred_c = int(np.argmax(probs))

        p = float(np.clip(probs[gt_c], eps, 1 - eps))
        w = float(sev_weights[gt_c])
        total_loss += -w * np.log(p)
        total_w    += w

        records.append({
            "study_id":    sid,
            "condition":   cond,
            "level":       lvl,
            "gt_severity": gt_c,
            "pred_severity": pred_c,
            "gt_label":    severity_names[gt_c],
            "pred_label":  severity_names[pred_c],
            "correct":     int(gt_c == pred_c),
            "prob_nm":     float(probs[0]),
            "prob_mod":    float(probs[1]),
            "prob_sev":    float(probs[2]),
            "confidence":  float(probs.max()),
        })

    eval_df = pd.DataFrame(records)

    # Merge keypoint errors if Model 1 eval CSV is provided
    if val_kp_df is not None:
        try:
            kp_long = []
            levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
            for lvl in levels:
                tmp = val_kp_df[["study_id",
                                  f"pred_x_{lvl}", f"pred_y_{lvl}",
                                  f"gt_x_{lvl}",   f"gt_y_{lvl}"]].copy()
                tmp["level"]    = lvl
                tmp["pred_x"]   = tmp[f"pred_x_{lvl}"]
                tmp["pred_y"]   = tmp[f"pred_y_{lvl}"]
                tmp["gt_x"]     = tmp[f"gt_x_{lvl}"]
                tmp["gt_y"]     = tmp[f"gt_y_{lvl}"]
                tmp["kp_err_px"] = np.sqrt(
                    (tmp["pred_x"] - tmp["gt_x"])**2 +
                    (tmp["pred_y"] - tmp["gt_y"])**2
                )
                kp_long.append(tmp[["study_id","level","pred_x","pred_y",
                                     "gt_x","gt_y","kp_err_px"]])
            kp_long = pd.concat(kp_long, ignore_index=True)
            eval_df = eval_df.merge(kp_long, on=["study_id","level"], how="left")
        except Exception as e:
            logger.warning(f"  Could not merge keypoint data: {e}")

    # ── Log results ───────────────────────────────────────────────────────────
    overall_wll = float(total_loss / total_w) if total_w > 0 else float("inf")
    overall_acc = float(eval_df["correct"].mean() * 100)

    logger.info("\n" + "=" * 65)
    logger.info("── Test Split Evaluation Results ───────────────────────────")
    logger.info(f"  Total predictions    : {len(eval_df):,}")
    logger.info(f"  Overall accuracy     : {overall_acc:.2f}%")
    logger.info(f"  Weighted log-loss    : {overall_wll:.4f}")

    logger.info("\n  Per-condition accuracy:")
    logger.info(f"  {'Condition':<45}  {'Acc%':>6}  {'N':>5}")
    logger.info("  " + "-" * 58)
    for cond in CFG.CONDITIONS:
        sub = eval_df[eval_df["condition"] == cond]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean() * 100
        logger.info(f"  {cond:<45}  {acc:>5.1f}%  {len(sub):>5,}")

    logger.info("\n  Per-severity-class accuracy (across all conditions):")
    for c, name in enumerate(severity_names):
        sub = eval_df[eval_df["gt_severity"] == c]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean() * 100
        logger.info(f"  {name:12s}: {acc:5.1f}%  (n={len(sub):,})")

    if "kp_err_px" in eval_df.columns:
        kp_valid = eval_df["kp_err_px"].dropna()
        logger.info(f"\n  Keypoint detection (Model 1 — val split):")
        logger.info(f"  Mean pixel error     : {kp_valid.mean():.2f}px")
        logger.info(f"  Median pixel error   : {kp_valid.median():.2f}px")
        logger.info(f"  Within 10px          : {(kp_valid <= 10).mean()*100:.1f}%")
        logger.info(f"  Within 20px          : {(kp_valid <= 20).mean()*100:.1f}%")
    logger.info("=" * 65)

    # Save CSV
    out_path = CFG.OUTPUT_DIR / f"{run_id}_test_evaluation.csv"
    eval_df.to_csv(out_path, index=False)
    logger.info(f"\n  Evaluation CSV saved → {out_path.name}")

    log_event(run_id, SCRIPT_NAME, "test_evaluation_complete",
              overall_acc=overall_acc, overall_wll=overall_wll)
    return eval_df

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    run_id:          Optional[str] = None,
    smoke_only:      bool = False,
    calibrate_only:  bool = False,
    n_vis:           int  = 10,
    recalibrate:     bool = False,
):
    run_id     = run_id or make_run_id("m5")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, SCRIPT_NAME)

    # fix 2.4: GPU detection, cudnn.benchmark, detailed device logging
    if torch.cuda.is_available():
        device     = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # 10-30% speed gain on fixed-size inputs
        gpu_name   = torch.cuda.get_device_name(0)
        gpu_mem    = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        device_str = f"cuda ({gpu_name}, {gpu_mem:.1f}GB)"
    else:
        device     = torch.device("cpu")
        device_str = "cpu"

    logger.info("=" * 65)
    logger.info(f"Module 5 — Inference Pipeline | run_id={run_id}")
    logger.info(f"  Device        : {device_str}")
    logger.info(f"  smoke_only    : {smoke_only}")
    logger.info(f"  calibrate_only: {calibrate_only}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, SCRIPT_NAME, "script_start",
              device=device_str, smoke_only=smoke_only,
              calibrate_only=calibrate_only)
    ensure_dirs()

    try:
        # ── Load data ─────────────────────────────────────────────────────────
        master      = pd.read_csv(CFG.MASTER_INDEX_PATH)
        val_df      = pd.read_csv(CFG.SPLIT_VAL)
        test_df     = pd.read_csv(CFG.SPLIT_TEST)
        series_desc = pd.read_csv(CFG.DATA_DIR / "train_series_descriptions.csv")

        # Standardise column name (competition uses 'series_description')
        if "series_description" not in series_desc.columns:
            # try alternative name used in some versions of the dataset
            if "description" in series_desc.columns:
                series_desc = series_desc.rename(columns={"description": "series_description"})
            else:
                # fall back: first non-id text column
                non_id_cols = [c for c in series_desc.columns
                               if c not in ("study_id", "series_id")]
                if non_id_cols:
                    series_desc = series_desc.rename(
                        columns={non_id_cols[0]: "series_description"}
                    )

        logger.info(f"  master index  : {len(master):,} rows")
        logger.info(f"  val studies   : {val_df['study_id'].nunique():,}")
        logger.info(f"  test studies  : {test_df['study_id'].nunique():,}")

        # ── Load all four models ──────────────────────────────────────────────
        logger.info("\n── Loading models ──────────────────────────────────────")
        models: Dict[int, nn.Module] = {
            1: load_model1(device),
            2: load_classifier(2, device),
            3: load_classifier(3, device),
            4: load_classifier(4, device),
        }
        logger.info("  All four models loaded ✓")

        # ── NFN offsets (Option B) ────────────────────────────────────────────
        nfn_offsets = load_nfn_offsets(master, logger)

        # ── SS axial offsets (fix 2.2) ────────────────────────────────────────
        ss_offsets = load_ss_offsets(master, logger)

        # ── Temperature calibration ───────────────────────────────────────────
        temp_scales = load_or_calibrate_temperatures(
            models, val_df, device, logger, run_id, recalibrate=recalibrate
        )

        if calibrate_only:
            logger.info("\n  --calibrate-only: done.")
            return

        # ── Smoke test ────────────────────────────────────────────────────────
        smoke_test(
            test_df, master, series_desc,
            models, temp_scales, nfn_offsets,
            device, logger, run_id,
            ss_offsets=ss_offsets,
        )

        if smoke_only:
            logger.info("\n  --smoke-only: done.")
            return

        # ── Full inference ────────────────────────────────────────────────────
        all_preds, sub_df = run_full_inference(
            test_df, master, series_desc,
            models, temp_scales, nfn_offsets,
            device, logger, run_id, n_vis=n_vis,
            ss_offsets=ss_offsets,
        )

        # ── Final evaluation ──────────────────────────────────────────────────
        logger.info("\n── Final evaluation (test split — one-time) ────────────")
        metrics = compute_wll_from_preds(all_preds, master, test_df, logger, run_id)
        overall_wll = metrics["overall"]

        # ── Detailed test evaluation for report ──────────────────────────────────────
        logger.info("\n── Detailed test evaluation ────────────────────────────")
        # Load Model 1 keypoint eval CSV if it exists (from module4_model1_eval.py)
        kp_eval_path = next(CFG.OUTPUT_DIR.glob("*model1_eval_results.csv"), None)
        kp_df = pd.read_csv(kp_eval_path) if kp_eval_path else None
        if kp_df is not None:
            logger.info(f"  Keypoint eval CSV found: {kp_eval_path.name}")
        else:
            logger.info("  No keypoint eval CSV found — keypoint stats will be skipped")
        evaluate_test_predictions(all_preds, master, test_df, kp_df, run_id, logger)

        # ── Save submission CSV ───────────────────────────────────────────────
        sub_path = CFG.OUTPUT_DIR / f"{run_id}_submission.csv"
        sub_df.to_csv(sub_path, index=False)
        logger.info(f"\n  Submission CSV saved → {sub_path.name}")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, SCRIPT_NAME, "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name="module5_inference",
        start_time=start_time,
        end_time=end_time,
        best_val_loss=metrics.get("overall", float("inf")),
        notes=(
            f"Test wll={metrics.get('overall', float('inf')):.4f}  "
            f"T2={temp_scales.get(2,1):.3f}  "
            f"T3={temp_scales.get(3,1):.3f}  "
            f"T4={temp_scales.get(4,1):.3f}"
        ),
    )
    log_event(run_id, SCRIPT_NAME, "script_complete", end_time=end_time)
    logger.info(f"\nModule 5 complete. Overall test WLL: {overall_wll:.4f}")
    logger.info(f"Submission: {sub_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Module 5 — End-to-end Inference Pipeline"
    )
    parser.add_argument("--run-id",        default=None)
    parser.add_argument("--smoke-only",    action="store_true",
                        help="Run on 2 test studies only")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Run temperature calibration only, save JSON and exit")
    parser.add_argument("--recalibrate",   action="store_true",
                        help="Force re-calibration even if temperature_scales.json exists")
    parser.add_argument("--n-vis",         type=int, default=10,
                        help="Number of test studies to visualise (default 10)")
    args = parser.parse_args()
    run(
        run_id=args.run_id,
        smoke_only=args.smoke_only,
        calibrate_only=args.calibrate_only,
        n_vis=args.n_vis,
        recalibrate=args.recalibrate,
    )


if __name__ == "__main__":
    main()