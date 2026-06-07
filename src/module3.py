"""
src/module3.py
==============
Module 3: Binding CSV Content and Images Together — Dataset, Augmentation, DataLoaders

Steps (exactly as specified in the strategy document):
  1. Build the final sample table from the master index
  2. Verify alignment — load 20 random .npy patches and display with metadata
  3. Dataset class — loads .npy, applies augmentation (train only), returns (tensor, label)
  4. Split-aware loading — hard check: no study_id in more than one split
  5. Visualisation checkpoints:
       a. 4×4 grid of random training patches per condition group
       b. Augmented versions of the same patch (multiple times) to confirm structure preserved
       c. 3-channel side-by-side to confirm spatial coherence across slices

Run (standalone verification):
    python -m src.module3
    python -m src.module3 --smoke-only
    python -m src.module3 --run-id my_run_001
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline — operates on [0, 1] float BEFORE
    ImageNet normalisation.

    Two-step pipeline in __getitem__:
      Step 1 (this function): spatial + intensity augmentations on [0, 1] float
      Step 2 (imagenet_normalise): subtract mean, divide std

    Keeping augmentation pre-normalisation fixes two known failure modes:
      - ShiftScaleRotate with border_mode=0 pads with 0.0 (valid black border),
        not the ImageNet-shifted ~-2.1 which turns shifted patches nearly black.
      - GaussNoise magnitude is correct relative to [0, 1] signal; on
        ImageNet-normalised data the same variance is ~10x too large.

    All transforms apply identically to all 3 channels simultaneously.
    """
    return A.Compose([
        A.HorizontalFlip(p=CFG.AUG_HFLIP_P),
        A.ShiftScaleRotate(
            shift_limit=CFG.AUG_SSR_SHIFT,
            scale_limit=CFG.AUG_SSR_SCALE,
            rotate_limit=CFG.AUG_SSR_ROTATE,
            border_mode=0,   # pads with 0.0 = black, correct on [0,1] float
            p=CFG.AUG_SSR_P,
        ),
        A.RandomBrightnessContrast(p=CFG.AUG_BRIGHTNESS_P),
        A.RandomGamma(p=CFG.AUG_GAMMA_P),
        A.GaussNoise(
            std_range=(0.01, 0.05),  # sensible on [0,1] float
            p=CFG.AUG_NOISE_P,
        ),
    ])


def imagenet_normalise(patch_01: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalisation as the FINAL step after augmentation.
    Input : (3, H, W) float32 in [0, 1]
    Output: (3, H, W) float32, ImageNet-normalised
    """
    mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
    return (patch_01 - mean) / std


def build_val_transforms() -> None:
    """
    Validation / test: no augmentation.
    Returns None as the sentinel — the Dataset checks for None and skips aug.
    """
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Patch path resolver — no master index involvement
# ─────────────────────────────────────────────────────────────────────────────

def resolve_patch_paths(df: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
    """
    Derive patch_path for every row from the filename convention used by module 2:
        {PATCHES_DIR}/{study_id}_{series_id}_{instance_number}_{condition}_{level}.npy

    Does NOT read master_index.csv — the master index is read-only after module 1.
    Rows whose .npy file does not exist on disk get patch_path = None.

    Warns if more than 5% of rows resolve to None — silent mass-drop would
    cause training on a fraction of data without any visible error.

    Returns a copy of df with patch_path column populated.
    """
    df = df.copy()

    def _path(row) -> Optional[str]:
        fname = (
            f"{int(row['study_id'])}"
            f"_{int(row['series_id'])}"
            f"_{int(row['instance_number'])}"
            f"_{row['condition']}"
            f"_{row['level']}"
            f".npy"
        )
        fpath = CFG.PATCHES_DIR / fname
        return str(fpath) if fpath.exists() else None

    df["patch_path"] = df.apply(_path, axis=1)

    n_total   = len(df)
    n_missing = df["patch_path"].isna().sum()
    pct_missing = 100 * n_missing / n_total if n_total > 0 else 0
    label = f"[{split_name}] " if split_name else ""

    if n_total > 0 and pct_missing > 5.0:
        import warnings
        warnings.warn(
            f"resolve_patch_paths {label}"
            f"{n_missing:,}/{n_total:,} rows ({pct_missing:.1f}%) have no matching "
            f".npy file in {CFG.PATCHES_DIR}. "
            f"If this is unexpected, check that module2 has been run and that "
            f"filenames have not changed.",
            UserWarning,
            stacklevel=2,
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RSNADataset(Dataset):
    """
    Loads pre-computed .npy patches and returns (patch_tensor, label_int).

    Parameters
    ----------
    df          : rows from the master index filtered to a single split
    transform   : albumentations Compose pipeline or None
    condition_filter : if given, only return rows matching this condition
                       (used to build per-model datasets for models 2/3/4)

    Label mapping (from CFG):
        Normal/Mild → 0,  Moderate → 1,  Severe → 2

    Input tensor shape  : (3, 128, 128) float32
    Label               : int64 scalar
    """

    def __init__(
        self,
        df:               pd.DataFrame,
        transform:        Optional[A.Compose] = None,
        condition_filter: Optional[str]       = None,
    ):
        if condition_filter is not None:
            df = df[df["condition"] == condition_filter].copy()

        # Drop rows with missing patch_path
        if "patch_path" not in df.columns:
            df["patch_path"] = None
        missing = df["patch_path"].isna()
        if missing.any():
            df = df[~missing].copy()

        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row   = self.df.iloc[idx]
        patch = np.load(row["patch_path"]).astype(np.float32)   # (3, H, W) ImageNet-normed
        label = int(row["severity_label"])

        if self.transform is not None:
            # Patches on disk are ImageNet-normalised.
            # Augmentation must run on [0, 1] float — undo norm first.
            mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
            std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
            patch_01  = np.clip(patch * std + mean, 0.0, 1.0)   # back to [0, 1]

            # albumentations expects (H, W, C)
            patch_hwc = patch_01.transpose(1, 2, 0)
            augmented = self.transform(image=patch_hwc)
            patch_01  = augmented["image"].transpose(2, 0, 1)   # (3, H, W) [0,1]

            # Re-apply ImageNet normalisation after augmentation
            patch = imagenet_normalise(patch_01)

        tensor = torch.from_numpy(patch)                         # (3, H, W) float32
        return tensor, torch.tensor(label, dtype=torch.long)

    # ── convenience accessors ─────────────────────────────────────────────────

    def label_counts(self) -> dict:
        """Return {0: n, 1: n, 2: n} counts for class-weight computation."""
        counts = self.df["severity_label"].value_counts().to_dict()
        return {k: counts.get(k, 0) for k in [0, 1, 2]}

    def sample_rows(self, n: int = 16, seed: int = 42) -> pd.DataFrame:
        return self.df.sample(min(n, len(self.df)), random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    split_dfs:         dict,          # {"train": df, "val": df, "test": df}
    condition_filter:  Optional[str] = None,
    batch_size:        int            = None,
    num_workers:       int            = 4,
) -> dict:
    """
    Build DataLoaders for all three splits.

    Returns
    -------
    {
      "train": DataLoader,   # shuffled, with augmentation
      "val":   DataLoader,   # sequential, no augmentation
      "test":  DataLoader,   # sequential, no augmentation  (do not use until final eval)
    }
    Also returns the train Dataset so callers can access .label_counts().
    """
    bs = batch_size or CFG.BATCH_SIZE

    train_ds = RSNADataset(split_dfs["train"],
                           transform=build_train_transforms(),
                           condition_filter=condition_filter)
    val_ds   = RSNADataset(split_dfs["val"],
                           transform=build_val_transforms(),
                           condition_filter=condition_filter)
    test_ds  = RSNADataset(split_dfs["test"],
                           transform=build_val_transforms(),
                           condition_filter=condition_filter)

    loaders = {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
    return loaders, train_ds


# ─────────────────────────────────────────────────────────────────────────────
# Split integrity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(split_dfs: dict, logger, run_id: str) -> bool:
    """
    Hard check: no study_id appears in more than one split.
    Logs an error and returns False if any overlap found.
    """
    sets  = {name: set(df["study_id"].unique()) for name, df in split_dfs.items()}
    names = list(sets.keys())
    ok    = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                logger.error(
                    f"  LEAKAGE: {len(overlap)} studies appear in both "
                    f"'{names[i]}' and '{names[j]}': {list(overlap)[:5]}"
                )
                log_event(run_id, "module3", "split_leakage",
                          split_a=names[i], split_b=names[j],
                          n_overlap=len(overlap),
                          sample_ids=list(overlap)[:10])
                ok = False
    if ok:
        logger.info("  Split leakage check PASSED — no study_id in multiple splits. ✓")
        log_event(run_id, "module3", "leakage_check_passed")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Alignment verification (20 random rows)
# ─────────────────────────────────────────────────────────────────────────────

def verify_alignment(master: pd.DataFrame, logger, run_id: str,
                     n: int = 20) -> bool:
    """
    For n random rows across all splits:
      - Load the .npy patch
      - Display it alongside metadata
      - Check patch shape and dtype
    Saves a grid image to CFG.VIS_DIR.
    """
    logger.info(f"── Alignment verification ({n} random samples) ──────────────")

    sample = master.dropna(subset=["patch_path"]).sample(
        min(n, len(master)), random_state=CFG.RANDOM_SEED
    )

    ok = True
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes_flat = axes.flatten()

    mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]

    for i, (_, row) in enumerate(sample.iterrows()):
        if i >= len(axes_flat):
            break
        try:
            patch = np.load(row["patch_path"]).astype(np.float32)
        except Exception as exc:
            logger.error(f"  Failed to load {row['patch_path']}: {exc}")
            ok = False
            axes_flat[i].axis("off")
            continue

        if patch.shape != (3, CFG.PATCH_SIZE_PX, CFG.PATCH_SIZE_PX):
            logger.error(
                f"  Wrong shape: {patch.shape} expected "
                f"(3, {CFG.PATCH_SIZE_PX}, {CFG.PATCH_SIZE_PX})"
            )
            ok = False

        # De-normalise ch1 (annotated slice) for display
        display = np.clip(patch * std + mean, 0, 1)[1]

        ax = axes_flat[i]
        ax.imshow(display, cmap="gray")
        sev = CFG.SEVERITY_INV.get(int(row["severity_label"]), "?")
        title = (
            f"{row['condition'].replace('_',' ')[:18]}\n"
            f"{row['level']}  [{sev}]  {row.get('split','?')}"
        )
        ax.set_title(title, fontsize=6.5)
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("Alignment check — 20 random samples", fontsize=10, y=1.01)
    plt.tight_layout()
    out = CFG.VIS_DIR / "alignment_check.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Alignment grid saved → {out}")
    log_event(run_id, "module3", "alignment_check",
              passed=ok, n_samples=len(sample))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _denorm(patch: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation for display. Returns (3,H,W) in [0,1]."""
    mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
    return np.clip(patch * std + mean, 0, 1)


def vis_patch_grid(
    ds:         RSNADataset,
    run_id:     str,
    logger,
    condition:  str = "all",
    n:          int = 16,
) -> None:
    """
    Visualisation checkpoint A:
    4×4 grid of random training patches (ch1 = annotated slice),
    labelled with condition / level / severity.
    One grid saved per condition group.
    """
    sample_rows = ds.sample_rows(n=n, seed=CFG.RANDOM_SEED)

    cols = 4
    rows = max(1, (len(sample_rows) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = axes.flatten() if rows > 1 else axes

    for i, (_, row) in enumerate(sample_rows.iterrows()):
        if i >= len(axes_flat):
            break
        patch   = np.load(row["patch_path"]).astype(np.float32)
        display = _denorm(patch)[1]    # channel 1 = annotated slice

        ax = axes_flat[i]
        ax.imshow(display, cmap="gray")
        sev = CFG.SEVERITY_INV.get(int(row["severity_label"]), "?")
        ax.set_title(
            f"{row['condition'].replace('_',' ')[:18]}\n"
            f"{row['level']}  [{sev}]",
            fontsize=7,
        )
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    label = condition.replace(" ", "_")
    plt.suptitle(f"Training patches — {condition}", fontsize=10, y=1.01)
    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_patch_grid_{label}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Patch grid saved → {out.name}")


def vis_augmentation_check(
    ds:      RSNADataset,
    run_id:  str,
    logger,
    n_aug:   int = 8,
) -> None:
    """
    Visualisation checkpoint B:
    Pick one sample. Show original + n_aug augmented versions side by side.
    Confirms augmentations are not destroying anatomical structure.
    Each column = one augmented version; rows = 3 channels.
    """
    row   = ds.df.iloc[0]
    patch = np.load(row["patch_path"]).astype(np.float32)   # ImageNet-normed

    # Undo normalisation once — augment on [0, 1] for every version
    mean     = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std      = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
    patch_01 = np.clip(patch * std + mean, 0.0, 1.0)

    aug     = build_train_transforms()
    results = [patch]                  # column 0 = original (normalised, for _denorm display)

    for _ in range(n_aug):
        hwc       = patch_01.transpose(1, 2, 0)
        augmented = aug(image=hwc)
        aug_01    = augmented["image"].transpose(2, 0, 1)   # (3,H,W) [0,1]
        results.append(imagenet_normalise(aug_01))          # store normalised for display

    n_cols = len(results)             # 1 original + n_aug
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.5, 3 * 2.5))

    col_labels = ["original"] + [f"aug {i+1}" for i in range(n_aug)]
    row_labels  = ["ch0 (inst-1)", "ch1 (inst)", "ch2 (inst+1)"]

    for col, p in enumerate(results):
        disp = _denorm(p)
        for row_idx in range(3):
            ax = axes[row_idx, col]
            ax.imshow(disp[row_idx], cmap="gray")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_labels[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=7)

    cond = row["condition"].replace("_", " ")
    plt.suptitle(
        f"Augmentation check — {cond} {row['level']}  "
        f"[{CFG.SEVERITY_INV.get(int(row['severity_label']), '?')}]",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_augmentation_check.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Augmentation check saved → {out.name}")


def vis_channel_coherence(
    ds:     RSNADataset,
    run_id: str,
    logger,
    n:      int = 6,
) -> None:
    """
    Visualisation checkpoint C:
    For n samples, display all 3 channels side by side.
    Adjacent slices should look like small spatial shifts of the same anatomy,
    not random images — confirms the 3-slice stacking is correct.
    """
    sample_rows = ds.sample_rows(n=n, seed=CFG.RANDOM_SEED + 1)

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))

    for row_idx, (_, row) in enumerate(sample_rows.iterrows()):
        patch = _denorm(np.load(row["patch_path"]).astype(np.float32))
        sev   = CFG.SEVERITY_INV.get(int(row["severity_label"]), "?")
        label = f"{row['condition'].replace('_',' ')[:18]}  {row['level']}  [{sev}]"

        for ch, ch_label in enumerate(["inst-1", "inst (annotated)", "inst+1"]):
            ax = axes[row_idx, ch]
            ax.imshow(patch[ch], cmap="gray")
            ax.axis("off")
            title = ch_label if row_idx == 0 else ""
            if title:
                ax.set_title(title, fontsize=8)
            if ch == 0:
                ax.set_ylabel(label, fontsize=6.5)

    plt.suptitle("3-channel coherence check — adjacent slices", fontsize=9, y=1.01)
    plt.tight_layout()
    out = CFG.VIS_DIR / f"{run_id}_channel_coherence.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Channel coherence saved → {out.name}")



# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(run_id: str, logger) -> bool:
    """
    Smoke test for Module 3. Runs before any full processing.

    Checks performed
    ----------------
    1. Required files exist: master_index.csv, all three split CSVs
    2. Split CSVs have patch_path column and at least some paths are non-null
    3. Split leakage: no study_id appears in more than one split
    4. Load 4 random .npy patches — check shape (3,128,128), dtype float32,
       value range is reasonable (normalised values typically in [-3, 3])
    5. Build a tiny Dataset (8 rows) and pull one DataLoader batch
       — confirm tensor shape, dtype, and label values are in {0, 1, 2}
    6. Run augmentation pipeline once — confirm output shape matches input
    7. Save all 3 visualisation checks on the small sample
       (patch grid, augmentation, channel coherence)
    """
    logger.info("=" * 65)
    logger.info("Smoke test — Module 3")
    logger.info("=" * 65)
    ensure_dirs()
    issues = []

    # ── 1. File existence ─────────────────────────────────────────────────────
    logger.info("\n── Check 1: Required files ─────────────────────────────────")
    required = {
        "master_index": CFG.MASTER_INDEX_PATH,
        "split_train":  CFG.SPLIT_TRAIN,
        "split_val":    CFG.SPLIT_VAL,
        "split_test":   CFG.SPLIT_TEST,
    }
    for name, path in required.items():
        if path.exists():
            logger.info(f"  {name:15s} ✓  {path}")
        else:
            msg = f"{name} not found: {path}"
            logger.error(f"  {name:15s} ✗  {path}")
            issues.append(msg)

    if issues:
        log_event(run_id, "module3", "smoke_test_failed",
                  stage="file_existence", issues=issues)
        logger.error("  Cannot continue — required files missing.")
        return False

    # ── 2. Load split CSVs, resolve patch_path from filenames, validate train+val
    logger.info("\n── Check 2: CSV structure & patch_path (train + val only) ──")
    split_dfs = {
        "train": pd.read_csv(CFG.SPLIT_TRAIN),
        "val":   pd.read_csv(CFG.SPLIT_VAL),
        "test":  pd.read_csv(CFG.SPLIT_TEST),
    }

    # Resolve patch paths from the .npy filename convention — no master index read.
    # Test split intentionally has no patches — patches are built at inference time.
    for name in ["train", "val"]:
        split_dfs[name] = resolve_patch_paths(split_dfs[name], split_name=name)

    # Validate patch_path presence for train and val.
    # A partial fill (e.g. only smoke studies processed) is acceptable for the
    # smoke test — we only hard-fail if BOTH train and val have zero patches.
    n_filled_by_split = {}
    for name in ["train", "val"]:
        df       = split_dfs[name]
        n_filled = (~df["patch_path"].isna()).sum()
        n_total  = len(df)
        pct      = 100 * n_filled / n_total if n_total > 0 else 0
        n_filled_by_split[name] = n_filled
        status   = "✓" if n_filled > 0 else "⚠"
        logger.info(
            f"  {name:5s}  patch_path filled={n_filled:,}/{n_total:,} "
            f"({pct:.1f}%)  {status}"
        )
        if n_filled == 0:
            logger.warning(
                f"  {name}: 0 patches found — module2 smoke test may have "
                f"saved patches into the other split. Continuing."
            )

    if n_filled_by_split["train"] == 0 and n_filled_by_split["val"] == 0:
        issues.append(
            f"No .npy patches found in either train or val — run module2 first"
        )

    # Log test split status without requiring patches
    test_df = split_dfs["test"]
    logger.info(
        f"  test   patch_path skipped (no patches by design — "
        f"built at inference time)  {len(test_df):,} rows"
    )

    if issues:
        log_event(run_id, "module3", "smoke_test_failed",
                  stage="csv_structure", issues=issues)
        return False

    # ── 3. Split leakage ──────────────────────────────────────────────────────
    logger.info("\n── Check 3: Split leakage ──────────────────────────────────")
    leakage_ok = verify_no_leakage(split_dfs, logger, run_id)
    if not leakage_ok:
        issues.append("Split leakage detected")

    # ── 4. Load 4 random .npy patches (train+val, whichever has patches) ───────
    logger.info("\n── Check 4: Load 4 random .npy patches ─────────────────────")
    trainval = pd.concat([split_dfs["train"], split_dfs["val"]], ignore_index=True)
    valid    = trainval.dropna(subset=["patch_path"])
    logger.info(f"  {len(valid):,} rows with resolved patch_path across train+val")
    if len(valid) == 0:
        issues.append("No valid patches to load — skipping checks 4-7")
        log_event(run_id, "module3", "smoke_test_failed",
                  stage="no_patches", issues=issues)
        return False
    sample4  = valid.sample(min(4, len(valid)), random_state=CFG.RANDOM_SEED)

    for _, row in sample4.iterrows():
        path = row["patch_path"]
        try:
            patch = np.load(path).astype(np.float32)
            exp   = (3, CFG.PATCH_SIZE_PX, CFG.PATCH_SIZE_PX)
            shape_ok = patch.shape == exp
            range_ok = (-5.0 <= patch.min()) and (patch.max() <= 5.0)
            dtype_ok = patch.dtype == np.float32
            flags = (
                f"shape={'✓' if shape_ok else '✗'}{patch.shape}  "
                f"dtype={'✓' if dtype_ok else '✗'}{patch.dtype}  "
                f"range={'✓' if range_ok else '✗'}[{patch.min():.3f}, {patch.max():.3f}]"
            )
            logger.info(
                f"  study={row['study_id']}  {row['condition'][:20]}  "
                f"{row['level']}  {flags}"
            )
            if not shape_ok:
                issues.append(f"Wrong shape {patch.shape}, expected {exp}: {path}")
            if not range_ok:
                issues.append(f"Out-of-range values [{patch.min():.2f},{patch.max():.2f}]: {path}")
        except Exception as exc:
            issues.append(f"Failed to load {path}: {exc}")
            logger.error(f"  ✗ Load failed: {exc}")

    # ── 5. Tiny Dataset + DataLoader batch ────────────────────────────────────
    logger.info("\n── Check 5: Tiny Dataset + DataLoader batch ─────────────────")
    try:
        # Tiny splits from train+val only — test has no patches by design
        tiny_df  = valid.sample(min(8, len(valid)), random_state=CFG.RANDOM_SEED)
        n        = len(tiny_df)
        tr_df    = tiny_df.iloc[:max(1, n * 2 // 3)].copy()
        vl_df    = tiny_df.iloc[max(1, n * 2 // 3):].copy()
        te_df    = pd.DataFrame(columns=vl_df.columns)   # empty — no patches
        tiny_splits = {"train": tr_df, "val": vl_df, "test": te_df}

        tiny_loaders, tiny_train_ds = build_dataloaders(
            tiny_splits, batch_size=min(4, len(tr_df)), num_workers=0
        )
        batch_x, batch_y = next(iter(tiny_loaders["train"]))

        shape_ok  = batch_x.ndim == 4 and batch_x.shape[1] == 3
        dtype_ok  = batch_x.dtype == torch.float32
        label_ok  = all(int(l) in {0, 1, 2} for l in batch_y)

        logger.info(f"  batch x : {tuple(batch_x.shape)}  dtype={batch_x.dtype}  "
                    f"{'✓' if shape_ok and dtype_ok else '✗'}")
        logger.info(f"  batch y : {batch_y.tolist()}  "
                    f"all valid labels={'✓' if label_ok else '✗'}")
        logger.info(f"  x range : [{batch_x.min():.4f}, {batch_x.max():.4f}]")

        if not (shape_ok and dtype_ok and label_ok):
            issues.append("DataLoader batch shape/dtype/label check failed")

        log_event(run_id, "module3", "smoke_dataloader_ok",
                  batch_shape=list(batch_x.shape),
                  labels=batch_y.tolist())

    except Exception as exc:
        issues.append(f"DataLoader failed: {exc}")
        logger.error(f"  ✗ DataLoader error: {exc}")
        tiny_train_ds = None

    # ── 6. Augmentation pipeline check ───────────────────────────────────────
    logger.info("\n── Check 6: Augmentation pipeline ──────────────────────────")
    try:
        aug         = build_train_transforms()
        test_patch  = np.load(sample4.iloc[0]["patch_path"]).astype(np.float32)
        hwc         = test_patch.transpose(1, 2, 0)
        aug_result  = aug(image=hwc)
        aug_patch   = aug_result["image"].transpose(2, 0, 1)
        shape_match = aug_patch.shape == test_patch.shape
        logger.info(
            f"  input shape : {test_patch.shape}  "
            f"→  output shape : {aug_patch.shape}  "
            f"{'✓' if shape_match else '✗'}"
        )
        if not shape_match:
            issues.append(f"Augmentation changed shape: {test_patch.shape} → {aug_patch.shape}")
    except Exception as exc:
        issues.append(f"Augmentation pipeline error: {exc}")
        logger.error(f"  ✗ Augmentation error: {exc}")

    # ── 7. Visualisation checkpoints on small sample ──────────────────────────
    logger.info("\n── Check 7: Visualisation checkpoints ──────────────────────")
    try:
        # valid is already train+val only — correct source for visualisation
        smoke_ds = RSNADataset(valid.sample(
            min(16, len(valid)), random_state=CFG.RANDOM_SEED
        ), transform=None)

        if len(smoke_ds) > 0:
            vis_patch_grid(smoke_ds, run_id, logger, condition="smoke_sample")
            vis_augmentation_check(smoke_ds, run_id, logger, n_aug=4)
            vis_channel_coherence(smoke_ds, run_id, logger,
                                  n=min(4, len(smoke_ds)))
            logger.info("  All 3 visualisation checks saved. ✓")
        else:
            logger.warning("  No valid patches for visualisation checks.")
    except Exception as exc:
        issues.append(f"Visualisation error: {exc}")
        logger.error(f"  ✗ Visualisation error: {exc}")

    # ── Result ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    if issues:
        logger.error(f"Smoke test FAILED — {len(issues)} issue(s):")
        for iss in issues:
            logger.error(f"  • {iss}")
        log_event(run_id, "module3", "smoke_test_failed",
                  n_issues=len(issues), issues=issues)
        return False
    else:
        logger.info("Smoke test PASSED — all checks OK. ✓")
        log_event(run_id, "module3", "smoke_test_passed")
        return True

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, smoke_only: bool = False):
    run_id     = run_id or make_run_id("m3")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, "module3")

    logger.info("=" * 65)
    logger.info(f"Module 3 — Dataset & DataLoaders | run_id={run_id}")
    logger.info(f"  PATCHES_DIR : {CFG.PATCHES_DIR}")
    logger.info(f"  BATCH_SIZE  : {CFG.BATCH_SIZE}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, "module3", "script_start",
              batch_size=CFG.BATCH_SIZE,
              aug={
                  "hflip_p":      CFG.AUG_HFLIP_P,
                  "ssr_p":        CFG.AUG_SSR_P,
                  "ssr_shift":    CFG.AUG_SSR_SHIFT,
                  "ssr_scale":    CFG.AUG_SSR_SCALE,
                  "ssr_rotate":   CFG.AUG_SSR_ROTATE,
                  "brightness_p": CFG.AUG_BRIGHTNESS_P,
                  "gamma_p":      CFG.AUG_GAMMA_P,
                  "noise_p":      CFG.AUG_NOISE_P,
              })

    ensure_dirs()

    try:
        # ── smoke test (always runs first) ────────────────────────────────────
        smoke_ok = smoke_test(run_id, logger)
        if smoke_only:
            logger.info("--smoke-only: stopping after smoke test.")
            return
        if not smoke_ok:
            logger.warning(
                "Smoke test had issues — review above before proceeding."
            )

        # ── load split CSVs ────────────────────────────────────────────────────
        for path in [CFG.SPLIT_TRAIN, CFG.SPLIT_VAL, CFG.SPLIT_TEST]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {path}. Run module1 first."
                )

        split_dfs = {
            "train": pd.read_csv(CFG.SPLIT_TRAIN),
            "val":   pd.read_csv(CFG.SPLIT_VAL),
            "test":  pd.read_csv(CFG.SPLIT_TEST),
        }

        # Resolve patch_path from .npy filenames — no master index read.
        # Test has no patches by design; skip it here.
        for name in ["train", "val"]:
            split_dfs[name] = resolve_patch_paths(split_dfs[name], split_name=name)

        for name, df in split_dfs.items():
            if name == "test":
                logger.info(
                    f"  {name:5s} : {df['study_id'].nunique():,} studies  "
                    f"{len(df):,} annotations  "
                    f"(no patches — built at inference time)"
                )
            else:
                n_filled = (~df["patch_path"].isna()).sum()
                logger.info(
                    f"  {name:5s} : {df['study_id'].nunique():,} studies  "
                    f"{len(df):,} annotations  "
                    f"patch_path resolved: {n_filled:,}"
                )

        # ── step 2: alignment verification ────────────────────────────────────
        logger.info("\n── Step 2: Alignment verification ─────────────────────────")
        # Alignment verification uses train only — test has no patches, val can influence training decisions so excluded here
        verify_alignment(split_dfs["train"], logger, run_id)

        # ── build datasets ─────────────────────────────────────────────────────
        logger.info("\n── Step 3: Building datasets ───────────────────────────────")
        loaders, train_ds = build_dataloaders(split_dfs)

        label_counts = train_ds.label_counts()
        total        = sum(label_counts.values())
        logger.info(f"\n  Training set label counts (for class-weight reference):")
        for cls, count in label_counts.items():
            sev    = CFG.SEVERITY_INV[cls]
            weight = total / (3 * count) if count > 0 else float("inf")
            logger.info(f"    {sev:12s} (cls {cls}): {count:,}  weight={weight:.4f}")

        log_event(run_id, "module3", "label_counts",
                  train_counts=label_counts)

        # Quick DataLoader sanity check
        logger.info("\n  DataLoader batch sanity check …")
        batch_x, batch_y = next(iter(loaders["train"]))
        logger.info(f"    batch x shape : {tuple(batch_x.shape)}")
        logger.info(f"    batch y shape : {tuple(batch_y.shape)}")
        logger.info(f"    x dtype       : {batch_x.dtype}")
        logger.info(f"    x min/max     : {batch_x.min():.4f} / {batch_x.max():.4f}")
        logger.info(f"    y unique      : {batch_y.unique().tolist()}")

        log_event(run_id, "module3", "dataloader_sanity",
                  batch_shape=list(batch_x.shape),
                  x_min=float(batch_x.min()),
                  x_max=float(batch_x.max()))

        # ── visualisation checkpoints ──────────────────────────────────────────
        logger.info("\n── Step 5: Visualisation checkpoints ───────────────────────")

        # A: patch grids per condition group
        for cond in CFG.CONDITIONS:
            cond_ds = RSNADataset(
                split_dfs["train"],
                transform=None,
                condition_filter=cond,
            )
            if len(cond_ds) == 0:
                logger.warning(f"  No train patches for condition: {cond}")
                continue
            vis_patch_grid(cond_ds, run_id, logger, condition=cond)

        # B: augmentation check (one sample, 8 augmented versions)
        if len(train_ds) > 0:
            vis_augmentation_check(train_ds, run_id, logger)

        # C: 3-channel coherence (6 samples)
        if len(train_ds) > 0:
            vis_channel_coherence(train_ds, run_id, logger)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, "module3", "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name="module3_dataset",
        start_time=start_time,
        end_time=end_time,
        notes="Dataset and DataLoader verification complete",
    )
    log_event(run_id, "module3", "script_complete", end_time=end_time)
    logger.info(f"\nModule 3 complete. Visualisations in: {CFG.VIS_DIR}")


def get_split_dataloaders(
    condition_filter: Optional[str] = None,
    batch_size:       Optional[int] = None,
    num_workers:      int           = 4,
) -> Tuple[dict, RSNADataset]:
    """
    Convenience function imported by module4.
    Loads split CSVs from disk, resolves patch_path from .npy filenames,
    and returns (loaders_dict, train_dataset).
    Does NOT read master_index.csv — master index is read-only after module 1.

    Usage in module4:
        from src.module3 import get_split_dataloaders
        loaders, train_ds = get_split_dataloaders(condition_filter="spinal_canal_stenosis")
    """
    split_dfs = {}
    for name, path in [("train", CFG.SPLIT_TRAIN),
                       ("val",   CFG.SPLIT_VAL),
                       ("test",  CFG.SPLIT_TEST)]:
        df = pd.read_csv(path)
        # Resolve patch paths for train and val; test has no patches by design
        if name in ("train", "val"):
            df = resolve_patch_paths(df, split_name=name)
        split_dfs[name] = df

    return build_dataloaders(
        split_dfs,
        condition_filter=condition_filter,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Module 3 — Dataset, Augmentation & DataLoader verification"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run smoke test checks then exit")
    args = parser.parse_args()
    run(run_id=args.run_id, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()