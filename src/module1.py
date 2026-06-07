"""
src/module1.py
==============
Module 1: Data Collection & Preprocessing

Steps (exactly as specified in the strategy document):
  1. Build master index — join all 3 CSVs into single source of truth
  2. Verify every (study_id, condition, level) has a coordinate row; log mismatches
  3. Smoke test on 1-2 studies before full run
  4. Dataset split — study-level 70/15/15 stratified on severity distribution
  5. Save split_train.csv, split_val.csv, split_test.csv
  6. Visualisation checkpoint — label distribution bar chart

Run:
    python -m src.module1
    python -m src.module1 --smoke-only
    python -m src.module1 --run-id my_run_001
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import CFG, ensure_dirs
from src.logger import get_logger, log_event, make_run_id, write_run_summary


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build master index
# ─────────────────────────────────────────────────────────────────────────────

def build_master_index(logger, run_id: str) -> pd.DataFrame:
    """
    Join train.csv + train_series_descriptions.csv + train_label_coordinates.csv.

    Every row in the returned DataFrame contains:
        study_id, series_id, series_type, instance_number,
        condition, level, x, y, severity_label
    """
    logger.info("── Step 1: Building master index ──────────────────────────")

    # Load
    train_df  = pd.read_csv(CFG.TRAIN_CSV)
    series_df = pd.read_csv(CFG.SERIES_CSV)
    coords_df = pd.read_csv(CFG.COORDS_CSV)

    logger.info(f"  train.csv       : {len(train_df):,} rows  "
                f"({train_df['study_id'].nunique():,} studies)")
    logger.info(f"  series_desc.csv : {len(series_df):,} rows")
    logger.info(f"  coords.csv      : {len(coords_df):,} rows")

    log_event(run_id, "module1", "csvs_loaded",
              train_rows=len(train_df),
              series_rows=len(series_df),
              coords_rows=len(coords_df),
              n_studies=int(train_df["study_id"].nunique()))

    # ── melt train.csv: wide → long ──────────────────────────────────────────
    label_cols = [c for c in train_df.columns if c != "study_id"]

    def _parse_col(col: str):
        for lvl in CFG.LEVELS:
            if col.endswith("_" + lvl):
                cond = col[: -(len(lvl) + 1)]
                return cond, lvl
        return None, None

    train_long = (
        train_df
        .melt(id_vars="study_id", value_vars=label_cols,
              var_name="condition_level", value_name="severity_label_str")
        .dropna(subset=["severity_label_str"])
    )
    parsed = train_long["condition_level"].apply(
        lambda x: pd.Series(_parse_col(x), index=["condition", "level"])
    )
    train_long = pd.concat([train_long, parsed], axis=1)
    train_long = train_long.dropna(subset=["condition", "level"]).copy()
    train_long["severity_label"] = train_long["severity_label_str"].map(CFG.SEVERITY_MAP)
    train_long = train_long[["study_id", "condition", "level",
                               "severity_label_str", "severity_label"]]

    logger.info(f"  train_long      : {len(train_long):,} rows after melt")

    # ── normalise coords condition + level to match train_long format ────────
    # coords.csv stores:  "Spinal Canal Stenosis", "L1/L2"
    # train.csv columns:  "spinal_canal_stenosis",  "l1_l2"
    coords_df = coords_df.copy()
    coords_df["condition"] = (
        coords_df["condition"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    coords_df["level"] = (
        coords_df["level"]
        .str.strip()
        .str.lower()
        .str.replace("/", "_", regex=False)
    )
    logger.info(f"  coords condition samples : {coords_df['condition'].unique()[:3].tolist()}")
    logger.info(f"  coords level     samples : {coords_df['level'].unique()[:3].tolist()}")
    logger.info(f"  train_long cond  samples : {train_long['condition'].unique()[:3].tolist()}")
    logger.info(f"  train_long level samples : {train_long['level'].unique()[:3].tolist()}")

    # ── enrich coords with series type ───────────────────────────────────────
    coords_enriched = coords_df.merge(
        series_df[["study_id", "series_id", "series_description"]],
        on=["study_id", "series_id"],
        how="left",
    ).rename(columns={"series_description": "series_type"})

    # ── join labels ← coords ──────────────────────────────────────────────────
    master = train_long.merge(
        coords_enriched[[
            "study_id", "series_id", "series_type",
            "instance_number", "condition", "level", "x", "y"
        ]],
        on=["study_id", "condition", "level"],
        how="left",
    )

    logger.info(f"  master index    : {len(master):,} rows")
    log_event(run_id, "module1", "master_index_built",
              total_rows=len(master),
              n_studies=int(master["study_id"].nunique()))

    return master


# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify coordinates
# ─────────────────────────────────────────────────────────────────────────────

def verify_coordinates(master: pd.DataFrame, logger, run_id: str) -> pd.DataFrame:
    """
    Check every (study_id, condition, level) has x, y, instance_number.
    Log mismatches to run_log.jsonl.  Drop missing rows and log impact.
    """
    logger.info("── Step 2: Verifying coordinates ───────────────────────────")

    missing_mask     = master[["x", "y", "instance_number"]].isnull().any(axis=1)
    n_missing        = int(missing_mask.sum())
    affected_studies = master.loc[missing_mask, "study_id"].unique().tolist()

    if n_missing > 0:
        logger.warning(
            f"  {n_missing:,} rows missing coordinates "
            f"({len(affected_studies)} studies affected)"
        )
        logger.warning(f"  Affected study IDs: {affected_studies[:10]}"
                       + (" …" if len(affected_studies) > 10 else ""))
        log_event(run_id, "module1", "missing_coordinates",
                  n_missing=n_missing,
                  n_affected_studies=len(affected_studies),
                  affected_study_ids=affected_studies)
    else:
        logger.info("  All rows have coordinates. ✓")
        log_event(run_id, "module1", "coordinate_check_passed", n_missing=0)

    clean = master.loc[~missing_mask].copy()
    logger.info(f"  Rows remaining after drop: {len(clean):,}")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# 3. Smoke test on 1–2 studies
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(master: pd.DataFrame, logger, run_id: str,
               n_studies: int = None) -> bool:
    """
    Load CFG.SMOKE_N_STUDIES (default 2) studies and:
      - Print all their rows
      - Confirm series types are correctly identified
      - Confirm instance numbers exist as files on disk
      - Confirm (x, y) fall within image dimensions (requires pydicom)

    Returns True if all checks pass, False otherwise.
    """
    n = n_studies or CFG.SMOKE_N_STUDIES
    logger.info(f"── Step 3: Smoke test ({n} studies) ────────────────────────")

    # Pick studies that BOTH exist in the cleaned master AND have their DICOM
    # series directory present on disk.  The first N unique IDs in the master
    # can be among the studies that had missing coordinates and were already
    # dropped, which would leave the smoke test with an empty DataFrame.
    all_ids   = np.sort(master["study_id"].unique())
    cutoff    = int(len(all_ids) * CFG.TRAIN_RATIO)  # first 70% = future train
    safe_ids  = all_ids[:cutoff]
    valid_ids = []
    for sid in safe_ids:
        series_ids = master.loc[master["study_id"] == sid, "series_id"].unique()
        has_dicom  = any(
            (CFG.DICOM_ROOT / str(int(sid)) / str(int(serid))).exists()
            for serid in series_ids
        )
        if has_dicom:
            valid_ids.append(sid)
        if len(valid_ids) == n:
            break
    logger.info(
        f"  Candidate pool: first {cutoff:,} of {len(all_ids):,} studies "
        f"(future train ~70%) — val/test studies never selected here."
    )

    if not valid_ids:
        logger.error("  Smoke test: no studies with DICOM data found on disk.")
        log_event(run_id, "module1", "smoke_test_no_data")
        return False

    sample_ids = valid_ids
    sample_df  = master[master["study_id"].isin(sample_ids)]

    logger.info(f"  Selected studies (with DICOM on disk): {sample_ids}")
    logger.info(f"\nMaster index rows for studies: {sample_ids}")
    print(sample_df.to_string(index=False))

    issues = []

    for _, row in sample_df.iterrows():
        sid   = row["study_id"]
        serid = row["series_id"]
        inst  = row["instance_number"]
        x, y  = row["x"], row["y"]

        # Series type identified?
        if pd.isna(row["series_type"]):
            issues.append(f"study={sid} series={serid}: series_type is NaN")

        # DICOM file exists on disk?
        dcm_path = CFG.DICOM_ROOT / str(int(sid)) / str(int(serid)) / f"{int(inst)}.dcm"
        if not dcm_path.exists():
            issues.append(f"Missing DICOM: {dcm_path}")
            continue   # can't check dims without the file

        # (x, y) within image bounds?
        try:
            import pydicom
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
            arr_shape = ds.pixel_array.shape
            h = arr_shape[0]
            w = arr_shape[1] if len(arr_shape) > 1 else arr_shape[0]
            if not (0 <= x < w and 0 <= y < h):
                issues.append(
                    f"study={sid} coord ({x:.0f},{y:.0f}) "
                    f"outside image bounds ({w}×{h})"
                )
        except ImportError:
            logger.warning("  pydicom not installed — skipping pixel-bounds check")
        except Exception as exc:
            issues.append(f"study={sid} DICOM read error: {exc}")

    if issues:
        for iss in issues:
            logger.error(f"  SMOKE ISSUE: {iss}")
        log_event(run_id, "module1", "smoke_test_failed",
                  n_studies=n, issues=issues)
        logger.warning("  Smoke test finished WITH issues — review before full run.")
        return False
    else:
        logger.info("  Smoke test PASSED — all checks OK. ✓")
        log_event(run_id, "module1", "smoke_test_passed", n_studies=n)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Dataset split + save
# ─────────────────────────────────────────────────────────────────────────────

def _stratification_key(master: pd.DataFrame) -> pd.Series:
    """
    Build a per-study key capturing severity distribution across all 25 conditions.
    Encodes severity_label for each (condition × level) as a joined string
    e.g. "0_0_1_0_2_0_0_0_1_0_…" (25 values).
    Rare keys (count < 2) are collapsed to the study's max severity.
    """
    study_labels = (
        master
        .groupby(["study_id", "condition", "level"])["severity_label"]
        .first()
        .unstack(["condition", "level"])
        .fillna(0)
        .astype(int)
    )
    key = study_labels.apply(lambda r: "_".join(r.astype(str)), axis=1)

    # Collapse rare combos so sklearn stratified split doesn't fail
    rare = key.value_counts()[lambda s: s < 2].index
    if len(rare):
        max_sev = master.groupby("study_id")["severity_label"].max().astype(str)
        key[key.isin(rare)] = max_sev[key.isin(rare)]
    return key


def split_and_save(master: pd.DataFrame, logger, run_id: str):
    """
    Study-level 70/15/15 stratified split.
    Saves split_train.csv, split_val.csv, split_test.csv.
    Returns (train_ids, val_ids, test_ids).
    """
    logger.info("── Steps 4–5: Dataset split ────────────────────────────────")

    ensure_dirs()
    study_ids = master["study_id"].unique()
    strat_key = _stratification_key(master).reindex(study_ids)

    # Split 1: train vs (val + test)
    train_ids, valtest_ids = train_test_split(
        study_ids,
        test_size=(CFG.VAL_RATIO + CFG.TEST_RATIO),
        stratify=strat_key.loc[study_ids],
        random_state=CFG.RANDOM_SEED,
    )

    # Split 2: val vs test
    strat_vt = _stratification_key(
        master[master["study_id"].isin(valtest_ids)]
    ).reindex(valtest_ids)

    val_ids, test_ids = train_test_split(
        valtest_ids,
        test_size=CFG.TEST_RATIO / (CFG.VAL_RATIO + CFG.TEST_RATIO),
        stratify=strat_vt,
        random_state=CFG.RANDOM_SEED,
    )

    logger.info(f"  Train: {len(train_ids):,} studies")
    logger.info(f"  Val  : {len(val_ids):,} studies")
    logger.info(f"  Test : {len(test_ids):,} studies  (LOCKED)")

    # Save
    def _save(ids, path, split_name):
        df = master[master["study_id"].isin(ids)].copy()
        df.insert(0, "split", split_name)
        df.to_csv(path, index=False)
        logger.info(f"  Saved {path.name}  ({len(df):,} rows)")

    _save(train_ids, CFG.SPLIT_TRAIN, "train")
    _save(val_ids,   CFG.SPLIT_VAL,   "val")
    _save(test_ids,  CFG.SPLIT_TEST,  "test")

    # Also save master index
    master.to_csv(CFG.MASTER_INDEX_PATH, index=False)
    logger.info(f"  Saved master_index.csv ({len(master):,} rows)")

    log_event(run_id, "module1", "splits_saved",
              n_train=len(train_ids),
              n_val=len(val_ids),
              n_test=len(test_ids))

    return list(train_ids), list(val_ids), list(test_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Split integrity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(train_ids, val_ids, test_ids, logger, run_id: str):
    """Hard check: no study_id appears in more than one split."""
    sets  = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    ok    = True
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                logger.error(
                    f"  LEAKAGE: {len(overlap)} studies in both "
                    f"{names[i]} and {names[j]}: {list(overlap)[:5]}"
                )
                log_event(run_id, "module1", "split_leakage",
                          split_a=names[i], split_b=names[j],
                          overlap_ids=list(overlap))
                ok = False
    if ok:
        logger.info("  Split leakage check PASSED. ✓")
        log_event(run_id, "module1", "leakage_check_passed")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def visualise_label_distribution(master: pd.DataFrame, run_id: str, logger):
    """
    Print and plot label distribution across all 25 conditions.
    Saves PNG to CFG.VIS_DIR.
    """
    logger.info("── Step 6: Label distribution visualisation ─────────────────")

    ensure_dirs()
    all_conds = [f"{c}_{l}" for c in CFG.CONDITIONS for l in CFG.LEVELS]

    # Build counts
    counts = (
        master
        .assign(cond_lvl=master["condition"] + "_" + master["level"])
        .groupby(["cond_lvl", "severity_label"])["study_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(all_conds, fill_value=0)
        .rename(columns=CFG.SEVERITY_INV)
        .reindex(columns=["Normal/Mild", "Moderate", "Severe"], fill_value=0)
    )

    # Print table
    logger.info("\n" + counts.to_string())
    log_event(run_id, "module1", "label_distribution",
              totals={
                  "Normal/Mild": int(counts["Normal/Mild"].sum()),
                  "Moderate":    int(counts["Moderate"].sum()),
                  "Severe":      int(counts["Severe"].sum()),
              })

    # Plot
    fig, ax = plt.subplots(figsize=(20, 7))
    x     = np.arange(len(all_conds))
    w     = 0.28
    cols  = {"Normal/Mild": "#4CAF50", "Moderate": "#FF9800", "Severe": "#F44336"}
    for i, (sev, col) in enumerate(cols.items()):
        vals = counts[sev].values if sev in counts.columns else np.zeros(len(all_conds))
        ax.bar(x + i * w, vals, w, label=sev, color=col, alpha=0.85, edgecolor="none")

    ax.set_xticks(x + w)
    ax.set_xticklabels(all_conds, rotation=75, ha="right", fontsize=7.5)
    ax.set_ylabel("Number of studies", fontsize=11)
    ax.set_title("Severity label distribution — all 25 conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_path = CFG.VIS_DIR / f"{run_id}_label_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Chart saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(run_id: str = None, smoke_only: bool = False):
    """
    Execute all Module 1 steps.

    Parameters
    ----------
    run_id     : unique identifier for this run (auto-generated if None)
    smoke_only : if True, stop after the smoke test
    """
    run_id     = run_id or make_run_id("m1")
    start_time = datetime.now().isoformat()
    logger     = get_logger(run_id, "module1")

    logger.info("=" * 65)
    logger.info(f"Module 1 — Data Collection & Preprocessing | run_id={run_id}")
    logger.info(f"  DATA_DIR  : {CFG.DATA_DIR}")
    logger.info(f"  DICOM_ROOT: {CFG.DICOM_ROOT}")
    logger.info(f"  OUTPUT_DIR: {CFG.OUTPUT_DIR}")
    logger.info("=" * 65 + "\n")

    log_event(run_id, "module1", "script_start",
              data_dir=str(CFG.DATA_DIR),
              dicom_root=str(CFG.DICOM_ROOT),
              random_seed=CFG.RANDOM_SEED)

    try:
        # 1. Master index
        master = build_master_index(logger, run_id)

        # 2. Verify coordinates
        master = verify_coordinates(master, logger, run_id)

        # 3. Smoke test
        smoke_passed = smoke_test(master, logger, run_id)
        if smoke_only:
            logger.info("--smoke-only: stopping after smoke test.")
            return

        if not smoke_passed:
            logger.warning(
                "Smoke test had issues but continuing with full run. "
                "Investigate the issues logged above."
            )

        # 4 + 5. Split and save
        train_ids, val_ids, test_ids = split_and_save(master, logger, run_id)

        # Leakage check
        verify_no_leakage(train_ids, val_ids, test_ids, logger, run_id)

        # 6. Visualisation
        visualise_label_distribution(master, run_id, logger)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"FATAL ERROR:\n{tb}")
        log_event(run_id, "module1", "fatal_error",
                  error=str(exc), traceback=tb)
        raise

    # Wrap up
    end_time = datetime.now().isoformat()
    write_run_summary(
        run_id=run_id,
        model_name="module1_data_prep",
        start_time=start_time,
        end_time=end_time,
        notes="Data collection & preprocessing complete",
    )
    log_event(run_id, "module1", "script_complete", end_time=end_time)
    logger.info(f"\nModule 1 complete. Outputs in: {CFG.OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Module 1 — Data Collection & Preprocessing"
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Unique run identifier (auto-generated if not given)"
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run smoke test on 2 studies then exit"
    )
    args = parser.parse_args()
    run(run_id=args.run_id, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()