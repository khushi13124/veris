"""
config/settings.py
==================
Single source of truth for every path, constant, and hyperparameter used
across the project.  Every module does:

    from config.settings import CFG

and reads values from CFG.  Nothing is hard-coded anywhere else.
"""

from pathlib import Path


class CFG:
    # ── project root ──────────────────────────────────────────────────────────
    # Resolved relative to this file so the project is portable.
    ROOT: Path = Path(__file__).resolve().parent.parent

    # ── raw data ──────────────────────────────────────────────────────────────
    DATA_DIR:          Path = ROOT / "data" / "raw"
    #DATA_DIR:          Path = ROOT / "data" / "extracted"
    TRAIN_CSV:         Path = DATA_DIR / "train.csv"
    SERIES_CSV:        Path = DATA_DIR / "train_series_descriptions.csv"
    COORDS_CSV:        Path = DATA_DIR / "train_label_coordinates.csv"
    DICOM_ROOT:        Path = DATA_DIR / "train_images"   # /{study_id}/{series_id}/{inst}.dcm

    # ── derived data ──────────────────────────────────────────────────────────
    PATCHES_DIR:       Path = ROOT / "data" / "patches"   # .npy files written by module2
    MASTER_INDEX_PATH: Path = ROOT / "outputs" / "master_index.csv"

    # ── outputs ───────────────────────────────────────────────────────────────
    OUTPUT_DIR:        Path = ROOT / "outputs"
    SPLITS_DIR:        Path = ROOT / "outputs" / "splits"
    LOGS_DIR:          Path = ROOT / "outputs" / "logs"
    VIS_DIR:           Path = ROOT / "outputs" / "vis"

    SPLIT_TRAIN:       Path = SPLITS_DIR / "split_train.csv"
    SPLIT_VAL:         Path = SPLITS_DIR / "split_val.csv"
    SPLIT_TEST:        Path = SPLITS_DIR / "split_test.csv"  # LOCKED after creation

    # ── dataset split ratios ──────────────────────────────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO:   float = 0.15
    TEST_RATIO:  float = 0.15
    RANDOM_SEED: int   = 42

    # ── label schema ─────────────────────────────────────────────────────────
    CONDITIONS: list = [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    LEVELS: list = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

    SEVERITY_MAP: dict = {
        "Normal/Mild": 0,
        "Moderate":    1,
        "Severe":      2,
    }
    SEVERITY_INV: dict = {0: "Normal/Mild", 1: "Moderate", 2: "Severe"}

    # Competition sample weights (used in log-loss metric)
    SEVERITY_WEIGHTS: dict = {0: 1, 1: 2, 2: 4}

    # ── image processing (module 2) ───────────────────────────────────────────
    CROP_SIZE_PX:    int   = 160    # initial square crop centred on (x, y)
    PATCH_SIZE_PX:   int   = 128    # resize target for each channel
    N_SLICES:        int   = 3      # [inst-1, inst, inst+1] stacked as channels
    IMAGENET_MEAN: list    = [0.485, 0.456, 0.406]
    IMAGENET_STD:  list    = [0.229, 0.224, 0.225]
    PERCENTILE_LOW:  float = 1.0    # fallback windowing lower percentile
    PERCENTILE_HIGH: float = 99.0   # fallback windowing upper percentile

    # ── augmentation (module 3) ───────────────────────────────────────────────
    AUG_HFLIP_P:          float = 0.5
    AUG_SSR_P:            float = 0.5   # ShiftScaleRotate
    AUG_SSR_SHIFT:        float = 0.1
    AUG_SSR_SCALE:        float = 0.1
    AUG_SSR_ROTATE:       int   = 10    # degrees
    AUG_BRIGHTNESS_P:     float = 0.3
    AUG_GAMMA_P:          float = 0.3
    AUG_NOISE_P:          float = 0.2

    # ── training (module 4) ───────────────────────────────────────────────────
    BATCH_SIZE:       int   = 16
    NUM_EPOCHS:       int   = 60
    LEARNING_RATE:    float = 1e-4
    WEIGHT_DECAY:     float = 1e-4
    WARMUP_EPOCHS:    int   = 5
    WARMUP_START_LR:  float = 1e-6
    PHASE2_LR:        float = 1e-5      # lr after backbone unfreeze
    PHASE2_START:     int   = 6         # epoch (1-indexed) to unfreeze layer4
    EARLY_STOP_PATIENCE:  int   = 15
    EARLY_STOP_MIN_DELTA: float = 1e-4
    CHECKPOINT_EVERY: int   = 10        # save epoch checkpoint every N epochs

    # ── model checkpoints ────────────────────────────────────────────────────
    CKPT_DIR: Path = ROOT / "outputs" / "checkpoints"

    # ── inference (module 5) ─────────────────────────────────────────────────
    TTA_ROTATIONS: list = [0, 5, -5]    # degrees; hflip handled separately
    SMOKE_N_STUDIES: int = 2            # studies used for smoke tests

    # ── series type labels (as they appear in train_series_descriptions.csv) ──
    SERIES_SAG_T1: str = "Sagittal T1"
    SERIES_SAG_T2: str = "Sagittal T2/STIR"
    SERIES_AXL_T2: str = "Axial T2"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        CFG.PATCHES_DIR,
        CFG.SPLITS_DIR,
        CFG.LOGS_DIR,
        CFG.VIS_DIR,
        CFG.CKPT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
