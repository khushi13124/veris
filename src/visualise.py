"""
src/visualise.py
================
Visualisation helpers used across modules.

Public API
----------
show_crops(slice_image, annotations_list, save_path)
    Renders a full windowed slice with one 160×160 bounding box per annotation,
    colour-coded by condition.  Every condition annotated on the same slice
    appears simultaneously.

show_patch_pipeline(raw, rescaled, windowed, crop_ch0, crop_ch1, crop_ch2,
                    final_patch, save_path)
    Side-by-side panel: raw → windowed → cropped channels 0/1/2 → final
    (3,128,128) stack displayed as a 3-row grid.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config.settings import CFG

# One distinct colour per condition
_CONDITION_COLOURS = {
    "spinal_canal_stenosis":             "#E74C3C",   # red
    "left_neural_foraminal_narrowing":   "#3498DB",   # blue
    "right_neural_foraminal_narrowing":  "#2ECC71",   # green
    "left_subarticular_stenosis":        "#F39C12",   # orange
    "right_subarticular_stenosis":       "#9B59B6",   # purple
}
_DEFAULT_COLOUR = "#AAAAAA"


# ─────────────────────────────────────────────────────────────────────────────

def show_crops(
    slice_image: np.ndarray,
    annotations_list: List[Tuple],
    save_path: Path,
    title: str = "",
) -> None:
    """
    Draw every annotation bounding box on a single full-slice image.

    Parameters
    ----------
    slice_image      : 2-D uint8 numpy array — the full windowed slice
    annotations_list : list of (x, y, condition, level, severity_label) tuples
                       x, y are pixel coordinates of the annotation centre
    save_path        : where to write the PNG
    title            : optional figure title
    """
    half = CFG.CROP_SIZE_PX // 2   # 80 px

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(slice_image, cmap="gray", origin="upper")
    ax.set_title(title or "Annotation crop regions", fontsize=11, pad=8)
    ax.axis("off")

    legend_handles = {}

    for (x, y, condition, level, severity_label) in annotations_list:
        colour = _CONDITION_COLOURS.get(condition, _DEFAULT_COLOUR)
        sev    = CFG.SEVERITY_INV.get(int(severity_label), str(severity_label))

        # Bounding box top-left, clamped to image
        bx = max(0, int(x) - half)
        by = max(0, int(y) - half)

        rect = mpatches.Rectangle(
            (bx, by), CFG.CROP_SIZE_PX, CFG.CROP_SIZE_PX,
            linewidth=1.5, edgecolor=colour, facecolor="none",
        )
        ax.add_patch(rect)

        # Small label just above the box
        ax.text(
            bx + 2, max(0, by - 3),
            f"{condition.replace('_', ' ')} {level} [{sev}]",
            color=colour, fontsize=6.5, va="bottom",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),
        )

        # Cross-hair at annotation centre
        ax.plot(x, y, "+", color=colour, markersize=8, markeredgewidth=1.2)

        # Build legend entry (one per condition, not per annotation)
        if condition not in legend_handles:
            legend_handles[condition] = mpatches.Patch(
                color=colour,
                label=condition.replace("_", " "),
            )

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            loc="lower right", fontsize=7,
            framealpha=0.6, edgecolor="none",
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def show_patch_pipeline(
    raw:         np.ndarray,
    windowed:    np.ndarray,
    crops:       List[np.ndarray],   # [ch0, ch1, ch2]  each (160,160) uint8
    final_patch: np.ndarray,         # (3, 128, 128) float32, already normalised
    save_path:   Path,
    title:       str = "",
) -> None:
    """
    Side-by-side panel showing every stage of the pipeline for one annotation.

    Layout (2 rows):
      Row 1: raw | windowed | crop ch0 | crop ch1 | crop ch2
      Row 2: final patch ch0 | ch1 | ch2  (de-normalised for display)
    """
    # De-normalise final patch for display: undo ImageNet norm → [0,1] → uint8
    mean = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
    display_patch = np.clip(final_patch * std + mean, 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle(title or "Patch pipeline", fontsize=11, y=1.01)

    def _show(ax, img, ttl, cmap="gray"):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, origin="upper")
        else:
            ax.imshow(img, origin="upper")
        ax.set_title(ttl, fontsize=8)
        ax.axis("off")
        vmin, vmax = float(img.min()), float(img.max())
        ax.set_xlabel(f"min={vmin:.3f}  max={vmax:.3f}", fontsize=7)
        ax.xaxis.set_label_position("top")

    _show(axes[0, 0], raw,            "raw (after load)")
    _show(axes[0, 1], windowed,       "windowed uint8")
    for i, (ch, name) in enumerate(zip(crops, ["crop ch0\n(inst-1)", "crop ch1\n(inst)", "crop ch2\n(inst+1)"])):
        _show(axes[0, 2 + i], ch, name)

    _show(axes[1, 0], display_patch[0], "final ch0\n(de-normed)")
    _show(axes[1, 1], display_patch[1], "final ch1\n(de-normed)")
    _show(axes[1, 2], display_patch[2], "final ch2\n(de-normed)")

    # Combined 3-ch view in the last two cells
    rgb = np.stack([display_patch[0], display_patch[1], display_patch[2]], axis=-1)
    _show(axes[1, 3], rgb, "all 3 ch\n(RGB view)")

    # Value ranges at each stage printed as a table
    axes[1, 4].axis("off")
    stage_ranges = [
        ("raw",        raw),
        ("windowed",   windowed),
        ("crop ch1",   crops[1]),
        ("final ch1",  final_patch[1]),
    ]
    table_text = "\n".join(
        f"{name:12s}  [{a.min():7.3f}, {a.max():7.3f}]"
        for name, a in stage_ranges
    )
    axes[1, 4].text(
        0.05, 0.95, "Value ranges:\n\n" + table_text,
        transform=axes[1, 4].transAxes,
        va="top", fontsize=7.5, fontfamily="monospace",
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
