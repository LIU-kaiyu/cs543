"""Qualitative panels comparing baseline vs Auto-Restormer preprocessing.

Each panel row = one KITTI-C image and shows:
  corrupted RGB | enhanced RGB | baseline depth | Restormer depth | GT depth | error-reduction map

Selects:
  * snow (success: largest baseline-better-than-experiment AbsRel gain)
  * dark (mild success)
  * impulse_noise (failure: largest regression)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.preprocessing import build_preprocessor
from src.datasets.transforms import load_kitti_depth, load_rgb
from src.evaluation.align import align_scale_shift

BASELINE_CSV = PROJECT_ROOT / "outputs/metrics/kittic_results.csv"
EXP_CSV = PROJECT_ROOT / "outputs/metrics/kittic_results_auto-restormer_g0p7.csv"
OUT_DIR = PROJECT_ROOT / "outputs/figures/restormer_analysis"
REPORT_FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

DEPTH_MIN = 1e-3
DEPTH_MAX = 80.0
DEPTH_CMAP = "magma"


def load_joined() -> pd.DataFrame:
    base = pd.read_csv(BASELINE_CSV)
    exp = pd.read_csv(EXP_CSV)
    keep = ["image_path", "gt_path", "corruption_type", "severity", "pred_path", "abs_rel", "rmse", "delta1"]
    base = base[keep].rename(columns={"pred_path": "pred_path_base", "abs_rel": "abs_rel_base", "rmse": "rmse_base", "delta1": "delta1_base"})
    exp = exp[keep].rename(columns={"pred_path": "pred_path_exp", "abs_rel": "abs_rel_exp", "rmse": "rmse_exp", "delta1": "delta1_exp"})
    joined = base.merge(exp, on=["image_path", "gt_path", "corruption_type", "severity"], how="inner")
    joined["abs_rel_drop"] = joined["abs_rel_base"] - joined["abs_rel_exp"]
    return joined


def pick_examples(joined: pd.DataFrame) -> dict[str, list[pd.Series]]:
    """Pick the most informative examples per corruption type."""
    picks: dict[str, list[pd.Series]] = {}
    snow_sev = joined[(joined.corruption_type == "snow") & (joined.severity >= 3)]
    snow_top = snow_sev.sort_values("abs_rel_drop", ascending=False).head(2)
    picks["snow"] = [row for _, row in snow_top.iterrows()]
    dark_sev = joined[(joined.corruption_type == "dark") & (joined.severity >= 3)]
    dark_top = dark_sev.sort_values("abs_rel_drop", ascending=False).head(2)
    picks["dark"] = [row for _, row in dark_top.iterrows()]
    noise = joined[(joined.corruption_type == "impulse_noise") & (joined.severity >= 3)]
    noise_top = noise.sort_values("abs_rel_drop", ascending=True).head(2)
    picks["impulse_noise"] = [row for _, row in noise_top.iterrows()]
    return picks


def to_uint8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)


def prep_preview(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    corrupted = load_rgb(row.image_path)
    preprocessor = build_preprocessor("auto-restormer", corruption_type=row.corruption_type, gamma=0.7)
    if preprocessor is None:
        enhanced = corrupted.copy()
    else:
        enhanced = preprocessor(corrupted.copy())
    return to_uint8(corrupted), to_uint8(enhanced)


def aligned_depth(pred_npy: Path, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred = np.load(pred_npy).astype(np.float32)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    aligned = align_scale_shift(pred, gt, valid)
    aligned = np.clip(aligned, DEPTH_MIN, DEPTH_MAX)
    return aligned


def error_reduction_map(base_aligned: np.ndarray, exp_aligned: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    base_err = np.abs(base_aligned - gt)
    exp_err = np.abs(exp_aligned - gt)
    diff = (base_err - exp_err)
    diff[~valid] = 0.0
    return diff


def draw_row(axes, row: pd.Series, vmin: float, vmax: float) -> None:
    corrupted, enhanced = prep_preview(row)
    gt = load_kitti_depth(row.gt_path)
    valid = (gt > DEPTH_MIN) & (gt < DEPTH_MAX)
    base_aligned = aligned_depth(Path(row.pred_path_base), gt, valid)
    exp_aligned = aligned_depth(Path(row.pred_path_exp), gt, valid)
    err_red = error_reduction_map(base_aligned, exp_aligned, gt, valid)

    gt_viz = np.where(valid, gt, np.nan)

    axes[0].imshow(corrupted)
    axes[0].set_title(f"corrupted RGB\n({row.corruption_type}, sev {int(row.severity)})", fontsize=9)
    axes[1].imshow(enhanced)
    axes[1].set_title("Restormer-enhanced RGB", fontsize=9)
    axes[2].imshow(base_aligned, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"baseline depth\nAbsRel={row.abs_rel_base:.3f}", fontsize=9)
    axes[3].imshow(exp_aligned, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax)
    axes[3].set_title(f"Auto-Restormer depth\nAbsRel={row.abs_rel_exp:.3f}", fontsize=9)
    axes[4].imshow(gt_viz, cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax)
    axes[4].set_title("KITTI ground truth", fontsize=9)
    cmap = LinearSegmentedColormap.from_list("err_red", ["#d62728", "#ffffff", "#2ca02c"])
    err_clip = np.clip(err_red, -3.0, 3.0)
    axes[5].imshow(err_clip, cmap=cmap, vmin=-3.0, vmax=3.0)
    drop_pct = (row.abs_rel_base - row.abs_rel_exp) / max(row.abs_rel_base, 1e-9) * 100.0
    axes[5].set_title(f"error reduction\n(green=Restormer better, red=worse)\nDelta AbsRel = {drop_pct:+.1f}%", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def panel_for_corruption(rows: list[pd.Series], title: str, out_name: str) -> None:
    fig, axes = plt.subplots(len(rows), 6, figsize=(18, 3 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    vmin, vmax = 0.0, 60.0
    for i, row in enumerate(rows):
        draw_row(axes[i], row, vmin, vmax)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(REPORT_FIG_DIR / f"{out_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def combined_panel(picks: dict[str, list[pd.Series]], out_name: str) -> None:
    selected = [
        ("Snow (Restormer-deraining helps)", picks["snow"][0]),
        ("Dark (gamma helps)", picks["dark"][0]),
        ("Impulse noise (Restormer-denoising hurts)", picks["impulse_noise"][0]),
    ]
    fig, axes = plt.subplots(len(selected), 6, figsize=(18, 3 * len(selected)))
    vmin, vmax = 0.0, 60.0
    for i, (label, row) in enumerate(selected):
        draw_row(axes[i], row, vmin, vmax)
        axes[i, 0].set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center", labelpad=80)
    fig.suptitle("Qualitative comparison: Auto-Restormer vs Baseline on KITTI-C", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(REPORT_FIG_DIR / f"{out_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    joined = load_joined()
    picks = pick_examples(joined)
    for c, rows in picks.items():
        for r in rows:
            print(f"  [{c}] sev={int(r.severity)} AbsRel base={r.abs_rel_base:.4f} exp={r.abs_rel_exp:.4f} drop={r.abs_rel_drop:+.4f}  {Path(r.image_path).name}")
    panel_for_corruption(picks["snow"], "Snow (success case): Auto-Restormer vs Baseline", "fig_qualitative_snow")
    panel_for_corruption(picks["dark"], "Dark (mild success): Auto-Restormer vs Baseline", "fig_qualitative_dark")
    panel_for_corruption(picks["impulse_noise"], "Impulse noise (failure case): Auto-Restormer vs Baseline", "fig_qualitative_noise_failure")
    combined_panel(picks, "fig_qualitative_combined")
    print("Figures saved under", OUT_DIR, "and", REPORT_FIG_DIR)


if __name__ == "__main__":
    main()
