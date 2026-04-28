#!/usr/bin/env bash
# Export report-ready DIODE failure case panels.
set -e

python - <<'EOF'
import sys
from pathlib import Path

project_root = Path.cwd()
if not (project_root / "configs" / "dataset_paths.yaml").exists():
    raise RuntimeError("Run this script from the repository root.")

sys.path.insert(0, str(project_root))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.failure_slices import get_worst_n
from src.datasets.transforms import load_diode_depth
from src.evaluation.align import align_scale_shift


def _resize_gt(gt: np.ndarray, mask: np.ndarray, target_hw: tuple[int, int]):
    if gt.shape == target_hw:
        return gt, mask

    width = target_hw[1]
    height = target_hw[0]
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return gt, mask


def _depth_range(gt: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    valid = gt[valid_mask]
    if valid.size == 0:
        return 0.0, 1.0

    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max()) if float(valid.max()) > lo else lo + 1.0
    return lo, hi


def _error_range(error: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = error[valid_mask]
    if valid.size == 0:
        return 1.0
    vmax = float(np.percentile(valid, 95))
    return vmax if vmax > 0 else 1.0


def _plot_case_grid(df: pd.DataFrame, title: str, out_path: Path):
    n = len(df)
    fig, axes = plt.subplots(n, 4, figsize=(18, max(3.5 * n, 4.5)), squeeze=False)
    fig.suptitle(title, fontsize=20, y=0.995)

    column_titles = ["RGB", "GT Depth", "MiDaS Prediction", "Error Map"]
    for col, col_title in enumerate(column_titles):
        axes[0, col].set_title(col_title, fontsize=16, pad=12)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        rgb = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        pred = np.load(row["pred_path"]).astype(np.float32)
        gt, valid_mask = load_diode_depth(row["depth_path"], row["mask_path"])
        gt, valid_mask = _resize_gt(gt, valid_mask, pred.shape)
        aligned = align_scale_shift(pred, gt, valid_mask)

        lo, hi = _depth_range(gt, valid_mask)
        error = np.abs(aligned - gt)
        err_vmax = _error_range(error, valid_mask)

        gt_vis = gt.copy()
        pred_vis = aligned.copy()
        err_vis = error.copy()
        gt_vis[~valid_mask] = np.nan
        pred_vis[~valid_mask] = np.nan
        err_vis[~valid_mask] = np.nan

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 1].imshow(gt_vis, cmap="inferno", vmin=lo, vmax=hi)
        axes[row_idx, 2].imshow(pred_vis, cmap="inferno", vmin=lo, vmax=hi)
        axes[row_idx, 3].imshow(err_vis, cmap="magma", vmin=0.0, vmax=err_vmax)

        label = f"[{row['domain']}] {row['frame_id']}\nabs_rel={row['abs_rel']:.4f}  rmse={row['rmse']:.3f}"
        axes[row_idx, 0].text(
            0.0,
            1.03,
            label,
            transform=axes[row_idx, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
        )

        for col in range(4):
            axes[row_idx, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_single_case(row: pd.Series, title: str, out_path: Path):
    rgb = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
    pred = np.load(row["pred_path"]).astype(np.float32)
    gt, valid_mask = load_diode_depth(row["depth_path"], row["mask_path"])
    gt, valid_mask = _resize_gt(gt, valid_mask, pred.shape)
    aligned = align_scale_shift(pred, gt, valid_mask)

    lo, hi = _depth_range(gt, valid_mask)
    error = np.abs(aligned - gt)
    err_vmax = _error_range(error, valid_mask)

    gt_vis = gt.copy()
    pred_vis = aligned.copy()
    err_vis = error.copy()
    gt_vis[~valid_mask] = np.nan
    pred_vis[~valid_mask] = np.nan
    err_vis[~valid_mask] = np.nan

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=20, y=0.98)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=16)
    axes[1].imshow(gt_vis, cmap="inferno", vmin=lo, vmax=hi)
    axes[1].set_title("GT Depth", fontsize=16)
    axes[2].imshow(pred_vis, cmap="inferno", vmin=lo, vmax=hi)
    axes[2].set_title("MiDaS Prediction", fontsize=16)
    axes[3].imshow(err_vis, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[3].set_title("Error Map", fontsize=16)

    axes[0].text(
        0.0,
        -0.12,
        f"[{row['domain']}] {row['frame_id']}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    axes[2].text(
        0.0,
        -0.12,
        f"abs_rel={row['abs_rel']:.4f}  rmse={row['rmse']:.3f}  delta1={row['delta1']:.4f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


metrics_csv = Path("outputs/metrics/diode_results.csv")
out_dir = Path("outputs/figures/diode")
top_n = 6

if not metrics_csv.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_csv}")

df = pd.read_csv(metrics_csv)

overall_worst = get_worst_n(df, metric="abs_rel", n=top_n)
outdoor_worst = get_worst_n(df[df["domain"] == "outdoor"], metric="abs_rel", n=top_n)
indoor_worst = get_worst_n(df[df["domain"] == "indoors"], metric="abs_rel", n=top_n)

_plot_case_grid(
    overall_worst,
    title="Top Failure Cases — DIODE Overall",
    out_path=out_dir / "diode_top_failure_cases_overall.png",
)
_plot_case_grid(
    outdoor_worst,
    title="Top Failure Cases — DIODE Outdoor",
    out_path=out_dir / "diode_top_failure_cases_outdoor.png",
)
_plot_case_grid(
    indoor_worst,
    title="Top Failure Cases — DIODE Indoors",
    out_path=out_dir / "diode_top_failure_cases_indoors.png",
)
_plot_single_case(
    overall_worst.iloc[0],
    title="Worst Single Failure Case — DIODE",
    out_path=out_dir / "diode_worst_single_case.png",
)

print(f"Failure panels exported to {out_dir}/")
EOF
