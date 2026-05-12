"""
Create report figures for snow, dark, and noise corruption examples.

Each output figure has six columns:
  corrupted | preprocessed | baseline depth | improved depth | GT | error
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.midas_adapter import MiDaSAdapter  # noqa: E402
from src.adapters.preprocessing import apply_clahe, apply_denoising, apply_gamma  # noqa: E402
from src.datasets.transforms import load_kitti_depth, load_rgb  # noqa: E402
from src.evaluation.align import align_scale_shift  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402


BASELINE_CSV = PROJECT_ROOT / "outputs" / "metrics" / "kittic_results.csv"
CONSERVATIVE_CSV = PROJECT_ROOT / "outputs" / "metrics" / "kittic_results_auto-conservative.csv"
FIGURE_DIR = PROJECT_ROOT / "figures"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions" / "kitti_c"


def baseline_pred_path(row: pd.Series) -> Path:
    return (
        PRED_DIR
        / row["corruption_type"]
        / str(int(row["severity"]))
        / row["sequence"]
        / f"{int(row['frame_id']):010d}.npy"
    )


def load_comparison_rows() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_CSV)
    conservative = pd.read_csv(CONSERVATIVE_CSV)
    keys = ["image_path", "corruption_type", "severity", "sequence", "frame_id"]
    merged = baseline.merge(
        conservative,
        on=keys,
        suffixes=("_base", "_improved"),
        how="inner",
    )
    merged["absrel_gain"] = merged["abs_rel_base"] - merged["abs_rel_improved"]
    merged["baseline_pred_path"] = merged.apply(baseline_pred_path, axis=1).astype(str)
    merged = merged[merged["baseline_pred_path"].map(lambda path: Path(path).exists())]
    merged = merged[merged["gt_path_base"].map(lambda path: Path(path).exists())]
    merged = merged[merged["image_path"].map(lambda path: Path(path).exists())]
    return merged.reset_index(drop=True)


def choose_row(df: pd.DataFrame, corruption_filter: list[str]) -> pd.Series:
    sub = df[df["corruption_type"].isin(corruption_filter)].copy()
    sub = sub[sub["absrel_gain"] > 0]
    if sub.empty:
        sub = df[df["corruption_type"].isin(corruption_filter)].copy()
    sub = sub.sort_values("absrel_gain", ascending=False)
    return sub.iloc[0]


def normalize_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def depth_image(ax, depth: np.ndarray, valid_mask: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    shown = depth.astype(np.float32).copy()
    shown[~valid_mask] = np.nan
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")
    ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def error_image(
    ax,
    baseline_depth: np.ndarray,
    improved_depth: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    eps = 1e-6
    base_error = np.zeros_like(gt, dtype=np.float32)
    improved_error = np.zeros_like(gt, dtype=np.float32)
    base_error[valid_mask] = np.abs(gt[valid_mask] - baseline_depth[valid_mask]) / (gt[valid_mask] + eps)
    improved_error[valid_mask] = np.abs(gt[valid_mask] - improved_depth[valid_mask]) / (gt[valid_mask] + eps)
    # Positive values mean preprocessing reduced relative error.
    reduction = base_error - improved_error
    shown = reduction.copy()
    shown[~valid_mask] = np.nan
    limit = np.nanpercentile(np.abs(shown), 95)
    if not np.isfinite(limit) or limit <= 0:
        limit = 0.2
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("black")
    ax.imshow(shown, cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_title("error\n(green=better)", fontsize=10)
    ax.axis("off")


def make_figure(
    adapter: MiDaSAdapter,
    row: pd.Series,
    preprocessing_name: str,
    preprocessor,
    output_path: Path,
) -> None:
    image = load_rgb(row["image_path"])
    preprocessed = preprocessor(image)

    baseline_raw = np.load(row["baseline_pred_path"])
    improved_raw = adapter.predict(image, preprocessor=preprocessor)
    gt = load_kitti_depth(row["gt_path_base"])

    if gt.shape != baseline_raw.shape:
        gt = cv2.resize(gt, (baseline_raw.shape[1], baseline_raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    if improved_raw.shape != baseline_raw.shape:
        improved_raw = cv2.resize(
            improved_raw,
            (baseline_raw.shape[1], baseline_raw.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    valid_mask = (gt > 0) & (gt >= 1e-3) & (gt <= 80.0)
    baseline_aligned = np.clip(align_scale_shift(baseline_raw, gt, valid_mask), 1e-3, 80.0)
    improved_aligned = np.clip(align_scale_shift(improved_raw, gt, valid_mask), 1e-3, 80.0)

    baseline_metrics = compute_all_metrics(baseline_aligned, gt, valid_mask)
    improved_metrics = compute_all_metrics(improved_aligned, gt, valid_mask)

    vmin = float(np.nanpercentile(gt[valid_mask], 2))
    vmax = float(np.nanpercentile(gt[valid_mask], 98))

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4), constrained_layout=True)
    axes[0].imshow(normalize_image(image))
    axes[0].set_title("corrupted", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(normalize_image(preprocessed))
    axes[1].set_title(preprocessing_name, fontsize=10)
    axes[1].axis("off")

    depth_image(axes[2], baseline_aligned, valid_mask, "baseline\ndepth", vmin, vmax)
    depth_image(axes[3], improved_aligned, valid_mask, "improved\ndepth", vmin, vmax)
    depth_image(axes[4], gt, valid_mask, "GT", vmin, vmax)
    error_image(axes[5], baseline_aligned, improved_aligned, gt, valid_mask)

    title = (
        f"{row['corruption_type']} severity {int(row['severity'])}, "
        f"AbsRel {baseline_metrics['abs_rel']:.3f} -> {improved_metrics['abs_rel']:.3f}, "
        f"RMSE {baseline_metrics['rmse']:.2f} -> {improved_metrics['rmse']:.2f}"
    )
    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def main() -> None:
    df = load_comparison_rows()
    figure_specs = [
        (
            "snow",
            ["snow"],
            "CLAHE",
            lambda image: apply_clahe(image, clip_limit=2.0),
            FIGURE_DIR / "figure1_snow_corruption.png",
        ),
        (
            "dark",
            ["dark"],
            "gamma corrected",
            lambda image: apply_gamma(image, gamma=0.7),
            FIGURE_DIR / "figure2_dark_corruption.png",
        ),
        (
            "noise",
            ["gaussian_noise", "shot_noise", "impulse_noise", "iso_noise"],
            "Denoised",
            apply_denoising,
            FIGURE_DIR / "figure3_noise_corruption.png",
        ),
    ]

    rows = [(name, choose_row(df, filters), label, preprocessor, path) for name, filters, label, preprocessor, path in figure_specs]
    print("Selected samples:")
    for name, row, _, _, _ in rows:
        print(
            f"  {name}: {row['corruption_type']} severity={int(row['severity'])} "
            f"frame={int(row['frame_id']):010d} absrel_gain={row['absrel_gain']:.4f}"
        )

    adapter = MiDaSAdapter(model_type="dpt_hybrid_384")
    for _, row, label, preprocessor, path in rows:
        make_figure(adapter, row, label, preprocessor, path)


if __name__ == "__main__":
    main()
