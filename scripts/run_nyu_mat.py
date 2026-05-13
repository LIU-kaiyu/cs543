import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.midas_adapter import MiDaSAdapter


MODEL_TYPE = "dpt_hybrid_384"

MAT_PATH = PROJECT_ROOT / "data" / "nyu_depth_v2" / "nyu_depth_v2_labeled.mat"

OUT_DIR = PROJECT_ROOT / "outputs" / "nyu_full"
PRED_DIR = OUT_DIR / "predictions"
VIS_DIR = OUT_DIR / "visualizations"
TOP_DIR = OUT_DIR / "top_errors"

PRED_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
TOP_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "nyu_metrics_full.csv"
TOP_CSV = OUT_DIR / "nyu_top_errors_full.csv"

MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
CROP = 20


def make_valid_mask(gt, pred):
    mask = (
        (gt > MIN_DEPTH)
        & (gt < MAX_DEPTH)
        & np.isfinite(gt)
        & np.isfinite(pred)
    )

    if CROP > 0:
        mask[:CROP, :] = False
        mask[-CROP:, :] = False
        mask[:, :CROP] = False
        mask[:, -CROP:] = False

    return mask


def align_scale_shift(pred, gt, mask):
    x = pred[mask].reshape(-1).astype(np.float64)
    y = gt[mask].reshape(-1).astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

    pred_aligned = scale * pred + shift
    pred_aligned = np.clip(pred_aligned, MIN_DEPTH, MAX_DEPTH)

    return pred_aligned.astype(np.float32)


def compute_metrics(pred, gt):
    mask = make_valid_mask(gt, pred)
    pred_aligned = align_scale_shift(pred, gt, mask)

    gt_valid = gt[mask]
    pred_valid = pred_aligned[mask]

    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    delta1 = np.mean(thresh < 1.25)

    return abs_rel, rmse, delta1, pred_aligned, mask


def crop_for_vis(arr):
    return arr[CROP:-CROP, CROP:-CROP]


def save_vis(img, gt, pred, mask, idx, abs_rel, rmse, delta1):
    error = np.zeros_like(gt)
    error[mask] = np.abs(gt[mask] - pred[mask])

    img = crop_for_vis(img)
    gt = crop_for_vis(gt)
    pred = crop_for_vis(pred)
    error = crop_for_vis(error)
    mask = crop_for_vis(mask)

    vmin = np.percentile(gt[mask], 1)
    vmax = np.percentile(gt[mask], 99)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title(f"RGB\nIdx={idx}")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gt, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.title("GT Depth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="plasma", vmin=vmin, vmax=vmax)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(error, cmap="magma")
    plt.title(f"AbsRel={abs_rel:.3f}, RMSE={rmse:.3f}\nDelta1={delta1:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(VIS_DIR / f"nyu_{idx:04d}.png", dpi=150)
    plt.close()


def save_top10_grid(images, depths, df):
    top10 = df.sort_values("abs_rel", ascending=False).head(10)

    fig, axes = plt.subplots(10, 4, figsize=(16, 40))

    for row, (_, r) in enumerate(top10.iterrows()):
        idx = int(r["index"])

        img = images[idx].astype(np.uint8)
        gt = depths[idx].astype(np.float32)
        pred = np.load(r["pred_path"]).astype(np.float32)

        mask = make_valid_mask(gt, pred)

        error = np.zeros_like(gt)
        error[mask] = np.abs(gt[mask] - pred[mask])

        img = crop_for_vis(img)
        gt = crop_for_vis(gt)
        pred = crop_for_vis(pred)
        error = crop_for_vis(error)
        mask = crop_for_vis(mask)

        vmin = np.percentile(gt[mask], 1)
        vmax = np.percentile(gt[mask], 99)

        metrics_text = (
            f"Idx={idx}\n"
            f"AbsRel={r['abs_rel']:.3f}\n"
            f"RMSE={r['rmse']:.3f}\n"
            f"Delta1={r['delta1']:.3f}"
        )

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"RGB\n{metrics_text}", fontsize=10)

        axes[row, 1].imshow(gt, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("GT Depth", fontsize=10)

        axes[row, 2].imshow(pred, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[row, 2].set_title("Prediction", fontsize=10)

        axes[row, 3].imshow(error, cmap="magma")
        axes[row, 3].set_title("Absolute Error", fontsize=10)

        for col in range(4):
            axes[row, col].axis("off")

    plt.suptitle(
        "Top 10 Highest Error Cases (NYU Depth V2 + MiDaS)",
        fontsize=20,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(OUT_DIR / "nyu_top10_full_grid.png", dpi=150)
    plt.close()


def save_top30_rgb(images, df):
    top30 = df.sort_values("abs_rel", ascending=False).head(30)

    fig, axes = plt.subplots(5, 6, figsize=(20, 14))
    axes = axes.flatten()

    for ax, (_, r) in zip(axes, top30.iterrows()):
        idx = int(r["index"])
        img = images[idx].astype(np.uint8)

        ax.imshow(img)
        ax.set_title(
            f"Idx={idx}\n"
            f"AbsRel={r['abs_rel']:.3f}\n"
            f"RMSE={r['rmse']:.3f}, Delta1={r['delta1']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(top30):]:
        ax.axis("off")

    plt.suptitle(
        "Top 30 Failure Cases (NYU Depth V2 + MiDaS, RGB Only)",
        fontsize=20,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(OUT_DIR / "nyu_top30_rgb.png", dpi=150)
    plt.close()


def main():
    print("Loading NYU Depth V2...")

    with h5py.File(MAT_PATH, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])

    images = np.transpose(images, (0, 3, 2, 1))
    depths = np.transpose(depths, (0, 2, 1))

    print(f"Total images: {len(images)}")

    adapter = MiDaSAdapter(model_type=MODEL_TYPE)

    records = []

    for i in tqdm(range(len(images))):
        img = images[i].astype(np.uint8)
        gt = depths[i].astype(np.float32)

        pred = adapter.predict(img).astype(np.float32)

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        abs_rel, rmse, delta1, pred_aligned, mask = compute_metrics(pred, gt)

        pred_path = PRED_DIR / f"nyu_{i:04d}.npy"
        np.save(pred_path, pred_aligned)

        save_vis(img, gt, pred_aligned, mask, i, abs_rel, rmse, delta1)

        records.append({
            "index": i,
            "dataset": "nyu_depth_v2",
            "model_name": MODEL_TYPE,
            "pred_path": str(pred_path),
            "abs_rel": abs_rel,
            "rmse": rmse,
            "delta1": delta1,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    top_df = df.sort_values("abs_rel", ascending=False).head(30)
    top_df.to_csv(TOP_CSV, index=False)

    save_top10_grid(images, depths, df)
    save_top30_rgb(images, df)

    print("\n=== FINAL RESULTS (NYU FULL) ===")
    print(f"AbsRel: {df['abs_rel'].mean():.4f}")
    print(f"RMSE:   {df['rmse'].mean():.4f}")
    print(f"Delta1: {df['delta1'].mean():.4f}")

    print(f"\nSaved to: {OUT_DIR}")
    print(f"Top10 grid: {OUT_DIR / 'nyu_top10_full_grid.png'}")
    print(f"Top30 RGB:  {OUT_DIR / 'nyu_top30_rgb.png'}")

    final_metrics = {
        "dataset": "nyu_depth_v2",
        "model_name": MODEL_TYPE,
        "abs_rel": df["abs_rel"].mean(),
        "rmse": df["rmse"].mean(),
        "delta1": df["delta1"].mean(),
    }

    final_df = pd.DataFrame([final_metrics])
    final_df.to_csv(OUT_DIR / "nyu_final_results.csv", index=False)


if __name__ == "__main__":
    main()
