#!/usr/bin/env bash
# Run MiDaS dpt_hybrid_384 over the full KITTI-C manifest.
# Requires: data/manifests/kitti_c_manifest.csv to exist (build via notebook 01).
set -e

python - <<'EOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from pathlib import Path
from src.adapters.midas_adapter import MiDaSAdapter
from src.evaluation.align import align_scale_shift
from src.evaluation.metrics import compute_all_metrics
from src.utils.io import save_metrics_csv
import numpy as np

MANIFEST = Path("data/manifests/kitti_c_manifest.csv")
PRED_DIR = Path("outputs/predictions/kitti_c")
METRICS_OUT = Path("outputs/metrics/kittic_results.csv")
MODEL_TYPE = "dpt_hybrid_384"

if not MANIFEST.exists():
    print(f"Manifest not found: {MANIFEST}")
    print("Run notebook 01_build_kittic_manifest.ipynb first.")
    sys.exit(1)

df = pd.read_csv(MANIFEST)
df = df[df["gt_path"].notna()].reset_index(drop=True)
print(f"Loaded manifest: {len(df)} samples")

adapter = MiDaSAdapter(model_type=MODEL_TYPE)
records = []

for idx, row in df.iterrows():
    try:
        depth = adapter.predict(row["image_path"])
        pred_path = PRED_DIR / row["corruption_type"] / str(row["severity"]) / f"{row['frame_id']}.npy"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(pred_path), depth)

        from src.datasets.transforms import load_kitti_depth
        gt = load_kitti_depth(row["gt_path"])

        import cv2
        if gt.shape != depth.shape:
            gt = cv2.resize(gt, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        valid_mask = gt > 0
        aligned = align_scale_shift(depth, gt, valid_mask)
        m = compute_all_metrics(aligned, gt, valid_mask)

        record = {
            "image_path": row["image_path"],
            "gt_path": row["gt_path"],
            "pred_path": str(pred_path),
            "dataset": "kitti_c",
            "corruption_type": row["corruption_type"],
            "severity": row["severity"],
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
            "model_name": MODEL_TYPE,
            **m,
        }
        records.append(record)

        if idx % 100 == 0:
            print(f"  {idx}/{len(df)} — abs_rel={m['abs_rel']:.4f}")
            save_metrics_csv(records, METRICS_OUT)
            records = []

    except Exception as e:
        print(f"  ERROR [{idx}] {row['image_path']}: {e}")

if records:
    save_metrics_csv(records, METRICS_OUT)

print(f"\nDone. Results saved to {METRICS_OUT}")
EOF
