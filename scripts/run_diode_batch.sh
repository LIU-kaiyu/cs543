#!/usr/bin/env bash
# Run MiDaS dpt_hybrid_384 over the DIODE validation manifest.
# Requires: data/raw/diode/val to exist (download via scripts/download_diode_val.sh).
set -e

python - <<'EOF'
import sys
from pathlib import Path

project_root = Path.cwd()
if not (project_root / "configs" / "dataset_paths.yaml").exists():
    raise RuntimeError("Run this script from the repository root.")

sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd

from src.adapters.midas_adapter import MiDaSAdapter
from src.datasets.diode import build_manifest
from src.datasets.transforms import load_diode_depth
from src.evaluation.align import align_scale_shift
from src.evaluation.metrics import compute_all_metrics
from src.utils.io import save_metrics_csv
from src.utils.paths import get_dataset_path, get_output_path

val_root = get_dataset_path("diode", "val_root")
manifests_dir = get_output_path("manifests")
pred_root = get_output_path("predictions") / "diode"
metrics_out = get_output_path("metrics") / "diode_results.csv"
manifest_path = manifests_dir / "diode_val_manifest.csv"
model_type = "dpt_hybrid_384"
min_depth = 1e-3
max_depth = 300.0

if not val_root.exists():
    raise FileNotFoundError(
        f"DIODE validation root not found: {val_root}\n"
        "Run scripts/download_diode_val.sh first or update configs/dataset_paths.yaml."
    )

if manifest_path.exists():
    df = pd.read_csv(manifest_path)
else:
    df = build_manifest(val_root, split="val")
    if df.empty:
        raise RuntimeError(f"No DIODE samples found under {val_root}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")

print(f"Loaded DIODE manifest with {len(df)} samples")

adapter = MiDaSAdapter(model_type=model_type)
records = []

if metrics_out.exists():
    metrics_out.unlink()

for idx, row in df.iterrows():
    try:
        pred_path = pred_root / row["domain"] / row["scene"] / row["scan"] / f"{row['frame_id']}.npy"
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        if pred_path.exists():
            pred = np.load(str(pred_path))
        else:
            pred = adapter.predict(row["image_path"])
            np.save(str(pred_path), pred)

        gt, valid_mask = load_diode_depth(row["depth_path"], row["mask_path"])
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (pred.shape[1], pred.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        aligned = align_scale_shift(pred, gt, valid_mask)
        metrics = compute_all_metrics(aligned, gt, valid_mask, min_depth=min_depth, max_depth=max_depth)

        records.append(
            {
                "image_path": row["image_path"],
                "depth_path": row["depth_path"],
                "mask_path": row["mask_path"],
                "pred_path": str(pred_path),
                "dataset": "diode",
                "corruption_type": row["corruption_type"],
                "severity": row["severity"],
                "domain": row["domain"],
                "scene": row["scene"],
                "scan": row["scan"],
                "frame_id": row["frame_id"],
                "model_name": model_type,
                **metrics,
            }
        )

        if idx % 100 == 0:
            print(f"  {idx}/{len(df)} — {row['domain']} abs_rel={metrics['abs_rel']:.4f}")
            save_metrics_csv(records, metrics_out)
            records = []

    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR [{idx}] {row['image_path']}: {exc}")

if records:
    save_metrics_csv(records, metrics_out)

print(f"\nDone. Results saved to {metrics_out}")
EOF
