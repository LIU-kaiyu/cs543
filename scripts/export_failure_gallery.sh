#!/usr/bin/env bash
# Export failure galleries (worst / median / best images per corruption type).
# Requires: outputs/metrics/kittic_results.csv
set -e

python - <<'EOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.failure_slices import get_worst_n, get_best_n, get_median_n

METRICS_CSV = Path("outputs/metrics/kittic_results.csv")
GALLERY_DIR = Path("outputs/galleries/kitti_c")
N = 20

if not METRICS_CSV.exists():
    print(f"Metrics file not found: {METRICS_CSV}")
    sys.exit(1)

df = pd.read_csv(METRICS_CSV)
corruptions = df["corruption_type"].unique()

for ct in corruptions:
    for label, fn in [("worst", get_worst_n), ("median", get_median_n), ("best", get_best_n)]:
        subset = fn(df, metric="abs_rel", corruption_type=ct, n=N)
        out_dir = GALLERY_DIR / ct / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(subset.iterrows()):
            try:
                import cv2
                img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
                pred = np.load(row["pred_path"])

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title(f"RGB  [{row['corruption_type']} sev={row['severity']}]")
                axes[0].axis("off")
                axes[1].imshow(pred, cmap="inferno")
                axes[1].set_title(f"Depth  abs_rel={row['abs_rel']:.4f}")
                axes[1].axis("off")
                plt.tight_layout()
                plt.savefig(out_dir / f"{rank:02d}_{row['frame_id']}.png", dpi=100)
                plt.close(fig)
            except Exception as e:
                print(f"  Could not save gallery image: {e}")

print(f"Galleries exported to {GALLERY_DIR}/")
EOF
