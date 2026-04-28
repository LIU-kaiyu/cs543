#!/usr/bin/env bash
# Export DIODE failure galleries grouped by indoor/outdoor domain.
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

from src.analysis.failure_slices import get_best_n, get_median_n, get_worst_n

metrics_csv = Path("outputs/metrics/diode_results.csv")
gallery_dir = Path("outputs/galleries/diode")
n = 12

if not metrics_csv.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_csv}")

df = pd.read_csv(metrics_csv)
domains = sorted(df["domain"].unique())

for domain in domains:
    for label, fn in [("worst", get_worst_n), ("median", get_median_n), ("best", get_best_n)]:
        subset = fn(df, metric="abs_rel", corruption_type=domain, n=n)
        out_dir = gallery_dir / domain / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(subset.iterrows()):
            img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
            pred = np.load(row["pred_path"])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title(f"RGB [{row['domain']}]")
            axes[0].axis("off")
            axes[1].imshow(pred, cmap="inferno")
            axes[1].set_title(f"Depth abs_rel={row['abs_rel']:.4f}")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"{rank:02d}_{row['frame_id']}.png", dpi=100)
            plt.close(fig)

print(f"Galleries exported to {gallery_dir}/")
EOF
