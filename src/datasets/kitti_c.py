"""
KITTI-C dataset manifest builder and PyTorch Dataset.

Expected directory layout (RoboDepth convention):
  <root>/
    <corruption_type>/     e.g. brightness, fog, motion_blur ...
      <severity>/          1, 2, 3, 4, 5
        <seq>/
          image_02/data/
            <frame>.png

Ground-truth depth layout:
  <gt_root>/
    <seq>/
      proj_depth/groundtruth/image_02/
        <frame>.png       (16-bit KITTI depth PNG, value/256 = metres)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.transforms import load_rgb, load_kitti_depth, resize_depth


# ------------------------------------------------------------------
# Manifest builder
# ------------------------------------------------------------------

def build_manifest(
    root_dir: str | Path,
    gt_dir: str | Path,
    split: str = "test",
) -> pd.DataFrame:
    """
    Scan a KITTI-C root directory and build a manifest DataFrame.

    Args:
        root_dir: KITTI-C root (corruption folders at top level)
        gt_dir: clean KITTI ground-truth depth root
        split: label to attach to every row

    Returns:
        DataFrame with columns:
            image_path, gt_path, corruption_type, severity, sequence,
            frame_id, split, dataset
    """
    root_dir = Path(root_dir)
    gt_dir = Path(gt_dir)
    records = []

    for corruption_dir in sorted(root_dir.iterdir()):
        if not corruption_dir.is_dir():
            continue
        corruption_type = corruption_dir.name

        for severity_dir in sorted(corruption_dir.iterdir()):
            if not severity_dir.is_dir():
                continue
            try:
                severity = int(severity_dir.name)
            except ValueError:
                continue

            for seq_dir in sorted(severity_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                seq = seq_dir.name
                img_dir = seq_dir / "image_02" / "data"
                if not img_dir.exists():
                    continue

                for img_path in sorted(img_dir.glob("*.png")):
                    frame_id = img_path.stem
                    gt_path = (
                        gt_dir
                        / seq
                        / "proj_depth"
                        / "groundtruth"
                        / "image_02"
                        / f"{frame_id}.png"
                    )
                    records.append(
                        {
                            "image_path": str(img_path),
                            "gt_path": str(gt_path) if gt_path.exists() else None,
                            "corruption_type": corruption_type,
                            "severity": severity,
                            "sequence": seq,
                            "frame_id": frame_id,
                            "split": split,
                            "dataset": "kitti_c",
                        }
                    )

    df = pd.DataFrame(records)
    return df


# ------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------

class KittiCDataset(Dataset):
    """
    Dataset that loads (image, gt_depth) pairs from a KITTI-C manifest.

    Args:
        manifest: DataFrame (or path to CSV) produced by build_manifest
        corruption_filter: if given, only include rows with this corruption_type
        severity_filter: if given, only include rows with this severity level
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        corruption_filter: Optional[str] = None,
        severity_filter: Optional[int] = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(str(manifest))

        df = manifest.copy()
        if corruption_filter is not None:
            df = df[df["corruption_type"] == corruption_filter]
        if severity_filter is not None:
            df = df[df["severity"] == severity_filter]

        # Only keep rows where gt_path exists
        df = df[df["gt_path"].notna()].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = load_rgb(row["image_path"])          # (H, W, 3) float32
        gt = load_kitti_depth(row["gt_path"])        # (H, W) float32 metres

        # Resize gt to match image if needed
        if gt.shape != image.shape[:2]:
            gt = resize_depth(gt, image.shape[:2])

        valid_mask = gt > 0

        return {
            "image": image,
            "gt_depth": gt,
            "valid_mask": valid_mask,
            "image_path": row["image_path"],
            "gt_path": row["gt_path"],
            "corruption_type": row["corruption_type"],
            "severity": int(row["severity"]),
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
        }
