"""
KITTI-C dataset manifest builder and PyTorch Dataset.

Expected directory layout (RoboDepth / actual download convention):
  <root>/
    <corruption_type>/     e.g. brightness, fog, motion_blur ...
      <severity>/          1, 2, 3, 4, 5   (omitted for 'clean')
        kitti_data/
          <date>/          e.g. 2011_09_26
            <seq>/         e.g. 2011_09_26_drive_0002_sync
              image_02/data/
                <frame>.png

Ground-truth depth layout (optional):
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
    gt_dir: str | Path | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """
    Scan a KITTI-C root directory and build a manifest DataFrame.

    Handles the RoboDepth download layout where images are nested under
    kitti_data/<date>/<seq>/image_02/data/.

    Args:
        root_dir: KITTI-C root (corruption folders at top level)
        gt_dir: clean KITTI ground-truth depth root (optional; gt_path
                column will be None when not provided or file not found)
        split: label to attach to every row

    Returns:
        DataFrame with columns:
            image_path, gt_path, corruption_type, severity, sequence,
            frame_id, split, dataset
    """
    root_dir = Path(root_dir)
    gt_dir = Path(gt_dir) if gt_dir is not None else None
    records = []

    for corruption_dir in sorted(root_dir.iterdir()):
        if not corruption_dir.is_dir():
            continue
        corruption_type = corruption_dir.name

        # 'clean' has no severity sub-folders — images are directly under kitti_data/
        if corruption_type == "clean":
            severity_dirs = [corruption_dir]
            severity_val = 0
        else:
            severity_dirs = sorted(corruption_dir.iterdir())
            severity_val = None  # set per sub-dir below

        for severity_dir in severity_dirs:
            if not severity_dir.is_dir():
                continue

            if corruption_type == "clean":
                severity = 0
            else:
                try:
                    severity = int(severity_dir.name)
                except ValueError:
                    continue

            # Walk kitti_data/<date>/<seq>/image_02/data/
            kitti_data_dir = severity_dir / "kitti_data"
            if not kitti_data_dir.exists():
                continue

            for date_dir in sorted(kitti_data_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                for seq_dir in sorted(date_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    seq = seq_dir.name
                    img_dir = seq_dir / "image_02" / "data"
                    if not img_dir.exists():
                        continue

                    for img_path in sorted(img_dir.glob("*.png")):
                        frame_id = img_path.stem
                        gt_path = None
                        if gt_dir is not None:
                            for gt_prefix in [gt_dir, gt_dir / "train", gt_dir / "val"]:
                                candidate = (
                                    gt_prefix
                                    / seq
                                    / "proj_depth"
                                    / "groundtruth"
                                    / "image_02"
                                    / f"{frame_id}.png"
                                )
                                if candidate.exists():
                                    gt_path = str(candidate)
                                    break

                        records.append(
                            {
                                "image_path": str(img_path),
                                "gt_path": gt_path,
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
