"""
DIODE dataset manifest builder and PyTorch Dataset.

Expected directory layout (official DIODE validation split):
  <val_root>/
    indoors|outdoor/
      <scene>/
        <scan>/
          <frame>.png
          <frame>_depth.npy
          <frame>_depth_mask.npy
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset

from src.datasets.transforms import load_diode_depth, load_rgb


def build_manifest(
    root_dir: str | Path,
    split: str = "val",
) -> pd.DataFrame:
    """
    Scan a DIODE split directory and build a manifest DataFrame.

    Args:
        root_dir: DIODE split root, typically data/raw/diode/val
        split: split label to attach to each row

    Returns:
        DataFrame with columns:
          image_path, depth_path, mask_path, corruption_type, severity, domain,
          scene, scan, frame_id, split, dataset
    """
    root_dir = Path(root_dir)
    records: list[dict] = []

    for image_path in sorted(root_dir.rglob("*.png")):
        if image_path.name.endswith(("_depth.png", "_depth_mask.png")):
            continue

        depth_path = image_path.with_name(f"{image_path.stem}_depth.npy")
        mask_path = image_path.with_name(f"{image_path.stem}_depth_mask.npy")
        if not depth_path.exists() or not mask_path.exists():
            continue

        rel = image_path.relative_to(root_dir)
        parts = rel.parts
        domain = parts[0] if len(parts) > 0 else "unknown"
        scene = parts[1] if len(parts) > 1 else ""
        scan = parts[2] if len(parts) > 2 else ""

        records.append(
            {
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "mask_path": str(mask_path),
                # Reuse the existing analysis stack by mapping domain -> corruption_type.
                "corruption_type": domain,
                "severity": 0,
                "domain": domain,
                "scene": scene,
                "scan": scan,
                "frame_id": image_path.stem,
                "split": split,
                "dataset": "diode",
            }
        )

    return pd.DataFrame(records)


class DIODEDataset(Dataset):
    """
    Dataset that loads (image, gt_depth, valid_mask) tuples from a DIODE manifest.

    Args:
        manifest: DataFrame or path to a manifest CSV produced by build_manifest
        domain_filter: if given, only keep "indoors" or "outdoor" rows
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        domain_filter: Optional[str] = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(str(manifest))

        df = manifest.copy()
        if domain_filter is not None:
            df = df[df["domain"] == domain_filter]
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = load_rgb(row["image_path"])
        gt_depth, valid_mask = load_diode_depth(row["depth_path"], row["mask_path"])

        return {
            "image": image,
            "gt_depth": gt_depth,
            "valid_mask": valid_mask,
            "image_path": row["image_path"],
            "depth_path": row["depth_path"],
            "mask_path": row["mask_path"],
            "domain": row["domain"],
            "scene": row["scene"],
            "scan": row["scan"],
            "frame_id": row["frame_id"],
        }
