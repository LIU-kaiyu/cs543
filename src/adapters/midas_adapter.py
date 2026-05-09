"""
MiDaS inference adapter.

Wraps third_party/MIDAS so the benchmark code never imports MiDaS directly.
Outputs are raw relative inverse depth maps saved as .npy files.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ---- FORCE THE MIDAS PATH FOR THIS MACHINE ----
# _MIDAS_DIR = Path("/home/kaiyul3/cs543/third_party/MIDAS")
_MIDAS_DIR = Path(__file__).resolve().parents[2] / "third_party" / "MIDAS"

if not (_MIDAS_DIR / "midas").is_dir():
    raise RuntimeError(
        f"MiDaS repo not found at: {_MIDAS_DIR}\n"
        f"Expected package folder: {_MIDAS_DIR / 'midas'}"
    )

if str(_MIDAS_DIR) not in sys.path:
    sys.path.insert(0, str(_MIDAS_DIR))

from midas.model_loader import load_model, default_models  # noqa: E402

# Ensure MiDaS is importable.
# Strategy: search sys.path entries for one whose parent contains third_party/MIDAS.
# This is more reliable than Path(__file__).resolve() on network filesystems
# (e.g. Google Drive FUSE mounts in Colab), where resolve() can return wrong paths.
def _find_midas_dir() -> Path:
    # 1. Check if sys.path already points directly to MiDaS
    for entry in sys.path:
        p = Path(entry)
        if (p / "midas").is_dir():
            return p

    # 2. Check if sys.path points to project root
    for entry in sys.path:
        for dirname in ("MIDAS", "MiDaS"):
            candidate = Path(entry) / "third_party" / dirname
            if (candidate / "midas").is_dir():
                return candidate

    # 3. Walk upward from this file
    try:
        for parent in Path(__file__).resolve().parents:
            for dirname in ("MIDAS", "MiDaS"):
                candidate = parent / "third_party" / dirname
                if (candidate / "midas").is_dir():
                    return candidate
    except Exception:
        pass

    raise RuntimeError(
        "Cannot locate third_party/MIDAS. "
        "Add the project root to sys.path before importing:\n"
        "  sys.path.insert(0, '/path/to/CS543 Project')"
    )

_MIDAS_DIR = _find_midas_dir()
if str(_MIDAS_DIR) not in sys.path:
    sys.path.insert(0, str(_MIDAS_DIR))

from midas.model_loader import load_model, default_models  # noqa: E402


def _read_image(path: str) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype(np.float32)


class MiDaSAdapter:
    """
    Loads a MiDaS model and runs inference over a list of image paths.

    Args:
        model_type: one of the keys in midas.model_loader.default_models
        weights_path: explicit path to weights file; defaults to the
                      path listed in default_models relative to the MiDaS dir
        device: torch device string (e.g. "cuda" or "cpu"); auto-detected if None
        optimize: use half-float on CUDA
    """

    def __init__(
        self,
        model_type: str = "dpt_hybrid_384",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        optimize: bool = False,
    ) -> None:
        self.model_type = model_type

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if weights_path is None:
            if model_type not in default_models:
                raise ValueError(f"Unknown model_type '{model_type}'. "
                                 f"Choose from: {list(default_models.keys())}")
            weights_path = str(_MIDAS_DIR / default_models[model_type])

        self.model, self.transform, self.net_w, self.net_h = load_model(
            self.device, weights_path, model_type, optimize
        )
        self.optimize = optimize
        print(f"[MiDaSAdapter] Model '{model_type}' loaded on {self.device}")

    def predict(self, image_path: str) -> np.ndarray:
        """
        Run inference on a single image.

        Returns:
            depth (np.ndarray): raw relative inverse depth map,
                                shape (H, W), float32, same spatial size as input
        """
        # img_rgb = _read_image(image_path)
        if isinstance(image_path, str):
            img_rgb = _read_image(image_path)
        else:
            img_rgb = image_path
        sample = self.transform({"image": img_rgb})["image"]


        with torch.no_grad():
            tensor = torch.from_numpy(sample).to(self.device).unsqueeze(0)

            if self.optimize and self.device == torch.device("cuda"):
                tensor = tensor.to(memory_format=torch.channels_last).half()

            raw = self.model.forward(tensor)
            depth = (
                torch.nn.functional.interpolate(
                    raw.unsqueeze(1),
                    size=img_rgb.shape[:2],  # (H, W)
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .float()
                .numpy()
            )

        return depth

    def predict_batch(
        self,
        image_paths: list[str],
        batch_size: int = 16,
        num_workers: int = 8,
    ) -> list[np.ndarray]:
        """
        Run batched GPU inference over a list of image paths.

        Images are loaded in parallel on CPU, batched for a single GPU forward
        pass, then each output is interpolated back to its original resolution.
        Uses AMP (fp16) on CUDA for ~2x throughput on A100.

        Returns:
            list of depth arrays (float32, shape matching input image H×W)
        """
        use_amp = self.device.type == "cuda"

        def _load(path: str):
            img = _read_image(path)
            orig_hw = img.shape[:2]
            tensor = torch.from_numpy(self.transform({"image": img})["image"])
            return tensor, orig_hw

        results = [None] * len(image_paths)

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start: start + batch_size]

            if num_workers and num_workers > 0:
                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    loaded = list(pool.map(_load, batch_paths))
            else:
                loaded = [_load(path) for path in batch_paths]

            tensors = torch.stack([t for t, _ in loaded]).to(self.device)
            orig_sizes = [hw for _, hw in loaded]

            with torch.no_grad(), torch.autocast(self.device.type, enabled=use_amp):
                raw = self.model.forward(tensors)

            for i, orig_hw in enumerate(orig_sizes):
                depth = (
                    torch.nn.functional.interpolate(
                        raw[i].unsqueeze(0).unsqueeze(0).float(),
                        size=orig_hw,
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                results[start + i] = depth

        return results

    def run_batch(
        self,
        image_paths: list[str],
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Run inference over a list of image paths.

        Args:
            image_paths: list of absolute or relative paths to RGB images
            output_dir: if provided, saves each prediction as <stem>.npy here
            verbose: print progress

        Returns:
            list of dicts with keys: image_path, pred_path (or None), runtime_s
        """
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

        records = []
        for idx, img_path in enumerate(image_paths):
            t0 = time.perf_counter()
            try:
                depth = self.predict(img_path)
                elapsed = time.perf_counter() - t0

                pred_path = None
                if output_dir is not None:
                    stem = Path(img_path).stem
                    pred_path = str(out / f"{stem}.npy")
                    np.save(pred_path, depth)

                records.append(
                    {"image_path": img_path, "pred_path": pred_path, "runtime_s": elapsed, "error": None}
                )

                if verbose:
                    print(f"  [{idx+1}/{len(image_paths)}] {Path(img_path).name} "
                          f"— {elapsed*1000:.0f} ms")

            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                records.append(
                    {"image_path": img_path, "pred_path": None, "runtime_s": elapsed, "error": str(exc)}
                )
                print(f"  [{idx+1}/{len(image_paths)}] ERROR: {exc}")

        return records
