"""
MiDaS inference adapter.

Wraps third_party/MiDaS so the benchmark code never imports MiDaS directly.
Outputs are raw relative inverse depth maps saved as .npy files.
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def _iter_midas_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.environ.get("MIDAS_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        candidates.append(path)
        candidates.append(path / "third_party" / "MiDaS")
        candidates.append(path / "third_party" / "MIDAS")

    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "third_party" / "MiDaS")
        candidates.append(parent / "third_party" / "MIDAS")

    # Preserve order but drop duplicates.
    ordered: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            ordered.append(candidate)
            seen.add(key)
    return ordered


def _find_midas_dir() -> Path:
    """
    Locate an upstream MiDaS checkout.

    Accepted layouts:
      - <candidate>/midas/...
      - <candidate>/MiDaS/midas/...     (when a parent directory is given)
    """
    for candidate in _iter_midas_candidates():
        direct = candidate
        nested = candidate / "MiDaS"

        if (direct / "midas").is_dir():
            return direct
        if (nested / "midas").is_dir():
            return nested

    raise RuntimeError(
        "Cannot locate an upstream MiDaS checkout.\n"
        "Expected one of these layouts:\n"
        "  third_party/MiDaS/midas/\n"
        "  third_party/MIDAS/midas/\n"
        "You can also point MIDAS_DIR at the repo root."
    )


_MIDAS_DIR = _find_midas_dir()
if str(_MIDAS_DIR) not in sys.path:
    sys.path.insert(0, str(_MIDAS_DIR))

from midas.model_loader import default_models, load_model  # noqa: E402


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
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
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
        img_rgb = _read_image(image_path)
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
