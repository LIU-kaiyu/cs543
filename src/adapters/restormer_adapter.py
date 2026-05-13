"""
Restormer inference adapter.

Wraps third_party/Restormer so the benchmark code never imports it directly.
Each task model is loaded lazily on first call and cached for the session.

Supported tasks:
  deraining        – snow, fog, frost weather removal
  real_denoising   – sensor/shot/impulse noise removal
"""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_RESTORMER_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Restormer"

_TASK_CONFIG: dict[str, dict] = {
    "deraining": {
        "weights": _RESTORMER_DIR / "Deraining" / "pretrained_models" / "deraining.pth",
        "params": {
            "inp_channels": 3, "out_channels": 3, "dim": 48,
            "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66,
            "bias": False, "LayerNorm_type": "WithBias", "dual_pixel_task": False,
        },
    },
    "real_denoising": {
        "weights": _RESTORMER_DIR / "Denoising" / "pretrained_models" / "real_denoising.pth",
        "params": {
            "inp_channels": 3, "out_channels": 3, "dim": 48,
            "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66,
            "bias": False, "LayerNorm_type": "BiasFree", "dual_pixel_task": False,
        },
    },
}

_model_cache: dict[str, tuple[nn.Module, torch.device]] = {}


def _get_model(task: str) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task in _model_cache:
        model, cached_device = _model_cache[task]
        if cached_device == device:
            return model, device

    if task not in _TASK_CONFIG:
        raise ValueError(f"Unknown Restormer task: {task!r}. Choose from {list(_TASK_CONFIG)}")

    config = _TASK_CONFIG[task]
    weights_path: Path = config["weights"]

    if not _RESTORMER_DIR.is_dir():
        raise RuntimeError(
            f"Restormer repo not found at {_RESTORMER_DIR}\n"
            "Clone it with:\n"
            "  git clone --depth 1 https://github.com/swz30/Restormer third_party/Restormer"
        )
    if not weights_path.exists():
        raise RuntimeError(
            f"Restormer weights not found at {weights_path}\n"
            "Download them with:\n"
            "  bash scripts/download_restormer_weights.sh"
        )

    arch_path = _RESTORMER_DIR / "basicsr" / "models" / "archs" / "restormer_arch.py"
    load_arch = run_path(str(arch_path))
    model = load_arch["Restormer"](**config["params"])

    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint["params"])
    model.eval()
    model.to(device)

    _model_cache[task] = (model, device)
    print(f"[Restormer] task={task!r} loaded on {device}")
    return model, device


def apply_restormer(image: np.ndarray, task: str = "deraining") -> np.ndarray:
    """
    Restore a degraded image using Restormer.

    Accepts and returns float32 RGB images in [0, 1] (HWC), matching the
    MiDaS adapter input convention. Pads input to a multiple of 8 as
    required by the Restormer architecture, then crops back.

    Args:
        image: float32 RGB [0, 1] HWC numpy array
        task:  one of "deraining" or "real_denoising"
    """
    model, device = _get_model(task)

    h, w = image.shape[:2]

    # HWC float32 [0, 1] → BCHW tensor
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(tensor)

    out = torch.clamp(out, 0.0, 1.0)

    # Crop back to original size
    if pad_h or pad_w:
        out = out[:, :, :h, :w]

    return out.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
