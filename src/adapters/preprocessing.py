"""
Image preprocessing options for corruption-robust MiDaS inference.

All functions accept and return RGB float32 images in [0, 1], matching the
MiDaS adapter input convention.
"""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


Preprocessor = Callable[[np.ndarray], np.ndarray]


CLAHE_CORRUPTIONS = {
    "brightness",
    "contrast",
    "fog",
    "frost",
    "snow",
}
CONSERVATIVE_CLAHE_CORRUPTIONS = {
    "snow",
}
GAMMA_CORRUPTIONS = {
    "dark",
}
DENOISE_CORRUPTIONS = {
    "gaussian_noise",
    "impulse_noise",
    "iso_noise",
    "shot_noise",
}
RESTORMER_WEATHER_CORRUPTIONS = {
    "fog",
    "frost",
    "snow",
}
RESTORMER_NOISE_CORRUPTIONS = {
    "gaussian_noise",
    "impulse_noise",
    "iso_noise",
    "shot_noise",
}


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _to_float(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Improve local contrast by applying CLAHE to the LAB luminance channel."""
    rgb8 = _to_uint8(image)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return _to_float(enhanced_rgb)


def apply_gamma(image: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Apply gamma correction; gamma < 1 brightens low-light images."""
    return np.power(np.clip(image, 0.0, 1.0), gamma).astype(np.float32)


def apply_denoising(
    image: np.ndarray,
    h: float = 7.0,
    h_color: float = 7.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Reduce color noise with OpenCV non-local means denoising."""
    rgb8 = _to_uint8(image)
    denoised = cv2.fastNlMeansDenoisingColored(
        rgb8,
        None,
        h,
        h_color,
        template_window_size,
        search_window_size,
    )
    return _to_float(denoised)


def compose_preprocessors(preprocessors: list[Preprocessor]) -> Preprocessor:
    def _composed(image: np.ndarray) -> np.ndarray:
        out = image
        for preprocessor in preprocessors:
            out = preprocessor(out)
        return out

    return _composed


def build_preprocessor(
    name: str,
    corruption_type: str | None = None,
    gamma: float = 0.7,
    clahe_clip_limit: float = 2.0,
) -> Preprocessor | None:
    """
    Return a preprocessing function for the requested strategy.

    Supported names:
      none, auto, auto-conservative, clahe, gamma, denoise,
      clahe-gamma, clahe-denoise, restormer, auto-restormer
    """
    normalized = name.lower().replace("_", "-")
    corruption = (corruption_type or "").lower()

    if normalized in {"none", "off", "baseline"}:
        return None

    if normalized == "auto":
        if corruption in GAMMA_CORRUPTIONS:
            return lambda image: apply_gamma(image, gamma=gamma)
        if corruption in DENOISE_CORRUPTIONS:
            return apply_denoising
        if corruption in CLAHE_CORRUPTIONS:
            return lambda image: apply_clahe(image, clip_limit=clahe_clip_limit)
        return None

    if normalized in {"auto-conservative", "conservative"}:
        if corruption in GAMMA_CORRUPTIONS:
            return lambda image: apply_gamma(image, gamma=gamma)
        if corruption in DENOISE_CORRUPTIONS:
            return apply_denoising
        if corruption in CONSERVATIVE_CLAHE_CORRUPTIONS:
            return lambda image: apply_clahe(image, clip_limit=clahe_clip_limit)
        return None

    if normalized == "clahe":
        return lambda image: apply_clahe(image, clip_limit=clahe_clip_limit)
    if normalized == "gamma":
        return lambda image: apply_gamma(image, gamma=gamma)
    if normalized == "denoise":
        return apply_denoising
    if normalized == "clahe-gamma":
        return compose_preprocessors(
            [
                lambda image: apply_clahe(image, clip_limit=clahe_clip_limit),
                lambda image: apply_gamma(image, gamma=gamma),
            ]
        )
    if normalized == "clahe-denoise":
        return compose_preprocessors(
            [
                lambda image: apply_clahe(image, clip_limit=clahe_clip_limit),
                apply_denoising,
            ]
        )

    if normalized == "restormer":
        from src.adapters.restormer_adapter import apply_restormer
        return apply_restormer

    if normalized == "auto-restormer":
        if corruption in GAMMA_CORRUPTIONS:
            return lambda image: apply_gamma(image, gamma=gamma)
        if corruption in RESTORMER_WEATHER_CORRUPTIONS:
            from src.adapters.restormer_adapter import apply_restormer
            return apply_restormer
        if corruption in RESTORMER_NOISE_CORRUPTIONS:
            from src.adapters.restormer_adapter import apply_restormer
            return lambda image: apply_restormer(image, task="real_denoising")
        return None

    raise ValueError(f"Unknown preprocessing strategy: {name}")


def prediction_tag(name: str, gamma: float = 0.7, clahe_clip_limit: float = 2.0) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized in {"none", "off", "baseline"}:
        return "baseline"
    parts = [normalized]
    if "gamma" in normalized or normalized == "auto":
        parts.append(f"g{gamma:g}")
    if "clahe" in normalized or normalized == "auto":
        parts.append(f"c{clahe_clip_limit:g}")
    if normalized == "auto-restormer":
        parts.append(f"g{gamma:g}")
    return "_".join(parts).replace(".", "p")
