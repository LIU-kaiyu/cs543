"""
Shared image preprocessing utilities for loading ground-truth depth maps.
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    """Load an RGB image as float32 in [0, 1], shape (H, W, 3)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_kitti_depth(path: str) -> np.ndarray:
    """
    Load a KITTI depth PNG (16-bit, depth = pixel_value / 256).
    Returns float32 depth in metres; 0 means invalid.
    """
    depth_png = np.array(Image.open(str(path)), dtype=np.uint16)
    depth = depth_png.astype(np.float32) / 256.0
    return depth


def load_nyu_depth(path: str) -> np.ndarray:
    """
    Load an NYU depth PNG (16-bit, depth = pixel_value / 1000).
    Returns float32 depth in metres; 0 means invalid.
    """
    depth_png = np.array(Image.open(str(path)), dtype=np.uint16)
    depth = depth_png.astype(np.float32) / 1000.0
    return depth


def load_diode_depth(depth_npy: str, valid_mask_npy: str):
    """
    Load DIODE depth (stored as .npy) and its binary valid mask.
    Returns (depth, mask) both float32/bool.
    """
    depth = np.load(str(depth_npy)).squeeze().astype(np.float32)
    mask = np.load(str(valid_mask_npy)).squeeze().astype(bool)
    return depth, mask


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a depth map using nearest-neighbour to avoid blending invalid pixels."""
    return cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
