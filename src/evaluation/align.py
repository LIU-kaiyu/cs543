"""
Scale-and-shift alignment of MiDaS relative inverse depth to metric ground truth.

MiDaS outputs relative inverse depth (disparity-like values), not calibrated
metric depth. Direct comparison to GT depth will partly measure scale mismatch
rather than scene-understanding quality. Alignment is REQUIRED before metrics.

Two strategies are provided:
  - align_scale_shift: least-squares fit of (scale, shift) — recommended
  - align_scale_only:  median-ratio alignment — simpler, less accurate
"""
import numpy as np


def align_scale_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Fit pred_aligned = scale * pred + shift to minimise MSE on valid GT pixels
    using a closed-form least-squares solution.

    Args:
        pred: raw MiDaS output, shape (H, W), float32, any positive range
        gt: metric ground-truth depth, shape (H, W), float32, metres
        valid_mask: boolean mask of pixels with valid GT, shape (H, W)

    Returns:
        pred_aligned: aligned depth map, same shape as pred
    """
    p = pred[valid_mask].astype(np.float64)
    g = gt[valid_mask].astype(np.float64)

    if p.size == 0:
        return pred.astype(np.float32, copy=True)

    # Closed-form least-squares fit for g ~= scale * p + shift.
    # This is equivalent to np.linalg.lstsq([p, 1], g), but avoids building
    # a large dense design matrix for every KITTI-C image.
    p_mean = p.mean()
    g_mean = g.mean()
    p_centered = p - p_mean
    denom = np.sum(p_centered * p_centered)
    if denom <= np.finfo(np.float64).eps:
        scale = 0.0
        shift = g_mean
    else:
        scale = np.sum(p_centered * (g - g_mean)) / denom
        shift = g_mean - scale * p_mean

    pred_aligned = (scale * pred + shift).astype(np.float32)
    return pred_aligned


def align_scale_only(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Scale pred by the median ratio gt/pred on valid pixels.
    Faster but less accurate than align_scale_shift.

    Args:
        pred: raw MiDaS output, shape (H, W)
        gt: metric ground-truth depth, shape (H, W)
        valid_mask: boolean mask, shape (H, W)

    Returns:
        pred_aligned: scaled depth map, same shape as pred
    """
    p = pred[valid_mask]
    g = gt[valid_mask]

    # Avoid division by zero
    nonzero = p > 0
    if nonzero.sum() == 0:
        return pred.copy()

    scale = np.median(g[nonzero] / p[nonzero])
    return (scale * pred).astype(np.float32)
