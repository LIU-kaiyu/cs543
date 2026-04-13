"""
Standard monocular depth evaluation metrics.

All functions expect inputs that are already aligned (see evaluation/align.py)
and clipped to a valid depth range.
"""
import numpy as np


def abs_rel(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Absolute relative error: mean(|gt - pred| / gt)"""
    p, g = pred[valid_mask], gt[valid_mask]
    return float(np.mean(np.abs(g - p) / g))


def sq_rel(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Squared relative error: mean((gt - pred)^2 / gt)"""
    p, g = pred[valid_mask], gt[valid_mask]
    return float(np.mean((g - p) ** 2 / g))


def rmse(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Root mean squared error."""
    p, g = pred[valid_mask], gt[valid_mask]
    return float(np.sqrt(np.mean((g - p) ** 2)))


def rmse_log(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """Root mean squared error in log space."""
    p, g = pred[valid_mask], gt[valid_mask]
    pos = (p > 0) & (g > 0)
    return float(np.sqrt(np.mean((np.log(g[pos]) - np.log(p[pos])) ** 2)))


def delta_threshold(
    pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray, threshold: float = 1.25
) -> float:
    """Percentage of pixels where max(gt/pred, pred/gt) < threshold."""
    p, g = pred[valid_mask], gt[valid_mask]
    pos = (p > 0) & (g > 0)
    ratio = np.maximum(g[pos] / p[pos], p[pos] / g[pos])
    return float((ratio < threshold).mean())


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> dict:
    """
    Compute all standard metrics after clipping predictions to [min_depth, max_depth].

    Args:
        pred: aligned depth map (H, W)
        gt: ground-truth depth (H, W)
        valid_mask: boolean mask of valid GT pixels (H, W)
        min_depth: minimum valid depth (metres)
        max_depth: maximum valid depth (metres)

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3
    """
    # Further restrict mask to depth range
    depth_mask = valid_mask & (gt >= min_depth) & (gt <= max_depth)

    # Clip predictions to prevent log(0) and division issues
    pred_clipped = np.clip(pred, min_depth, max_depth)

    return {
        "abs_rel":  abs_rel(pred_clipped, gt, depth_mask),
        "sq_rel":   sq_rel(pred_clipped, gt, depth_mask),
        "rmse":     rmse(pred_clipped, gt, depth_mask),
        "rmse_log": rmse_log(pred_clipped, gt, depth_mask),
        "delta1":   delta_threshold(pred_clipped, gt, depth_mask, 1.25),
        "delta2":   delta_threshold(pred_clipped, gt, depth_mask, 1.25 ** 2),
        "delta3":   delta_threshold(pred_clipped, gt, depth_mask, 1.25 ** 3),
        "n_valid":  int(depth_mask.sum()),
    }
