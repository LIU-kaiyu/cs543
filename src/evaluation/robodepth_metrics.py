"""
RoboDepth-style evaluation wrapper.

Thin wrapper around the logic in third_party/RoboDepth/competition/evaluator_npy.py.
We replicate the compute_errors function here so we can call it without needing
to launch the full evaluator CLI, while keeping results consistent with the
official RoboDepth competition submission format.
"""
import numpy as np
from skimage.transform import resize as sk_resize


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute the RoboDepth standard error metrics.
    Matches third_party/RoboDepth/competition/evaluator_npy.py::compute_errors.

    Args:
        gt: ground-truth depth (H, W), float32
        pred: predicted depth (H, W), float32 — already aligned and clipped

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())

    rmse_val = float(np.sqrt(((gt - pred) ** 2).mean()))

    log_valid = (gt > 0) & (pred > 0)
    rmse_log_val = float(
        np.sqrt(((np.log(gt[log_valid]) - np.log(pred[log_valid])) ** 2).mean())
    )

    abs_rel_val = float(np.mean(np.abs(gt - pred) / gt))
    sq_rel_val = float(np.mean(((gt - pred) ** 2) / gt))

    return {
        "abs_rel": abs_rel_val,
        "sq_rel": sq_rel_val,
        "rmse": rmse_val,
        "rmse_log": rmse_log_val,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


def evaluate_batch(
    pred_disps: np.ndarray,
    gt_depths: list[np.ndarray],
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
    use_median_scaling: bool = True,
    eigen_crop: bool = True,
) -> list[dict]:
    """
    Evaluate a batch of predicted disparities against GT depths.
    Mirrors the evaluation loop in evaluator_npy.py.

    Args:
        pred_disps: stacked predictions, shape (N, H, W) — raw disparity (inverse depth)
        gt_depths: list of N ground-truth depth arrays (may have varying sizes)
        min_depth: KITTI convention — 1e-3
        max_depth: KITTI convention — 80.0
        use_median_scaling: apply median scale factor (mono evaluation standard)
        eigen_crop: apply KITTI Eigen crop

    Returns:
        list of N dicts, each with the metrics from compute_errors
    """
    results = []

    for i in range(len(pred_disps)):
        gt_depth = gt_depths[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = sk_resize(pred_disps[i], (gt_h, gt_w))
        pred_depth = 1.0 / np.clip(pred_disp, 1e-8, None)

        mask = (gt_depth > min_depth) & (gt_depth < max_depth)

        if eigen_crop:
            crop = np.array(
                [
                    0.40810811 * gt_h,
                    0.99189189 * gt_h,
                    0.03594771 * gt_w,
                    0.96405229 * gt_w,
                ],
                dtype=np.int32,
            )
            crop_mask = np.zeros_like(mask)
            crop_mask[crop[0]: crop[1], crop[2]: crop[3]] = True
            mask = mask & crop_mask

        if mask.sum() == 0:
            results.append({k: float("nan") for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]})
            continue

        pred_depth_masked = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        if use_median_scaling:
            ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
            pred_depth_masked *= ratio

        pred_depth_masked = np.clip(pred_depth_masked, min_depth, max_depth)

        results.append(compute_errors(gt_depth_masked, pred_depth_masked))

    return results
