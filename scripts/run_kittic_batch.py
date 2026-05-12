"""
Build the KITTI-C manifest, run MiDaS predictions, and compute final metrics.

Examples:
  python scripts/run_kittic_batch.py --build-manifest --max-samples 10
  python scripts/run_kittic_batch.py --preprocess auto --max-samples 10
  python scripts/run_kittic_batch.py --eval-only
  python scripts/run_kittic_batch.py --build-manifest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.kitti_c import build_manifest  # noqa: E402
from src.datasets.transforms import load_kitti_depth  # noqa: E402
from src.evaluation.align import align_scale_shift  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.adapters.preprocessing import build_preprocessor, prediction_tag  # noqa: E402
from src.utils.paths import get_dataset_path, get_output_path  # noqa: E402


MODEL_TYPE = "dpt_hybrid_384"
MIN_DEPTH = 1e-3
MAX_DEPTH = 80.0


def prediction_path(row: pd.Series, pred_dir: Path) -> Path:
    return (
        pred_dir
        / row["corruption_type"]
        / str(row["severity"])
        / row["sequence"]
        / f"{row['frame_id']}.npy"
    )


def ensure_manifest(force_rebuild: bool = False) -> Path:
    manifest_path = get_output_path("manifests") / "kitti_c_manifest.csv"
    if manifest_path.exists() and not force_rebuild:
        return manifest_path

    root = get_dataset_path("kitti_c", "root")
    gt_root = get_dataset_path("kitti_c", "gt_path")
    print(f"Building manifest from {root}")
    print(f"Using GT root {gt_root}")

    df = build_manifest(root, gt_root, split="test")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    gt_count = int(df["gt_path"].notna().sum()) if len(df) else 0
    print(f"Saved manifest: {manifest_path}")
    print(f"Rows: {len(df)}; with GT: {gt_count}")
    return manifest_path


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def load_eval_rows(
    manifest_path: Path,
    max_samples: int | None,
    corruptions: list[str] | None,
    severities: list[str] | None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, dtype={"frame_id": str})
    df = df[df["gt_path"].notna()].reset_index(drop=True)
    if corruptions is not None:
        df = df[df["corruption_type"].isin(corruptions)].reset_index(drop=True)
    if severities is not None:
        severity_values = {int(severity) for severity in severities}
        df = df[df["severity"].astype(int).isin(severity_values)].reset_index(drop=True)
    if max_samples is not None:
        df = df.head(max_samples).reset_index(drop=True)
    return df


def run_inference(
    df: pd.DataFrame,
    pred_dir: Path,
    model_type: str,
    batch_size: int,
    num_workers: int,
    preprocess: str,
    gamma: float,
    clahe_clip_limit: float,
) -> None:
    from src.adapters.midas_adapter import MiDaSAdapter

    adapter = MiDaSAdapter(model_type=model_type)

    todo = [i for i, row in df.iterrows() if not prediction_path(row, pred_dir).exists()]
    if not todo:
        print("All requested predictions already exist.")
        return

    todo_rows = df.loc[todo].reset_index(drop=True)
    print(f"Running inference for {len(todo_rows)} images; skipping {len(df) - len(todo_rows)} existing.")

    image_paths = todo_rows["image_path"].tolist()
    preprocessors = [
        build_preprocessor(
            preprocess,
            corruption_type=row["corruption_type"],
            gamma=gamma,
            clahe_clip_limit=clahe_clip_limit,
        )
        for _, row in todo_rows.iterrows()
    ]
    for start in range(0, len(todo_rows), batch_size):
        batch_rows = todo_rows.iloc[start : start + batch_size]
        batch_paths = image_paths[start : start + batch_size]
        batch_preprocessors = preprocessors[start : start + batch_size]
        depths = adapter.predict_batch(
            batch_paths,
            batch_size=len(batch_paths),
            num_workers=num_workers,
            preprocessors=batch_preprocessors,
        )

        for depth, (_, row) in zip(depths, batch_rows.iterrows()):
            out_path = prediction_path(row, pred_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth)

        done = min(start + len(batch_rows), len(todo_rows))
        print(f"Predicted {done}/{len(todo_rows)}")


def metric_record(
    row: pd.Series,
    pred: np.ndarray,
    pred_path: Path | None,
    model_type: str,
    preprocess: str,
) -> dict:
    gt = load_kitti_depth(row["gt_path"])

    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid_mask = gt > 0
    aligned = align_scale_shift(pred, gt, valid_mask)
    metrics = compute_all_metrics(aligned, gt, valid_mask, MIN_DEPTH, MAX_DEPTH)

    return {
        "image_path": row["image_path"],
        "gt_path": row["gt_path"],
        "pred_path": "" if pred_path is None else str(pred_path),
        "dataset": "kitti_c",
        "corruption_type": row["corruption_type"],
        "severity": row["severity"],
        "sequence": row["sequence"],
        "frame_id": row["frame_id"],
        "model_name": model_type,
        "preprocess": preprocess,
        **metrics,
    }


def write_results(
    records: list[dict],
    metrics_out: Path,
) -> pd.DataFrame:
    results = pd.DataFrame(records)
    if results.empty:
        print("No metrics were computed because no matching prediction files were found.")
        return results

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_out, index=False)
    print(f"Saved {len(results)} metric rows to {metrics_out}")

    summary = (
        results.groupby("corruption_type")[["abs_rel", "rmse", "delta1"]]
        .mean()
        .sort_index()
    )
    print("\nMean metrics per corruption type:")
    print(summary.to_string())
    return results


def run_eval(
    df: pd.DataFrame,
    pred_dir: Path,
    metrics_out: Path,
    model_type: str,
    preprocess: str,
    progress_interval: int,
) -> pd.DataFrame:
    records = []
    missing = 0

    for idx, row in df.iterrows():
        pred_path = prediction_path(row, pred_dir)
        if not pred_path.exists():
            missing += 1
            continue

        try:
            pred = np.load(str(pred_path))
            records.append(metric_record(row, pred, pred_path, model_type, preprocess))
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR [{idx}] {row['image_path']}: {exc}")

        done = idx + 1
        if progress_interval > 0 and (done % progress_interval == 0 or done == len(df)):
            print(f"Evaluated {done}/{len(df)}; metrics rows={len(records)}; missing={missing}", flush=True)

    results = pd.DataFrame(records)
    if missing:
        print(f"Skipped {missing} rows without prediction files.")

    return write_results(records, metrics_out)


def run_stream_eval(
    df: pd.DataFrame,
    pred_dir: Path,
    metrics_out: Path,
    model_type: str,
    preprocess: str,
    gamma: float,
    clahe_clip_limit: float,
    batch_size: int,
    num_workers: int,
    progress_interval: int,
) -> pd.DataFrame:
    from src.adapters.midas_adapter import MiDaSAdapter

    adapter = MiDaSAdapter(model_type=model_type)
    records = []
    image_paths = df["image_path"].tolist()
    preprocessors = [
        build_preprocessor(
            preprocess,
            corruption_type=row["corruption_type"],
            gamma=gamma,
            clahe_clip_limit=clahe_clip_limit,
        )
        for _, row in df.iterrows()
    ]

    for start in range(0, len(df), batch_size):
        batch_rows = df.iloc[start : start + batch_size]
        batch_paths = image_paths[start : start + batch_size]
        batch_preprocessors = preprocessors[start : start + batch_size]

        cached_depths: list[np.ndarray | None] = []
        uncached_positions = []
        uncached_paths = []
        uncached_preprocessors = []

        for local_idx, (_, row) in enumerate(batch_rows.iterrows()):
            pred_path = prediction_path(row, pred_dir)
            if pred_path.exists():
                cached_depths.append(np.load(str(pred_path)))
            else:
                cached_depths.append(None)
                uncached_positions.append(local_idx)
                uncached_paths.append(batch_paths[local_idx])
                uncached_preprocessors.append(batch_preprocessors[local_idx])

        if uncached_paths:
            new_depths = adapter.predict_batch(
                uncached_paths,
                batch_size=len(uncached_paths),
                num_workers=num_workers,
                preprocessors=uncached_preprocessors,
            )
            for local_idx, depth in zip(uncached_positions, new_depths):
                cached_depths[local_idx] = depth

        for depth, (_, row) in zip(cached_depths, batch_rows.iterrows()):
            if depth is None:
                continue
            pred_path = prediction_path(row, pred_dir)
            records.append(
                metric_record(
                    row,
                    depth,
                    pred_path if pred_path.exists() else None,
                    model_type,
                    preprocess,
                )
            )

        done = min(start + len(batch_rows), len(df))
        if progress_interval > 0 and (done % progress_interval == 0 or done == len(df)):
            print(f"Stream-evaluated {done}/{len(df)}; metrics rows={len(records)}", flush=True)

    return write_results(records, metrics_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", action="store_true", help="Build or rebuild the KITTI-C manifest first.")
    parser.add_argument("--eval-only", action="store_true", help="Skip MiDaS inference and only evaluate existing .npy predictions.")
    parser.add_argument("--stream-eval", action="store_true", help="Predict and evaluate without saving new .npy files.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for smoke testing.")
    parser.add_argument("--corruptions", default=None, help="Comma-separated corruption filter, e.g. dark,snow,frost.")
    parser.add_argument("--severities", default=None, help="Comma-separated severity filter, e.g. 3,4,5.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel image-loading workers.")
    parser.add_argument("--model-type", default=MODEL_TYPE, help="MiDaS model type.")
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=[
            "none",
            "auto",
            "auto-conservative",
            "clahe",
            "gamma",
            "denoise",
            "clahe-gamma",
            "clahe-denoise",
        ],
        help="Image preprocessing strategy before MiDaS inference.",
    )
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma correction value; gamma < 1 brightens images.")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clipLimit.")
    parser.add_argument("--eval-progress-interval", type=int, default=500, help="Print evaluation progress every N rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = ensure_manifest(force_rebuild=args.build_manifest)
    tag = prediction_tag(args.preprocess, gamma=args.gamma, clahe_clip_limit=args.clahe_clip_limit)
    if tag == "baseline":
        pred_dir = get_output_path("predictions") / "kitti_c"
        metrics_out = get_output_path("metrics") / "kittic_results.csv"
    else:
        pred_dir = get_output_path("predictions") / "kitti_c" / tag
        metrics_out = get_output_path("metrics") / f"kittic_results_{tag}.csv"

    df = load_eval_rows(
        manifest_path,
        args.max_samples,
        _split_csv_arg(args.corruptions),
        _split_csv_arg(args.severities),
    )
    if df.empty:
        raise RuntimeError("Manifest has no rows with KITTI ground-truth depth.")

    print(f"Loaded {len(df)} evaluable rows from {manifest_path}")
    print(f"Preprocessing: {args.preprocess} -> predictions: {pred_dir}")
    if args.stream_eval and args.eval_only:
        raise ValueError("--stream-eval and --eval-only cannot be used together.")

    if args.stream_eval:
        run_stream_eval(
            df,
            pred_dir,
            metrics_out,
            args.model_type,
            args.preprocess,
            args.gamma,
            args.clahe_clip_limit,
            args.batch_size,
            args.num_workers,
            args.eval_progress_interval,
        )
        return

    if not args.eval_only:
        run_inference(
            df,
            pred_dir,
            args.model_type,
            args.batch_size,
            args.num_workers,
            args.preprocess,
            args.gamma,
            args.clahe_clip_limit,
        )
    run_eval(df, pred_dir, metrics_out, args.model_type, args.preprocess, args.eval_progress_interval)


if __name__ == "__main__":
    main()
