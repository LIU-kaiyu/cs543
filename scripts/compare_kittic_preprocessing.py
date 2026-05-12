"""
Compare baseline KITTI-C metrics against a preprocessing experiment.

Example:
  python scripts/compare_kittic_preprocessing.py \
    --baseline outputs/metrics/kittic_results.csv \
    --experiment outputs/metrics/kittic_results_auto_g0p7_c2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["abs_rel", "rmse", "delta1"]


def aggregate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Metrics file is empty: {path}")
    return df.groupby("corruption_type")[METRICS].mean().sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()

    baseline = aggregate(args.baseline)
    experiment = aggregate(args.experiment)

    joined = baseline.join(experiment, lsuffix="_baseline", rsuffix="_experiment", how="inner")
    joined["abs_rel_delta"] = joined["abs_rel_experiment"] - joined["abs_rel_baseline"]
    joined["rmse_delta"] = joined["rmse_experiment"] - joined["rmse_baseline"]
    joined["delta1_delta"] = joined["delta1_experiment"] - joined["delta1_baseline"]
    joined["abs_rel_improvement_pct"] = -100.0 * joined["abs_rel_delta"] / joined["abs_rel_baseline"]
    joined["rmse_improvement_pct"] = -100.0 * joined["rmse_delta"] / joined["rmse_baseline"]

    columns = [
        "abs_rel_baseline",
        "abs_rel_experiment",
        "abs_rel_delta",
        "abs_rel_improvement_pct",
        "rmse_baseline",
        "rmse_experiment",
        "rmse_delta",
        "rmse_improvement_pct",
        "delta1_baseline",
        "delta1_experiment",
        "delta1_delta",
    ]
    print(joined[columns].to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
