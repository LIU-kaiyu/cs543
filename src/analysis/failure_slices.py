"""
Failure slice extraction from a benchmark results DataFrame.

The results CSV is expected to have at minimum:
  image_path, corruption_type, severity, abs_rel, delta1, rmse, ...
"""
import pandas as pd


def get_worst_n(
    df: pd.DataFrame,
    metric: str = "abs_rel",
    corruption_type: str | None = None,
    severity: int | None = None,
    n: int = 20,
) -> pd.DataFrame:
    """Return the n rows with the highest (worst) metric value."""
    sub = _filter(df, corruption_type, severity)
    return sub.nlargest(n, metric)


def get_best_n(
    df: pd.DataFrame,
    metric: str = "abs_rel",
    corruption_type: str | None = None,
    severity: int | None = None,
    n: int = 20,
) -> pd.DataFrame:
    """Return the n rows with the lowest (best) metric value."""
    sub = _filter(df, corruption_type, severity)
    return sub.nsmallest(n, metric)


def get_median_n(
    df: pd.DataFrame,
    metric: str = "abs_rel",
    corruption_type: str | None = None,
    severity: int | None = None,
    n: int = 20,
) -> pd.DataFrame:
    """Return n rows closest to the median metric value."""
    sub = _filter(df, corruption_type, severity).copy()
    med = sub[metric].median()
    sub["_dist_to_median"] = (sub[metric] - med).abs()
    result = sub.nsmallest(n, "_dist_to_median").drop(columns=["_dist_to_median"])
    return result


def _filter(
    df: pd.DataFrame,
    corruption_type: str | None,
    severity: int | None,
) -> pd.DataFrame:
    sub = df.copy()
    if corruption_type is not None:
        sub = sub[sub["corruption_type"] == corruption_type]
    if severity is not None:
        sub = sub[sub["severity"] == severity]
    return sub
