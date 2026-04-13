"""
Report table generators for the benchmark results DataFrame.
"""
import pandas as pd
import numpy as np


_METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]


def corruption_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per corruption type (mean across all severities).

    Returns a DataFrame indexed by corruption_type with one column per metric.
    """
    available = [m for m in _METRICS if m in df.columns]
    return (
        df.groupby("corruption_type")[available]
        .mean()
        .sort_values("abs_rel", ascending=True)
    )


def severity_curve(
    df: pd.DataFrame, corruption_type: str, metric: str = "abs_rel"
) -> pd.DataFrame:
    """
    Return mean metric value per severity level for a given corruption type.

    Args:
        df: results DataFrame
        corruption_type: e.g. "fog", "motion_blur"
        metric: column name to aggregate

    Returns:
        DataFrame with columns [severity, <metric>], sorted by severity
    """
    sub = df[df["corruption_type"] == corruption_type]
    curve = sub.groupby("severity")[metric].mean().reset_index()
    return curve.sort_values("severity")


def model_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per model_name across all corruptions.
    Requires a 'model_name' column in df.
    """
    available = [m for m in _METRICS if m in df.columns]
    return (
        df.groupby("model_name")[available]
        .mean()
        .sort_values("abs_rel", ascending=True)
    )


def per_corruption_severity_pivot(
    df: pd.DataFrame, metric: str = "abs_rel"
) -> pd.DataFrame:
    """
    Build a (corruption_type x severity) pivot table for a given metric.
    """
    return df.pivot_table(
        index="corruption_type",
        columns="severity",
        values=metric,
        aggfunc="mean",
    )
