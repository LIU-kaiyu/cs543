"""Produce figures comparing baseline vs auto-restormer on KITTI-C."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs/figures/restormer_analysis"
OUT.mkdir(parents=True, exist_ok=True)

BASE = PROJECT_ROOT / "outputs/metrics/kittic_results.csv"
EXP = PROJECT_ROOT / "outputs/metrics/kittic_results_auto-restormer_g0p7.csv"

base = pd.read_csv(BASE)
exp = pd.read_csv(EXP)


def figure_per_corruption_absrel() -> None:
    g_b = base.groupby("corruption_type")["abs_rel"].mean()
    g_e = exp.groupby("corruption_type")["abs_rel"].mean()
    order = g_b.sort_values().index.tolist()
    g_b = g_b.reindex(order)
    g_e = g_e.reindex(order)
    y = np.arange(len(order))
    h = 0.4
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y - h / 2, g_b.values, h, label="Baseline", color="#999999")
    ax.barh(y + h / 2, g_e.values, h, label="Auto-Restormer", color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("AbsRel (lower is better)")
    ax.set_title("Per-corruption AbsRel: Baseline vs Auto-Restormer")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    for yi, b, e in zip(y, g_b.values, g_e.values):
        delta_pct = (e - b) / b * 100.0
        marker = "down" if delta_pct < 0 else "up"
        color = "#2ca02c" if delta_pct < -0.5 else ("#d62728" if delta_pct > 0.5 else "#555555")
        ax.text(max(b, e) + 0.005, yi, f"{marker} {abs(delta_pct):.1f}%", va="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_corruption_absrel.png", dpi=200)
    fig.savefig(OUT / "fig_per_corruption_absrel.pdf")
    plt.close(fig)


def figure_improvement_bar() -> None:
    g_b = base.groupby("corruption_type")["abs_rel"].mean()
    g_e = exp.groupby("corruption_type")["abs_rel"].mean()
    pct = ((g_b - g_e) / g_b * 100.0).sort_values()
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in pct.values]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(pct))
    ax.barh(y, pct.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(pct.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AbsRel relative improvement (%) -- positive = Auto-Restormer better")
    ax.set_title("Per-corruption AbsRel improvement from Auto-Restormer")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    for yi, v in zip(y, pct.values):
        offset = 0.6 if v >= 0 else -0.6
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, yi, f"{v:+.2f}%", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_corruption_improvement.png", dpi=200)
    fig.savefig(OUT / "fig_per_corruption_improvement.pdf")
    plt.close(fig)


def figure_severity_curves() -> None:
    severities = sorted(base["severity"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = ["Overall (all corruptions)", "Snow only", "Noise (gauss/shot/impulse/iso)"]
    selectors = [
        lambda df: df,
        lambda df: df[df["corruption_type"] == "snow"],
        lambda df: df[df["corruption_type"].isin(["gaussian_noise", "shot_noise", "impulse_noise", "iso_noise"])],
    ]
    for ax, title, sel in zip(axes, titles, selectors):
        b_vals = [sel(base[base["severity"] == s])["abs_rel"].mean() for s in severities]
        e_vals = [sel(exp[exp["severity"] == s])["abs_rel"].mean() for s in severities]
        ax.plot(severities, b_vals, marker="o", color="#999999", label="Baseline", linewidth=2)
        ax.plot(severities, e_vals, marker="s", color="#1f77b4", label="Auto-Restormer", linewidth=2)
        ax.set_xlabel("Severity")
        ax.set_ylabel("AbsRel")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_severity_curves.png", dpi=200)
    fig.savefig(OUT / "fig_severity_curves.pdf")
    plt.close(fig)


def figure_strategy_overall() -> None:
    strategies = ["Baseline", "Naive Auto", "Auto-Conservative", "Auto-Restormer"]
    absrel = [0.330317, 0.330889, 0.329626, 0.328243]
    rmse = [7.039359, 7.046996, 7.029620, 6.995024]
    delta1 = [0.472845, 0.471136, 0.473361, 0.474216]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [("AbsRel (lower better)", absrel), ("RMSE (lower better)", rmse), (r"$\delta_1$ (higher better)", delta1)]
    for ax, (name, values) in zip(axes, metrics):
        colors = ["#999999", "#ff7f0e", "#9467bd", "#1f77b4"]
        x = np.arange(len(strategies))
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, fontsize=9)
        ax.set_title(name)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        best = min(values) if "lower" in name else max(values)
        for bar, v in zip(bars, values):
            color = "#2ca02c" if v == best else "black"
            weight = "bold" if v == best else "normal"
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=color, fontweight=weight)
        margin = (max(values) - min(values)) * 0.5 + 1e-3
        ax.set_ylim(min(values) - margin, max(values) + margin * 2)
    fig.suptitle("Overall KITTI-C metrics across four preprocessing strategies (n=59,332)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_strategy_comparison.png", dpi=200)
    fig.savefig(OUT / "fig_strategy_comparison.pdf")
    plt.close(fig)


def figure_snow_severity_detail() -> None:
    severities = sorted(s for s in base["severity"].unique() if s > 0)
    metrics = ["abs_rel", "rmse", "delta1"]
    titles = ["Snow: AbsRel (lower better)", "Snow: RMSE (lower better)", r"Snow: $\delta_1$ (higher better)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, m, title in zip(axes, metrics, titles):
        b_vals = [base[(base["corruption_type"] == "snow") & (base["severity"] == s)][m].mean() for s in severities]
        e_vals = [exp[(exp["corruption_type"] == "snow") & (exp["severity"] == s)][m].mean() for s in severities]
        ax.plot(severities, b_vals, marker="o", color="#999999", label="Baseline", linewidth=2)
        ax.plot(severities, e_vals, marker="s", color="#1f77b4", label="Auto-Restormer", linewidth=2)
        ax.set_xlabel("Severity")
        ax.set_ylabel(m)
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_snow_severity_metrics.png", dpi=200)
    fig.savefig(OUT / "fig_snow_severity_metrics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_per_corruption_absrel()
    figure_improvement_bar()
    figure_severity_curves()
    figure_strategy_overall()
    figure_snow_severity_detail()
    print("All figures written to", OUT)
