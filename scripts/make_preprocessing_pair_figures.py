"""
Create 3x2 preprocessing comparison figures for the report.

Rows:
  1. Gamma correction for dark corruption
  2. Denoising for noise corruption
  3. CLAHE for snow corruption

Columns:
  corrupted image | preprocessed image
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.preprocessing import apply_clahe, apply_denoising, apply_gamma  # noqa: E402
from src.datasets.transforms import load_rgb  # noqa: E402


BASELINE_CSV = PROJECT_ROOT / "outputs" / "metrics" / "kittic_results.csv"
CONSERVATIVE_CSV = PROJECT_ROOT / "outputs" / "metrics" / "kittic_results_auto-conservative.csv"
FIGURE_DIR = PROJECT_ROOT / "figures"


def comparison_rows() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_CSV)
    conservative = pd.read_csv(CONSERVATIVE_CSV)
    keys = ["image_path", "corruption_type", "severity", "sequence", "frame_id"]
    merged = baseline.merge(
        conservative,
        on=keys,
        suffixes=("_base", "_improved"),
        how="inner",
    )
    merged["absrel_gain"] = merged["abs_rel_base"] - merged["abs_rel_improved"]
    merged = merged[merged["image_path"].map(lambda path: Path(path).exists())]
    return merged.reset_index(drop=True)


def select_examples(df: pd.DataFrame, corruption_filter: list[str], n: int) -> pd.DataFrame:
    sub = df[df["corruption_type"].isin(corruption_filter)].copy()
    sub = sub.sort_values(["absrel_gain", "severity"], ascending=[False, False])
    return sub.head(n).reset_index(drop=True)


def plot_pair(ax_left, ax_right, row: pd.Series, preprocessor, left_title: str, right_title: str) -> None:
    image = load_rgb(row["image_path"])
    processed = preprocessor(image)
    ax_left.imshow(image)
    ax_left.set_title(left_title, fontsize=10, pad=2)
    ax_left.axis("off")
    ax_left.set_anchor("C")
    ax_right.imshow(processed)
    ax_right.set_title(right_title, fontsize=10, pad=2)
    ax_right.axis("off")
    ax_right.set_anchor("C")


def make_figure(dark_row: pd.Series, noise_row: pd.Series, snow_row: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 5.35), constrained_layout=False)

    plot_pair(
        axes[0, 0],
        axes[0, 1],
        dark_row,
        lambda image: apply_gamma(image, gamma=0.7),
        "Dark corrupted",
        "Gamma corrected",
    )
    axes[0, 0].set_ylabel("Gamma correction", fontsize=11)

    plot_pair(
        axes[1, 0],
        axes[1, 1],
        noise_row,
        apply_denoising,
        f"{noise_row['corruption_type']} corrupted",
        "Denoised",
    )
    axes[1, 0].set_ylabel("Denoising", fontsize=11)

    plot_pair(
        axes[2, 0],
        axes[2, 1],
        snow_row,
        lambda image: apply_clahe(image, clip_limit=2.0),
        "Snow corrupted",
        "CLAHE enhanced",
    )
    axes[2, 0].set_ylabel("CLAHE", fontsize=11)

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.07, 0.5)

    fig.suptitle("Corruption-aware preprocessing examples", fontsize=13, y=0.992)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.935, bottom=0.005, wspace=0.012, hspace=0.035)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(output_path)


def main() -> None:
    df = comparison_rows()
    n_variants = 4
    dark = select_examples(df, ["dark"], n_variants)
    noise = select_examples(df, ["gaussian_noise", "shot_noise", "impulse_noise", "iso_noise"], n_variants)
    snow = select_examples(df, ["snow"], n_variants)

    for idx in range(n_variants):
        output = FIGURE_DIR / f"preprocessing_examples_3x2_v{idx + 1}.png"
        make_figure(dark.iloc[idx], noise.iloc[idx], snow.iloc[idx], output)

    # Stable alias for the figure we recommend using in the report.
    make_figure(dark.iloc[0], noise.iloc[0], snow.iloc[0], FIGURE_DIR / "preprocessing_examples_3x2_recommended.png")


if __name__ == "__main__":
    main()
