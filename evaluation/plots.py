"""Generates the benchmark plots required by the brief (section 11):
accuracy_vs_tolerance, error_cdf, error_distribution, pr_curve, and
accuracy broken down by noise/scale/rotation/family.

Styling follows a validated categorical palette (fixed hue order, never
cycled) and a single sequential hue for magnitude, rather than matplotlib
defaults - see the dataviz skill referenced in this project's session for
the underlying method.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import metrics as metrics_mod

# Fixed-order categorical palette (light-mode chart surface).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "axes.grid": True,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _series_color(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_vs_tolerance(df: pd.DataFrame, out_path: str, group_col: str = "split",
                                max_tol_px: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tolerances = np.arange(0, max_tol_px + 1, 1)
    for i, (group, sub) in enumerate(sorted(df.groupby(group_col))):
        err = sub["error_px"].to_numpy()
        acc = [np.mean(err <= t) for t in tolerances]
        ax.plot(tolerances, acc, color=_series_color(i), linewidth=2, label=str(group))
    ax.set_xlabel("Tolerance (px)")
    ax.set_ylabel("Accuracy (fraction of pairs)")
    ax.set_title("Accuracy vs. tolerance")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def plot_error_cdf(df: pd.DataFrame, out_path: str, group_col: str = "split") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (group, sub) in enumerate(sorted(df.groupby(group_col))):
        err = np.sort(sub["error_px"].to_numpy())
        cdf = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, cdf, color=_series_color(i), linewidth=2, label=str(group))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Error (px, symlog scale)")
    ax.set_ylabel("Cumulative fraction of pairs")
    ax.set_title("Error CDF")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out_path)


def plot_error_distribution(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    err = df["error_px"].to_numpy()
    bins = np.logspace(np.log10(max(err.min(), 1e-2)), np.log10(max(err.max(), 1.0)), 30)
    ax.hist(err, bins=bins, color=SEQUENTIAL_BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Error (px, log scale)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Error distribution (pooled across splits)")
    _save(fig, out_path)


def plot_pr_curve(df: pd.DataFrame, out_path: str, tolerance_px: float = 5.0) -> None:
    """Precision-recall curve using each prediction's classical confidence
    (ZNCC score) as the ranking signal and error_px <= tolerance_px as the
    correctness label - the same style of PR curve used to characterize
    the classical baseline in the prior systems studied during Phase 0."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ordered = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    correct = (ordered["error_px"] <= tolerance_px).to_numpy().astype(int)
    tp_cumsum = np.cumsum(correct)
    n = np.arange(1, len(ordered) + 1)
    precision = tp_cumsum / n
    recall = tp_cumsum / max(correct.sum(), 1)
    ax.plot(recall, precision, color=SEQUENTIAL_BLUE, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall (correct = error <= {tolerance_px:.0f}px, ranked by confidence)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _save(fig, out_path)


def _bar_breakdown(df: pd.DataFrame, group_col: str, out_path: str, title: str,
                    tolerance_px: float = 5.0) -> None:
    df2 = metrics_mod.add_breakdown_columns(df) if group_col not in df.columns else df
    groups = sorted(df2[group_col].dropna().unique())
    accs = [float(np.mean(df2.loc[df2[group_col] == g, "error_px"] <= tolerance_px)) for g in groups]
    counts = [int((df2[group_col] == g).sum()) for g in groups]

    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(groups)), 4.5))
    bars = ax.bar(range(len(groups)), accs, color=[_series_color(i) for i in range(len(groups))], width=0.6)
    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{accs[i]:.0%}\n(n={n})",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"Accuracy @ {tolerance_px:.0f}px")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    _save(fig, out_path)


def plot_accuracy_by_noise(df: pd.DataFrame, out_path: str) -> None:
    _bar_breakdown(df, "noise_level", out_path, "Accuracy by noise level")


def plot_accuracy_by_scale(df: pd.DataFrame, out_path: str) -> None:
    _bar_breakdown(df, "scale_condition", out_path, "Accuracy by scale condition")


def plot_accuracy_by_rotation(df: pd.DataFrame, out_path: str) -> None:
    _bar_breakdown(df, "rotation_condition", out_path, "Accuracy by rotation condition")


def plot_accuracy_by_family(df: pd.DataFrame, out_path: str) -> None:
    _bar_breakdown(df, "structural_family", out_path, "Accuracy by structural family")


def generate_all_plots(df: pd.DataFrame, out_dir: str) -> None:
    df = metrics_mod.add_breakdown_columns(df)
    os.makedirs(out_dir, exist_ok=True)
    plot_accuracy_vs_tolerance(df, os.path.join(out_dir, "accuracy_vs_tolerance.png"))
    plot_error_cdf(df, os.path.join(out_dir, "error_cdf.png"))
    plot_error_distribution(df, os.path.join(out_dir, "error_distribution.png"))
    plot_pr_curve(df, os.path.join(out_dir, "pr_curve.png"))
    plot_accuracy_by_noise(df, os.path.join(out_dir, "accuracy_by_noise.png"))
    plot_accuracy_by_scale(df, os.path.join(out_dir, "accuracy_by_scale.png"))
    plot_accuracy_by_rotation(df, os.path.join(out_dir, "accuracy_by_rotation.png"))
    plot_accuracy_by_family(df, os.path.join(out_dir, "accuracy_by_family.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark plots from a per-pair results file.")
    parser.add_argument("--results", default="outputs/reports/per_pair_results.csv")
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()
    generate_all_plots(pd.read_csv(args.results), args.out)
