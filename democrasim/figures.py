"""Matplotlib figures for the headline results.

Style follows a small set of chart rules: thin marks (2px lines), one axis,
recessive hairline grid, muted axis ink, categorical hues assigned in fixed
order by entity (measured world = blue, moment-matched toy = aqua,
homogeneous toy = yellow — never reassigned when a series drops out), a
legend plus selective direct labels, and text in ink colors rather than
series colors. Low-contrast slots (aqua, yellow) are always paired with
direct labels.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from democrasim.electorate import Electorate

# Reference palette (validated set; see dataviz skill palette.md).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "lines.linewidth": 2.0,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _new_axes(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="x", visible=False)
    return fig, ax


def _direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=INK_SECONDARY,
        annotation_clip=False,
    )
    ax.plot([x], [y], marker="o", markersize=5, color=color, zorder=5)


def impact_distribution(
    electorate: Electorate,
    *,
    clip: tuple[float, float] = (-6_000, 6_000),
    bins: int = 80,
) -> plt.Figure:
    """Small-multiple weighted histograms of household impacts per policy.

    The spike of voters with no stake is reported as text rather than drawn
    (it would flatten every other bin), and tail mass beyond the clip window
    is folded into the edge bins with a note.
    """
    with plt.rc_context(_RC):
        n = electorate.n_policies
        fig, axes = plt.subplots(
            n, 1, figsize=(7.0, 2.1 * n), sharex=True, constrained_layout=True
        )
        edges = np.linspace(clip[0], clip[1], bins + 1)
        total_weight = electorate.weights.sum()
        for j, (ax, label) in enumerate(
            zip(np.atleast_1d(axes), electorate.policy_labels, strict=True)
        ):
            ax.grid(axis="x", visible=False)
            delta = electorate.deltas[:, j]
            affected = np.abs(delta) > 1.0
            share_zero = 1.0 - electorate.weights[affected].sum() / total_weight
            clipped = np.clip(delta[affected], clip[0], clip[1])
            ax.hist(
                clipped,
                bins=edges,
                weights=electorate.weights[affected] / total_weight,
                color=SERIES[j],
                edgecolor=SURFACE,
                linewidth=0.4,
            )
            ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=0)
            ax.set_title(label)
            ax.text(
                0.99,
                0.82,
                f"{share_zero:.0%} of adults see ≈$0\n"
                "(not drawn; tails folded into edge bins)",
                transform=ax.transAxes,
                ha="right",
                fontsize=8.5,
                color=INK_SECONDARY,
            )
            ax.set_ylabel("share of adults")
        np.atleast_1d(axes)[-1].set_xlabel(
            "true change in household net income, dollars per year"
        )
        return fig


def margin_distribution(
    electorates: Sequence[Electorate],
    names: Sequence[str],
) -> plt.Figure:
    """Weighted ECDF of |true margin| between the two policies, log x.

    The x-position where each curve rises is the dollars actually at stake
    for voters in that world — the quantity perception noise competes with.
    """
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 3.6))
        for j, (electorate, name) in enumerate(zip(electorates, names, strict=True)):
            magnitude = np.abs(electorate.margins)
            order = np.argsort(magnitude)
            x = magnitude[order]
            cdf = np.cumsum(electorate.weights[order])
            cdf = cdf / cdf[-1]
            positive = x > 0
            ax.plot(
                np.maximum(x[positive], 1.0),
                cdf[positive],
                color=SERIES[j],
                label=name,
            )
            _direct_label(
                ax,
                float(np.maximum(x[-1], 1.0)),
                float(cdf[-1]),
                name,
                SERIES[j],
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("|true margin between policies|, dollars per year (log scale)")
        ax.set_ylabel("share of adults at or below")
        ax.set_title("How much is actually at stake for each voter")
        ax.legend(loc="upper left")
        return fig


def accuracy_curve(
    sweeps: Sequence[pd.DataFrame],
    names: Sequence[str],
    *,
    target: float | None = 0.9,
    thresholds: Sequence[float] | None = None,
) -> plt.Figure:
    """P(elects the welfare-optimal policy) vs mean ranking accuracy."""
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.2))
        for j, (sweep, name) in enumerate(zip(sweeps, names, strict=True)):
            frame = sweep.sort_values("mean_ranking_accuracy")
            x = frame["mean_ranking_accuracy"].to_numpy()
            y = frame["p_tracked"].to_numpy()
            se = frame["p_tracked_se"].to_numpy()
            ax.fill_between(
                x,
                y - 1.96 * se,
                y + 1.96 * se,
                color=SERIES[j],
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(x, y, color=SERIES[j], label=name)
            _direct_label(ax, float(x[-1]), float(y[-1]), name, SERIES[j])
        if target is not None:
            ax.axhline(target, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
            ax.text(
                ax.get_xlim()[0],
                target,
                f"  target {target:.0%}",
                va="bottom",
                fontsize=8.5,
                color=INK_MUTED,
            )
        if thresholds:
            for j, threshold in enumerate(thresholds):
                if np.isfinite(threshold):
                    ax.axvline(
                        threshold,
                        color=SERIES[j],
                        linewidth=1.0,
                        linestyle=(0, (2, 3)),
                        alpha=0.7,
                    )
        ax.set_xlabel("mean probability a voter correctly ranks the two policies")
        ax.set_ylabel("share of elections electing the\nwelfare-optimal policy")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Do elections track welfare as perception improves?")
        ax.legend(loc="lower right")
        return fig


def bias_curve(
    sweeps: Sequence[pd.DataFrame],
    names: Sequence[str],
    *,
    toward_label: str,
) -> plt.Figure:
    """P(the favored policy wins) vs systematic bias in its favor."""
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        for j, (sweep, name) in enumerate(zip(sweeps, names, strict=True)):
            toward = sweep.attrs["toward"]
            frame = sweep.sort_values("bias_dollars")
            x = frame["bias_dollars"].to_numpy()
            y = frame[f"p_win_{toward}"].to_numpy()
            ax.plot(x, y, color=SERIES[j], label=name)
            _direct_label(ax, float(x[-1]), float(y[-1]), name, SERIES[j])
        ax.axhline(0.5, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
        ax.set_xlabel(
            "systematic misperception favoring "
            f"{toward_label}, dollars per voter per year"
        )
        ax.set_ylabel(f"share of elections won by {toward_label}")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("How many perceived dollars of bias flip the election?")
        ax.legend(loc="lower right")
        return fig


def rule_comparison(sweeps: Sequence[pd.DataFrame], names: Sequence[str]) -> plt.Figure:
    """Welfare tracking by voting rule, on one electorate."""
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        for j, (sweep, name) in enumerate(zip(sweeps, names, strict=True)):
            frame = sweep.sort_values("mean_ranking_accuracy")
            x = frame["mean_ranking_accuracy"].to_numpy()
            y = frame["p_tracked"].to_numpy()
            ax.plot(x, y, color=SERIES[j], label=name)
            _direct_label(ax, float(x[-1]), float(y[-1]), name, SERIES[j])
        ax.set_xlabel("mean probability a voter correctly ranks the two policies")
        ax.set_ylabel("share of elections electing the\nwelfare-optimal policy")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Voting rules compared on the measured electorate")
        ax.legend(loc="lower right")
        return fig


def save_figures(figures: dict[str, plt.Figure], directory: Path | str) -> list[Path]:
    """Write each figure as PNG into ``directory``; returns paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = directory / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


__all__: Iterable[str] = [
    "accuracy_curve",
    "bias_curve",
    "impact_distribution",
    "margin_distribution",
    "rule_comparison",
    "save_figures",
]
