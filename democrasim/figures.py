"""Matplotlib figures for the headline results.

Style follows a small set of chart rules: thin marks (2px lines), one axis,
recessive hairline grid, muted axis ink, categorical hues assigned in fixed
order by entity (measured world = blue, moment-matched toy = aqua,
homogeneous toy = yellow — never reassigned when a series drops out), a
legend naming every series, and text in ink colors rather than series
colors. Series identity leans on the legend plus the CSV table views in
docs/results/ (several ECDF curves share an endpoint, so end-of-line
labels would collide).
"""

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # keep notebook inline rendering alive
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from democrasim.electorate import Electorate, FloatArray

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
            "true change in household net income, dollars per year "
            "(gross, before financing)"
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
        fig, ax = _new_axes((7.0, 3.9))
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
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("|true margin between policies|, dollars per year (log scale)")
        ax.set_ylabel("share of adults at or below")
        ax.set_title("How much is actually at stake for each voter")
        ax.legend(loc="upper left")
        return fig


def _tracking_band(ax: plt.Axes, frame: pd.DataFrame, x: FloatArray, j: int) -> None:
    """95% Wilson band (falls back to the Wald band for old frames)."""
    if {"p_tracked_lo", "p_tracked_hi"} <= set(frame.columns):
        lo = frame["p_tracked_lo"].to_numpy()
        hi = frame["p_tracked_hi"].to_numpy()
    else:
        y = frame["p_tracked"].to_numpy()
        se = frame["p_tracked_se"].to_numpy()
        lo, hi = y - 1.96 * se, y + 1.96 * se
    ax.fill_between(x, lo, hi, color=SERIES[j], alpha=0.15, linewidth=0)


def accuracy_curve(
    sweeps: Sequence[pd.DataFrame],
    names: Sequence[str],
    *,
    target: float | None = 0.9,
    thresholds: Sequence[float] | None = None,
    axis: str = "accuracy",
) -> plt.Figure:
    """P(elects the welfare-optimal policy) per world, with Wilson bands.

    ``axis="accuracy"`` plots against mean ranking accuracy (the derived,
    config-specific coordinate); ``axis="noise"`` plots against the
    perception-noise primitive σ on a log scale. The same sweeps feed both.
    """
    if axis not in ("accuracy", "noise"):
        raise ValueError("axis must be 'accuracy' or 'noise'")
    x_column = "mean_ranking_accuracy" if axis == "accuracy" else "noise_sd"
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.2))
        for j, (sweep, name) in enumerate(zip(sweeps, names, strict=True)):
            frame = sweep.sort_values(x_column)
            if axis == "noise":
                # log axis: place σ=0 at a nominal position below the grid
                frame = frame[frame["noise_sd"] > 0]
            x = frame[x_column].to_numpy()
            y = frame["p_tracked"].to_numpy()
            _tracking_band(ax, frame, x, j)
            ax.plot(x, y, color=SERIES[j], label=name)
        if axis == "noise":
            ax.set_xscale("log")
            ax.set_xlabel(
                "perception noise σ, dollars per year (log scale; σ=0 omitted)"
            )
        else:
            ax.set_xlabel("mean probability a voter correctly ranks the two policies")
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
        if thresholds and axis == "accuracy":
            for j, threshold in enumerate(thresholds):
                if np.isfinite(threshold):
                    ax.axvline(
                        threshold,
                        color=SERIES[j],
                        linewidth=1.0,
                        linestyle=(0, (2, 3)),
                        alpha=0.7,
                    )
        ax.set_ylabel("share of elections electing the\nwelfare-optimal policy")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Do elections track welfare as perception improves?")
        ax.legend(loc="lower right")
        return fig


def n_sensitivity(
    curves: Sequence[pd.DataFrame],
    names: Sequence[str],
) -> plt.Figure:
    """Analytic tracking vs noise for several electorate sizes.

    Expects frames from :func:`democrasim.analytic_plurality_curve`; the
    point is that Monte Carlo thresholds are properties of the electorate
    size, not of the electorate alone.
    """
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        for j, (curve, name) in enumerate(zip(curves, names, strict=True)):
            frame = curve[curve["noise_sd"] > 0].sort_values("noise_sd")
            ax.plot(
                frame["noise_sd"].to_numpy(),
                frame["p_tracked"].to_numpy(),
                color=SERIES[j % len(SERIES)],
                label=name,
            )
        ax.set_xscale("log")
        ax.axhline(0.9, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
        ax.set_xlabel("perception noise σ, dollars per year (log scale)")
        ax.set_ylabel("P(welfare-optimal policy wins), analytic")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Electorate size moves the tracking window")
        ax.legend(loc="lower left", title="sampled voters")
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
        ax.axhline(0.5, color=BASELINE, linewidth=1.0, linestyle=(0, (4, 4)))
        ax.set_xlabel(
            "systematic misperception favoring "
            f"{toward_label}, dollars per voter per year"
        )
        ax.set_ylabel(f"share of elections won by {toward_label}")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("How many perceived dollars of bias flip the election?")
        if len(sweeps) > 1:  # a single series is named by the title
            ax.legend(loc="lower right")
        return fig


def rule_comparison(sweeps: Sequence[pd.DataFrame], names: Sequence[str]) -> plt.Figure:
    """Welfare tracking by voting rule, on one electorate.

    Rules can coincide exactly (with two policies, instant runoff IS
    plurality), so coincident lines get distinct treatments: the first
    series draws as a wide translucent band underneath, later series as
    normal or dashed lines on top — all still color-by-entity.
    """
    styles = (
        {"linewidth": 4.5, "alpha": 0.35, "solid_capstyle": "round"},
        {"linewidth": 2.0},
        {"linewidth": 2.0, "linestyle": (0, (4, 3))},
    )
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        for j, (sweep, name) in enumerate(zip(sweeps, names, strict=True)):
            frame = sweep.sort_values("mean_ranking_accuracy")
            x = frame["mean_ranking_accuracy"].to_numpy()
            y = frame["p_tracked"].to_numpy()
            ax.plot(x, y, color=SERIES[j], label=name, **styles[j % len(styles)])
        ax.set_xlabel("mean probability a voter correctly ranks the two policies")
        ax.set_ylabel("share of elections electing the\nwelfare-optimal policy")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Voting rules compared on the measured electorate")
        ax.legend(loc="lower right")
        return fig


def strategic_positions(frame: pd.DataFrame) -> plt.Figure:
    """Equilibrium platform intensity vs perception noise, by selfish weight.

    Expects the strategic_equilibria frame: for each (selfish_weight, sigma),
    the mean enacted Policy-A intensity across pure Nash equilibria and
    candidate 1's iterated-best-response proposal.
    """
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        weights = sorted(frame["selfish_weight"].unique())
        # Coincident series (identical curves for different weights) get the
        # wide-translucent-underlay treatment so both stay visible.
        solid = (
            {"linewidth": 2.0},
            {"linewidth": 4.5, "alpha": 0.35},
            {"linewidth": 2.0},
        )
        for j, weight in enumerate(weights):
            sub = frame[frame["selfish_weight"] == weight].sort_values("sigma")
            positive = sub[sub["sigma"] > 0]
            ax.plot(
                positive["sigma"],
                positive["mean_enacted_alpha"],
                color=SERIES[j],
                label=f"selfish weight {weight:g} (enacted)",
                **solid[j % len(solid)],
            )
            ax.plot(
                positive["sigma"],
                positive["ibr_position_1_alpha"],
                color=SERIES[j],
                linewidth=1.2,
                linestyle=(0, (3, 3)),
                label=f"selfish weight {weight:g} (proposal)",
            )
        ax.set_xscale("log")
        ax.set_xlabel("perception noise σ, dollars per year (log scale; σ=0 omitted)")
        ax.set_ylabel("Policy-A intensity α")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Noise relaxes electoral discipline on self-serving platforms")
        ax.legend(loc="upper left", fontsize=8)
        return fig


def mixed_motive_tracking(frame: pd.DataFrame) -> plt.Figure:
    """Analytic tracking vs own-stake noise, by the voters' selfish weight.

    Expects a long frame with ``selfish_weight``, ``noise_sd`` and
    ``p_tracked`` columns (informed societal component, σ_soc = 0). At most
    six weights are drawn (one per palette hue) from the fixed display set;
    probe weights such as 0.98 stay in the CSV. The fully sociotropic
    series is constant at 1, so it draws as the wide translucent underlay.
    """
    display = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        weights = [w for w in display if w in set(frame["selfish_weight"])]
        for j, weight in enumerate(weights):
            sub = frame[
                (frame["selfish_weight"] == weight) & (frame["noise_sd"] > 0)
            ].sort_values("noise_sd")
            style = {"linewidth": 4.5, "alpha": 0.35} if weight == 0.0 else {}
            ax.plot(
                sub["noise_sd"],
                sub["p_tracked"],
                color=SERIES[j % len(SERIES)],
                label=f"{weight:g}",
                **style,
            )
        ax.set_xscale("log")
        ax.set_xlabel(
            "own-stake perception noise σ, dollars per year (log scale; σ=0 omitted)"
        )
        ax.set_ylabel("P(welfare-optimal policy wins), analytic")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("Societal motives substitute for accurate self-perception")
        ax.legend(loc="lower center", ncols=3, title="voters' selfish weight")
        return fig


def strategic_discipline(frame: pd.DataFrame) -> plt.Figure:
    """Equilibrium platform intensity vs the sociotropic voter share.

    Expects the heterogeneous-electorate equilibrium frame: purely selfish
    candidates against an electorate mixing noisy self-interested voters
    with informed sociotropic ones, one series per own-stake noise level.
    """
    with plt.rc_context(_RC):
        fig, ax = _new_axes((7.0, 4.0))
        sigmas = sorted(frame["sigma"].unique())
        for j, sigma in enumerate(sigmas):
            sub = frame[frame["sigma"] == sigma].sort_values("sociotropic_share")
            ax.plot(
                sub["sociotropic_share"],
                sub["mean_enacted_alpha"],
                color=SERIES[j % len(SERIES)],
                marker="o",
                markersize=4,
                label=f"σ = ${sigma:,.0f}",
            )
        ax.set_xlabel("share of voters who are informed and sociotropic")
        ax.set_ylabel("enacted Policy-A intensity α")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title("A sociotropic minority restores platform discipline")
        ax.legend(loc="upper right", title="own-stake noise")
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
    "n_sensitivity",
    "rule_comparison",
    "save_figures",
    "strategic_positions",
]
