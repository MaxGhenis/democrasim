"""Robustness runs behind docs/findings.md — every stress row, one command.

Reruns the headline accuracy sweep under alternative financing and welfare
assumptions, the exact-parity and sign-flip counterfactuals, the
rational-abstention variants, the analytic electorate-size sensitivity, and
the equivalence-scale spot checks. Writes tidy CSVs plus a summary JSON
(with provenance) to docs/results/. Run: uv run python scripts/robustness.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from democrasim import (
    ElectionSpec,
    Electorate,
    Isoelastic,
    LinearGaussianPerception,
    Plurality,
    Utilitarian,
    accuracy_sweep,
    analytic_plurality_curve,
    apply_financing,
    find_threshold,
    load_measured_electorate,
    run_elections,
    summarize_elections,
)
from democrasim.cli import DEFAULT_NOISE_GRID
from democrasim.data import ARTIFACT_STEM, _data_dir

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
N_ELECTIONS = 240
SEED = 42
N_GRID = [101, 1_001, 10_001, 100_001, 1_000_001, None]


def provenance() -> dict:
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/robustness.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "n_elections": N_ELECTIONS,
        "seed": SEED,
        "noise_grid": DEFAULT_NOISE_GRID,
    }


def sweep(electorate: Electorate, welfare, tag: str) -> dict:
    spec = ElectionSpec(perception=LinearGaussianPerception(), welfare=welfare)
    frame = accuracy_sweep(
        electorate,
        noise_sds=DEFAULT_NOISE_GRID,
        spec=spec,
        n_elections=N_ELECTIONS,
        seed=SEED,
    )
    frame.to_csv(OUT / f"robustness_{tag}.csv", index=False)
    at_zero = frame.loc[frame["noise_sd"] == 0.0].iloc[0]
    optimal = int(np.argmax(welfare.per_policy(electorate)))
    threshold = find_threshold(frame["mean_ranking_accuracy"], frame["p_tracked"], 0.9)
    print(
        f"{tag:>28}: optimal={electorate.policy_labels[optimal]} "
        f"p_tracked@perfect={at_zero['p_tracked']:.2f} "
        f"turnout@perfect={at_zero['mean_turnout']:.4f} "
        f"threshold@90%={threshold:.3f}"
    )
    return {
        "welfare_optimal": electorate.policy_labels[optimal],
        "p_tracked_at_perfect": float(at_zero["p_tracked"]),
        "turnout_at_perfect": float(at_zero["mean_turnout"]),
        "max_p_tracked": float(frame["p_tracked"].max()),
        "threshold_accuracy_90": (float(threshold) if np.isfinite(threshold) else None),
    }


def abstention_rows(financed: Electorate) -> dict:
    """Perfect-accuracy elections under plurality indifference bands."""
    out = {}
    for band in (10.0, 25.0, 100.0):
        spec = ElectionSpec(
            perception=LinearGaussianPerception(),
            rule=Plurality(abstain_below=band),
            welfare=Isoelastic(),
        )
        results = run_elections(
            financed, spec, N_ELECTIONS, np.random.default_rng(SEED)
        )
        summary = summarize_elections(results)
        out[f"abstain_below_{band:.0f}"] = {
            "p_tracked": summary["p_tracked"],
            "mean_turnout": round(summary["mean_turnout"], 4),
        }
        print(
            f"    abstain_below=${band:>5,.0f} @ perfect: "
            f"p_tracked={summary['p_tracked']:.2f} "
            f"turnout={summary['mean_turnout']:.2f}"
        )
    return out


def n_sensitivity(financed: Electorate) -> dict:
    """Analytic tracking curves and 90% bands by electorate size."""
    noise = [0.0, *np.geomspace(10.0, 100_000.0, 25).tolist()]
    frames = []
    bands = {}
    for n in N_GRID:
        curve = analytic_plurality_curve(
            financed, noise_sds=noise, n_voters=n, welfare=Isoelastic()
        )
        curve.insert(0, "n_voters", -1 if n is None else n)
        frames.append(curve)
        tracked = curve[curve["p_tracked"] >= 0.9]["mean_ranking_accuracy"]
        key = "inf" if n is None else str(n)
        bands[key] = (
            None
            if tracked.empty
            else [round(float(tracked.min()), 4), round(float(tracked.max()), 4)]
        )
        print(f"    n={key:>8}: 90% tracking accuracy band = {bands[key]}")
    pd.concat(frames, ignore_index=True).to_csv(
        OUT / "robustness_n_sensitivity.csv", index=False
    )
    return bands


def equivalence_scale_checks(gross: Electorate) -> dict:
    """Isoelastic (η=1) ranking under equivalized household income.

    Spot checks only: incomes and deltas are divided by an equivalence
    scale (household size, or its square root) before the welfare metric;
    the $1,000 floor then binds on the equivalized scale, which is part of
    what is being stress-tested. Voting is unaffected.
    """
    financed = apply_financing(gross, "per_capita")
    size = financed.demographics["household_size"].to_numpy().astype(float)
    out = {}
    for name, scale in (("per_capita", size), ("sqrt", np.sqrt(size))):
        equivalized = Electorate(
            deltas=financed.deltas / scale[:, None],
            weights=financed.weights,
            base_income=financed.base_income / scale,
            hh_adults=financed.hh_adults,
            policy_labels=financed.policy_labels,
            source=financed.source + f" | equivalized={name}",
        )
        values = Isoelastic().per_policy(equivalized)
        out[name] = financed.policy_labels[int(np.argmax(values))]
        print(f"    equivalence scale {name}: isoelastic η=1 ranks {out[name]}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gross = load_measured_electorate()
    financed = apply_financing(gross, "per_capita")
    summary: dict = {"provenance": provenance()}

    # Financing / welfare variants (headline = per_capita + isoelastic,
    # produced by `democrasim sweep`). "none_isoelastic" doubles as the
    # deficit-blind-perception case: voters see gross impacts.
    summary["proportional_isoelastic"] = sweep(
        apply_financing(gross, "proportional"),
        Isoelastic(),
        "proportional_isoelastic",
    )
    summary["none_utilitarian"] = sweep(gross, Utilitarian(), "none_utilitarian")
    summary["none_isoelastic"] = sweep(gross, Isoelastic(), "none_isoelastic")

    # Exact-parity counterfactual: rescale policy B's gross deltas so both
    # policies cost identically, then finance per capita. SYNTHETIC — shows
    # the knife-edge at perfect accuracy vanishing when the levy gap is 0.
    totals = gross.household_dollars()
    parity_deltas = gross.deltas.copy()
    parity_deltas[:, 1] *= totals[0] / totals[1]
    summary["exact_parity_per_capita"] = sweep(
        apply_financing(
            gross.with_deltas(parity_deltas, note="COUNTERFACTUAL exact cost parity"),
            "per_capita",
        ),
        Isoelastic(),
        "exact_parity_per_capita",
    )

    # Sign-flip counterfactual: rescale policy B to cost 3% MORE than
    # policy A, so the no-stakes bloc's levy-gap margin favors A instead.
    # SYNTHETIC — shows the perfect-accuracy outcome is set by the SIGN of
    # the residual cost gap, a welfare-irrelevant quantity.
    flip_deltas = gross.deltas.copy()
    flip_deltas[:, 1] *= (totals[0] / totals[1]) * 1.03
    summary["sign_flip_per_capita"] = sweep(
        apply_financing(
            gross.with_deltas(
                flip_deltas, note="COUNTERFACTUAL B costs 3% more than A"
            ),
            "per_capita",
        ),
        Isoelastic(),
        "sign_flip_per_capita",
    )

    print("  abstention bands (headline financing, perfect accuracy):")
    summary["abstention_at_perfect"] = abstention_rows(financed)

    print("  analytic electorate-size sensitivity:")
    summary["n_sensitivity_accuracy_bands"] = n_sensitivity(financed)

    print("  equivalence-scale spot checks:")
    summary["equivalence_scale_optimal"] = equivalence_scale_checks(gross)

    (OUT / "robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT / 'robustness_summary.json'}")


if __name__ == "__main__":
    main()
