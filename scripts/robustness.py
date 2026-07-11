"""Robustness runs behind docs/findings.md.

Reruns the headline accuracy sweep under alternative financing and welfare
assumptions, plus an exact-parity counterfactual, and writes tidy CSVs to
docs/results/. Run: uv run python scripts/robustness.py
"""

import json
from pathlib import Path

import numpy as np

from democrasim import (
    ElectionSpec,
    Isoelastic,
    LinearGaussianPerception,
    Utilitarian,
    accuracy_sweep,
    apply_financing,
    load_measured_electorate,
)
from democrasim.cli import DEFAULT_NOISE_GRID

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
N_ELECTIONS = 240
SEED = 42


def sweep(electorate, welfare, tag: str) -> dict:
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
    best = frame["p_tracked"].max()
    optimal = int(np.argmax(welfare.per_policy(electorate)))
    print(
        f"{tag:>28}: optimal={electorate.policy_labels[optimal]} "
        f"p_tracked@perfect={at_zero['p_tracked']:.2f} "
        f"turnout@perfect={at_zero['mean_turnout']:.2f} "
        f"max_p_tracked={best:.2f}"
    )
    return {
        "welfare_optimal": electorate.policy_labels[optimal],
        "p_tracked_at_perfect": float(at_zero["p_tracked"]),
        "turnout_at_perfect": float(at_zero["mean_turnout"]),
        "max_p_tracked": float(best),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gross = load_measured_electorate()
    summary = {}

    # Headline variant is financing=per_capita + isoelastic (in sweep dirs);
    # these are the assumption-stress runs.
    summary["proportional_isoelastic"] = sweep(
        apply_financing(gross, "proportional"),
        Isoelastic(),
        "proportional_isoelastic",
    )
    summary["none_utilitarian"] = sweep(gross, Utilitarian(), "none_utilitarian")
    summary["none_isoelastic"] = sweep(gross, Isoelastic(), "none_isoelastic")

    # Exact-parity counterfactual: rescale policy B's gross deltas so both
    # policies cost identically, then finance per capita. SYNTHETIC — used
    # only to show the knife-edge at perfect accuracy.
    totals = gross.household_dollars()
    scaled = gross.deltas.copy()
    scaled[:, 1] *= totals[0] / totals[1]
    parity = apply_financing(
        gross.with_deltas(scaled, note="COUNTERFACTUAL exact cost parity"),
        "per_capita",
    )
    summary["exact_parity_per_capita"] = sweep(
        parity, Isoelastic(), "exact_parity_per_capita"
    )

    (OUT / "robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
