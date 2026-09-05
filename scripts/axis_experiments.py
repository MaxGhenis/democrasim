"""Redistribution-axis experiments — docs/axis.md's numbers.

Four questions on the measured rate-to-transfer dial:

1. **The mapping.** What does the axis actually do to inequality, to the
   transfer, and to who gains? (descriptive, engine-computed)
2. **Demand formation.** Given an ideal inequality level and a belief about
   how far policy moves it, how much policy does a voter demand? Beliefs
   that understate a policy's reach should inflate demand.
3. **Where elections land.** Median ideal position, Nash equilibrium of the
   two-candidate position game, and the inequality-averse planner's
   optimum — on the same axis, in the same units.
4. **Self-interest versus values.** A purely self-interested electorate on
   measured incidence, against electorates that hold outcome targets.

Run: uv run python scripts/axis_experiments.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from democrasim.axis import (
    AxisCandidate,
    OutcomeType,
    RedistributionAxis,
    axis_equilibria,
    demanded_position,
    median_ideal_position,
    win_probability_table,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
GRID_POINTS = 41
N_VOTERS = 10_001
BELIEF_SLOPES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
#: Target inequality levels, as a share of the distance from the status-quo
#: Gini down to the lowest the axis reaches.
TARGET_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def provenance(axis: RedistributionAxis) -> dict:
    from democrasim.axis import AXIS_ARTIFACT_STEM
    from democrasim.data import _data_dir

    artifact = _data_dir() / f"{AXIS_ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/axis_experiments.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": AXIS_ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "grid_points": GRID_POINTS,
        "n_voters": N_VOTERS,
        "linearity_validation": "docs/results/axis_linearity_validation.json",
    }


def axis_map(axis: RedistributionAxis) -> pd.DataFrame:
    """E1: what the dial does, position by position."""
    household_weights = axis.household_weights
    households = household_weights.sum()
    rows = []
    for t in axis.positions:
        net = axis.net_deltas(float(t))
        transfer = -float(t) * float(
            (axis.gross_full * household_weights).sum() / axis.weights.sum()
        )
        rows.append(
            {
                "position": float(t),
                "gini": axis.gini(float(t)),
                "ede_change_dollars": axis.ede(float(t)),
                "transfer_per_adult": transfer,
                "share_households_gaining": float(
                    (household_weights[net > 1.0]).sum() / households
                ),
                "mean_gain_bottom_half": float(
                    np.average(
                        net[axis.base_income <= np.median(axis.base_income)],
                        weights=household_weights[
                            axis.base_income <= np.median(axis.base_income)
                        ],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def demand_formation(axis: RedistributionAxis, targets: dict) -> pd.DataFrame:
    """E2: demanded position by (ideal inequality, belief about the mapping)."""
    rows = []
    for name, target in targets.items():
        for slope in BELIEF_SLOPES:
            rows.append(
                {
                    "target_name": name,
                    "target_gini": target,
                    "belief_slope": slope,
                    "demanded_position": demanded_position(axis, target, slope),
                }
            )
            print(
                f"target {name} (G*={target:.4f}) belief x{slope:g}: "
                f"demands t={rows[-1]['demanded_position']:.3f}"
            )
    return pd.DataFrame(rows)


def where_elections_land(axis: RedistributionAxis, targets: dict) -> pd.DataFrame:
    """E3: median ideal, Nash equilibrium, and the planner's optimum."""
    ede = axis.ede_curve()
    planner = float(axis.positions[int(np.argmax(ede))])
    rows = []
    for name, target in targets.items():
        for slope in BELIEF_SLOPES:
            types = (OutcomeType(target_gini=target, belief_slope=slope),)
            table = win_probability_table(axis, types, n_voters=N_VOTERS)
            # Office-seekers: no policy preference, only the win.
            candidates = [
                AxisCandidate(
                    label,
                    preferences=OutcomeType(dollars_per_gini_point=0.0),
                    office_rent=10_000.0,
                )
                for label in ("Candidate 1", "Candidate 2")
            ]
            equilibria = axis_equilibria(
                axis, *candidates, types=types, n_voters=N_VOTERS, win_table=table
            )
            enacted = (
                float(np.mean([eq.enacted for eq in equilibria]))
                if equilibria
                else float("nan")
            )
            rows.append(
                {
                    "target_name": name,
                    "target_gini": target,
                    "belief_slope": slope,
                    "median_ideal": median_ideal_position(axis, types),
                    "n_equilibria": len(equilibria),
                    "enacted": enacted,
                    "enacted_gini": (
                        axis.gini(enacted) if np.isfinite(enacted) else float("nan")
                    ),
                    "planner_optimum": planner,
                }
            )
            print(
                f"land {name} belief x{slope:g}: median={rows[-1]['median_ideal']:.3f} "
                f"enacted={enacted:.3f} ({len(equilibria)} eq)"
            )
    return pd.DataFrame(rows)


def self_interest(axis: RedistributionAxis) -> dict:
    """E4: a purely self-interested electorate on measured incidence."""
    household_weights = axis.household_weights
    gaining = float(
        household_weights[axis.net_full > 0].sum() / household_weights.sum()
    )
    # Where the axis turns from gain to loss, by income decile. Household
    # composition means the crossing is a band, not a point: a large
    # household collects more transfers at the same income.
    order = np.argsort(axis.base_income)
    cumulative = np.cumsum(household_weights[order])
    decile = np.searchsorted(
        cumulative, np.linspace(0, cumulative[-1], 11)[1:-1], side="left"
    )
    bounds = np.concatenate([[0], decile, [len(order)]])
    by_decile = [
        {
            "decile": k + 1,
            "income_upper": float(axis.base_income[order][bounds[k + 1] - 1]),
            "mean_net_stake": float(
                np.average(
                    axis.net_full[order][bounds[k] : bounds[k + 1]],
                    weights=household_weights[order][bounds[k] : bounds[k + 1]],
                )
            ),
        }
        for k in range(10)
    ]
    rows = []
    for noise in (0.0, 1_000.0, 10_000.0):
        types = (
            OutcomeType(
                selfish_weight=1.0,
                household_index=0,
                own_noise_sd=noise,
                dollars_per_gini_point=0.0,
            ),
        )
        table = win_probability_table(axis, types, n_voters=N_VOTERS)
        candidates = [
            AxisCandidate(
                label,
                preferences=OutcomeType(dollars_per_gini_point=0.0),
                office_rent=10_000.0,
            )
            for label in ("Candidate 1", "Candidate 2")
        ]
        equilibria = axis_equilibria(
            axis, *candidates, types=types, n_voters=N_VOTERS, win_table=table
        )
        enacted = (
            float(np.mean([eq.enacted for eq in equilibria]))
            if equilibria
            else float("nan")
        )
        rows.append(
            {
                "own_noise_sd": noise,
                "enacted": enacted,
                "n_equilibria": len(equilibria),
            }
        )
        print(f"self-interest noise=${noise:,.0f}: enacted={enacted:.3f}")
    return {
        "share_households_gaining_at_full": gaining,
        "net_stake_by_income_decile": by_decile,
        "equilibria": rows,
    }


def belief_mixtures(axis: RedistributionAxis, target: float) -> pd.DataFrame:
    """E5: how an electorate aggregates disagreement about the mapping.

    Two blocs share one goal and differ only in what they believe the
    policy does. Sweeping the attenuated bloc's share asks whether
    competition averages the two beliefs or follows the median one.
    """
    rows = []
    for share in np.round(np.arange(0.0, 1.01, 0.1), 2):
        types = []
        if share > 0:
            types.append(
                OutcomeType(share=float(share), target_gini=target, belief_slope=0.25)
            )
        if share < 1:
            types.append(
                OutcomeType(
                    share=float(1 - share), target_gini=target, belief_slope=1.0
                )
            )
        types = tuple(types)
        table = win_probability_table(axis, types, n_voters=N_VOTERS)
        candidates = [
            AxisCandidate(
                label,
                preferences=OutcomeType(dollars_per_gini_point=0.0),
                office_rent=10_000.0,
            )
            for label in ("Candidate 1", "Candidate 2")
        ]
        equilibria = axis_equilibria(
            axis, *candidates, types=types, n_voters=N_VOTERS, win_table=table
        )
        enacted = (
            float(np.mean([eq.enacted for eq in equilibria]))
            if equilibria
            else float("nan")
        )
        rows.append(
            {
                "attenuated_share": float(share),
                "enacted": enacted,
                "mean_belief_slope": float(share * 0.25 + (1 - share) * 1.0),
                "n_equilibria": len(equilibria),
            }
        )
        print(f"belief mix q={share:.1f}: enacted={enacted:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    axis = RedistributionAxis.from_artifact(grid_points=GRID_POINTS)
    gini_curve = axis.gini_curve
    status_quo, floor = float(gini_curve[0]), float(gini_curve[-1])
    targets = {
        f"{fraction:.0%}_of_reach": float(status_quo - fraction * (status_quo - floor))
        for fraction in TARGET_FRACTIONS
    }
    print(f"Gini: status quo {status_quo:.4f} -> {floor:.4f} at t=1")

    mapping = axis_map(axis)
    mapping.to_csv(OUT / "axis_map.csv", index=False)

    demand = demand_formation(axis, targets)
    demand.to_csv(OUT / "axis_demand.csv", index=False)

    landing = where_elections_land(axis, targets)
    landing.to_csv(OUT / "axis_equilibria.csv", index=False)

    selfish = self_interest(axis)

    mixtures = belief_mixtures(axis, targets["50%_of_reach"])
    mixtures.to_csv(OUT / "axis_belief_mixtures.csv", index=False)

    summary = {
        "provenance": provenance(axis),
        "gini_status_quo": status_quo,
        "gini_at_full": floor,
        "ede_optimum_position": float(axis.positions[int(np.argmax(axis.ede_curve()))]),
        "targets": targets,
        "self_interest": selfish,
        "belief_mixture_target": targets["50%_of_reach"],
    }
    (OUT / "axis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    from democrasim import figures

    figures.save_figures(
        {
            "axis_map": figures.axis_map(mapping),
            "axis_demand": figures.axis_demand(demand),
        },
        OUT.parent / "figures",
    )
    print(f"wrote {OUT / 'axis_summary.json'}")


if __name__ == "__main__":
    main()
