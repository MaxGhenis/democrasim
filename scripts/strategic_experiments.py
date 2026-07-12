"""Equilibrium experiments for the strategic layer — docs/strategic.md's rows.

Sweeps perception noise and the candidates' selfish weight, computing exact
pure-Nash equilibria of the position game on the measured policy space, plus
an office-seeker variant. Writes tidy CSVs, a summary JSON with provenance,
and the equilibrium-map figure. Run: uv run python scripts/strategic_experiments.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

import democrasim as d
from democrasim.data import ARTIFACT_STEM, _data_dir
from democrasim.strategic import (
    Candidate,
    PolicySpace,
    iterated_best_response,
    pure_nash_equilibria,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
GRID_POINTS = 21
N_VOTERS = 10_001
SIGMAS = [0.0, 10.0, 50.0, 200.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0]
SELFISH_WEIGHTS = [0.0, 0.5, 1.0]


def provenance() -> dict:
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/strategic_experiments.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "grid_points": GRID_POINTS,
        "n_voters": N_VOTERS,
        "interpolation_validation": "docs/results/interpolation_validation.json",
    }


def candidate_households(gross: d.Electorate) -> dict:
    """Deterministic candidate households whose measured stakes oppose.

    Candidate 1: a two-adult household with 2+ children at the median
    income of such households — a gross Policy A winner. Candidate 2: the
    household at the weighted median income of Policy B's gross winners —
    selection by measured stake, not by an income-percentile proxy (the
    childless 95th percentile turns out to sit below the capped brackets
    and would *lose* from Policy B). Both selections are verified against
    the committed deltas before use.
    """
    kids = gross.demographics["n_children"].to_numpy()
    income = gross.base_income
    child_rows = np.flatnonzero((kids >= 2) & (gross.hh_adults == 2))
    child_median = d.weighted_quantile(
        income[child_rows], gross.weights[child_rows], [0.5]
    )[0]
    row_1 = int(child_rows[np.argmin(np.abs(income[child_rows] - child_median))])
    winners_b = np.flatnonzero(gross.deltas[:, 1] > 1.0)
    winner_median = d.weighted_quantile(
        income[winners_b], gross.weights[winners_b], [0.5]
    )[0]
    row_2 = int(winners_b[np.argmin(np.abs(income[winners_b] - winner_median))])
    assert gross.deltas[row_1, 0] > 0, "candidate 1 must gain from Policy A"
    assert gross.deltas[row_2, 1] > 0, "candidate 2 must gain from Policy B"

    def record(row: int) -> dict:
        return {
            "row": row,
            "income": round(float(income[row])),
            "children": int(kids[row]),
            "hh_adults": int(gross.hh_adults[row]),
            "gross_a": round(float(gross.deltas[row, 0])),
            "gross_b": round(float(gross.deltas[row, 1])),
        }

    # The rejected proxy, kept as evidence: the childless household at the
    # 95th income percentile sits below the capped brackets and would LOSE
    # from Policy B once financed.
    childless = np.flatnonzero(kids == 0)
    p95 = d.weighted_quantile(income[childless], gross.weights[childless], [0.95])[0]
    proxy = int(childless[np.argmin(np.abs(income[childless] - p95))])
    return {
        "candidate_1": record(row_1),
        "candidate_2": record(row_2),
        "rejected_childless_p95_proxy": record(proxy),
    }


def summarize(equilibria, space: PolicySpace, welfare_grid) -> dict:
    """Reduce an equilibrium set to the quantities the write-up quotes."""
    if not equilibria:
        return {"n_equilibria": 0}
    enacted = []
    for eq in equilibria:
        p = eq.p_win_1
        enacted.append(
            (
                p * eq.position_1[0] + (1 - p) * eq.position_2[0],
                p * eq.position_1[1] + (1 - p) * eq.position_2[1],
            )
        )
    enacted = np.array(enacted)
    welfare_of = {
        position: welfare_grid[index] for index, position in enumerate(space.positions)
    }
    eq_welfare = [
        eq.p_win_1 * welfare_of[eq.position_1]
        + (1 - eq.p_win_1) * welfare_of[eq.position_2]
        for eq in equilibria
    ]
    distance = [
        abs(eq.position_1[0] - eq.position_2[0])
        + abs(eq.position_1[1] - eq.position_2[1])
        for eq in equilibria
    ]
    return {
        "n_equilibria": len(equilibria),
        "mean_enacted_alpha": round(float(enacted[:, 0].mean()), 4),
        "mean_enacted_beta": round(float(enacted[:, 1].mean()), 4),
        "mean_expected_welfare": float(f"{np.mean(eq_welfare):.6g}"),
        "max_expected_welfare": float(f"{np.max(eq_welfare):.6g}"),
        "min_expected_welfare": float(f"{np.min(eq_welfare):.6g}"),
        "mean_position_distance": round(float(np.mean(distance)), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gross = d.load_measured_electorate()
    space = PolicySpace.from_gross_electorate(gross, grid_points=GRID_POINTS)
    households = candidate_households(gross)
    welfare_grid = space.societal_welfare_grid(d.Isoelastic())
    optimal = space.welfare_optimal_position(d.Isoelastic())

    rows = []
    for selfish_weight in SELFISH_WEIGHTS:
        candidate_1 = Candidate(
            "Candidate 1",
            households["candidate_1"]["row"],
            selfish_weight=selfish_weight,
        )
        candidate_2 = Candidate(
            "Candidate 2",
            households["candidate_2"]["row"],
            selfish_weight=selfish_weight,
        )
        for sigma in SIGMAS:
            equilibria = pure_nash_equilibria(
                space, candidate_1, candidate_2, sigma=sigma, n_voters=N_VOTERS
            )
            dynamics = iterated_best_response(
                space, candidate_1, candidate_2, sigma=sigma, n_voters=N_VOTERS
            )
            end = dynamics["path"][-1]
            rows.append(
                {
                    "selfish_weight": selfish_weight,
                    "sigma": sigma,
                    **summarize(equilibria, space, welfare_grid),
                    "ibr_converged": dynamics["converged"],
                    "ibr_cycle": dynamics["cycle"],
                    "ibr_position_1_alpha": end[0][0],
                    "ibr_position_1_beta": end[0][1],
                    "ibr_position_2_alpha": end[1][0],
                    "ibr_position_2_beta": end[1][1],
                }
            )
            print(
                f"lambda={selfish_weight} sigma={sigma:>7,.0f}: "
                f"{rows[-1]['n_equilibria']} eq, "
                f"enacted=({rows[-1].get('mean_enacted_alpha')}, "
                f"{rows[-1].get('mean_enacted_beta')}), "
                f"IBR end={end}"
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "strategic_equilibria.csv", index=False)

    # Office-seeker variant: a dollar rent large enough to dominate the
    # (near-zero, budget-neutral) utilitarian societal component.
    office_rows = []
    for sigma in SIGMAS:
        candidate_1 = Candidate(
            "Candidate 1",
            households["candidate_1"]["row"],
            selfish_weight=0.0,
            societal=d.Utilitarian(),
            office_rent=100_000.0,
        )
        candidate_2 = Candidate(
            "Candidate 2",
            households["candidate_2"]["row"],
            selfish_weight=0.0,
            societal=d.Utilitarian(),
            office_rent=100_000.0,
        )
        equilibria = pure_nash_equilibria(
            space, candidate_1, candidate_2, sigma=sigma, n_voters=N_VOTERS
        )
        convergent = [e for e in equilibria if e.position_1 == e.position_2]
        office_rows.append(
            {
                "sigma": sigma,
                "n_equilibria": len(equilibria),
                "n_convergent": len(convergent),
                "convergent_positions": sorted({e.position_1 for e in convergent})[:6],
            }
        )
        print(f"office-seekers sigma={sigma:>7,.0f}: {office_rows[-1]}")
    (OUT / "strategic_office_seekers.json").write_text(
        json.dumps(office_rows, indent=2, default=str) + "\n"
    )

    summary = {
        "provenance": provenance(),
        "candidate_households": households,
        "welfare_optimal_position_eta1": list(optimal),
        "sigmas": SIGMAS,
        "selfish_weights": SELFISH_WEIGHTS,
    }
    (OUT / "strategic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    from democrasim import figures

    figures.save_figures(
        {"strategic_positions": figures.strategic_positions(frame)},
        OUT.parent / "figures",
    )
    print(f"wrote {OUT / 'strategic_equilibria.csv'}")


if __name__ == "__main__":
    main()
