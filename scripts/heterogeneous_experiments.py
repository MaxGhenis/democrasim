"""Heterogeneous-preferences experiments — docs/heterogeneity.md's numbers.

Three questions, all on the measured electorate:

1. **Dose-response**: how much societal motive substitutes for how much
   self-perception accuracy, and how much cures the perfect-information
   knife-edge? (analytic, exact)
2. **Value pluralism**: when informed sociotropic voters disagree about
   inequality aversion, what wins — and under whose grading is that
   "correct"? (analytic, exact)
3. **Strategic discipline**: does a sociotropic minority restore the
   platform discipline that own-stake noise destroyed? (exact Nash)

Run: uv run python scripts/heterogeneous_experiments.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

import democrasim as d
from democrasim.data import ARTIFACT_STEM, _data_dir
from democrasim.preferences import VoterType
from democrasim.strategic import (
    Candidate,
    PolicySpace,
    iterated_best_response,
    pure_nash_equilibria,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
SIGMAS = [0.0, 10.0, 50.0, 200.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0]
SELFISH_WEIGHTS = [1.0, 0.98, 0.9, 0.75, 0.5, 0.25, 0.0]
SOCIOTROPIC_SHARES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DISCIPLINE_SIGMAS = [1_000.0, 10_000.0, 30_000.0]
GRID_POINTS = 21
N_VOTERS = 10_001


def provenance() -> dict:
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/heterogeneous_experiments.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "grid_points": GRID_POINTS,
        "n_voters": N_VOTERS,
    }


def tracking_dose_response(financed: d.Electorate) -> pd.DataFrame:
    """E1: analytic tracking curves per homogeneous selfish weight."""
    frames = []
    for weight in SELFISH_WEIGHTS:
        curve = d.analytic_mixed_plurality_curve(
            financed,
            SIGMAS,
            types=(VoterType(share=1.0, selfish_weight=weight, eta=1.0),),
            n_voters=N_VOTERS,
        )
        curve.insert(0, "selfish_weight", weight)
        frames.append(curve)
        print(
            f"dose-response s={weight:g}: tracked at sigma=0 -> "
            f"{curve.loc[0, 'p_tracked']:.3f}, at 30k -> "
            f"{curve.p_tracked.iloc[-1]:.3f}"
        )
    return pd.concat(frames, ignore_index=True)


def knife_edge_threshold(financed: d.Electorate) -> float:
    """E1b: the selfish weight below which σ=0 tracking flips to 1."""

    def tracked(weight: float) -> float:
        curve = d.analytic_mixed_plurality_curve(
            financed,
            [0.0],
            types=(VoterType(share=1.0, selfish_weight=weight, eta=1.0),),
            n_voters=N_VOTERS,
        )
        return float(curve.loc[0, "p_tracked"])

    low, high = 0.5, 1.0  # tracked at low, untracked at high
    assert tracked(low) > 0.5 and tracked(high) < 0.5
    for _ in range(40):
        mid = (low + high) / 2
        if tracked(mid) > 0.5:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def value_pluralism(financed: d.Electorate) -> dict:
    """E2: informed sociotropic voters split by inequality aversion."""
    base_eta = 1.0
    base_optimal = int(np.argmax(d.Isoelastic(eta=base_eta).per_policy(financed)))
    disagreeing = None
    for eta in (2.0, 2.5, 3.0, 4.0, 5.0):
        if int(np.argmax(d.Isoelastic(eta=eta).per_policy(financed))) != base_optimal:
            disagreeing = eta
            break
    if disagreeing is None:
        return {"disagreeing_eta": None}
    rows = []
    for share_high in np.round(np.arange(0.0, 1.01, 0.1), 2):
        types = []
        if share_high < 1.0:
            types.append(
                VoterType(share=1.0 - share_high, selfish_weight=0.0, eta=base_eta)
            )
        if share_high > 0.0:
            types.append(
                VoterType(share=share_high, selfish_weight=0.0, eta=disagreeing)
            )
        row = {"share_eta_high": float(share_high)}
        for grading_eta in (base_eta, disagreeing):
            curve = d.analytic_mixed_plurality_curve(
                financed,
                [0.0],
                types=tuple(types),
                n_voters=N_VOTERS,
                welfare=d.Isoelastic(eta=grading_eta),
            )
            row[f"p_tracked_grading_eta_{grading_eta:g}"] = float(
                curve.loc[0, "p_tracked"]
            )
        rows.append(row)
    ede = {
        f"eta_{eta:g}": [
            round(v, 4) for v in d.Isoelastic(eta=eta).dollar_equivalent(financed)
        ]
        for eta in (base_eta, disagreeing)
    }
    print(f"value pluralism: eta={disagreeing:g} flips the ranking vs eta=1")
    return {
        "base_eta": base_eta,
        "disagreeing_eta": disagreeing,
        "ede_dollars_per_household": ede,
        "rows": rows,
    }


def strategic_discipline(gross: d.Electorate, households: dict) -> pd.DataFrame:
    """E3: purely selfish candidates vs a sociotropic-minority electorate."""
    space = PolicySpace.from_gross_electorate(gross, grid_points=GRID_POINTS)
    candidate_1 = Candidate(
        "Candidate 1", households["candidate_1"]["row"], selfish_weight=1.0
    )
    candidate_2 = Candidate(
        "Candidate 2", households["candidate_2"]["row"], selfish_weight=1.0
    )
    rows = []
    for sigma in DISCIPLINE_SIGMAS:
        for share in SOCIOTROPIC_SHARES:
            types = []
            if share < 1.0:
                types.append(VoterType(share=1.0 - share, selfish_weight=1.0))
            if share > 0.0:
                types.append(VoterType(share=share, selfish_weight=0.0, eta=1.0))
            equilibria = pure_nash_equilibria(
                space,
                candidate_1,
                candidate_2,
                sigma=sigma,
                n_voters=N_VOTERS,
                voter_types=tuple(types),
            )
            dynamics = iterated_best_response(
                space,
                candidate_1,
                candidate_2,
                sigma=sigma,
                n_voters=N_VOTERS,
                voter_types=tuple(types),
            )
            row = {
                "sigma": sigma,
                "sociotropic_share": share,
                "n_equilibria": len(equilibria),
                "ibr_converged": dynamics["converged"],
                "ibr_cycle": dynamics["cycle"],
            }
            if equilibria:
                enacted = np.array(
                    [
                        (
                            eq.p_win_1 * eq.position_1[0]
                            + (1 - eq.p_win_1) * eq.position_2[0],
                            eq.p_win_1 * eq.position_1[1]
                            + (1 - eq.p_win_1) * eq.position_2[1],
                        )
                        for eq in equilibria
                    ]
                )
                row.update(
                    mean_enacted_alpha=float(enacted[:, 0].mean()),
                    mean_enacted_beta=float(enacted[:, 1].mean()),
                    max_proposal_alpha=max(eq.position_1[0] for eq in equilibria),
                )
            else:
                # No pure equilibrium: best responses cycle. Summarize play
                # by the cycle's enacted positions rather than inventing one.
                cycle_alphas = [
                    p1[0] for p1, _ in dynamics["path"][-8:]
                ]  # tail of the path
                row.update(
                    mean_enacted_alpha=float("nan"),
                    mean_enacted_beta=float("nan"),
                    max_proposal_alpha=max(cycle_alphas),
                )
            rows.append(row)
            print(
                f"discipline sigma={sigma:>7,.0f} q={share:.2f}: "
                f"{len(equilibria)} eq (cycle={row['ibr_cycle']}), "
                f"enacted alpha={row['mean_enacted_alpha']:.4f}"
            )
    return pd.DataFrame(rows)


def robustness_variants(gross: d.Electorate, households: dict) -> list[dict]:
    """E4: noisy sociotropic minority, and an interior-type electorate."""
    candidate_kwargs = dict(selfish_weight=1.0)
    candidate_1 = Candidate(
        "Candidate 1", households["candidate_1"]["row"], **candidate_kwargs
    )
    candidate_2 = Candidate(
        "Candidate 2", households["candidate_2"]["row"], **candidate_kwargs
    )
    rows = []

    def run(space, sigma, types, label, grid_points):
        equilibria = pure_nash_equilibria(
            space,
            candidate_1,
            candidate_2,
            sigma=sigma,
            n_voters=N_VOTERS,
            voter_types=types,
        )
        enacted_alpha = (
            float(
                np.mean(
                    [
                        eq.p_win_1 * eq.position_1[0]
                        + (1 - eq.p_win_1) * eq.position_2[0]
                        for eq in equilibria
                    ]
                )
            )
            if equilibria
            else float("nan")
        )
        rows.append(
            {
                "variant": label,
                "sigma": sigma,
                "grid_points": grid_points,
                "n_equilibria": len(equilibria),
                "mean_enacted_alpha": round(enacted_alpha, 4) if equilibria else None,
            }
        )
        print(f"variant {label} sigma={sigma:,.0f}: enacted alpha={enacted_alpha:.4f}")

    space_21 = PolicySpace.from_gross_electorate(gross, grid_points=GRID_POINTS)
    # A sociotropic minority that misreads the societal value by $500/hh.
    for sigma in (10_000.0, 30_000.0):
        run(
            space_21,
            sigma,
            (
                VoterType(share=0.7, selfish_weight=1.0),
                VoterType(
                    share=0.3, selfish_weight=0.0, eta=1.0, societal_noise_sd=500.0
                ),
            ),
            "noisy_sociotropic_30pct",
            GRID_POINTS,
        )
    # Interior selfish weights (the general table path), coarser grid, with
    # a same-grid pure-selfish baseline so the two are comparable.
    space_13 = PolicySpace.from_gross_electorate(gross, grid_points=13)
    for sigma in (10_000.0,):
        run(
            space_13,
            sigma,
            (VoterType(share=1.0, selfish_weight=1.0),),
            "selfish_baseline_g13",
            13,
        )
        run(
            space_13,
            sigma,
            (
                VoterType(
                    share=1.0, selfish_weight=0.5, eta=1.0, societal_noise_sd=500.0
                ),
            ),
            "homogeneous_half_selfish",
            13,
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gross = d.load_measured_electorate()
    financed = d.apply_financing(gross, "per_capita")

    from strategic_experiments import candidate_households

    households = candidate_households(gross)

    tracking = tracking_dose_response(financed)
    tracking.to_csv(OUT / "heterogeneous_tracking.csv", index=False)

    threshold = knife_edge_threshold(financed)
    print(f"knife-edge cure: sigma=0 tracking flips below s* = {threshold:.4f}")

    pluralism = value_pluralism(financed)

    discipline = strategic_discipline(gross, households)
    discipline.to_csv(OUT / "heterogeneous_discipline.csv", index=False)

    variants = robustness_variants(gross, households)

    summary = {
        "provenance": provenance(),
        "candidate_households": households,
        "knife_edge_selfish_weight_threshold": round(threshold, 4),
        "value_pluralism": pluralism,
        "robustness_variants": variants,
    }
    (OUT / "heterogeneous_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    from democrasim import figures

    figures.save_figures(
        {
            "mixed_motive_tracking": figures.mixed_motive_tracking(tracking),
            "strategic_discipline": figures.strategic_discipline(discipline),
        },
        OUT.parent / "figures",
    )
    print(f"wrote {OUT / 'heterogeneous_summary.json'}")


if __name__ == "__main__":
    main()
