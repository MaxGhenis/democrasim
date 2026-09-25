"""Voting rules compared on a three-option ballot — docs/rules.md's numbers.

The fixed-platform experiments offer voters two policies; the welfare
optimum (the status quo) is not on the ballot, and the rules comparison
degenerates — with two options, sincere ballot-normalized score IS
plurality. This experiment puts the status quo on the ballot as a third
option and runs every implemented rule across the noise grid, grading each
against the same welfare metric and pricing mistakes in dollars of
equally-distributed-equivalent income.

A rule that elects nobody (approval can: the status quo is never perceived
*better than* the status quo) leaves current law in place, so NO_WINNER
grades as the status quo.

Run: uv run python scripts/rules_experiments.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

import democrasim as d
from democrasim.data import ARTIFACT_STEM, _data_dir
from democrasim.electorate import Electorate
from democrasim.experiments import _substream, _wilson_interval
from democrasim.voting import NO_WINNER

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
SIGMAS = [0.0, 10.0, 50.0, 200.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0]
N_ELECTIONS = 240
N_VOTERS = 10_001

RULES = {
    "plurality": d.Plurality(),
    "approval": d.Approval(),
    "score_ballot_0_5": d.Score(normalize="ballot", levels=5),
    "score_stakes": d.Score(normalize="stakes", cap=1_000.0),
    "star_0_5": d.STAR(levels=5),
    "instant_runoff": d.InstantRunoff(indifferent_abstain=True),
    # Approval with a weakly-inclusive threshold: approve anything perceived
    # at least as good as the status quo (the -1e-9 admits exact zeros).
    # With the status quo on the ballot, the strict/weak convention decides
    # whether the near-indifferent mass approves it - appended last so the
    # other rules' RNG substreams are unchanged.
    "approval_inclusive": d.Approval(threshold=-1e-9),
}


def three_option_electorate() -> Electorate:
    """The financed two-policy electorate with the status quo on the ballot."""
    financed = d.apply_financing(d.load_measured_electorate(), "per_capita")
    return Electorate(
        deltas=np.column_stack([np.zeros(financed.n_voters), financed.deltas]),
        weights=financed.weights,
        base_income=financed.base_income,
        hh_adults=financed.hh_adults,
        policy_labels=("Status quo", *financed.policy_labels),
        source=financed.source + " | status quo on the ballot",
    )


def provenance() -> dict:
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/rules_experiments.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "n_elections": N_ELECTIONS,
        "n_voters": N_VOTERS,
        "rules": {name: repr(rule) for name, rule in RULES.items()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    electorate = three_option_electorate()
    metric = d.Isoelastic()
    welfare = metric.per_policy(electorate)
    ede = metric.dollar_equivalent(electorate)
    optimal = int(np.argmax(welfare))
    assert optimal == 0, "the status quo should be the welfare optimum here"

    rows = []
    for rule_index, (name, rule) in enumerate(RULES.items()):
        for sigma_index, sigma in enumerate(SIGMAS):
            spec = d.ElectionSpec(
                perception=d.LinearGaussianPerception(noise_sd=float(sigma)),
                rule=rule,
                welfare=metric,
                n_voters=N_VOTERS,
            )
            rng = _substream(20_260_712, rule_index * len(SIGMAS) + sigma_index)
            winners = np.empty(N_ELECTIONS, dtype=np.int64)
            turnout = np.empty(N_ELECTIONS)
            for k in range(N_ELECTIONS):
                result = d.run_election(
                    electorate, spec, rng, welfare_by_policy=welfare
                )
                winners[k] = result.winner
                turnout[k] = result.tally.turnout
            enacted = np.where(winners == NO_WINNER, 0, winners)
            tracked = enacted == optimal
            p_tracked = float(tracked.mean())
            lo, hi = _wilson_interval(p_tracked, N_ELECTIONS)
            regret = ede[optimal] - ede[enacted]
            rows.append(
                {
                    "rule": name,
                    "sigma": sigma,
                    "p_tracked": p_tracked,
                    "p_tracked_lo": lo,
                    "p_tracked_hi": hi,
                    "mean_regret_dollars": float(regret.mean()),
                    "mean_turnout": float(turnout.mean()),
                    "share_no_winner": float((winners == NO_WINNER).mean()),
                    "share_enacts_a": float((enacted == 1).mean()),
                    "share_enacts_b": float((enacted == 2).mean()),
                }
            )
            print(
                f"{name:>17} sigma={sigma:>7,.0f}: tracked={p_tracked:.3f} "
                f"regret=${rows[-1]['mean_regret_dollars']:.2f} "
                f"turnout={rows[-1]['mean_turnout']:.2f}"
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "rules_comparison.csv", index=False)

    summary = {
        "provenance": provenance(),
        "ballot": list(electorate.policy_labels),
        "welfare_metric": "Isoelastic(eta=1)",
        "ede_dollars_per_household": [round(v, 4) for v in ede],
        "no_winner_grades_as": "Status quo",
    }
    (OUT / "rules_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    from democrasim import figures

    figures.save_figures(
        {"rules_comparison": figures.rules_comparison(frame)},
        OUT.parent / "figures",
    )
    print(f"wrote {OUT / 'rules_comparison.csv'}")


if __name__ == "__main__":
    main()
