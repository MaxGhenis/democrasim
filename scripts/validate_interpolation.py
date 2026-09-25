"""Measure the linear-interpolation error the strategic layer relies on.

The strategic policy space scales the two committed incidence vectors:
position (α, β) is assumed to produce household deltas
``α·Δ_A + β·Δ_B``. Within-family linearity is an approximation — credit
phase-ins/phase-outs and bracket interactions bend it — so this script
runs the engine at the midpoints (CTC base $2,700 ≈ α=0.5; top rates
capped at 35.5% ≈ β=0.5) and compares engine truth to 0.5× the stored
endpoint deltas. Results land in docs/results/interpolation_validation.json.

Run: uv run python scripts/validate_interpolation.py
(orchestrates one subprocess per simulation; ~8 min each)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INTERMEDIATE = REPO / "data" / "intermediate"
OUT = REPO / "docs" / "results" / "interpolation_validation.json"
YEAR = 2026
PERIOD = "2026-01-01.2100-12-31"

MIDPOINTS = {
    "ctc_mid": {
        "reform": {"gov.irs.credits.ctc.amount.base[0].amount": {PERIOD: 2_700}},
        "endpoint_column": "delta_policy_a",
        "fraction": 0.5,
        "description": "CTC base $2,700 (midpoint of $2,200 -> $3,200)",
    },
    "cap_mid": {
        "reform": {
            "gov.irs.income.bracket.rates.7": {PERIOD: 0.355},
            "gov.irs.income.bracket.rates.6": {PERIOD: 0.345},
        },
        "endpoint_column": "delta_policy_b",
        "fraction": 0.5,
        "description": "Top rates 37%->35.5% and 35%->34.5% (half the cap depth)",
    },
}


def run_scenario(name: str) -> None:
    """One engine simulation; writes an intermediate parquet + stats json."""
    import policyengine as pe
    from policyengine_core.reforms import Reform

    spec = MIDPOINTS[name]
    simulation = pe.us.managed_microsimulation(
        reform=Reform.from_dict(spec["reform"], "policyengine_us")
    )
    net_income = simulation.calc("household_net_income", period=YEAR)
    household_id = simulation.calc("household_id", period=YEAR)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "household_id": np.asarray(household_id).astype(np.int64),
            "net_income": np.asarray(net_income, dtype=np.float64),
            "weight": np.asarray(net_income.weights, dtype=np.float64),
        }
    ).to_parquet(INTERMEDIATE / f"validate_{name}.households.parquet")
    (INTERMEDIATE / f"validate_{name}.stats.json").write_text(
        json.dumps(
            {
                "scenario": name,
                "microseries_net_income_total": float(net_income.sum()),
            }
        )
    )
    print(f"[{name}] total=${float(net_income.sum()) / 1e9:,.1f}B")


def compare() -> None:
    """Engine midpoint deltas vs 0.5x the committed endpoint deltas."""
    import democrasim as d

    baseline = pd.read_parquet(INTERMEDIATE / "baseline.households.parquet")
    gross = d.load_measured_electorate()
    # Household-level view of the committed artifact (one row per household).
    frame = gross.demographics[["household_id"]].copy()
    frame["weight"] = gross.weights
    frame["hh_adults"] = gross.hh_adults
    frame["delta_policy_a"] = gross.deltas[:, 0]
    frame["delta_policy_b"] = gross.deltas[:, 1]
    households = frame.groupby("household_id").first().reset_index()

    report: dict = {"year": YEAR, "midpoints": {}}
    for name, spec in MIDPOINTS.items():
        mid = pd.read_parquet(INTERMEDIATE / f"validate_{name}.households.parquet")
        merged = (
            baseline.rename(columns={"net_income": "base"})
            .merge(
                mid.rename(columns={"net_income": "mid"})[["household_id", "mid"]],
                on="household_id",
                validate="1:1",
            )
            .merge(
                households[["household_id", spec["endpoint_column"], "hh_adults"]],
                on="household_id",
                validate="1:1",
            )
        )
        engine_delta = merged["mid"] - merged["base"]
        predicted = spec["fraction"] * merged[spec["endpoint_column"]]
        error = engine_delta - predicted
        mass = merged["weight"]
        total_engine = float((engine_delta * mass).sum())
        total_predicted = float((predicted * mass).sum())
        abs_err = np.abs(error)
        report["midpoints"][name] = {
            "description": spec["description"],
            "engine_total_bn": round(total_engine / 1e9, 4),
            "interpolated_total_bn": round(total_predicted / 1e9, 4),
            "total_relative_error": round(
                abs(total_engine - total_predicted) / abs(total_engine), 5
            ),
            "share_households_within_1_dollar": round(
                float(mass[abs_err < 1.0].sum() / mass.sum()), 4
            ),
            "share_households_within_10_dollars": round(
                float(mass[abs_err < 10.0].sum() / mass.sum()), 4
            ),
            "p99_abs_error_dollars": round(
                float(
                    d.weighted_quantile(abs_err.to_numpy(), mass.to_numpy(), [0.99])[0]
                ),
                2,
            ),
            "max_abs_error_dollars": round(float(abs_err.max()), 2),
            "weighted_mean_abs_error_dollars": round(
                float((abs_err * mass).sum() / mass.sum()), 3
            ),
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--scenario":
        run_scenario(sys.argv[2])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare()
        return
    # Orchestrate: one simulation per subprocess (hard rule), then compare.
    for name in MIDPOINTS:
        print(f"=== scenario: {name} ===")
        subprocess.run(
            [sys.executable, str(Path(__file__)), "--scenario", name],
            check=True,
        )
    compare()


if __name__ == "__main__":
    main()
