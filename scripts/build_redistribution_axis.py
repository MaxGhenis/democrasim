"""Build the one-dimensional redistribution axis: rates up, revenue out.

A position on this axis is a single number ``t`` in [0, 1]: every federal
income tax bracket rate rises by ``t x 10`` percentage points, and the
revenue comes back as an equal per-adult transfer. ``t = 0`` is current
law; ``t = 1`` is a ten-point across-the-board increase funding the
largest transfer this axis offers. That is the classic linear-tax-plus-
demogrant dial of the redistribution literature, priced here on measured
household incidence through the real tax code instead of a stylized
economy.

Three simulations, one per subprocess (two engine sims in one kernel
fragment the allocator): baseline, ``t = 1``, and ``t = 0.5``. The
midpoint prices the axis's linearity the same way
``scripts/validate_interpolation.py`` prices the two-policy space — the
model interpolates gross incidence linearly in ``t``, and that
approximation is measured, not assumed.

Run: uv run python scripts/build_redistribution_axis.py
(needs the engine extra: uv sync --extra engine --group dev)
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = REPO_ROOT / "data" / "intermediate_axis"
OUT_DIR = REPO_ROOT / "democrasim" / "data"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
ARTIFACT_STEM = "us_2026_redistribution"

PERIOD = "2026-01-01.2100-12-31"
YEAR = 2026
#: Rate increase (percentage points, as a fraction) at t = 1.
FULL_RATE_INCREASE = 0.10
#: Intensities simulated: the endpoint and the linearity probe.
INTENSITIES = {"full": 1.0, "mid": 0.5}
SCENARIOS = ["baseline", "full", "mid"]


def reform_dict(intensity: float, baseline_rates: dict[int, float]) -> dict:
    """Raise every bracket rate by ``intensity x FULL_RATE_INCREASE``."""
    return {
        f"gov.irs.income.bracket.rates.{bracket}": {
            PERIOD: round(rate + intensity * FULL_RATE_INCREASE, 6)
        }
        for bracket, rate in baseline_rates.items()
    }


def baseline_rates() -> dict[int, float]:
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    instant = f"{YEAR}-01-01"
    return {
        bracket: float(
            getattr(parameters.gov.irs.income.bracket.rates, str(bracket))(instant)
        )
        for bracket in range(1, 8)
    }


def run_scenario(scenario: str, intermediate_dir: Path) -> None:
    """Run one simulation and write its household extract."""
    import policyengine as pe
    from policyengine_core.reforms import Reform

    rates = baseline_rates()
    if scenario == "baseline":
        simulation = pe.us.managed_microsimulation()
        reform = None
    else:
        reform = reform_dict(INTENSITIES[scenario], rates)
        simulation = pe.us.managed_microsimulation(
            reform=Reform.from_dict(reform, "policyengine_us")
        )

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    net_income = simulation.calc("household_net_income", period=YEAR)
    household_id = simulation.calc("household_id", period=YEAR)
    frame = pd.DataFrame(
        {
            "household_id": np.asarray(household_id).astype(np.int64),
            "net_income": np.asarray(net_income, dtype=np.float64),
            "weight": np.asarray(net_income.weights, dtype=np.float64),
        }
    )
    frame.to_parquet(intermediate_dir / f"{scenario}.households.parquet")
    stats = {
        "scenario": scenario,
        "intensity": INTENSITIES.get(scenario, 0.0),
        "baseline_rates": rates,
        "reform": reform,
        # MicroSeries total: ground truth the combine step must reproduce.
        "microseries_net_income_total": float(net_income.sum()),
        "n_households": len(frame),
    }
    (intermediate_dir / f"{scenario}.stats.json").write_text(
        json.dumps(stats, indent=2)
    )
    print(
        f"[{scenario}] households={len(frame):,} "
        f"total=${stats['microseries_net_income_total'] / 1e9:,.1f}B",
        flush=True,
    )


def combine(intermediate_dir: Path, out_dir: Path) -> None:
    """Merge the axis endpoint onto the committed adult-row structure."""
    from democrasim.data import _data_dir

    frames, stats = {}, {}
    for scenario in SCENARIOS:
        frames[scenario] = pd.read_parquet(
            intermediate_dir / f"{scenario}.households.parquet"
        )
        stats[scenario] = json.loads(
            (intermediate_dir / f"{scenario}.stats.json").read_text()
        )
        extracted = float(
            (frames[scenario]["net_income"] * frames[scenario]["weight"]).sum()
        )
        recorded = stats[scenario]["microseries_net_income_total"]
        if not np.isclose(extracted, recorded, rtol=1e-6):
            raise AssertionError(
                f"{scenario}: extracted {extracted:,.0f} != MicroSeries {recorded:,.0f}"
            )
        if frames[scenario]["household_id"].duplicated().any():
            raise AssertionError(f"{scenario}: duplicate household_id values")

    base = frames["baseline"]
    households = base.rename(columns={"net_income": "base_income"})
    for scenario in ("full", "mid"):
        merged = households.merge(
            frames[scenario][["household_id", "net_income", "weight"]],
            on="household_id",
            suffixes=("", f"_{scenario}"),
            validate="1:1",
        )
        if (merged["weight"] != merged[f"weight_{scenario}"]).any():
            raise AssertionError(f"{scenario}: household weights differ from baseline")
        households[f"delta_{scenario}"] = (
            merged["net_income"] - merged["base_income"]
        ).to_numpy()

    # Linearity of gross incidence in t: the model interpolates the axis
    # from the endpoint alone, so the midpoint is the measurement of that
    # approximation, not a free parameter.
    predicted = 0.5 * households["delta_full"].to_numpy()
    actual = households["delta_mid"].to_numpy()
    weights = households["weight"].to_numpy()
    error = actual - predicted
    total_actual = float((actual * weights).sum())
    validation = {
        "midpoint_total_actual_bn": total_actual / 1e9,
        "midpoint_total_predicted_bn": float((predicted * weights).sum()) / 1e9,
        "total_relative_error": abs(
            float((error * weights).sum()) / total_actual if total_actual else 0.0
        ),
        "share_households_within_1_dollar": float(
            np.average(np.abs(error) <= 1.0, weights=weights)
        ),
        "mean_absolute_error_dollars": float(
            np.average(np.abs(error), weights=weights)
        ),
        "p99_absolute_error_dollars": float(np.percentile(np.abs(error), 99)),
    }

    # Attach the axis endpoint to the committed adult-row artifact, which
    # already carries validated weights, adult counts, and demographics.
    adults = pd.read_parquet(_data_dir() / "us_2026_measured.parquet")
    axis = adults.merge(
        households[["household_id", "delta_full", "base_income"]].rename(
            columns={"base_income": "base_income_axis"}
        ),
        on="household_id",
        how="left",
        validate="m:1",
    )
    if axis["delta_full"].isna().any():
        raise AssertionError("some adult rows have no axis household match")
    income_gap = np.abs(axis["base_income"] - axis["base_income_axis"]).max()
    if income_gap > 1e-6:
        raise AssertionError(
            f"baseline net income differs from the committed artifact by "
            f"${income_gap:,.2f} — the engine or dataset moved"
        )
    # Keep only what the axis needs plus identifiers; the two-policy
    # columns already live in us_2026_measured and would be duplicated.
    axis = axis.rename(columns={"delta_full": "delta_axis_full"})[
        [
            "delta_axis_full",
            "weight",
            "base_income",
            "hh_adults",
            "n_children",
            "state",
            "household_id",
        ]
    ]

    gross_total = float(
        (households["delta_full"].to_numpy() * households["weight"].to_numpy()).sum()
    )
    meta = {
        "artifact": ARTIFACT_STEM,
        "source": (
            "Adult rows and demographics from us_2026_measured, plus the "
            "gross household net-income change under a "
            f"{FULL_RATE_INCREASE:.0%}-point increase in every federal "
            f"income tax bracket rate, policy year {YEAR}."
        ),
        "axis": {
            "description": (
                "Position t in [0, 1]: every bracket rate rises by "
                f"t x {FULL_RATE_INCREASE:.0%} points; the revenue returns "
                "as an equal per-adult transfer (applied by "
                "apply_financing's per_capita mode, whose levy is negative "
                "when the policy raises revenue)."
            ),
            "full_rate_increase": FULL_RATE_INCREASE,
            "baseline_rates": stats["baseline"]["baseline_rates"],
            "reform_at_t1": stats["full"]["reform"],
            "gross_revenue_bn": -gross_total / 1e9,
        },
        "linearity_validation": validation,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    axis.to_parquet(
        out_dir / f"{ARTIFACT_STEM}.parquet", compression="zstd", index=False
    )
    (out_dir / f"{ARTIFACT_STEM}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "axis_linearity_validation.json").write_text(
        json.dumps({"axis": meta["axis"], "validation": validation}, indent=2) + "\n"
    )
    print(f"wrote {out_dir / f'{ARTIFACT_STEM}.parquet'} ({len(axis):,} adult rows)")
    print(f"  gross revenue at t=1: ${-gross_total / 1e9:,.1f}B")
    print(f"  linearity: {validation['total_relative_error']:.2%} of totals, ")
    print(
        f"  {validation['share_households_within_1_dollar']:.1%} of households "
        f"within $1, mean |error| ${validation['mean_absolute_error_dollars']:,.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.scenario:
        run_scenario(args.scenario, args.intermediate_dir)
    elif args.combine:
        combine(args.intermediate_dir, args.out_dir)
    else:
        for scenario in SCENARIOS:
            print(f"=== scenario: {scenario} ===", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--scenario",
                    scenario,
                    "--intermediate-dir",
                    str(args.intermediate_dir),
                ],
                check=True,
            )
        combine(args.intermediate_dir, args.out_dir)


if __name__ == "__main__":
    main()
