"""Build the measured artifact: engine-computed impacts per real household.

Run as ``python -m democrasim.engine.build`` (or ``democrasim build-data``).
The default invocation orchestrates one subprocess per simulation — baseline
plus one per policy — then merges intermediates into the committed artifact.
One simulation per process is a hard rule: two engine simulations in one
kernel fragment the allocator and can exhaust memory even after ``del``.

Before writing, every scenario's extraction is checked against its
MicroSeries weighted aggregate; household IDs must be unique and identical
across scenarios and states; reform and person weights must exactly match
baseline household weights; adult-row dollars must reconcile to each
household; and excluded zero-adult dollars must remain negligible.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from democrasim.engine.reforms import POLICIES, YEAR

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_DIR = REPO_ROOT / "data" / "intermediate"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
ARTIFACT_STEM = "us_2026_measured"

BASELINE = "baseline"
SCENARIOS = [BASELINE] + [p["column"].removeprefix("delta_") for p in POLICIES]

#: Extracted aggregates must match MicroSeries aggregates to this rel. tol.
EXTRACTION_RTOL = 1e-6
#: Adult-row allocations must reproduce household dollars to this rel. tol.
ADULT_RECONCILIATION_RTOL = 1e-9
#: Max share of a policy's absolute dollars allowed in zero-adult households.
ZERO_ADULT_TOLERANCE = 0.005


def _policy_for_scenario(scenario: str) -> dict:
    for policy in POLICIES:
        if policy["column"] == f"delta_{scenario}":
            return policy
    raise ValueError(f"unknown scenario: {scenario!r}")


def _versions() -> dict[str, str]:
    from importlib import metadata

    return {
        package: metadata.version(package)
        for package in ("policyengine", "policyengine-us", "policyengine-core")
    }


def _release_manifest() -> dict:
    """The policyengine bundle manifest entry certifying US model + data."""
    from importlib import metadata

    path = Path(
        str(
            metadata.distribution("policyengine").locate_file(
                "policyengine/data/bundle/manifest.json"
            )
        )
    )
    if not path.exists():
        return {"note": "bundle manifest not found in policyengine package"}
    us = json.loads(path.read_text())["data_releases"]["us"]
    keep = (
        "bundle_id",
        "certification",
        "certified_data_artifact",
        "data_producer",
        "default_dataset",
    )
    return {k: us[k] for k in keep if k in us}


def _sanitize_bundle(value: object) -> object:
    """Replace machine-local absolute paths in bundle provenance."""
    if isinstance(value, str):
        return Path(value).name if value.startswith("/") else value
    if isinstance(value, dict):
        return {key: _sanitize_bundle(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_bundle(item) for item in value]
    return value


def verify() -> None:
    """Print engine provenance and current-law parameters (no simulation)."""
    print(json.dumps(_versions(), indent=2))
    print(json.dumps(_release_manifest(), indent=2))
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    instant = f"{YEAR}-01-01"
    ctc_base = parameters.gov.irs.credits.ctc.amount.base[0].amount(instant)
    print(f"CTC base amount {YEAR}:", ctc_base)
    for bracket in (5, 6, 7):
        rate = getattr(parameters.gov.irs.income.bracket.rates, str(bracket))
        print(f"bracket rate {bracket} {YEAR}:", rate(instant))


def run_scenario(scenario: str, intermediate_dir: Path) -> None:
    """Run one simulation and write its intermediate extracts."""
    import policyengine as pe
    from policyengine_core.reforms import Reform

    # managed_microsimulation pins the certified default dataset from the
    # policyengine.py release bundle and attaches provenance to the sim.
    if scenario == BASELINE:
        simulation = pe.us.managed_microsimulation()
    else:
        policy = _policy_for_scenario(scenario)
        reform = Reform.from_dict(policy["reform"], "policyengine_us")
        simulation = pe.us.managed_microsimulation(reform=reform)

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Household level: net income is the budgetary-cost variable (captures
    # cascading tax/benefit interactions, not just the reformed program).
    net_income = simulation.calc("household_net_income", period=YEAR)
    household_id = simulation.calc("household_id", period=YEAR)
    households = pd.DataFrame(
        {
            "household_id": np.asarray(household_id).astype(np.int64),
            "net_income": np.asarray(net_income, dtype=np.float64),
            "weight": np.asarray(net_income.weights, dtype=np.float64),
        }
    )
    households.to_parquet(intermediate_dir / f"{scenario}.households.parquet")

    stats = {
        "scenario": scenario,
        "year": YEAR,
        # The MicroSeries weighted total — ground truth the combine step
        # must reproduce from the extracted arrays.
        "microseries_net_income_total": float(net_income.sum()),
        "n_households": len(households),
        "bundle": _sanitize_bundle(getattr(simulation, "policyengine_bundle", None)),
    }

    if scenario == BASELINE:
        age = simulation.calc("age", period=YEAR)
        person_household_id = simulation.calc(
            "household_id", period=YEAR, map_to="person"
        )
        state = simulation.calc("state_code_str", period=YEAR)
        persons = pd.DataFrame(
            {
                "household_id": np.asarray(person_household_id).astype(np.int64),
                "age": np.asarray(age, dtype=np.float64),
                "weight": np.asarray(age.weights, dtype=np.float64),
            }
        )
        persons.to_parquet(intermediate_dir / "baseline.persons.parquet")
        households_state = pd.DataFrame(
            {
                "household_id": households["household_id"],
                "state": np.asarray(state).astype(str),
            }
        )
        households_state.to_parquet(intermediate_dir / "baseline.states.parquet")
        stats["n_persons"] = len(persons)
        stats["microseries_population"] = float(age.weights.sum())
        stats["versions"] = _versions()
        stats["release_manifest"] = _release_manifest()

    (intermediate_dir / f"{scenario}.stats.json").write_text(
        json.dumps(stats, indent=2)
    )
    print(
        f"[{scenario}] households={stats['n_households']}, "
        f"total=${stats['microseries_net_income_total'] / 1e9:,.1f}B"
    )


def _load_intermediates(
    intermediate_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = pd.read_parquet(intermediate_dir / "baseline.households.parquet")
    states = pd.read_parquet(intermediate_dir / "baseline.states.parquet")
    persons = pd.read_parquet(intermediate_dir / "baseline.persons.parquet")
    reformed = {
        scenario: pd.read_parquet(intermediate_dir / f"{scenario}.households.parquet")
        for scenario in SCENARIOS
        if scenario != BASELINE
    }
    stats = {}
    for scenario in SCENARIOS:
        scenario_stats = json.loads(
            (intermediate_dir / f"{scenario}.stats.json").read_text()
        )
        scenario_stats["bundle"] = _sanitize_bundle(scenario_stats.get("bundle"))
        stats[scenario] = scenario_stats

    scenario_frames = {BASELINE: baseline, "states": states, **reformed}
    for scenario, frame in scenario_frames.items():
        duplicate_rows = frame["household_id"].duplicated(keep=False)
        if duplicate_rows.any():
            raise AssertionError(
                f"{scenario}: {int(duplicate_rows.sum())} rows have "
                "duplicate household_id values"
            )

    baseline_ids = set(baseline["household_id"])
    for scenario, frame in scenario_frames.items():
        if scenario == BASELINE:
            continue
        scenario_ids = set(frame["household_id"])
        missing = baseline_ids - scenario_ids
        extra = scenario_ids - baseline_ids
        if missing or extra:
            raise AssertionError(
                f"{scenario}: household_id set differs from baseline "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

    for scenario, frame in {BASELINE: baseline, **reformed}.items():
        # Extraction fidelity: the extracted arrays must reproduce the
        # MicroSeries weighted total (same weights, same values).
        extracted_total = float((frame["net_income"] * frame["weight"]).sum())
        recorded = stats[scenario]["microseries_net_income_total"]
        if not np.isclose(extracted_total, recorded, rtol=EXTRACTION_RTOL):
            raise AssertionError(
                f"{scenario}: extracted total {extracted_total:,.0f} != "
                f"MicroSeries total {recorded:,.0f}"
            )

    for scenario, frame in reformed.items():
        compared_weights = baseline[["household_id", "weight"]].merge(
            frame[["household_id", "weight"]],
            on="household_id",
            suffixes=("_baseline", "_reform"),
            validate="1:1",
        )
        mismatched = (
            compared_weights["weight_baseline"] != compared_weights["weight_reform"]
        )
        if mismatched.any():
            raise AssertionError(
                f"{scenario}: {int(mismatched.sum())} household weights "
                "differ from baseline"
            )

    households = baseline.rename(columns={"net_income": "base_income"}).merge(
        states, on="household_id", validate="1:1"
    )
    for policy in POLICIES:
        scenario = policy["column"].removeprefix("delta_")
        scenario_frame = reformed[scenario].rename(
            columns={"net_income": f"income_{scenario}"}
        )
        households = households.merge(
            scenario_frame[["household_id", f"income_{scenario}"]],
            on="household_id",
            validate="1:1",
        )
        households[policy["column"]] = (
            households[f"income_{scenario}"] - households["base_income"]
        )
    return households, persons, stats


def combine(intermediate_dir: Path, out_dir: Path) -> None:
    """Merge intermediates into the adult-level artifact, with validation."""
    households, persons, stats = _load_intermediates(intermediate_dir)

    compared_person_weights = persons[["household_id", "weight"]].merge(
        households[["household_id", "weight"]],
        on="household_id",
        how="left",
        suffixes=("_person", "_household"),
        indicator=True,
        validate="m:1",
    )
    unknown_households = compared_person_weights["_merge"] != "both"
    if unknown_households.any():
        raise AssertionError(
            "persons: "
            f"{int(unknown_households.sum())} rows have household_id values "
            "absent from baseline"
        )
    person_weight_diffs = (
        compared_person_weights["weight_person"]
        - compared_person_weights["weight_household"]
    ).abs()
    person_weight_max_abs_diff = (
        float(person_weight_diffs.max()) if len(person_weight_diffs) else 0.0
    )
    mismatched_person_weights = (
        compared_person_weights["weight_person"]
        != compared_person_weights["weight_household"]
    )
    if mismatched_person_weights.any():
        raise AssertionError(
            "persons: "
            f"{int(mismatched_person_weights.sum())} weights differ from "
            "baseline household weights"
        )

    composition = (
        persons.assign(is_adult=persons["age"] >= 18)
        .groupby("household_id")
        .agg(
            household_size=("age", "size"),
            hh_adults=("is_adult", "sum"),
        )
        .reset_index()
    )
    composition["n_children"] = composition["household_size"] - composition["hh_adults"]
    households = households.merge(composition, on="household_id", validate="1:1")

    # Households with no voting-age adult cannot appear in the electorate;
    # their dollars are excluded. Enforce that this is negligible.
    zero_adult = households["hh_adults"] == 0
    exclusions = {}
    for policy in POLICIES:
        column = policy["column"]
        total = float((households[column].abs() * households["weight"]).sum())
        excluded = float(
            (
                households.loc[zero_adult, column].abs()
                * households.loc[zero_adult, "weight"]
            ).sum()
        )
        share = 0.0 if total == 0 else excluded / total
        if share > ZERO_ADULT_TOLERANCE:
            raise AssertionError(
                f"{column}: {share:.2%} of absolute dollars sit in "
                "households with no adults — investigate before building"
            )
        exclusions[column] = {
            "zero_adult_households": int(zero_adult.sum()),
            "share_of_abs_dollars_excluded": share,
        }
    households = households[~zero_adult]

    adults = persons[persons["age"] >= 18].merge(
        households[
            [
                "household_id",
                "base_income",
                "state",
                "household_size",
                "hh_adults",
                "n_children",
            ]
            + [p["column"] for p in POLICIES]
        ],
        on="household_id",
        validate="m:1",
    )

    # Consistency: summing each household's dollars once across adult rows
    # must reproduce the household-level weighted totals.
    costs = {}
    reconciliation_max_relative_discrepancy = {}
    for policy in POLICIES:
        column = policy["column"]
        household_total = float((households[column] * households["weight"]).sum())
        adult_total = float(
            (adults[column] * adults["weight"] / adults["hh_adults"]).sum()
        )
        if not np.isclose(
            adult_total,
            household_total,
            rtol=ADULT_RECONCILIATION_RTOL,
            atol=0.0,
        ):
            raise AssertionError(
                f"{column}: adult-row total {adult_total:,.0f} != "
                f"household total {household_total:,.0f}"
            )

        adult_dollars = (
            adults.assign(
                _allocated_dollars=(
                    adults[column] * adults["weight"] / adults["hh_adults"]
                )
            )
            .groupby("household_id", sort=False)["_allocated_dollars"]
            .sum()
        )
        household_dollars = (
            households.set_index("household_id")[column]
            * (households.set_index("household_id")["weight"])
        )
        adult_dollars = adult_dollars.reindex(household_dollars.index)
        absolute_discrepancy = (adult_dollars - household_dollars).abs()
        relative_discrepancy = np.divide(
            absolute_discrepancy.to_numpy(),
            household_dollars.abs().to_numpy(),
            out=np.zeros(len(household_dollars), dtype=np.float64),
            where=household_dollars.to_numpy() != 0,
        )
        zero_dollars_mismatch = (household_dollars.to_numpy() == 0) & (
            absolute_discrepancy.to_numpy() != 0
        )
        relative_discrepancy[zero_dollars_mismatch] = np.inf
        reconciled = np.isclose(
            adult_dollars.to_numpy(),
            household_dollars.to_numpy(),
            rtol=ADULT_RECONCILIATION_RTOL,
            atol=0.0,
        )
        if not reconciled.all():
            raise AssertionError(
                f"{column}: {int((~reconciled).sum())} households do not "
                "reconcile across adult rows"
            )
        reconciliation_max_relative_discrepancy[column] = float(
            relative_discrepancy.max(initial=0.0)
        )
        costs[policy["label"]] = {
            "total_household_dollars_bn": household_total / 1e9,
            "share_households_gaining": float(
                np.average(households[column] > 1.0, weights=households["weight"])
            ),
        }

    artifact = pd.DataFrame(
        {
            **{p["column"]: adults[p["column"]].astype(np.float64) for p in POLICIES},
            "weight": adults["weight"].astype(np.float64),
            "base_income": adults["base_income"].astype(np.float64),
            "hh_adults": adults["hh_adults"].astype(np.int16),
            "age": adults["age"].astype(np.int16),
            "state": pd.Categorical(adults["state"]),
            "household_size": adults["household_size"].astype(np.int16),
            "n_children": adults["n_children"].astype(np.int16),
            "household_id": adults["household_id"].astype(np.int64),
        }
    )

    versions = stats[BASELINE]["versions"]
    manifest = stats[BASELINE]["release_manifest"]
    bundle = stats[BASELINE].get("bundle")
    meta = {
        "artifact": ARTIFACT_STEM,
        "source": (
            f"PolicyEngine US {versions['policyengine-us']} "
            f"(policyengine {versions['policyengine']}) on its certified "
            f"default US dataset, policy year {YEAR}; rows are adults 18+"
        ),
        "row_definition": (
            "One row per voting-age adult (18+). Impact columns are the "
            "adult's household-level change in annual net income vs current "
            "law; hh_adults supports counting each household once."
        ),
        "year": YEAR,
        "policies": [{**policy, **costs[policy["label"]]} for policy in POLICIES],
        "engine": {
            "versions": versions,
            "release_bundle": bundle,
            "release_manifest": manifest,
        },
        "counts": {
            "households": len(households),
            "adult_rows": len(artifact),
            "adult_population": float(artifact["weight"].sum()),
        },
        "validations": {
            "extraction_rtol": EXTRACTION_RTOL,
            "person_household_weight_max_abs_diff": person_weight_max_abs_diff,
            "adult_household_reconciliation_rtol": ADULT_RECONCILIATION_RTOL,
            "adult_household_reconciliation_max_relative_discrepancy": (
                reconciliation_max_relative_discrepancy
            ),
            "zero_adult_exclusions": exclusions,
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(
        out_dir / f"{ARTIFACT_STEM}.parquet", compression="zstd", index=False
    )
    (out_dir / f"{ARTIFACT_STEM}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(
        f"wrote {out_dir / f'{ARTIFACT_STEM}.parquet'} ({len(artifact):,} adult rows)"
    )
    for label, cost in costs.items():
        print(
            f"  {label}: ${cost['total_household_dollars_bn']:,.1f}B, "
            f"{cost['share_households_gaining']:.1%} of households gain"
        )


def report(intermediate_dir: Path) -> None:
    """Print calibration diagnostics from intermediates (no simulation)."""
    households, _, _ = _load_intermediates(intermediate_dir)
    weights = households["weight"].to_numpy()
    base = households["base_income"].to_numpy()
    order = np.argsort(base)
    cum = np.cumsum(weights[order])
    decile = np.empty(len(households), dtype=int)
    decile[order] = np.minimum((10 * cum / cum[-1]).astype(int) + 1, 10)
    for policy in POLICIES:
        column = policy["column"]
        dollars = households[column].to_numpy() * weights
        total = dollars.sum()
        top_decile_share = dollars[decile == 10].sum() / total if total else 0
        bottom_half_share = dollars[decile <= 5].sum() / total if total else 0
        gaining = float(np.average(households[column] > 1.0, weights=weights))
        print(
            f"{policy['label']} ({column}): ${total / 1e9:,.1f}B | "
            f"top-decile share {top_decile_share:.1%} | "
            f"bottom-half share {bottom_half_share:.1%} | "
            f"households gaining {gaining:.1%}"
        )


def orchestrate(intermediate_dir: Path, out_dir: Path) -> None:
    """Run every scenario in its own subprocess, then combine."""
    for scenario in SCENARIOS:
        print(f"=== scenario: {scenario} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "democrasim.engine.build",
                "--scenario",
                scenario,
                "--intermediate-dir",
                str(intermediate_dir),
            ],
            check=True,
        )
    combine(intermediate_dir, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.verify:
        verify()
    elif args.scenario:
        run_scenario(args.scenario, args.intermediate_dir)
    elif args.combine:
        combine(args.intermediate_dir, args.out_dir)
    elif args.report:
        report(args.intermediate_dir)
    else:
        orchestrate(args.intermediate_dir, args.out_dir)


if __name__ == "__main__":
    main()
