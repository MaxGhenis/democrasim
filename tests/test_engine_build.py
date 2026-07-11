"""Tests for the artifact builder that don't require running simulations.

The reform-dictionary checks run everywhere (reforms are plain data); the
``engine``-marked tests need the policyengine stack installed and are
skipped otherwise.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from democrasim.engine.build import (
    ADULT_RECONCILIATION_RTOL,
    BASELINE,
    SCENARIOS,
    _policy_for_scenario,
    combine,
)
from democrasim.engine.reforms import PERIOD, POLICIES, YEAR

HAS_ENGINE = importlib.util.find_spec("policyengine") is not None


def _write_synthetic_intermediates(
    intermediate_dir: Path,
    *,
    zero_adult: bool = False,
    zero_adult_policy_a_delta: float = 0.0,
) -> None:
    intermediate_dir.mkdir()
    household_ids = np.arange(101, 107, dtype=np.int64)
    weights = np.array([1.0, 2.0, 1.5, 3.0, 0.5, 1.25])
    base_income = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0])
    deltas = {
        "policy_a": np.array([10.0, 0.0, 20.0, -5.0, 15.0, zero_adult_policy_a_delta]),
        "policy_b": np.array([0.0, 12.0, -3.0, 8.0, 5.0, 0.0]),
    }
    scenario_frames = {
        BASELINE: pd.DataFrame(
            {
                "household_id": household_ids,
                "net_income": base_income,
                "weight": weights,
            }
        ),
        **{
            scenario: pd.DataFrame(
                {
                    "household_id": household_ids,
                    "net_income": base_income + scenario_delta,
                    "weight": weights,
                }
            )
            for scenario, scenario_delta in deltas.items()
        },
    }

    ages_by_household = {
        101: [40, 38, 10],
        102: [25],
        103: [17, 35],
        104: [70, 68],
        105: [18, 4],
        106: [12, 15] if zero_adult else [20, 15],
    }
    persons = pd.DataFrame(
        [
            {"household_id": household_id, "age": age, "weight": weights[index]}
            for index, household_id in enumerate(household_ids)
            for age in ages_by_household[int(household_id)]
        ]
    )
    persons.to_parquet(intermediate_dir / "baseline.persons.parquet", index=False)
    pd.DataFrame(
        {
            "household_id": household_ids,
            "state": ["CA", "NY", "TX", "FL", "WA", "ME"],
        }
    ).to_parquet(intermediate_dir / "baseline.states.parquet", index=False)

    bundle = {
        "runtime_dataset_source": "/Users/example/cache/synthetic.h5",
        "runtime_dataset_uri": "hf://example/synthetic.h5@revision",
    }
    for scenario, frame in scenario_frames.items():
        frame.to_parquet(
            intermediate_dir / f"{scenario}.households.parquet", index=False
        )
        stats = {
            "scenario": scenario,
            "year": YEAR,
            "microseries_net_income_total": float(
                (frame["net_income"] * frame["weight"]).sum()
            ),
            "n_households": len(frame),
            "bundle": bundle,
        }
        if scenario == BASELINE:
            stats.update(
                {
                    "n_persons": len(persons),
                    "microseries_population": float(persons["weight"].sum()),
                    "versions": {
                        "policyengine": "test",
                        "policyengine-us": "test",
                        "policyengine-core": "test",
                    },
                    "release_manifest": {"bundle_id": "synthetic"},
                }
            )
        (intermediate_dir / f"{scenario}.stats.json").write_text(
            json.dumps(stats, indent=2)
        )


def _read_stats(intermediate_dir: Path, scenario: str) -> dict:
    return json.loads((intermediate_dir / f"{scenario}.stats.json").read_text())


def _write_stats(intermediate_dir: Path, scenario: str, stats: dict) -> None:
    (intermediate_dir / f"{scenario}.stats.json").write_text(
        json.dumps(stats, indent=2)
    )


class TestReformDefinitions:
    def test_labels_are_generic(self):
        for policy in POLICIES:
            assert policy["label"].startswith("Policy "), (
                "platform labels must stay generic — never candidates or parties"
            )

    def test_periods_have_explicit_start_and_end(self):
        start, dot, end = PERIOD.partition(".")
        assert dot == "."
        assert start == f"{YEAR}-01-01"
        assert end > start
        for policy in POLICIES:
            for periods in policy["reform"].values():
                assert set(periods) == {PERIOD}

    def test_scenarios_cover_baseline_plus_policies(self):
        assert SCENARIOS[0] == BASELINE
        assert len(SCENARIOS) == 1 + len(POLICIES)

    def test_policy_lookup(self):
        assert _policy_for_scenario("policy_a")["label"] == "Policy A"
        with pytest.raises(ValueError, match="unknown scenario"):
            _policy_for_scenario("policy_z")


class TestCombineValidation:
    def test_happy_path_records_validations_and_household_id(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        out_dir = tmp_path / "output"
        _write_synthetic_intermediates(intermediate_dir)

        combine(intermediate_dir, out_dir)

        artifact = pd.read_parquet(out_dir / "us_2026_measured.parquet")
        meta = json.loads((out_dir / "us_2026_measured.meta.json").read_text())
        validations = meta["validations"]
        assert artifact["household_id"].dtype == np.int64
        assert validations["person_household_weight_max_abs_diff"] == 0.0
        assert (
            validations["adult_household_reconciliation_rtol"]
            == ADULT_RECONCILIATION_RTOL
        )
        assert set(
            validations["adult_household_reconciliation_max_relative_discrepancy"]
        ) == {policy["column"] for policy in POLICIES}
        assert (
            meta["engine"]["release_bundle"]["runtime_dataset_source"] == "synthetic.h5"
        )
        assert meta["engine"]["release_bundle"]["runtime_dataset_uri"].startswith(
            "hf://"
        )

    def test_baseline_total_mismatch_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        stats = _read_stats(intermediate_dir, BASELINE)
        stats["microseries_net_income_total"] *= 1.1
        _write_stats(intermediate_dir, BASELINE, stats)

        with pytest.raises(AssertionError, match="baseline: extracted total"):
            combine(intermediate_dir, tmp_path / "output")

    def test_reform_total_mismatch_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        stats = _read_stats(intermediate_dir, "policy_a")
        stats["microseries_net_income_total"] *= 1.1
        _write_stats(intermediate_dir, "policy_a", stats)

        with pytest.raises(AssertionError, match="policy_a: extracted total"):
            combine(intermediate_dir, tmp_path / "output")

    def test_missing_reform_household_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        path = intermediate_dir / "policy_a.households.parquet"
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)

        with pytest.raises(
            AssertionError,
            match=r"policy_a: household_id set differs.*missing=1, extra=0",
        ):
            combine(intermediate_dir, tmp_path / "output")

    def test_duplicate_household_id_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        path = intermediate_dir / "policy_b.households.parquet"
        frame = pd.read_parquet(path)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(
            path, index=False
        )

        with pytest.raises(AssertionError, match=r"policy_b: .*duplicate household_id"):
            combine(intermediate_dir, tmp_path / "output")

    def test_reform_weight_mismatch_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        path = intermediate_dir / "policy_a.households.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "weight"] += 0.25
        frame.to_parquet(path, index=False)
        stats = _read_stats(intermediate_dir, "policy_a")
        stats["microseries_net_income_total"] = float(
            (frame["net_income"] * frame["weight"]).sum()
        )
        _write_stats(intermediate_dir, "policy_a", stats)

        with pytest.raises(
            AssertionError, match=r"policy_a: .*weights differ from baseline"
        ):
            combine(intermediate_dir, tmp_path / "output")

    def test_person_weight_mismatch_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(intermediate_dir)
        path = intermediate_dir / "baseline.persons.parquet"
        persons = pd.read_parquet(path)
        persons.loc[0, "weight"] += 0.25
        persons.to_parquet(path, index=False)

        with pytest.raises(
            AssertionError,
            match=r"persons: .*weights differ from baseline household weights",
        ):
            combine(intermediate_dir, tmp_path / "output")

    def test_zero_adult_household_below_tolerance_is_excluded(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        out_dir = tmp_path / "output"
        _write_synthetic_intermediates(intermediate_dir, zero_adult=True)

        combine(intermediate_dir, out_dir)

        artifact = pd.read_parquet(out_dir / "us_2026_measured.parquet")
        meta = json.loads((out_dir / "us_2026_measured.meta.json").read_text())
        assert 106 not in set(artifact["household_id"])
        for exclusion in meta["validations"]["zero_adult_exclusions"].values():
            assert exclusion == {
                "zero_adult_households": 1,
                "share_of_abs_dollars_excluded": 0.0,
            }

    def test_zero_adult_household_above_tolerance_raises(self, tmp_path):
        intermediate_dir = tmp_path / "intermediate"
        _write_synthetic_intermediates(
            intermediate_dir,
            zero_adult=True,
            zero_adult_policy_a_delta=1_000.0,
        )

        with pytest.raises(
            AssertionError, match=r"delta_policy_a: .*households with no adults"
        ):
            combine(intermediate_dir, tmp_path / "output")


@pytest.mark.engine
@pytest.mark.skipif(not HAS_ENGINE, reason="policyengine not installed")
class TestEngineProvenance:
    def test_bundle_manifest_certifies_us_data(self):
        from democrasim.engine.build import _release_manifest

        manifest = _release_manifest()
        assert manifest["certification"]["certified_for_model_version"]
        assert manifest["certified_data_artifact"]["dataset"]
        assert manifest["certified_data_artifact"]["sha256"]

    def test_installed_model_matches_certification(self):
        from importlib import metadata

        from democrasim.engine.build import _release_manifest

        certified = _release_manifest()["certification"]["certified_for_model_version"]
        assert metadata.version("policyengine-us") == certified
