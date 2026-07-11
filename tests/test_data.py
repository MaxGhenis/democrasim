"""Tests of the committed measured artifact.

These run wherever the artifact exists (it is committed to the repo); they
skip only if it has not been built yet. They pin the *structure* the rest of
the package depends on — opposite incidence, nonzero impacts, provenance —
without pinning exact dollar values, which move with engine releases.
"""

import numpy as np
import pytest

import democrasim
from democrasim.data import ARTIFACT_STEM, _data_dir


def artifact_exists() -> bool:
    return (_data_dir() / f"{ARTIFACT_STEM}.parquet").exists()


pytestmark = pytest.mark.skipif(
    not artifact_exists(), reason="measured artifact not built yet"
)


@pytest.fixture(scope="module")
def measured():
    return democrasim.load_measured_electorate()


@pytest.fixture(scope="module")
def meta():
    return democrasim.artifact_metadata()


def _nested_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _nested_strings(key)
            yield from _nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_strings(item)


class TestArtifactStructure:
    def test_two_generic_policies(self, measured):
        assert measured.policy_labels == ("Policy A", "Policy B")

    def test_population_is_us_adult_scale(self, measured):
        # US voting-age population is roughly 260M; allow a wide band.
        assert 2.0e8 < measured.population < 3.2e8

    def test_demographics_present(self, measured):
        assert measured.demographics is not None
        assert {"age", "state", "n_children"} <= set(measured.demographics.columns)
        assert (measured.demographics["age"] >= 18).all()

    def test_source_names_the_engine(self, measured):
        assert "PolicyEngine" in measured.source

    def test_household_once_values_are_verifiable(self, measured, meta):
        household_frame = measured.demographics[["household_id"]].copy()
        household_frame["weight"] = measured.weights
        household_frame["base_income"] = measured.base_income
        household_frame["hh_adults"] = measured.hh_adults
        policy_columns = []
        for index, policy in enumerate(meta["policies"]):
            column = policy["column"]
            policy_columns.append(column)
            household_frame[column] = measured.deltas[:, index]

        grouped = household_frame.groupby("household_id")
        for column in ["weight", "base_income", "hh_adults", *policy_columns]:
            assert grouped[column].nunique(dropna=False).eq(1).all()
        assert grouped.size().eq(grouped["hh_adults"].first()).all()

        adult_totals = np.array(
            [
                (
                    household_frame["weight"]
                    * household_frame[column]
                    / household_frame["hh_adults"]
                ).sum()
                for column in policy_columns
            ]
        )
        households_once = grouped.first()
        household_totals = np.array(
            [
                (households_once["weight"] * households_once[column]).sum()
                for column in policy_columns
            ]
        )
        np.testing.assert_allclose(adult_totals, household_totals, rtol=1e-9, atol=0.0)


class TestMeasuredImpacts:
    def test_both_policies_move_money(self, measured):
        totals = measured.household_dollars()
        assert np.all(totals > 1e9), "each policy should move > $1B"

    def test_opposite_incidence(self, measured):
        """Policy A's dollars go to child households low in the income
        distribution; policy B's to the top of it."""
        share = measured.weights / measured.hh_adults
        dollars_a = share * measured.deltas[:, 0]
        dollars_b = share * measured.deltas[:, 1]

        has_children = measured.demographics["n_children"].to_numpy() > 0
        assert dollars_a[has_children].sum() / dollars_a.sum() > 0.95

        top_decile_cut = democrasim.weighted_quantile(
            measured.base_income, measured.weights, [0.9]
        )[0]
        top = measured.base_income >= top_decile_cut
        assert dollars_b[top].sum() / dollars_b.sum() > 0.5
        assert dollars_a[top].sum() / dollars_a.sum() < 0.25

    def test_rough_budget_parity(self, measured, meta):
        totals = measured.household_dollars()
        ratio = totals.max() / totals.min()
        assert ratio < 1.25, f"policies should be budget-comparable: {totals}"

    def test_artifact_matches_engine_totals(self, measured, meta):
        for j, policy in enumerate(meta["policies"]):
            recorded = policy["total_household_dollars_bn"] * 1e9
            assert measured.household_dollars()[j] == pytest.approx(recorded, rel=1e-6)


class TestProvenance:
    def test_meta_records_reforms_and_engine(self, meta):
        assert meta["year"] == 2026
        for policy in meta["policies"]:
            assert policy["reform"], "reform dict must be recorded"
            for period_values in policy["reform"].values():
                for period in period_values:
                    start, _, end = period.partition(".")
                    assert start and end, "explicit start AND end dates"
        assert meta["engine"]["versions"]["policyengine-us"]
        assert meta["engine"]["release_bundle"] or meta["engine"]["release_manifest"]

    def test_meta_counts_match_artifact(self, measured, meta):
        assert meta["counts"]["adult_rows"] == measured.n_voters
        assert meta["counts"]["adult_population"] == pytest.approx(measured.population)

    def test_meta_records_builder_validations(self, meta):
        validations = meta["validations"]
        assert validations["person_household_weight_max_abs_diff"] == 0.0
        assert validations["adult_household_reconciliation_rtol"] == 1e-9
        discrepancies = validations[
            "adult_household_reconciliation_max_relative_discrepancy"
        ]
        assert set(discrepancies) == {"delta_policy_a", "delta_policy_b"}
        assert max(discrepancies.values()) <= 1e-9

    def test_meta_contains_no_machine_local_user_paths(self, meta):
        assert not any("/Users/" in value for value in _nested_strings(meta))
