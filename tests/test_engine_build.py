"""Tests for the artifact builder that don't require running simulations.

The reform-dictionary checks run everywhere (reforms are plain data); the
``engine``-marked tests need the policyengine stack installed and are
skipped otherwise.
"""

import importlib.util

import pytest

from democrasim.engine.build import BASELINE, SCENARIOS, _policy_for_scenario
from democrasim.engine.reforms import PERIOD, POLICIES, YEAR

HAS_ENGINE = importlib.util.find_spec("policyengine") is not None


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
