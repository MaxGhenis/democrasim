"""Regression pins for the findings: fail loudly if a rebuild moves the world.

The knife-edge results in docs/findings.md rest on facts of the committed
artifact — most fragilely, the *sign* of the residual cost gap between the
two policies. A routine ``build-data`` rerun against a newer engine could
flip that sign and silently invert every headline while ordinary tests stay
green. These tests pin the load-bearing quantities (deterministic given the
artifact — no Monte Carlo), so any rebuild that moves the findings' world
fails here and forces the findings note to be re-verified.
"""

import numpy as np
import pytest

import democrasim as d
from democrasim.data import ARTIFACT_STEM, _data_dir


def artifact_exists() -> bool:
    return (_data_dir() / f"{ARTIFACT_STEM}.parquet").exists()


pytestmark = pytest.mark.skipif(
    not artifact_exists(), reason="measured artifact not built yet"
)


@pytest.fixture(scope="module")
def gross():
    return d.load_measured_electorate()


@pytest.fixture(scope="module")
def financed(gross):
    return d.apply_financing(gross, "per_capita")


class TestCostGapKnifeEdge:
    def test_policy_a_costs_more(self, gross):
        """The findings' perfect-accuracy direction rests on this SIGN."""
        totals = gross.household_dollars()
        assert totals[0] > totals[1], (
            "Policy A no longer costs more than Policy B — the levy-gap "
            "sign has flipped and every perfect-accuracy result in "
            "docs/findings.md must be re-derived"
        )

    def test_budget_parity_within_five_percent(self, gross):
        totals = gross.household_dollars()
        assert totals.max() / totals.min() < 1.05

    def test_levy_gap_scale(self, gross):
        levy = gross.household_dollars() / gross.population
        gap = levy[0] - levy[1]
        assert 2.0 < gap < 7.0, f"levy gap ${gap:.2f} left its documented band"


class TestPreferenceStructure:
    def test_financed_majority_prefers_policy_b(self, financed):
        share_b = float(
            financed.weights[financed.margins < 0].sum() / financed.weights.sum()
        )
        assert 0.74 < share_b < 0.83, (
            "the no-stakes bloc's preference share moved; findings §2-3 "
            f"quote ~78.7%, artifact now gives {share_b:.1%}"
        )

    def test_gross_indifference_is_the_modal_state(self, gross):
        indifferent = np.abs(gross.margins) <= 1.0
        share = float(gross.weights[indifferent].sum() / gross.weights.sum())
        assert 0.70 < share < 0.82

    def test_high_stakes_pro_a_bloc_is_child_households(self, financed):
        pro_a = financed.margins > 500.0
        with_children = financed.demographics["n_children"].to_numpy() > 0
        weights = financed.weights
        share_kids = float(weights[pro_a & with_children].sum() / weights[pro_a].sum())
        assert share_kids > 0.99


class TestWelfareOrdering:
    def test_isoelastic_eta1_ranks_a_first(self, financed):
        values = d.Isoelastic(eta=1.0).per_policy(financed)
        assert values[0] > values[1]

    def test_both_policies_below_status_quo_at_eta1(self, financed):
        values = d.Isoelastic(eta=1.0).per_policy(financed)
        assert np.all(values < 0.0), (
            "findings state both financed policies score below the status "
            "quo under log welfare"
        )

    def test_eta3_reverses_the_ranking(self, financed):
        """Documented sensitivity: high inequality-aversion with the $1,000
        floor ranks Policy B first (findings 'what would move' section)."""
        values = d.Isoelastic(eta=3.0).per_policy(financed)
        assert values[1] > values[0]


class TestDeterministicElections:
    def test_perfect_accuracy_full_population_elects_b(self, financed):
        """The knife-edge itself, deterministically (no sampling)."""
        spec = d.ElectionSpec(
            perception=d.PERFECT_PERCEPTION,
            welfare=d.Isoelastic(),
            n_voters=None,
        )
        result = d.run_election(financed, spec, np.random.default_rng(0))
        assert result.winner == 1
        assert not result.tracked

    def test_approval_at_perfect_accuracy_elects_a_on_low_turnout(self, financed):
        spec = d.ElectionSpec(
            perception=d.PERFECT_PERCEPTION,
            rule=d.Approval(),
            welfare=d.Isoelastic(),
            n_voters=None,
        )
        result = d.run_election(financed, spec, np.random.default_rng(0))
        assert result.winner == 0
        assert 0.20 < result.tally.turnout < 0.27

    def test_abstention_band_repairs_perfect_accuracy(self, financed):
        spec = d.ElectionSpec(
            perception=d.PERFECT_PERCEPTION,
            rule=d.Plurality(abstain_below=25.0),
            welfare=d.Isoelastic(),
            n_voters=None,
        )
        result = d.run_election(financed, spec, np.random.default_rng(0))
        assert result.winner == 0
        assert result.tracked

    def test_analytic_large_population_limit_matches_knife_edge(self, financed):
        curve = d.analytic_plurality_curve(
            financed, noise_sds=[0.0, 1_000.0], n_voters=None
        )
        # Perfect accuracy: the levy-gap majority elects B with certainty.
        assert curve.loc[0, "p_tracked"] == 0.0
        # Moderate noise: the stake-weighted signal elects A with certainty.
        assert curve.loc[1, "p_tracked"] == 1.0
