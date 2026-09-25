"""Heterogeneous preferences: dollar EDE values and mixed-motive voters."""

import numpy as np
import pytest

import democrasim as d
from democrasim.electorate import Electorate
from democrasim.preferences import MixedMotivePerception, VoterType


def two_voter_electorate() -> Electorate:
    return Electorate(
        deltas=np.array([[1_000.0, 0.0], [0.0, 0.0]]),
        weights=np.ones(2),
        base_income=np.array([10_000.0, 40_000.0]),
        hh_adults=np.ones(2),
        policy_labels=("Policy A", "Policy B"),
        source="TOY hand-checkable pair",
    )


def toy_electorate() -> Electorate:
    """Six voters, opposed incidence, welfare-lens disagreement built in:
    B distributes more total dollars (utilitarian-better) but pays them to
    the two rich households (inequality-averse-worse)."""
    deltas = np.column_stack(
        [
            np.array([100.0, 100.0, 100.0, 100.0, -60.0, -60.0]),
            np.array([-30.0, -30.0, -30.0, -30.0, 250.0, 250.0]),
        ]
    )
    return Electorate(
        deltas=deltas,
        weights=np.ones(6),
        base_income=np.array([20e3, 25e3, 30e3, 35e3, 400e3, 500e3]),
        hh_adults=np.ones(6),
        policy_labels=("Policy A", "Policy B"),
        source="TOY six-voter opposed incidence",
    )


class TestDollarEquivalent:
    def test_utilitarian_is_mean_household_dollars(self):
        electorate = toy_electorate()
        expected = electorate.household_dollars() / 6.0
        np.testing.assert_allclose(
            d.Utilitarian().dollar_equivalent(electorate), expected
        )

    def test_isoelastic_log_case_is_geometric_mean_change(self):
        electorate = two_voter_electorate()
        base = np.sqrt(10_000.0 * 40_000.0)
        reformed = np.sqrt(11_000.0 * 40_000.0)
        values = d.Isoelastic(eta=1.0).dollar_equivalent(electorate)
        assert values[0] == pytest.approx(reformed - base, rel=1e-12)
        assert values[1] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0, 3.0])
    def test_ranks_policies_exactly_like_welfare(self, eta):
        """The EDE is a strictly increasing transform of mean utility."""
        electorate = d.apply_financing(d.load_measured_electorate(), "per_capita")
        metric = d.Isoelastic(eta=eta)
        dollars = metric.dollar_equivalent(electorate)
        welfare = metric.per_policy(electorate)
        assert np.argmax(dollars) == np.argmax(welfare)
        np.testing.assert_array_equal(np.sign(dollars), np.sign(welfare))


class TestVoterType:
    def test_share_and_weight_bounds(self):
        with pytest.raises(ValueError):
            VoterType(share=0.0)
        with pytest.raises(ValueError):
            VoterType(share=0.5, selfish_weight=1.5)
        with pytest.raises(ValueError):
            VoterType(share=0.5, societal_noise_sd=-1.0)
        with pytest.raises(ValueError):
            VoterType(share=0.5, own_noise_sd=-1.0)

    def test_mixture_shares_must_sum_to_one(self):
        with pytest.raises(ValueError):
            MixedMotivePerception(types=(VoterType(share=0.5), VoterType(share=0.4)))


class TestMixedMotivePerception:
    def test_satisfies_the_perception_protocol(self):
        assert isinstance(MixedMotivePerception(), d.PerceptionModel)

    def test_pure_selfish_noiseless_reproduces_true_deltas(self):
        electorate = toy_electorate()
        model = MixedMotivePerception.homogeneous(selfish_weight=1.0)
        perceived = model.perceive(electorate, np.random.default_rng(0))
        np.testing.assert_array_equal(perceived, electorate.deltas)

    def test_sociotropic_informed_sees_the_ede_values(self):
        electorate = toy_electorate()
        model = MixedMotivePerception.homogeneous(selfish_weight=0.0, eta=1.0)
        perceived = model.perceive(electorate, np.random.default_rng(0))
        expected = d.Isoelastic(eta=1.0).dollar_equivalent(electorate)
        np.testing.assert_allclose(perceived, np.tile(expected, (6, 1)))

    def test_runs_through_the_election_harness(self):
        electorate = toy_electorate()
        spec = d.ElectionSpec(
            perception=MixedMotivePerception.homogeneous(
                own=d.LinearGaussianPerception(noise_sd=100.0), selfish_weight=0.5
            ),
            n_voters=101,
        )
        result = d.run_election(electorate, spec, np.random.default_rng(1))
        assert result.winner in (0, 1, d.NO_WINNER)

    def test_vote_probabilities_selfish_match_probit_formula(self):
        electorate = toy_electorate()
        sigma = 300.0
        model = MixedMotivePerception.homogeneous(
            own=d.LinearGaussianPerception(noise_sd=sigma), selfish_weight=1.0
        )
        p_first, p_indifferent = model.vote_probabilities(electorate)
        from democrasim.perception import _phi

        np.testing.assert_allclose(
            p_first, _phi(electorate.margins / (sigma * np.sqrt(2.0)))
        )
        np.testing.assert_array_equal(p_indifferent, np.zeros(6))

    def test_vote_probabilities_match_monte_carlo(self):
        electorate = toy_electorate()
        model = MixedMotivePerception(
            own=d.LinearGaussianPerception(noise_sd=300.0),
            types=(
                VoterType(share=0.5, selfish_weight=1.0),
                VoterType(share=0.5, selfish_weight=0.5, societal_noise_sd=200.0),
            ),
        )
        p_first, _ = model.vote_probabilities(electorate)
        rng = np.random.default_rng(7)
        draws = 4_000
        first = np.zeros(6)
        for _ in range(draws):
            perceived = model.perceive(electorate, rng)
            first += perceived[:, 0] > perceived[:, 1]
        np.testing.assert_allclose(first / draws, p_first, atol=0.03)

    def test_own_noise_override_silences_the_shared_model(self):
        electorate = toy_electorate()
        model = MixedMotivePerception(
            own=d.LinearGaussianPerception(noise_sd=10_000.0),
            types=(VoterType(share=1.0, selfish_weight=1.0, own_noise_sd=0.0),),
        )
        perceived = model.perceive(electorate, np.random.default_rng(0))
        np.testing.assert_array_equal(perceived, electorate.deltas)


class TestAnalyticMixedCurve:
    def test_single_selfish_type_reduces_to_baseline_curve(self):
        electorate = d.apply_financing(d.load_measured_electorate(), "per_capita")
        noise = [0.0, 200.0, 1_000.0]
        mixed = d.analytic_mixed_plurality_curve(
            electorate, noise, types=(VoterType(share=1.0),)
        )
        baseline = d.analytic_plurality_curve(electorate, noise)
        for column in (
            "p_tracked",
            "expected_share_optimal",
            "expected_share_other",
            "expected_share_abstain",
        ):
            np.testing.assert_allclose(mixed[column], baseline[column], atol=1e-12)

    def test_informed_altruists_track_perfectly_at_any_own_noise(self):
        electorate = d.apply_financing(d.load_measured_electorate(), "per_capita")
        curve = d.analytic_mixed_plurality_curve(
            electorate,
            [0.0, 30_000.0],
            types=(VoterType(share=1.0, selfish_weight=0.0, eta=1.0),),
            welfare=d.Isoelastic(eta=1.0),
        )
        np.testing.assert_allclose(curve.p_tracked, 1.0)
        np.testing.assert_allclose(curve.expected_share_optimal, 1.0)

    def test_value_pluralism_types_disagree_when_etas_disagree(self):
        """Two informed sociotropic types with different η can split the
        electorate whenever their welfare lenses rank the policies
        differently — misperception-free disagreement."""
        electorate = toy_electorate()
        eta_low, eta_high = 0.0, 3.0
        low = d.Isoelastic(eta=eta_low).dollar_equivalent(electorate)
        high = d.Isoelastic(eta=eta_high).dollar_equivalent(electorate)
        assert np.argmax(low) != np.argmax(high), "toy chosen to disagree"
        curve = d.analytic_mixed_plurality_curve(
            electorate,
            [0.0],
            types=(
                VoterType(share=0.5, selfish_weight=0.0, eta=eta_low),
                VoterType(share=0.5, selfish_weight=0.0, eta=eta_high),
            ),
            welfare=d.Isoelastic(eta=eta_high),
        )
        assert curve.loc[0, "expected_share_optimal"] == pytest.approx(0.5)
        assert curve.loc[0, "p_tracked"] == pytest.approx(0.5)
