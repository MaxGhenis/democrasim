import numpy as np
import pytest

from democrasim import (
    NO_WINNER,
    PERFECT_PERCEPTION,
    ElectionSpec,
    Electorate,
    LinearGaussianPerception,
    Plurality,
    Utilitarian,
    run_election,
)


def concentrated_stakes_electorate() -> Electorate:
    """A 30% minority gains $10,000 under policy 0; a 70% majority gains
    $10 under policy 1. Policy 0 is utilitarian-optimal by 400×, but the
    majority prefers policy 1 — a perfectly informed plurality election
    does NOT track welfare here."""
    return Electorate(
        deltas=np.array([[10_000.0, 0.0]] * 30 + [[0.0, 10.0]] * 70),
        weights=np.ones(100),
        base_income=np.full(100, 50_000.0),
        hh_adults=np.ones(100),
        policy_labels=("Policy A", "Policy B"),
        source="TEST",
    )


class TestRunElection:
    def test_perfect_perception_elects_majority_choice(self, rng):
        electorate = concentrated_stakes_electorate()
        spec = ElectionSpec(
            perception=PERFECT_PERCEPTION,
            welfare=Utilitarian(),
            n_voters=501,
        )
        result = run_election(electorate, spec, rng)
        assert result.winner == 1
        assert result.welfare_optimal == 0
        assert not result.tracked
        assert result.regret == pytest.approx(
            result.welfare_by_policy[0] - result.welfare_by_policy[1]
        )

    def test_tracked_when_majority_and_welfare_align(self, rng):
        electorate = Electorate(
            deltas=np.array([[500.0, -500.0]] * 10),
            weights=np.ones(10),
            base_income=np.full(10, 50_000.0),
            hh_adults=np.ones(10),
            policy_labels=("Policy A", "Policy B"),
            source="TEST",
        )
        spec = ElectionSpec(
            perception=PERFECT_PERCEPTION, welfare=Utilitarian(), n_voters=101
        )
        result = run_election(electorate, spec, rng)
        assert result.winner == 0
        assert result.tracked
        assert result.regret == 0.0

    def test_full_population_mode(self, rng):
        electorate = concentrated_stakes_electorate()
        spec = ElectionSpec(
            perception=PERFECT_PERCEPTION, welfare=Utilitarian(), n_voters=None
        )
        result = run_election(electorate, spec, rng)
        assert result.winner == 1
        assert result.tally.shares[1] == pytest.approx(0.7)

    def test_no_winner_leaves_status_quo(self, rng):
        electorate = Electorate(
            deltas=np.zeros((5, 2)),
            weights=np.ones(5),
            base_income=np.full(5, 50_000.0),
            hh_adults=np.ones(5),
            policy_labels=("Policy A", "Policy B"),
            source="TEST",
        )
        spec = ElectionSpec(
            perception=PERFECT_PERCEPTION,
            rule=Plurality(),
            welfare=Utilitarian(),
            n_voters=None,
        )
        result = run_election(electorate, spec, rng)
        assert result.winner == NO_WINNER
        assert not result.tracked
        # Status quo realizes zero welfare; regret is the forgone optimum.
        assert result.regret == pytest.approx(result.welfare_by_policy.max())

    def test_precomputed_welfare_matches(self, rng):
        electorate = concentrated_stakes_electorate()
        spec = ElectionSpec(
            perception=LinearGaussianPerception(noise_sd=50.0),
            welfare=Utilitarian(),
            n_voters=201,
        )
        welfare = spec.welfare.per_policy(electorate)
        seed = np.random.default_rng(7)
        direct = run_election(electorate, spec, np.random.default_rng(7))
        precomputed = run_election(electorate, spec, seed, welfare_by_policy=welfare)
        assert direct.winner == precomputed.winner
        np.testing.assert_allclose(
            direct.welfare_by_policy, precomputed.welfare_by_policy
        )
