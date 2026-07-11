from math import erf, sqrt

import numpy as np
import pytest

from democrasim import (
    PERFECT_PERCEPTION,
    Electorate,
    GroupedPerception,
    LinearGaussianPerception,
    ranking_accuracy,
)


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


class TestLinearGaussian:
    def test_perfect_is_identity(self, small_electorate, rng):
        perceived = PERFECT_PERCEPTION.perceive(small_electorate, rng)
        np.testing.assert_array_equal(perceived, small_electorate.deltas)

    def test_bias_shifts_exactly_without_noise(self, small_electorate, rng):
        model = LinearGaussianPerception(noise_sd=0.0, bias=(100.0, -50.0))
        perceived = model.perceive(small_electorate, rng)
        np.testing.assert_allclose(
            perceived, small_electorate.deltas + np.array([100.0, -50.0])
        )

    def test_attenuation_scales(self, small_electorate, rng):
        model = LinearGaussianPerception(noise_sd=0.0, attenuation=0.5)
        perceived = model.perceive(small_electorate, rng)
        np.testing.assert_allclose(perceived, small_electorate.deltas * 0.5)

    def test_noise_has_requested_scale(self, small_electorate, rng):
        model = LinearGaussianPerception(noise_sd=1_000.0)
        draws = np.stack(
            [
                model.perceive(small_electorate, rng) - small_electorate.deltas
                for _ in range(2_000)
            ]
        )
        assert draws.std() == pytest.approx(1_000.0, rel=0.05)
        assert abs(draws.mean()) < 50.0

    def test_negative_noise_rejected(self):
        with pytest.raises(ValueError):
            LinearGaussianPerception(noise_sd=-1.0)

    def test_wrong_bias_length_rejected(self, small_electorate, rng):
        model = LinearGaussianPerception(bias=(1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match="bias"):
            model.perceive(small_electorate, rng)


class TestRankingAccuracy:
    def test_matches_normal_cdf(self, small_electorate):
        # Voter 0 margin $1,500, sigma 500: P = Φ(1500 / (500·√2)).
        model = LinearGaussianPerception(noise_sd=500.0)
        per_voter = model.analytic_ranking_accuracy(small_electorate)
        assert per_voter[0] == pytest.approx(phi(1_500 / (500 * sqrt(2))))

    def test_zero_margin_is_nan_and_excluded(self, small_electorate):
        model = LinearGaussianPerception(noise_sd=500.0)
        per_voter = model.analytic_ranking_accuracy(small_electorate)
        assert np.isnan(per_voter[3])
        mean = ranking_accuracy(model, small_electorate)
        weights = small_electorate.weights[:3]
        assert mean == pytest.approx(float(np.average(per_voter[:3], weights=weights)))

    def test_all_zero_margins_are_all_nan(self, small_electorate):
        tied = small_electorate.with_deltas(
            np.repeat(small_electorate.deltas[:, :1], 2, axis=1)
        )
        per_voter = LinearGaussianPerception(noise_sd=500.0).analytic_ranking_accuracy(
            tied
        )
        assert per_voter.dtype == np.float64
        assert np.isnan(per_voter).all()

    def test_noiseless_is_step(self, small_electorate):
        exact = PERFECT_PERCEPTION.analytic_ranking_accuracy(small_electorate)
        np.testing.assert_array_equal(exact[:3], [1.0, 1.0, 1.0])

    def test_bias_can_defeat_noiseless_perception(self, small_electorate):
        # A pro-policy-1 bias larger than voter 0's $1,500 margin flips them.
        model = LinearGaussianPerception(noise_sd=0.0, bias=(0.0, 2_000.0))
        per_voter = model.analytic_ranking_accuracy(small_electorate)
        assert per_voter[0] == 0.0  # margin +1500 overwhelmed by bias
        assert per_voter[2] == 1.0  # margin −2000, bias helps policy 1

    def test_monte_carlo_agrees_with_analytic(self, small_electorate, rng):
        class OpaqueModel:
            """Same behavior, but hides the analytic method."""

            inner = LinearGaussianPerception(noise_sd=500.0)

            def perceive(self, electorate, rng):
                return self.inner.perceive(electorate, rng)

        analytic = ranking_accuracy(
            LinearGaussianPerception(noise_sd=500.0), small_electorate
        )
        estimated = ranking_accuracy(
            OpaqueModel(), small_electorate, rng=rng, n_draws=3_000
        )
        assert estimated == pytest.approx(analytic, abs=0.02)

    def test_opaque_model_requires_draws(self, small_electorate):
        class OpaqueModel:
            def perceive(self, electorate, rng):
                return electorate.deltas

        with pytest.raises(ValueError, match="n_draws"):
            ranking_accuracy(OpaqueModel(), small_electorate)

    def test_opaque_model_rejects_three_policies(self, small_electorate, rng):
        electorate = Electorate(
            deltas=np.column_stack(
                (small_electorate.deltas, np.zeros(small_electorate.n_voters))
            ),
            weights=small_electorate.weights,
            base_income=small_electorate.base_income,
            hh_adults=small_electorate.hh_adults,
            policy_labels=("Policy A", "Policy B", "Policy C"),
            source="TEST three-policy electorate",
        )

        class OpaqueModel:
            def perceive(self, electorate, rng):
                return electorate.deltas

        with pytest.raises(
            ValueError, match=r"^ranking accuracy is defined for two policies$"
        ):
            ranking_accuracy(OpaqueModel(), electorate, rng=rng, n_draws=1)


class TestGroupedPerception:
    def test_groups_get_their_own_models(self, small_electorate, rng):
        model = GroupedPerception(
            column="group",
            models={"x": LinearGaussianPerception(bias=(500.0, 0.0))},
            default=PERFECT_PERCEPTION,
        )
        perceived = model.perceive(small_electorate, rng)
        np.testing.assert_allclose(
            perceived[:2, 0], small_electorate.deltas[:2, 0] + 500.0
        )
        np.testing.assert_allclose(perceived[2:], small_electorate.deltas[2:])

    def test_analytic_accuracy_dispatches_per_group(self, small_electorate):
        noisy = LinearGaussianPerception(noise_sd=500.0)
        model = GroupedPerception(
            column="group", models={"x": noisy}, default=PERFECT_PERCEPTION
        )
        per_voter = model.analytic_ranking_accuracy(small_electorate)
        assert per_voter[0] == pytest.approx(phi(1_500 / (500 * sqrt(2))))
        assert per_voter[2] == 1.0

    def test_requires_demographics(self, small_electorate, rng):
        bare = small_electorate.with_deltas(small_electorate.deltas)
        object.__setattr__(bare, "demographics", None)
        model = GroupedPerception(column="group", models={}, default=PERFECT_PERCEPTION)
        with pytest.raises(ValueError, match="demographics"):
            model.perceive(bare, rng)
