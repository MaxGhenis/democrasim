import numpy as np
import pytest

from democrasim import homogeneous_toy, moment_matched_toy


class TestMomentMatchedToy:
    def test_matches_weighted_moments(self, small_electorate, rng):
        toy = moment_matched_toy(small_electorate, rng, n=200_000)
        target_mean = np.average(
            small_electorate.deltas, axis=0, weights=small_electorate.weights
        )
        target_cov = np.cov(
            small_electorate.deltas.T,
            aweights=small_electorate.weights,
            ddof=0,
        )
        np.testing.assert_allclose(toy.deltas.mean(axis=0), target_mean, atol=15.0)
        np.testing.assert_allclose(
            np.cov(toy.deltas.T, ddof=0), target_cov, rtol=0.05, atol=2_000.0
        )

    def test_is_labeled_toy(self, small_electorate, rng):
        toy = moment_matched_toy(small_electorate, rng, n=100)
        assert toy.source.startswith("TOY")
        assert small_electorate.source in toy.source

    def test_one_adult_uniform_weights(self, small_electorate, rng):
        toy = moment_matched_toy(small_electorate, rng, n=100)
        assert np.all(toy.hh_adults == 1.0)
        assert np.all(toy.weights == 1.0)

    def test_incomes_positive(self, small_electorate, rng):
        toy = moment_matched_toy(small_electorate, rng, n=10_000)
        assert np.all(toy.base_income > 0)

    def test_preserves_impact_income_correlation(self, rng):
        # Incidence is the welfare signal: a toy that dropped the
        # impact-income correlation would rank policies arbitrarily.
        # Build a source where policy 0 pays the poor and policy 1 the rich.
        n = 5_000
        income = rng.lognormal(11, 0.7, n)
        deltas = np.column_stack(
            [
                2_000.0 * (income < np.median(income)),
                2_000.0 * (income >= np.median(income)),
            ]
        )
        from democrasim import Electorate

        source = Electorate(
            deltas=deltas,
            weights=np.ones(n),
            base_income=income,
            hh_adults=np.ones(n),
            policy_labels=("Policy A", "Policy B"),
            source="TEST",
        )
        toy = moment_matched_toy(source, rng, n=100_000)

        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1])

        log_income_source = np.log(source.base_income)
        log_income_toy = np.log(toy.base_income)
        for j in range(2):
            got = corr(toy.deltas[:, j], log_income_toy)
            want = corr(source.deltas[:, j], log_income_source)
            assert got == pytest.approx(want, abs=0.05)
        # And the signs are the economically meaningful part.
        assert corr(toy.deltas[:, 0], log_income_toy) < -0.3
        assert corr(toy.deltas[:, 1], log_income_toy) > 0.3


class TestHomogeneousToy:
    def test_every_voter_has_the_same_margin(self):
        toy = homogeneous_toy(margin=800.0, n=50)
        np.testing.assert_allclose(toy.margins, 800.0)
        np.testing.assert_allclose(toy.deltas[:, 0], 400.0)
        np.testing.assert_allclose(toy.deltas[:, 1], -400.0)

    def test_is_labeled_toy(self):
        assert homogeneous_toy(margin=100.0, n=5).source.startswith("TOY")

    def test_rejects_nonpositive_margin(self):
        with pytest.raises(ValueError):
            homogeneous_toy(margin=0.0)
