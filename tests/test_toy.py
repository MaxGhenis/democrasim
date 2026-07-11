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
