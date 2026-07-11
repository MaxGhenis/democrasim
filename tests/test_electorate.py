import numpy as np
import pytest

from democrasim import Electorate, weighted_quantile


def _electorate(**overrides):
    base = dict(
        deltas=np.array([[1.0, 2.0], [3.0, 4.0]]),
        weights=np.array([1.0, 1.0]),
        base_income=np.array([50_000.0, 60_000.0]),
        hh_adults=np.array([1.0, 1.0]),
        policy_labels=("Policy A", "Policy B"),
        source="TEST",
    )
    base.update(overrides)
    return Electorate(**base)


class TestValidation:
    def test_rejects_single_policy(self):
        with pytest.raises(ValueError, match="two policies"):
            _electorate(deltas=np.array([[1.0], [2.0]]), policy_labels=("A",))

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError, match="weights"):
            _electorate(weights=np.array([1.0]))

    def test_rejects_nonpositive_weights(self):
        with pytest.raises(ValueError, match="positive"):
            _electorate(weights=np.array([1.0, 0.0]))

    def test_rejects_zero_adults(self):
        with pytest.raises(ValueError, match="hh_adults"):
            _electorate(hh_adults=np.array([1.0, 0.0]))

    def test_rejects_nonfinite_deltas(self):
        with pytest.raises(ValueError, match="finite"):
            _electorate(deltas=np.array([[np.nan, 2.0], [3.0, 4.0]]))

    def test_rejects_label_mismatch(self):
        with pytest.raises(ValueError, match="labels"):
            _electorate(policy_labels=("A", "B", "C"))

    def test_arrays_are_immutable(self):
        electorate = _electorate()
        with pytest.raises(ValueError):
            electorate.deltas[0, 0] = 99.0


class TestSampling:
    def test_sample_follows_weights(self):
        rng = np.random.default_rng(0)
        electorate = _electorate(weights=np.array([9.0, 1.0]))
        sampled = electorate.sample(20_000, rng)
        share_first = (sampled.deltas[:, 0] == 1.0).mean()
        assert share_first == pytest.approx(0.9, abs=0.01)

    def test_sample_has_uniform_weights(self):
        rng = np.random.default_rng(0)
        sampled = _electorate().sample(50, rng)
        assert sampled.n_voters == 50
        assert np.all(sampled.weights == 1.0)

    def test_sample_carries_demographics(self, small_electorate):
        rng = np.random.default_rng(0)
        sampled = small_electorate.sample(10, rng)
        assert sampled.demographics is not None
        assert len(sampled.demographics) == 10


class TestAggregation:
    def test_household_dollars_counts_households_once(self, small_electorate):
        # Household 0: two adults, weight 10 each, +$1,000 under policy 0
        # -> 10 × $1,000 (not 20 ×). Household 1: weight 30, $0.
        totals = small_electorate.household_dollars()
        assert totals[0] == pytest.approx(10 * 1_000.0)
        assert totals[1] == pytest.approx(10 * -500.0 + 30 * 2_000.0)

    def test_margins(self, small_electorate):
        assert small_electorate.margins[0] == pytest.approx(1_500.0)
        assert small_electorate.margins[3] == 0.0

    def test_impact_summary_shares(self, small_electorate):
        summary = small_electorate.impact_summary()
        row = summary[summary["policy"] == "Policy A"].iloc[0]
        # 20 of 100 weight gains, nobody loses, 80 unaffected.
        assert row["share_gaining"] == pytest.approx(0.2)
        assert row["share_losing"] == 0.0
        assert row["share_unaffected"] == pytest.approx(0.8)


def test_weighted_quantile_median():
    # Uniform weights: interpolated median is exact.
    values = np.array([1.0, 2.0, 3.0])
    assert weighted_quantile(values, np.ones(3), [0.5])[0] == pytest.approx(2.0)
    # Weight concentrated on one value pulls the median to it.
    skewed = weighted_quantile(
        np.array([1.0, 2.0, 10.0]), np.array([1.0, 1.0, 98.0]), [0.5]
    )[0]
    assert skewed == pytest.approx(10.0, abs=0.5)
