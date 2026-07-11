import numpy as np
import pytest

from democrasim import (
    ElectionSpec,
    LinearGaussianPerception,
    Utilitarian,
    accuracy_sweep,
    bias_sweep,
    find_threshold,
    homogeneous_toy,
    run_elections,
    summarize_elections,
)


@pytest.fixture(scope="module")
def toy():
    return homogeneous_toy(margin=1_000.0, n=20_000)


@pytest.fixture(scope="module")
def toy_spec():
    return ElectionSpec(
        perception=LinearGaussianPerception(),
        welfare=Utilitarian(),
        n_voters=501,
    )


class TestAccuracySweep:
    def test_tracking_falls_as_noise_rises(self, toy, toy_spec):
        sweep = accuracy_sweep(
            toy,
            noise_sds=[10.0, 50_000.0],
            spec=toy_spec,
            n_elections=100,
            seed=0,
        )
        # Near-perfect accuracy: unanimity elects the better policy.
        assert sweep.loc[0, "p_tracked"] == 1.0
        # Enormous noise: the election is close to a coin flip.
        assert sweep.loc[1, "p_tracked"] < 0.8
        # Accuracy metric is monotone in noise.
        assert (
            sweep.loc[0, "mean_ranking_accuracy"]
            > sweep.loc[1, "mean_ranking_accuracy"]
        )

    def test_records_labels_and_source(self, toy, toy_spec):
        sweep = accuracy_sweep(
            toy, noise_sds=[100.0], spec=toy_spec, n_elections=5, seed=0
        )
        assert sweep.attrs["policy_labels"] == ("Policy A", "Policy B")
        assert "TOY" in sweep.attrs["source"]
        assert {
            "noise_sd",
            "mean_ranking_accuracy",
            "p_tracked",
            "p_tracked_se",
            "mean_regret",
            "p_win_0",
            "p_win_1",
        } <= set(sweep.columns)


class TestBiasSweep:
    def test_large_bias_flips_the_election(self, toy, toy_spec):
        sweep = bias_sweep(
            toy,
            bias_dollars=[0.0, 5_000.0],
            toward=1,
            noise_sd=200.0,
            spec=toy_spec,
            n_elections=50,
            seed=0,
        )
        # Unbiased: the truly-better policy 0 wins everything.
        assert sweep.loc[0, "p_win_0"] == 1.0
        # A $5,000 perceived bonus for policy 1 dwarfs the $1,000 margin.
        assert sweep.loc[1, "p_win_1"] == 1.0

    def test_bias_below_margin_is_harmless_when_noiseless(self, toy, toy_spec):
        sweep = bias_sweep(
            toy,
            bias_dollars=[500.0],
            toward=1,
            noise_sd=0.0,
            spec=toy_spec,
            n_elections=20,
            seed=0,
        )
        # Margin is $1,000; a $500 bias cannot flip any voter.
        assert sweep.loc[0, "p_win_0"] == 1.0


class TestFindThreshold:
    def test_interpolates_crossing(self):
        assert find_threshold(
            [0.5, 0.7, 0.9], [0.2, 0.6, 1.0], target=0.8
        ) == pytest.approx(0.8)

    def test_returns_nan_when_never_reached(self):
        assert np.isnan(find_threshold([0.5, 0.9], [0.2, 0.6], target=0.95))

    def test_handles_unsorted_input(self):
        assert find_threshold(
            [0.9, 0.5, 0.7], [1.0, 0.2, 0.6], target=0.8
        ) == pytest.approx(0.8)

    def test_immediate_crossing_returns_first_x(self):
        assert find_threshold([0.5, 0.9], [0.95, 1.0], target=0.9) == 0.5


def test_summarize_elections_counts(toy, toy_spec, rng):
    results = run_elections(toy, toy_spec, n_elections=20, rng=rng)
    summary = summarize_elections(results)
    assert summary["p_tracked"] == 1.0  # margin $1,000, perfect perception
    assert summary["p_win_0"] == 1.0
    assert summary["mean_turnout"] == 1.0
    assert summary["p_no_winner"] == 0.0
