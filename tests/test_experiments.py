import numpy as np
import pytest

from democrasim import (
    ElectionSpec,
    LinearGaussianPerception,
    Utilitarian,
    accuracy_sweep,
    analytic_plurality_curve,
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


class TestStatistics:
    def test_wilson_interval_nonzero_at_boundaries(self, toy, toy_spec):
        sweep = accuracy_sweep(
            toy, noise_sds=[10.0], spec=toy_spec, n_elections=100, seed=0
        )
        row = sweep.iloc[0]
        assert row["p_tracked"] == 1.0
        # Wald SE collapses to zero at the boundary; Wilson must not.
        assert row["p_tracked_se"] == 0.0
        assert row["p_tracked_hi"] == 1.0
        assert 0.94 < row["p_tracked_lo"] < 1.0

    def test_substreams_are_grid_edit_stable(self, toy, toy_spec):
        # Editing one grid point must not perturb another: point i draws
        # from a position-keyed substream, not a shared sequential RNG.
        full = accuracy_sweep(
            toy,
            noise_sds=[100.0, 2_000.0],
            spec=toy_spec,
            n_elections=30,
            seed=7,
        )
        edited = accuracy_sweep(
            toy,
            noise_sds=[999.0, 2_000.0],
            spec=toy_spec,
            n_elections=30,
            seed=7,
        )
        assert full.loc[1, "p_tracked"] == edited.loc[1, "p_tracked"]
        assert full.loc[1, "p_win_0"] == edited.loc[1, "p_win_0"]

    def test_bias_sweep_rejects_bad_target(self, toy, toy_spec):
        with pytest.raises(ValueError, match="toward"):
            bias_sweep(
                toy,
                bias_dollars=[100.0],
                toward=5,
                spec=toy_spec,
                n_elections=2,
                seed=0,
            )


class TestAnalyticPluralityCurve:
    def test_agrees_with_simulation(self, toy):
        analytic = analytic_plurality_curve(
            toy,
            noise_sds=[500.0, 5_000.0, 50_000.0],
            n_voters=501,
            welfare=Utilitarian(),
        )
        simulated = accuracy_sweep(
            toy,
            noise_sds=[500.0, 5_000.0, 50_000.0],
            spec=ElectionSpec(
                perception=LinearGaussianPerception(),
                welfare=Utilitarian(),
                n_voters=501,
            ),
            n_elections=400,
            seed=3,
        )
        for i in range(3):
            assert analytic.loc[i, "p_tracked"] == pytest.approx(
                simulated.loc[i, "p_tracked"], abs=0.06
            )

    def test_large_population_limit_is_deterministic(self, toy):
        # Any per-voter accuracy above 0.5 wins with certainty at infinite n.
        curve = analytic_plurality_curve(
            toy, noise_sds=[50_000.0], n_voters=None, welfare=Utilitarian()
        )
        assert curve.loc[0, "mean_ranking_accuracy"] < 0.51
        assert curve.loc[0, "p_tracked"] == 1.0

    def test_thresholds_tighten_with_n(self, toy):
        # Jury-theorem scaling: the accuracy needed for 90% tracking falls
        # toward 0.5 as the electorate grows.
        noise = np.geomspace(100.0, 200_000.0, 25)
        thresholds = {}
        for n in (101, 10_001, 1_000_001):
            curve = analytic_plurality_curve(
                toy, noise_sds=noise, n_voters=n, welfare=Utilitarian()
            )
            thresholds[n] = find_threshold(
                curve["mean_ranking_accuracy"], curve["p_tracked"], 0.9
            )
        assert thresholds[101] > thresholds[10_001] > thresholds[1_000_001]

    def test_all_abstain_leaves_status_quo(self):
        electorate = homogeneous_toy(margin=100.0, n=50).with_deltas(np.zeros((50, 2)))
        curve = analytic_plurality_curve(
            electorate, noise_sds=[0.0], n_voters=None, welfare=Utilitarian()
        )
        assert curve.loc[0, "expected_share_abstain"] == 1.0
        assert curve.loc[0, "p_tracked"] == 0.0

    def test_rejects_three_policies(self, rng):
        from democrasim import Electorate

        electorate = Electorate(
            deltas=np.zeros((4, 3)),
            weights=np.ones(4),
            base_income=np.full(4, 50_000.0),
            hh_adults=np.ones(4),
            policy_labels=("Policy A", "Policy B", "Policy C"),
            source="TEST",
        )
        with pytest.raises(ValueError, match="two policies"):
            analytic_plurality_curve(electorate, noise_sds=[100.0])


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
