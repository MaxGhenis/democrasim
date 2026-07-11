import numpy as np
import pytest

import democrasim as d
from democrasim.strategic import (
    Candidate,
    PolicySpace,
    _payoff_matrices,
    iterated_best_response,
    objective_grid,
    pure_nash_equilibria,
    win_probability,
)


def toy_space(grid_points: int = 5) -> PolicySpace:
    """Hand-checkable space: 6 voters; A helps the poor majority, B the rich.

    net_a gives +$100 to four low-income voters and −$50 to two rich ones;
    net_b mirrors it. Weights uniform, one adult each.
    """
    net_a = np.array([100.0, 100.0, 100.0, 100.0, -50.0, -50.0])
    net_b = np.array([-25.0, -25.0, -25.0, -25.0, 200.0, 200.0])
    grid = np.linspace(0.0, 1.0, grid_points)
    return PolicySpace(
        net_a=net_a,
        net_b=net_b,
        weights=np.ones(6),
        base_income=np.array([20e3, 25e3, 30e3, 35e3, 400e3, 500e3]),
        hh_adults=np.ones(6),
        alphas=grid,
        betas=grid.copy(),
    )


class TestWinProbability:
    def test_equal_positions_are_a_coin_flip(self):
        margins = np.zeros(10)
        assert win_probability(margins, np.ones(10), 100.0, 10_001) == 0.5
        assert win_probability(margins, np.ones(10), 0.0, None) == 0.5

    def test_matches_analytic_plurality_curve(self):
        gross = d.load_measured_electorate()
        financed = d.apply_financing(gross, "per_capita")
        for sigma, n in ((1_000.0, 10_001), (0.0, None), (10_000.0, 101)):
            curve = d.analytic_plurality_curve(financed, noise_sds=[sigma], n_voters=n)
            # analytic curve reports P(welfare-optimal = Policy A wins);
            # win_probability takes A-minus-B margins directly.
            assert win_probability(
                financed.margins, financed.weights, sigma, n
            ) == pytest.approx(float(curve.loc[0, "p_tracked"]), abs=1e-9)

    def test_more_favorable_margins_win_more(self):
        weights = np.ones(5)
        low = win_probability(np.full(5, 10.0), weights, 1_000.0, 1_001)
        high = win_probability(np.full(5, 500.0), weights, 1_000.0, 1_001)
        assert high > low > 0.5


class TestPolicySpace:
    def test_net_deltas_match_financing_ground_truth(self):
        """Linearity theorem: interpolated net stakes equal financing the
        scaled gross deltas directly."""
        gross = d.load_measured_electorate()
        space = PolicySpace.from_gross_electorate(gross, grid_points=5)
        alpha, beta = 0.5, 0.75
        scaled = gross.with_deltas(
            np.column_stack(
                [
                    alpha * gross.deltas[:, 0] + beta * gross.deltas[:, 1],
                    np.zeros(gross.n_voters),
                ]
            )
        )
        truth = d.apply_financing(scaled, "per_capita").deltas[:, 0]
        np.testing.assert_allclose(
            space.net_deltas(alpha, beta), truth, rtol=0, atol=1e-9
        )

    def test_compression_preserves_win_probabilities(self):
        gross = d.load_measured_electorate()
        space = PolicySpace.from_gross_electorate(gross, grid_points=5)
        rng = np.random.default_rng(3)
        for _ in range(4):
            da, db = rng.uniform(-1, 1, 2)
            full = win_probability(
                da * space.net_a + db * space.net_b,
                space.weights,
                1_000.0,
                10_001,
            )
            compressed = win_probability(
                da * space.comp_a + db * space.comp_b,
                space.comp_weights,
                1_000.0,
                10_001,
            )
            assert compressed == pytest.approx(full, abs=1e-6)

    def test_societal_grid_matches_fixed_platform_welfare(self):
        gross = d.load_measured_electorate()
        space = PolicySpace.from_gross_electorate(gross, grid_points=3)
        financed = d.apply_financing(gross, "per_capita")
        expected = d.Isoelastic().per_policy(financed)
        grid = space.societal_welfare_grid(d.Isoelastic())
        # (alpha=1, beta=0) is full Policy A; (0, 1) is full Policy B.
        assert grid[space.position_index(1.0, 0.0)] == pytest.approx(
            expected[0], rel=1e-9
        )
        assert grid[space.position_index(0.0, 1.0)] == pytest.approx(
            expected[1], rel=1e-9
        )
        assert grid[space.position_index(0.0, 0.0)] == 0.0


class TestObjectives:
    def test_pure_selfish_maximizes_own_household(self):
        space = toy_space()
        candidate = Candidate("Candidate 1", household_index=0, selfish_weight=1.0)
        grid = objective_grid(space, candidate)
        best = space.positions[int(np.argmax(grid))]
        # Voter 0 gains from A (+100/unit) and loses from B: corner (1, 0).
        assert best == (1.0, 0.0)

    def test_pure_societal_maximizes_welfare_metric(self):
        space = toy_space()
        candidate = Candidate(
            "Candidate 1",
            household_index=4,  # rich household, but weight is on society
            selfish_weight=0.0,
            societal=d.Isoelastic(eta=2.0),
        )
        grid = objective_grid(space, candidate)
        welfare = space.societal_welfare_grid(d.Isoelastic(eta=2.0))
        assert int(np.argmax(grid)) == int(np.argmax(welfare))

    def test_selfish_weight_bounds(self):
        with pytest.raises(ValueError):
            Candidate("Candidate 1", household_index=0, selfish_weight=1.5)


class TestEquilibrium:
    def test_office_seekers_converge(self):
        """Pure office motivation: both chase votes; equilibrium positions
        coincide (Downsian convergence) on the toy space."""
        space = toy_space()
        flat = d.Utilitarian()  # ~constant on a budget-neutral space
        c1 = Candidate(
            "Candidate 1", 0, selfish_weight=0.0, societal=flat, office_rent=10.0
        )
        c2 = Candidate(
            "Candidate 2", 4, selfish_weight=0.0, societal=flat, office_rent=10.0
        )
        equilibria = pure_nash_equilibria(space, c1, c2, sigma=50.0, n_voters=1_001)
        assert equilibria, "office-seeker game should have a pure equilibrium"
        assert any(e.position_1 == e.position_2 for e in equilibria)

    def test_returned_profiles_survive_deviation_check(self):
        space = toy_space(grid_points=4)
        c1 = Candidate("Candidate 1", 0, selfish_weight=0.7)
        c2 = Candidate("Candidate 2", 4, selfish_weight=0.7)
        payoff_1, payoff_2, _ = _payoff_matrices(space, c1, c2, 200.0, 1_001)
        equilibria = pure_nash_equilibria(space, c1, c2, sigma=200.0, n_voters=1_001)
        assert equilibria
        for eq in equilibria:
            i = space.position_index(*eq.position_1)
            j = space.position_index(*eq.position_2)
            assert payoff_1[:, j].max() <= payoff_1[i, j] + 1e-12
            assert payoff_2[:, i].max() <= payoff_2[j, i] + 1e-12

    def test_opposed_selfish_candidates_diverge_under_noise(self):
        """With heavy noise, win probabilities flatten and purely selfish
        candidates sit at their own corners."""
        space = toy_space()
        c1 = Candidate("Candidate 1", 0, selfish_weight=1.0)
        c2 = Candidate("Candidate 2", 4, selfish_weight=1.0)
        equilibria = pure_nash_equilibria(space, c1, c2, sigma=100_000.0, n_voters=101)
        assert any(
            e.position_1 == (1.0, 0.0) and e.position_2 == (0.0, 1.0)
            for e in equilibria
        )

    def test_iterated_best_response_lands_on_a_nash(self):
        space = toy_space()
        c1 = Candidate("Candidate 1", 0, selfish_weight=0.5)
        c2 = Candidate("Candidate 2", 4, selfish_weight=0.5)
        result = iterated_best_response(space, c1, c2, sigma=200.0, n_voters=1_001)
        assert result["converged"]
        final = result["path"][-1]
        profiles = {
            (e.position_1, e.position_2)
            for e in pure_nash_equilibria(space, c1, c2, sigma=200.0, n_voters=1_001)
        }
        assert (final[0], final[1]) in profiles
