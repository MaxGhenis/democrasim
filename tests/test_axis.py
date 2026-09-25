"""The redistribution axis: outcomes, ideal points, and beliefs."""

import numpy as np
import pytest

from democrasim.axis import (
    EGALITARIAN,
    AxisCandidate,
    OutcomeType,
    RedistributionAxis,
    _vote_shares,
    axis_equilibria,
    demanded_position,
    median_ideal_position,
    weighted_gini,
    win_probability_table,
)


def toy_axis(grid_points: int = 11) -> RedistributionAxis:
    """Six one-adult households; the dial taxes 10% of income and rebates it.

    Gross stakes are -10% of baseline income, so the per-adult transfer is
    $11,833.33 and the poorest four households gain.
    """
    base_income = np.array([10e3, 20e3, 30e3, 50e3, 100e3, 500e3])
    return RedistributionAxis(
        gross_full=-0.1 * base_income,
        weights=np.ones(6),
        base_income=base_income,
        hh_adults=np.ones(6),
        grid_points=grid_points,
        source="TOY six-household axis",
    )


class TestWeightedGini:
    def test_equal_incomes_have_no_inequality(self):
        assert weighted_gini(np.full(200, 40e3), np.ones(200)) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_two_point_case_matches_the_closed_form(self):
        # One unit at 0+, one at 1: Gini of an equal-weight two-point
        # distribution is (x2 - x1) / (2 * mean) * (n-1)/n = 0.5 here.
        values = np.array([1e-9, 1.0])
        assert weighted_gini(values, np.ones(2)) == pytest.approx(0.5, abs=1e-6)

    def test_weights_are_respected(self):
        values = np.array([1.0, 3.0])
        replicated = weighted_gini(np.array([1.0, 1.0, 3.0]), np.ones(3))
        weighted = weighted_gini(values, np.array([2.0, 1.0]))
        assert weighted == pytest.approx(replicated, rel=1e-12)

    def test_rejects_a_non_positive_total(self):
        with pytest.raises(ValueError):
            weighted_gini(np.array([-1.0, -2.0]), np.ones(2))


class TestAxis:
    def test_the_transfer_returns_exactly_the_revenue(self):
        axis = toy_axis()
        household_dollars = float(axis.net_full @ axis.household_weights)
        assert household_dollars == pytest.approx(0.0, abs=1e-6)

    def test_net_stakes_are_linear_and_progressive(self):
        axis = toy_axis()
        np.testing.assert_allclose(axis.net_deltas(0.5), 0.5 * axis.net_full)
        np.testing.assert_allclose(axis.net_deltas(0.0), np.zeros(6))
        # A 10%-of-income levy rebated equally per adult: every household
        # below the $118,333 break-even gains, only the top one loses.
        assert (axis.net_full[:5] > 0).all()
        assert axis.net_full[5] < 0

    def test_inequality_falls_monotonically_along_the_axis(self):
        axis = toy_axis()
        curve = axis.gini_curve
        assert (np.diff(curve) < 0).all()
        assert curve[0] > curve[-1]

    def test_incomes_are_floored(self):
        axis = RedistributionAxis(
            gross_full=np.array([-100e3, 0.0]),
            weights=np.ones(2),
            base_income=np.array([50e3, 50e3]),
            hh_adults=np.ones(2),
            income_floor=1_000.0,
        )
        assert axis.incomes(1.0).min() == pytest.approx(1_000.0)


class TestOutcomePreferences:
    def test_an_interior_target_gives_an_interior_ideal_point(self):
        axis = toy_axis(grid_points=21)
        midpoint = float(axis.gini_curve[10])
        voter = OutcomeType(target_gini=midpoint)
        assert voter.ideal_position(axis) == pytest.approx(0.5, abs=1e-9)

    def test_wanting_full_equality_is_a_monotone_preference(self):
        axis = toy_axis()
        curve = OutcomeType(target_gini=0.0).utility_curve(axis)
        assert np.argmax(curve) == len(curve) - 1
        assert (np.diff(curve) > 0).all()

    def test_disbelieving_the_policy_flattens_the_outcome_term(self):
        axis = toy_axis()
        blind = OutcomeType(target_gini=0.2, belief_slope=0.0)
        values = blind.outcome_value(axis)
        np.testing.assert_allclose(values, values[0])

    def test_selfish_weight_needs_a_household(self):
        with pytest.raises(ValueError):
            OutcomeType(selfish_weight=0.5)
        assert OutcomeType(selfish_weight=0.5, household_index=0).selfish_weight == 0.5

    def test_validation(self):
        with pytest.raises(ValueError):
            OutcomeType(share=0.0)
        with pytest.raises(ValueError):
            OutcomeType(target_gini=1.5)
        with pytest.raises(ValueError):
            OutcomeType(dollars_per_gini_point=-1.0)


class TestBeliefs:
    def test_attenuated_beliefs_demand_more_policy(self):
        """A voter who thinks the policy does less than it does asks for
        more of it — the demand-inflation channel."""
        axis = toy_axis(grid_points=41)
        target = float(axis.gini_curve[10])  # reachable at t = 0.25
        demands = [
            demanded_position(axis, target, slope) for slope in (1.0, 0.75, 0.5, 0.25)
        ]
        assert demands[0] == pytest.approx(0.25, abs=1e-9)
        assert demands == sorted(demands)
        assert demands[-1] > demands[0]

    def test_overconfident_beliefs_demand_less(self):
        axis = toy_axis(grid_points=41)
        target = float(axis.gini_curve[20])
        assert demanded_position(axis, target, 2.0) < demanded_position(
            axis, target, 1.0
        )

    def test_closed_form_matches_the_argmax(self):
        axis = toy_axis(grid_points=41)
        target = float(axis.gini_curve[16])
        for slope in (0.5, 1.0, 1.5):
            voter = OutcomeType(target_gini=target, belief_slope=slope)
            assert voter.ideal_position(axis) == pytest.approx(
                demanded_position(axis, target, slope), abs=1e-9
            )


class TestVoting:
    def test_identical_positions_are_a_coin_flip(self):
        axis = toy_axis()
        table = win_probability_table(axis, (EGALITARIAN,), n_voters=1_001)
        np.testing.assert_allclose(np.diag(table), 0.5)

    def test_the_table_is_antisymmetric(self):
        axis = toy_axis()
        table = win_probability_table(axis, (EGALITARIAN,), n_voters=1_001)
        np.testing.assert_allclose(table + table.T, 1.0, atol=1e-12)

    def test_unanimous_preference_wins_outright(self):
        axis = toy_axis()
        table = win_probability_table(axis, (EGALITARIAN,), n_voters=1_001)
        # Egalitarians want maximum redistribution: the last position beats
        # every other one.
        np.testing.assert_allclose(table[-1, :-1], 1.0)

    def test_a_selfish_voter_ignores_the_distance_between_positions(self):
        """With a linear axis and a gradient belief, both the mean and the
        spread of a selfish voter's comparison scale with the gap, so their
        direction preference is scale-free."""
        axis = toy_axis(grid_points=11)
        types = (
            OutcomeType(
                selfish_weight=1.0,
                household_index=0,
                own_noise_sd=5_000.0,
            ),
        )
        near, _ = _vote_shares(axis, types, 6, 5)
        far, _ = _vote_shares(axis, types, 10, 0)
        assert near == pytest.approx(far, rel=1e-12)

    def test_mixture_shares_are_linear(self):
        axis = toy_axis()
        low = (OutcomeType(target_gini=0.0),)
        high = (OutcomeType(target_gini=0.5),)
        mixed = (
            OutcomeType(share=0.25, target_gini=0.0),
            OutcomeType(share=0.75, target_gini=0.5),
        )
        a, _ = _vote_shares(axis, low, 8, 2)
        b, _ = _vote_shares(axis, high, 8, 2)
        m, _ = _vote_shares(axis, mixed, 8, 2)
        assert m == pytest.approx(0.25 * a + 0.75 * b, rel=1e-12)


class TestEquilibrium:
    def test_office_seekers_converge_on_the_median_ideal(self):
        """Two candidates who only want to win adopt the electorate's
        median ideal position — the Downsian benchmark, recovered on
        measured incidence."""
        axis = toy_axis(grid_points=21)
        target = float(axis.gini_curve[8])
        types = (OutcomeType(target_gini=target),)
        flat = OutcomeType(dollars_per_gini_point=0.0)
        candidates = [
            AxisCandidate(label, preferences=flat, office_rent=10_000.0)
            for label in ("Candidate 1", "Candidate 2")
        ]
        equilibria = axis_equilibria(axis, *candidates, types=types, n_voters=1_001)
        assert equilibria
        median = median_ideal_position(axis, types)
        assert any(
            eq.position_1 == pytest.approx(median)
            and eq.position_2 == pytest.approx(median)
            for eq in equilibria
        )

    def test_policy_motivated_candidates_still_face_the_electorate(self):
        axis = toy_axis(grid_points=21)
        electorate = (OutcomeType(target_gini=float(axis.gini_curve[4])),)
        candidates = (
            AxisCandidate("Candidate 1", OutcomeType(target_gini=0.0)),
            AxisCandidate(
                "Candidate 2", OutcomeType(target_gini=float(axis.gini_curve[0]))
            ),
        )
        equilibria = axis_equilibria(
            axis, *candidates, types=electorate, n_voters=1_001
        )
        assert equilibria
        for eq in equilibria:
            assert 0.0 <= eq.enacted <= 1.0

    def test_equilibria_survive_a_deviation_check(self):
        axis = toy_axis(grid_points=11)
        types = (OutcomeType(target_gini=float(axis.gini_curve[5])),)
        c1 = AxisCandidate("Candidate 1", OutcomeType(target_gini=0.0))
        c2 = AxisCandidate("Candidate 2", OutcomeType(target_gini=0.4))
        table = win_probability_table(axis, types, n_voters=1_001)
        equilibria = axis_equilibria(
            axis, c1, c2, types=types, n_voters=1_001, win_table=table
        )
        assert equilibria
        u1, u2 = c1.utility_curve(axis), c2.utility_curve(axis)
        payoff_1 = table * u1[:, None] + (1 - table) * u1[None, :]
        payoff_2 = (1 - table.T) * u2[:, None] + table.T * u2[None, :]
        positions = list(axis.positions)
        for eq in equilibria:
            i = positions.index(pytest.approx(eq.position_1))
            j = positions.index(pytest.approx(eq.position_2))
            assert payoff_1[:, j].max() <= payoff_1[i, j] + 1e-9
            assert payoff_2[:, i].max() <= payoff_2[j, i] + 1e-9
