import numpy as np
import pytest

from democrasim import NO_WINNER, Approval, InstantRunoff, Plurality


class TestPlurality:
    def test_weighted_majority_wins(self):
        perceived = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
        tally = Plurality().tally(perceived, np.array([5.0, 2.0, 2.0]))
        assert tally.winner == 0
        assert tally.shares[0] == pytest.approx(5 / 9)
        assert tally.turnout == 1.0

    def test_exact_indifference_abstains(self):
        perceived = np.array([[0.0, 0.0], [10.0, 0.0]])
        tally = Plurality().tally(perceived, np.array([1.0, 1.0]))
        assert tally.turnout == pytest.approx(0.5)
        assert tally.winner == 0

    def test_abstain_below_threshold(self):
        perceived = np.array([[100.0, 0.0], [30.0, 0.0]])
        tally = Plurality(abstain_below=50.0).tally(perceived, np.array([1.0, 1.0]))
        assert tally.turnout == pytest.approx(0.5)

    def test_no_ballots_means_no_winner(self):
        perceived = np.zeros((3, 2))
        tally = Plurality().tally(perceived, np.ones(3))
        assert tally.winner == NO_WINNER
        assert tally.turnout == 0.0

    def test_three_policies(self):
        perceived = np.array([[1.0, 5.0, 2.0], [0.0, 4.0, 9.0]])
        tally = Plurality().tally(perceived, np.array([3.0, 1.0]))
        assert tally.winner == 1


class TestApproval:
    def test_approves_above_status_quo(self):
        perceived = np.array([[100.0, 50.0], [-10.0, 40.0], [-5.0, -5.0]])
        tally = Approval().tally(perceived, np.ones(3))
        assert tally.shares[0] == pytest.approx(1 / 3)
        assert tally.shares[1] == pytest.approx(2 / 3)
        assert tally.winner == 1
        assert tally.turnout == pytest.approx(2 / 3)

    def test_threshold_raises_the_bar(self):
        perceived = np.array([[100.0, 50.0]])
        tally = Approval(threshold=75.0).tally(perceived, np.ones(1))
        assert tally.shares[0] == 1.0
        assert tally.shares[1] == 0.0

    def test_multiple_approvals_per_ballot(self):
        perceived = np.array([[100.0, 50.0]])
        tally = Approval().tally(perceived, np.ones(1))
        # Approval need not sum to 1 across policies.
        assert tally.shares.sum() == pytest.approx(2.0)


class TestInstantRunoff:
    def test_two_policies_reduce_to_plurality(self):
        perceived = np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
        tally = InstantRunoff().tally(perceived, np.array([5.0, 2.0, 2.0]))
        assert tally.winner == 0

    def test_elimination_transfers_ballots(self):
        # Policy 2 is eliminated first; its supporter's next choice (1)
        # decides the election.
        perceived = np.array(
            [
                [3.0, 2.0, 1.0],
                [3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0],
                [1.0, 3.0, 2.0],
                [2.0, 3.0, 1.0],
            ]
        )
        tally = InstantRunoff().tally(perceived, np.ones(5))
        assert tally.winner == 1
        # Shares report the first round, before transfers.
        assert tally.shares[2] == pytest.approx(0.2)

    def test_elimination_tie_eliminates_lowest_policy_index(self):
        # Policies 0 and 1 each start with weight 2, behind policy 2's weight
        # 3. Eliminating policy 0 transfers its weight to policy 1, which wins.
        perceived = np.array(
            [
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [2.0, 1.0, 3.0],
                [2.0, 1.0, 3.0],
            ]
        )
        tally = InstantRunoff().tally(perceived, np.array([2.0, 2.0, 1.0, 2.0]))
        assert tally.winner == 1
        np.testing.assert_allclose(tally.shares, [2 / 7, 2 / 7, 3 / 7])

    def test_indifferent_abstention_matches_plurality(self):
        perceived = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        weights = np.array([5.0, 2.0, 3.0])

        plurality = Plurality().tally(perceived, weights)
        runoff = InstantRunoff(indifferent_abstain=True).tally(perceived, weights)

        assert runoff.winner == plurality.winner == 1
        assert runoff.turnout == pytest.approx(plurality.turnout)
        np.testing.assert_allclose(runoff.shares, plurality.shares)

    def test_default_indifference_force_votes_for_policy_zero(self):
        perceived = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        weights = np.array([5.0, 2.0, 3.0])

        tally = InstantRunoff(indifferent_abstain=False).tally(perceived, weights)

        assert tally.winner == 0
        assert tally.turnout == 1.0
        np.testing.assert_allclose(tally.shares, [0.7, 0.3])

    def test_immediate_majority_short_circuits(self):
        perceived = np.array([[9.0, 1.0, 0.0]] * 3 + [[0.0, 9.0, 1.0]])
        tally = InstantRunoff().tally(perceived, np.ones(4))
        assert tally.winner == 0
        assert tally.turnout == 1.0
