import numpy as np
import pytest

from democrasim import (
    NO_WINNER,
    STAR,
    Approval,
    InstantRunoff,
    Plurality,
    Score,
    VotingRule,
)


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


class TestScore:
    def test_ballot_normalized_two_options_matches_plurality(self):
        """With two options, sincere ballot-normalized score IS plurality:
        every participating ballot scores the preferred option 1, the other
        0."""
        rng = np.random.default_rng(5)
        perceived = rng.normal(0.0, 500.0, size=(40, 2))
        weights = rng.uniform(0.5, 2.0, size=40)
        score = Score().tally(perceived, weights)
        plurality = Plurality().tally(perceived, weights)
        assert score.winner == plurality.winner
        assert score.turnout == pytest.approx(plurality.turnout)
        np.testing.assert_allclose(score.shares, plurality.shares)

    def test_ballot_normalized_hand_check_three_options(self):
        perceived = np.array([[0.0, 100.0, 50.0], [0.0, -80.0, -40.0]])
        weights = np.array([1.0, 1.0])
        tally = Score().tally(perceived, weights)
        # Voter 1: scores (0, 1, .5); voter 2: (1, 0, .5) -> totals (1, 1, 1)
        np.testing.assert_allclose(tally.shares, [0.5, 0.5, 0.5])
        assert tally.winner == 0  # exact tie resolves to the lowest index
        assert tally.turnout == 1.0

    def test_indifferent_voters_abstain(self):
        perceived = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 5.0]])
        tally = Score().tally(perceived, np.array([9.0, 1.0]))
        assert tally.turnout == pytest.approx(0.1)
        assert tally.winner == 1

    def test_stakes_normalization_centers_and_saturates(self):
        perceived = np.array([[0.0, 500.0, -2_000.0]])
        tally = Score(normalize="stakes", cap=1_000.0).tally(perceived, np.ones(1))
        np.testing.assert_allclose(tally.shares, [0.5, 0.75, 0.0])
        assert tally.winner == 1

    def test_levels_round_to_ballot_marks(self):
        perceived = np.array([[0.0, 61.0, 100.0]])
        tally = Score(levels=5).tally(perceived, np.ones(1))
        np.testing.assert_allclose(tally.shares, [0.0, 0.6, 1.0])

    def test_validation(self):
        with pytest.raises(ValueError):
            Score(normalize="range")
        with pytest.raises(ValueError):
            Score(cap=0.0)
        with pytest.raises(ValueError):
            Score(levels=0)

    def test_satisfies_the_protocol(self):
        assert isinstance(Score(), VotingRule)
        assert isinstance(STAR(), VotingRule)


class TestSTAR:
    def test_runoff_can_overturn_the_score_leader(self):
        """Two mild fans of option 1 lose the runoff to three voters who
        narrowly prefer option 0 - scores pick the finalists, majorities
        pick between them."""
        perceived = np.array(
            [[100.0, 90.0, 0.0]] * 3 + [[0.0, 100.0, 20.0]] * 2,
        )
        tally = STAR(levels=5).tally(perceived, np.ones(5))
        # Score totals (0-1 scale): option 1 leads with 4.4 (3 x 0.8 + 2 x 1)
        # over option 0's 3.0 - but the runoff between them splits 3-2 for
        # option 0, whose supporters scored it strictly higher.
        assert tally.winner == 0

    def test_two_options_is_a_majority_vote(self):
        perceived = np.array([[10.0, 0.0]] * 3 + [[0.0, 500.0]] * 2)
        tally = STAR().tally(perceived, np.ones(5))
        assert tally.winner == 0
        assert tally.turnout == 1.0

    def test_runoff_tie_falls_back_to_the_score_leader(self):
        perceived = np.array([[100.0, 60.0, 0.0], [0.0, 60.0, 100.0]])
        tally = STAR(levels=5).tally(perceived, np.ones(2))
        # Score totals: option 1 leads (1.2 vs 1.0 vs 1.0); finalists are
        # options 1 and 0, and their runoff ties 1-1 - the tie falls back
        # to the score leader.
        assert tally.winner == 1
        assert tally.turnout == 1.0

    def test_indifferent_ballots_stay_home(self):
        perceived = np.array([[5.0, 5.0, 5.0], [0.0, 10.0, 20.0]])
        tally = STAR().tally(perceived, np.array([100.0, 1.0]))
        assert tally.winner == 2
        assert tally.turnout == pytest.approx(1.0 / 101.0)
