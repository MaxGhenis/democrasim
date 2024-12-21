import unittest
import numpy as np
from democrasim.core.voter import Voter, VoterParams


class TestVoter(unittest.TestCase):
    def test_voter_init(self):
        params = VoterParams(
            weight_alpha=2.0,
            weight_beta=5.0,
            accuracy_mu=0.5,
            accuracy_sigma=0.5,
        )
        voter = Voter(params=params)
        self.assertIsNotNone(voter.preferences)
        self.assertTrue(len(voter.preferences.weights) > 0)
        self.assertTrue(0 <= voter.base_turnout_prob <= 1)

    def test_voter_evaluation(self):
        voter = Voter()
        policy = {"economic": 1.0, "social": 0.5, "environmental": 2.0}
        utility = voter.evaluate_candidate(policy, candidate_id="X")
        # We can't predict the exact utility but can check that it runs
        self.assertIsInstance(utility, float)

    def test_turnout_with_intervention(self):
        voter = Voter()
        base_p = voter.get_effective_turnout_prob()
        voter.apply_intervention("civics_program", {"turnout_boost": 0.1})
        new_p = voter.get_effective_turnout_prob()
        self.assertTrue(new_p > base_p)


if __name__ == "__main__":
    unittest.main()
