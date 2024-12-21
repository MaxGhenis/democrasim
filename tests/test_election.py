import unittest
from democrasim.core.election import run_election
from democrasim.core.voter import Voter


class TestElection(unittest.TestCase):
    def test_run_election(self):
        # create 5 voters
        voters = [Voter() for _ in range(5)]
        candidate_policies = {"A": {"economic": 1.0}, "B": {"economic": 0.8}}
        winner = run_election(voters, candidate_policies)
        self.assertIn(winner, ["A", "B"])
