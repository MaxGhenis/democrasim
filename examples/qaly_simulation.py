"""
qaly_simulation.py

Example usage of democrasim to simulate an election where
the outcome is measured in QALYs.
"""

import numpy as np
from democrasim.core.voter import Voter
from democrasim.core.election import run_election
from democrasim.models.candidate import Candidate
from democrasim.core.policy import Policy
from democrasim.models.outcome import compute_outcome


def main():
    # create some voters
    voters = [Voter() for _ in range(1000)]

    # define candidate policies
    polA = Policy("A_policy", {"economic": 1.0, "health": 1.0})
    polB = Policy("B_policy", {"economic": 1.2, "health": 0.8})

    candA = Candidate("A", polA, baseline_popularity=0.1)
    candB = Candidate("B", polB, baseline_popularity=-0.05)

    candidate_policies = {
        candA.candidate_id: candA.get_policy_dimensions(),
        candB.candidate_id: candB.get_policy_dimensions(),
    }
    candidate_baseline = {
        candA.candidate_id: candA.baseline_popularity,
        candB.candidate_id: candB.baseline_popularity,
    }

    winner = run_election(voters, candidate_policies, candidate_baseline)
    # Suppose outcome if A wins = 10k QALYs, B wins = 8k QALYs
    outcomes = {"A": 10_000, "B": 8_000}
    total_qalys = compute_outcome(winner, outcomes)
    print(f"Winner is {winner}, total QALYs = {total_qalys}")


if __name__ == "__main__":
    main()
