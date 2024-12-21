"""
inequality_simulation.py

A quick example of using the democrasim package with an inequality metric.
"""

import numpy as np
from democrasim.core.voter import Voter
from democrasim.core.election import run_election
from democrasim.models.candidate import Candidate
from democrasim.core.policy import Policy
from democrasim.metrics.inequality import gini_coefficient


def main():
    # create voters
    voters = [Voter() for _ in range(500)]

    # define candidate policies
    candA = Candidate(
        "A",
        Policy("A_policy", {"economic": 1.0, "social": 0.5}),
        baseline_popularity=0.0,
    )
    candB = Candidate(
        "B",
        Policy("B_policy", {"economic": 0.8, "social": 1.2}),
        baseline_popularity=0.1,
    )

    candidate_policies = {
        candA.candidate_id: candA.get_policy_dimensions(),
        candB.candidate_id: candB.get_policy_dimensions(),
    }

    # run election
    winner = run_election(voters, candidate_policies)
    # Suppose if A wins => we have some distribution of incomes
    if winner == "A":
        incomes = np.random.normal(50_000, 10_000, 5000)
    else:
        incomes = np.random.normal(45_000, 12_000, 5000)

    gini = gini_coefficient(incomes)
    print(f"Winner = {winner}, Gini = {gini:.3f}")


if __name__ == "__main__":
    main()
