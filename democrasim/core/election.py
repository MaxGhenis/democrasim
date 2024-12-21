import numpy as np
from typing import Dict, List
from democrasim.core.voter import Voter


def run_election(
    voters: List[Voter],
    candidate_policies: Dict[str, Dict[str, float]],
    candidate_baseline: Dict[str, float] = None,
) -> str:
    """
    Conduct a single election among all voters, returning the winning candidate's ID.

    Args:
        voters: list of Voter objects
        candidate_policies: e.g. {
            'A': {'economic': 1.0, 'environmental': 0.5},
            'B': {'economic': 0.8, 'environmental': 1.2}
        }
        candidate_baseline: e.g. {'A': 0.1, 'B': -0.2} capturing non-policy preference
    """
    if candidate_baseline is None:
        # default baseline preference is 0 for each candidate
        candidate_baseline = {cid: 0.0 for cid in candidate_policies.keys()}

    votes = {cid: 0 for cid in candidate_policies.keys()}

    for voter in voters:
        if voter.will_vote():
            # Evaluate each candidate
            best_candidate = None
            best_utility = -1e9  # or float('-inf')

            for cid, policies in candidate_policies.items():
                base_utility = candidate_baseline.get(cid, 0.0)
                u = voter.evaluate_candidate(
                    policies,
                    candidate_id=cid,
                    candidate_baseline_utility=base_utility,
                )
                if u > best_utility:
                    best_utility = u
                    best_candidate = cid

            votes[best_candidate] += 1

    # pick the candidate with the most votes
    winner = max(votes.items(), key=lambda x: x[1])[0]
    return winner
