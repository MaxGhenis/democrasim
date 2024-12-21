from typing import Dict
from democrasim.core.policy import Policy


class Candidate:
    """
    Represents a candidate with a given policy platform and optional baseline popularity.
    """

    def __init__(
        self,
        candidate_id: str,
        policy: Policy,
        baseline_popularity: float = 0.0,
    ):
        self.candidate_id = candidate_id
        self.policy = policy
        self.baseline_popularity = baseline_popularity

    def get_policy_dimensions(self) -> Dict[str, float]:
        return self.policy.dimensions

    def __repr__(self):
        return f"Candidate({self.candidate_id}, baseline={self.baseline_popularity}, policy={self.policy})"
