from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np

from democrasim.utils.distributions import sample_beta, sample_lognormal

PolicyValue = Union[float, Dict[str, float]]


@dataclass
class VoterPreferences:
    """
    Represents a voter's policy preferences across multiple dimensions:
    e.g. economic, social, environmental.
    """

    weights: Dict[str, float] = field(default_factory=dict)
    accuracies: Dict[str, float] = field(default_factory=dict)
    biases: Dict[str, float] = field(default_factory=dict)


@dataclass
class VoterParams:
    """
    Defines distributions for generating a random voter.
    """

    # Beta distribution parameters for preference weights
    weight_alpha: float = 2.0
    weight_beta: float = 5.0

    # Lognormal parameters for accuracies
    accuracy_mu: float = 0.5
    accuracy_sigma: float = 0.5

    # Beta distribution parameters for turnout
    base_turnout_alpha: float = 4.0
    base_turnout_beta: float = 6.0

    # Age range
    min_age: float = 18.0
    max_age: float = 80.0

    # Which policy dimensions exist
    policy_dimensions: List[str] = field(
        default_factory=lambda: ["economic", "social", "environmental"]
    )


class Voter:
    """
    Represents an individual voter, including:
     - Weighted preferences over multiple policy dimensions
     - Accuracy (how well they perceive each dimension's "true" value)
     - Potential biases
     - Turnout probability
     - Age
     - Potential "intervention" effects
    """

    def __init__(
        self,
        params: Optional[VoterParams] = None,
        age: Optional[float] = None,
    ):
        if params is None:
            params = VoterParams()

        self.params = params
        self.preferences = VoterPreferences()
        self.intervention_effects: Dict[str, Dict] = {}

        # Generate random dimension-level weights, accuracies, biases
        for dim in params.policy_dimensions:
            w = sample_beta(params.weight_alpha, params.weight_beta)
            a = sample_lognormal(params.accuracy_mu, params.accuracy_sigma)
            b = np.random.normal(0, 0.2)  # small random bias

            self.preferences.weights[dim] = w
            self.preferences.accuracies[dim] = a
            self.preferences.biases[dim] = b

        # Normalize sum of weights to 1
        total_w = sum(self.preferences.weights.values())
        for dim in self.preferences.weights:
            self.preferences.weights[dim] /= total_w

        # Turnout
        self.base_turnout_prob = sample_beta(
            params.base_turnout_alpha, params.base_turnout_beta
        )

        # Age
        self.age = (
            age
            if age is not None
            else np.random.uniform(params.min_age, params.max_age)
        )

    def get_effective_accuracy(self, dimension: str) -> float:
        """
        Combine base accuracy with any intervention multipliers for that dimension.
        """
        base = self.preferences.accuracies[dimension]
        multiplier = 1.0

        for eff in self.intervention_effects.values():
            if (
                "accuracy_multipliers" in eff
                and dimension in eff["accuracy_multipliers"]
            ):
                multiplier *= eff["accuracy_multipliers"][dimension]

        return base * multiplier

    def perceive_policy_value(
        self, dimension: str, true_value: float
    ) -> float:
        """
        Return perceived policy value after adding noise + bias.
        Noise is ~ Normal(0, 1/accuracy).
        """
        acc = self.get_effective_accuracy(dimension)
        # if acc is large, noise is small; if acc is small, noise is big
        noise_std = 1.0 / acc if acc > 1e-6 else 1e6
        noise = np.random.normal(0, noise_std)

        bias = self.preferences.biases[dimension]
        perceived = true_value + noise + bias

        return perceived

    def get_effective_weight(self, dimension: str) -> float:
        """
        Combine base weight with any intervention multipliers.
        """
        base = self.preferences.weights[dimension]
        multiplier = 1.0

        for eff in self.intervention_effects.values():
            if (
                "weight_multipliers" in eff
                and dimension in eff["weight_multipliers"]
            ):
                multiplier *= eff["weight_multipliers"][dimension]

        return base * multiplier

    def get_effective_turnout_prob(self) -> float:
        """
        Combine base turnout probability with intervention-based additions or multipliers.
        """
        p = self.base_turnout_prob
        for eff in self.intervention_effects.values():
            if "turnout_boost" in eff:
                # Add or multiply. As an example, let's do additive:
                p += eff["turnout_boost"]
            if "turnout_mult" in eff:
                p *= eff["turnout_mult"]

        return max(0.0, min(1.0, p))

    def will_vote(self) -> bool:
        """Determines if the voter turns out this election."""
        return np.random.random() < self.get_effective_turnout_prob()

    def evaluate_candidate(
        self,
        candidate_policy: Dict[str, float],
        candidate_id: str,
        candidate_baseline_utility: float = 0.0,
    ) -> float:
        """
        Compute the utility for the candidate's policy across dimensions.
        candidate_policy: e.g. {"economic": 1.2, "social": 0.5, ...}
        candidate_baseline_utility: captures non-policy / personal preference for the candidate
        """
        utility = candidate_baseline_utility
        for dim, true_val in candidate_policy.items():
            perceived_val = self.perceive_policy_value(dim, true_val)
            w = self.get_effective_weight(dim)
            utility += w * perceived_val
        return utility

    def apply_intervention(self, intervention_id: str, effect: Dict):
        """
        Store an intervention effect, e.g.:
          effect = {
            "accuracy_multipliers": {"economic": 1.1},
            "weight_multipliers": {"environmental": 1.2},
            "turnout_boost": 0.05
          }
        """
        self.intervention_effects[intervention_id] = effect

    def remove_intervention(self, intervention_id: str):
        """Remove a previously applied intervention's effect."""
        self.intervention_effects.pop(intervention_id, None)

    def age_one_year(self):
        """Increment this voter's age by one."""
        self.age += 1
