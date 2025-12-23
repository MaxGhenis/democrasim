"""
Strategic model with endogenous candidate positioning.

Models a game where:
- Voters have ideal outcomes and vote for the candidate whose perceived
  policy outcome is closest to their ideal
- Candidates choose policy positions to maximize expected policy outcome,
  considering both P(win) and the outcome if they win vs. lose
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm


@dataclass
class PolicyOutcomeFunction:
    """
    Maps policy instrument τ ∈ [0,1] to outcome g ∈ [0,1].

    Default is a simple linear relationship, but can be customized
    (e.g., with PolicyEngine for tax→gini mapping).
    """

    func: Callable[[float], float] = None

    def __post_init__(self):
        if self.func is None:
            # Default: higher tax → lower gini (more redistribution)
            # g(τ) = 0.6 - 0.4*τ  (gini ranges from 0.2 to 0.6)
            self.func = lambda tau: 0.6 - 0.4 * tau

    def __call__(self, tau: float) -> float:
        """Return true outcome for policy τ."""
        return np.clip(self.func(tau), 0, 1)

    def with_noise(self, tau: float, noise_std: float) -> float:
        """Return perceived outcome with Gaussian noise."""
        true_g = self(tau)
        perceived = true_g + np.random.normal(0, noise_std)
        return np.clip(perceived, 0, 1)


@dataclass
class StrategicVoter:
    """
    A voter with:
    - Ideal outcome g* (their preferred Gini index)
    - Noise level σ for perceiving policy→outcome mapping
    """

    ideal_outcome: float  # g* ∈ [0,1]
    noise_std: float = 0.05  # σ for perception noise

    def perceive_outcome(
        self, tau: float, policy_func: PolicyOutcomeFunction
    ) -> float:
        """Perceive the outcome of policy τ with noise."""
        return policy_func.with_noise(tau, self.noise_std)

    def utility(self, outcome: float) -> float:
        """Quadratic loss from ideal outcome."""
        return -((outcome - self.ideal_outcome) ** 2)

    def vote(
        self,
        tau_a: float,
        tau_b: float,
        policy_func: PolicyOutcomeFunction,
    ) -> str:
        """Vote for candidate whose perceived outcome is closer to ideal."""
        perceived_a = self.perceive_outcome(tau_a, policy_func)
        perceived_b = self.perceive_outcome(tau_b, policy_func)

        dist_a = abs(perceived_a - self.ideal_outcome)
        dist_b = abs(perceived_b - self.ideal_outcome)

        return "A" if dist_a < dist_b else "B"


@dataclass
class StrategicCandidate:
    """
    A policy-motivated candidate who:
    - Has ideal outcome g* (their preferred Gini index)
    - Chooses policy τ to maximize expected outcome
    - May have noise in perceiving the policy→outcome mapping
    """

    candidate_id: str
    ideal_outcome: float  # g* ∈ [0,1]
    noise_std: float = 0.0  # Candidate's perception noise (expertise)

    def utility(self, outcome: float) -> float:
        """Quadratic loss from ideal outcome."""
        return -((outcome - self.ideal_outcome) ** 2)

    def perceived_outcome(
        self, tau: float, policy_func: PolicyOutcomeFunction
    ) -> float:
        """Candidate's belief about outcome from policy τ."""
        if self.noise_std > 0:
            return policy_func.with_noise(tau, self.noise_std)
        return policy_func(tau)


def compute_win_probability(
    tau_a: float,
    tau_b: float,
    voters: List[StrategicVoter],
    policy_func: PolicyOutcomeFunction,
    n_samples: int = 1000,
) -> float:
    """
    Estimate P(A wins) via Monte Carlo simulation.

    Each voter's perception is stochastic, so we simulate many elections.
    """
    a_wins = 0

    for _ in range(n_samples):
        votes_a = sum(
            1 for v in voters
            if v.vote(tau_a, tau_b, policy_func) == "A"
        )
        if votes_a > len(voters) / 2:
            a_wins += 1

    return a_wins / n_samples


def compute_win_probability_analytical(
    tau_a: float,
    tau_b: float,
    voters: List[StrategicVoter],
    policy_func: PolicyOutcomeFunction,
) -> float:
    """
    Compute P(A wins) analytically (approximate).

    For each voter, compute P(voter votes A) based on their noise distribution,
    then use normal approximation for sum of Bernoulli trials.
    """
    g_a = policy_func(tau_a)
    g_b = policy_func(tau_b)

    # For each voter, P(vote A) = P(|g_a + η - g*| < |g_b + η - g*|)
    # This is complex with different noise per voter, so we approximate

    vote_probs = []
    for v in voters:
        # Approximate: P(vote A) based on distance and noise
        dist_a = abs(g_a - v.ideal_outcome)
        dist_b = abs(g_b - v.ideal_outcome)

        if v.noise_std < 1e-9:
            # No noise: deterministic vote
            p_a = 1.0 if dist_a < dist_b else 0.0
        else:
            # With noise: use probit-style approximation
            # P(vote A) ≈ Φ((dist_b - dist_a) / (sqrt(2) * σ))
            diff = dist_b - dist_a
            p_a = norm.cdf(diff / (np.sqrt(2) * v.noise_std))

        vote_probs.append(p_a)

    # Expected votes for A and variance
    expected_votes_a = sum(vote_probs)
    variance = sum(p * (1 - p) for p in vote_probs)

    # P(A wins) = P(votes_A > N/2) ≈ Φ((E[votes_A] - N/2) / sqrt(Var))
    n = len(voters)
    if variance < 1e-9:
        return 1.0 if expected_votes_a > n / 2 else 0.0

    z = (expected_votes_a - n / 2) / np.sqrt(variance)
    return norm.cdf(z)


def candidate_expected_utility(
    tau_own: float,
    tau_opponent: float,
    candidate: StrategicCandidate,
    voters: List[StrategicVoter],
    policy_func: PolicyOutcomeFunction,
    is_candidate_a: bool = True,
    use_analytical: bool = True,
) -> float:
    """
    Compute candidate's expected utility:
    E[U] = P(win) * u(g(τ_own)) + (1-P(win)) * u(g(τ_opponent))
    """
    if is_candidate_a:
        p_win = (
            compute_win_probability_analytical(tau_own, tau_opponent, voters, policy_func)
            if use_analytical
            else compute_win_probability(tau_own, tau_opponent, voters, policy_func)
        )
    else:
        p_win = 1 - (
            compute_win_probability_analytical(tau_opponent, tau_own, voters, policy_func)
            if use_analytical
            else compute_win_probability(tau_opponent, tau_own, voters, policy_func)
        )

    # Outcomes (candidate may perceive with noise, but for EU calculation
    # we typically use their belief about the mapping)
    g_own = candidate.perceived_outcome(tau_own, policy_func)
    g_opponent = candidate.perceived_outcome(tau_opponent, policy_func)

    u_win = candidate.utility(g_own)
    u_lose = candidate.utility(g_opponent)

    return p_win * u_win + (1 - p_win) * u_lose


def find_best_response(
    candidate: StrategicCandidate,
    tau_opponent: float,
    voters: List[StrategicVoter],
    policy_func: PolicyOutcomeFunction,
    is_candidate_a: bool = True,
) -> float:
    """
    Find candidate's best response given opponent's policy.

    Returns τ* that maximizes expected utility.
    """
    def neg_utility(tau):
        return -candidate_expected_utility(
            tau, tau_opponent, candidate, voters, policy_func, is_candidate_a
        )

    result = minimize_scalar(neg_utility, bounds=(0, 1), method='bounded')
    return result.x


def find_equilibrium(
    candidate_a: StrategicCandidate,
    candidate_b: StrategicCandidate,
    voters: List[StrategicVoter],
    policy_func: PolicyOutcomeFunction,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> Tuple[float, float, dict]:
    """
    Find Nash equilibrium in policy positions via iterated best response.

    Returns:
        (τ_a*, τ_b*, info_dict)
    """
    # Initialize at candidates' "naive" ideal policies
    # (the τ that would achieve their ideal g, ignoring winning)
    # For default f: g = 0.6 - 0.4*τ → τ = (0.6 - g) / 0.4

    def ideal_tau(g_star):
        # Inverse of default function, clamped to [0,1]
        return np.clip((0.6 - g_star) / 0.4, 0, 1)

    tau_a = ideal_tau(candidate_a.ideal_outcome)
    tau_b = ideal_tau(candidate_b.ideal_outcome)

    history = [(tau_a, tau_b)]

    for i in range(max_iterations):
        # Best response for A given B's position
        tau_a_new = find_best_response(
            candidate_a, tau_b, voters, policy_func, is_candidate_a=True
        )

        # Best response for B given A's new position
        tau_b_new = find_best_response(
            candidate_b, tau_a_new, voters, policy_func, is_candidate_a=False
        )

        history.append((tau_a_new, tau_b_new))

        # Check convergence
        if abs(tau_a_new - tau_a) < tolerance and abs(tau_b_new - tau_b) < tolerance:
            return tau_a_new, tau_b_new, {
                "converged": True,
                "iterations": i + 1,
                "history": history,
            }

        tau_a, tau_b = tau_a_new, tau_b_new

    return tau_a, tau_b, {
        "converged": False,
        "iterations": max_iterations,
        "history": history,
    }


def generate_strategic_voters(
    n: int,
    ideal_mean: float = 0.4,
    ideal_std: float = 0.15,
    noise_std: float = 0.05,
) -> List[StrategicVoter]:
    """
    Generate a population of strategic voters.

    Args:
        n: Number of voters
        ideal_mean: Mean of ideal outcome distribution
        ideal_std: Std of ideal outcome distribution
        noise_std: Perception noise (same for all voters, or pass array)
    """
    ideals = np.random.normal(ideal_mean, ideal_std, n)
    ideals = np.clip(ideals, 0, 1)

    if isinstance(noise_std, (int, float)):
        noise_stds = [noise_std] * n
    else:
        noise_stds = noise_std

    return [
        StrategicVoter(ideal_outcome=g, noise_std=σ)
        for g, σ in zip(ideals, noise_stds)
    ]


def run_strategic_election(
    voters: List[StrategicVoter],
    candidate_a: StrategicCandidate,
    candidate_b: StrategicCandidate,
    policy_func: PolicyOutcomeFunction,
    find_equilibrium_first: bool = True,
) -> dict:
    """
    Run a full strategic election simulation.

    If find_equilibrium_first=True, candidates optimize their positions.
    Otherwise, uses their "naive" ideal policies.

    Returns dict with results including winner, outcomes, and equilibrium info.
    """
    if find_equilibrium_first:
        tau_a, tau_b, eq_info = find_equilibrium(
            candidate_a, candidate_b, voters, policy_func
        )
    else:
        # Naive: each candidate picks τ closest to their ideal outcome
        tau_a = np.clip((0.6 - candidate_a.ideal_outcome) / 0.4, 0, 1)
        tau_b = np.clip((0.6 - candidate_b.ideal_outcome) / 0.4, 0, 1)
        eq_info = {"converged": None, "iterations": 0, "history": []}

    # Run the election
    votes = {"A": 0, "B": 0}
    for v in voters:
        vote = v.vote(tau_a, tau_b, policy_func)
        votes[vote] += 1

    winner = "A" if votes["A"] > votes["B"] else "B"
    winning_tau = tau_a if winner == "A" else tau_b
    realized_outcome = policy_func(winning_tau)

    # Compute expected outcome (averaging over voter noise)
    p_a_wins = compute_win_probability_analytical(tau_a, tau_b, voters, policy_func)
    expected_outcome = p_a_wins * policy_func(tau_a) + (1 - p_a_wins) * policy_func(tau_b)

    # Median voter's ideal (for comparison)
    median_ideal = np.median([v.ideal_outcome for v in voters])

    return {
        "winner": winner,
        "votes": votes,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "outcome_if_a": policy_func(tau_a),
        "outcome_if_b": policy_func(tau_b),
        "realized_outcome": realized_outcome,
        "expected_outcome": expected_outcome,
        "p_a_wins": p_a_wins,
        "median_voter_ideal": median_ideal,
        "equilibrium_info": eq_info,
        "candidate_a_ideal": candidate_a.ideal_outcome,
        "candidate_b_ideal": candidate_b.ideal_outcome,
    }
