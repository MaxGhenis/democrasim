"""Tests for strategic candidate model."""

import unittest
import numpy as np
from democrasim.core.strategic import (
    PolicyOutcomeFunction,
    StrategicVoter,
    StrategicCandidate,
    compute_win_probability_analytical,
    find_best_response,
    find_equilibrium,
    generate_strategic_voters,
    run_strategic_election,
)


class TestPolicyOutcomeFunction(unittest.TestCase):
    def test_default_function(self):
        f = PolicyOutcomeFunction()
        # Default: g = 0.6 - 0.4*τ
        self.assertAlmostEqual(f(0), 0.6)
        self.assertAlmostEqual(f(1), 0.2)
        self.assertAlmostEqual(f(0.5), 0.4)

    def test_custom_function(self):
        f = PolicyOutcomeFunction(func=lambda tau: tau ** 2)
        self.assertAlmostEqual(f(0), 0)
        self.assertAlmostEqual(f(0.5), 0.25)
        self.assertAlmostEqual(f(1), 1)

    def test_clipping(self):
        # Function that would go out of bounds
        f = PolicyOutcomeFunction(func=lambda tau: 2 * tau - 0.5)
        self.assertAlmostEqual(f(0), 0)  # Clipped from -0.5
        self.assertAlmostEqual(f(1), 1)  # Clipped from 1.5


class TestStrategicVoter(unittest.TestCase):
    def test_deterministic_vote(self):
        """With zero noise, voter should vote deterministically."""
        f = PolicyOutcomeFunction()
        voter = StrategicVoter(ideal_outcome=0.3, noise_std=0)

        # τ=0.75 → g=0.3, τ=0.25 → g=0.5
        # Voter with ideal 0.3 should prefer τ=0.75
        np.random.seed(42)
        vote = voter.vote(tau_a=0.75, tau_b=0.25, policy_func=f)
        self.assertEqual(vote, "A")

    def test_utility_function(self):
        voter = StrategicVoter(ideal_outcome=0.4, noise_std=0.05)
        self.assertEqual(voter.utility(0.4), 0)  # At ideal
        self.assertAlmostEqual(voter.utility(0.5), -0.01)  # Distance of 0.1


class TestStrategicCandidate(unittest.TestCase):
    def test_utility(self):
        candidate = StrategicCandidate("A", ideal_outcome=0.3)
        self.assertEqual(candidate.utility(0.3), 0)
        self.assertAlmostEqual(candidate.utility(0.4), -0.01)


class TestWinProbability(unittest.TestCase):
    def test_no_noise_deterministic(self):
        """With zero noise, win probability should be 0 or 1."""
        f = PolicyOutcomeFunction()
        # All voters want g=0.35
        voters = [StrategicVoter(ideal_outcome=0.35, noise_std=0) for _ in range(10)]

        # τ=0.625 → g=0.35, τ=0.25 → g=0.5
        # All voters should vote for A
        p_a = compute_win_probability_analytical(
            tau_a=0.625, tau_b=0.25, voters=voters, policy_func=f
        )
        self.assertGreater(p_a, 0.99)

    def test_symmetric_voters_fifty_fifty(self):
        """With symmetric voter distribution and equal distances, p ≈ 0.5."""
        f = PolicyOutcomeFunction()
        # Voters uniformly distributed, both candidates equidistant from median
        np.random.seed(42)
        voters = [
            StrategicVoter(ideal_outcome=0.4 + 0.1 * (i - 5) / 5, noise_std=0.05)
            for i in range(11)
        ]  # ideals from 0.3 to 0.5, median at 0.4

        # Both candidates at equal distance from median ideal of 0.4
        # τ_a=0.4 → g=0.44, τ_b=0.6 → g=0.36
        # Distances: |0.44-0.4|=0.04, |0.36-0.4|=0.04
        p_a = compute_win_probability_analytical(
            tau_a=0.4, tau_b=0.6, voters=voters, policy_func=f
        )
        # Should be close to 0.5 given symmetry
        self.assertGreater(p_a, 0.3)
        self.assertLess(p_a, 0.7)


class TestEquilibrium(unittest.TestCase):
    def test_convergence(self):
        """Equilibrium finder should converge."""
        np.random.seed(42)
        f = PolicyOutcomeFunction()
        voters = generate_strategic_voters(n=100, ideal_mean=0.4, noise_std=0.05)

        candidate_a = StrategicCandidate("A", ideal_outcome=0.3)
        candidate_b = StrategicCandidate("B", ideal_outcome=0.5)

        tau_a, tau_b, info = find_equilibrium(candidate_a, candidate_b, voters, f)

        self.assertTrue(info["converged"])
        self.assertGreaterEqual(tau_a, 0)
        self.assertLessEqual(tau_a, 1)
        self.assertGreaterEqual(tau_b, 0)
        self.assertLessEqual(tau_b, 1)

    def test_candidates_move_toward_center(self):
        """Candidates should moderate from their ideal positions."""
        np.random.seed(42)
        f = PolicyOutcomeFunction()
        # Voters centered at g=0.4
        voters = generate_strategic_voters(n=100, ideal_mean=0.4, noise_std=0.03)

        candidate_a = StrategicCandidate("A", ideal_outcome=0.25)  # Wants low gini
        candidate_b = StrategicCandidate("B", ideal_outcome=0.55)  # Wants higher gini

        tau_a, tau_b, info = find_equilibrium(candidate_a, candidate_b, voters, f)

        # A's ideal τ would be (0.6-0.25)/0.4 = 0.875
        # B's ideal τ would be (0.6-0.55)/0.4 = 0.125
        # Both should move toward center (τ≈0.5 for g≈0.4)

        ideal_tau_a = (0.6 - 0.25) / 0.4  # 0.875
        ideal_tau_b = (0.6 - 0.55) / 0.4  # 0.125

        # A should have moved down from 0.875 toward center
        self.assertLess(tau_a, ideal_tau_a)
        # B should have moved up from 0.125 toward center
        self.assertGreater(tau_b, ideal_tau_b)


class TestFullElection(unittest.TestCase):
    def test_run_election(self):
        """Full election should return sensible results."""
        np.random.seed(42)
        f = PolicyOutcomeFunction()
        voters = generate_strategic_voters(n=50, ideal_mean=0.4, noise_std=0.05)

        candidate_a = StrategicCandidate("A", ideal_outcome=0.3)
        candidate_b = StrategicCandidate("B", ideal_outcome=0.5)

        result = run_strategic_election(voters, candidate_a, candidate_b, f)

        self.assertIn(result["winner"], ["A", "B"])
        self.assertEqual(sum(result["votes"].values()), 50)
        self.assertGreaterEqual(result["realized_outcome"], 0)
        self.assertLessEqual(result["realized_outcome"], 1)
        self.assertIn("equilibrium_info", result)


if __name__ == "__main__":
    unittest.main()
