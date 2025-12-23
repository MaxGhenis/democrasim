"""
strategic_equilibrium.py

Demonstrates how voter information (noise) affects equilibrium candidate
positioning and policy outcomes.

Key finding: Lower voter noise → candidates converge to median → better
aggregate welfare. Higher noise → candidates stay extreme → polarization.
"""

import numpy as np
from democrasim.core.strategic import (
    PolicyOutcomeFunction,
    StrategicCandidate,
    generate_strategic_voters,
    find_equilibrium,
    run_strategic_election,
)


def run_noise_experiment():
    """
    Compare equilibrium outcomes under different voter noise levels.
    """
    np.random.seed(42)

    # Policy function: higher tax → lower gini
    f = PolicyOutcomeFunction()

    # Candidates with divergent preferences
    candidate_a = StrategicCandidate("A", ideal_outcome=0.25)  # Wants equality
    candidate_b = StrategicCandidate("B", ideal_outcome=0.55)  # Tolerates inequality

    # Voter population centered at g*=0.4 (moderate preference)
    n_voters = 200
    voter_ideal_mean = 0.40
    voter_ideal_std = 0.10

    noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20]

    print("=" * 70)
    print("STRATEGIC EQUILIBRIUM: Effect of Voter Information")
    print("=" * 70)
    print(f"\nCandidate A ideal outcome (Gini): {candidate_a.ideal_outcome}")
    print(f"Candidate B ideal outcome (Gini): {candidate_b.ideal_outcome}")
    print(f"Median voter ideal outcome: {voter_ideal_mean}")
    print(f"\nPolicy function: Gini = 0.6 - 0.4 * τ (tax rate)")
    print("  → τ=0 gives Gini=0.6 (high inequality)")
    print("  → τ=1 gives Gini=0.2 (low inequality)")

    # Ideal τ for each candidate (what they'd choose if certain to win)
    ideal_tau_a = (0.6 - candidate_a.ideal_outcome) / 0.4
    ideal_tau_b = (0.6 - candidate_b.ideal_outcome) / 0.4
    median_tau = (0.6 - voter_ideal_mean) / 0.4

    print(f"\nNaive (certain-win) policies:")
    print(f"  A would choose τ={ideal_tau_a:.3f} → Gini={candidate_a.ideal_outcome}")
    print(f"  B would choose τ={ideal_tau_b:.3f} → Gini={candidate_b.ideal_outcome}")
    print(f"  Median voter wants τ={median_tau:.3f} → Gini={voter_ideal_mean}")

    print("\n" + "-" * 70)
    print("EQUILIBRIUM ANALYSIS BY VOTER NOISE LEVEL")
    print("-" * 70)

    results = []

    for noise in noise_levels:
        voters = generate_strategic_voters(
            n=n_voters,
            ideal_mean=voter_ideal_mean,
            ideal_std=voter_ideal_std,
            noise_std=noise,
        )

        tau_a, tau_b, info = find_equilibrium(
            candidate_a, candidate_b, voters, f
        )

        g_a = f(tau_a)
        g_b = f(tau_b)

        # Distance from median voter's ideal
        dist_a = abs(g_a - voter_ideal_mean)
        dist_b = abs(g_b - voter_ideal_mean)

        # Run election to get outcome
        result = run_strategic_election(
            voters, candidate_a, candidate_b, f, find_equilibrium_first=True
        )

        # Welfare: negative distance from median ideal (higher = better)
        welfare = -abs(result["expected_outcome"] - voter_ideal_mean)

        results.append({
            "noise": noise,
            "tau_a": tau_a,
            "tau_b": tau_b,
            "g_a": g_a,
            "g_b": g_b,
            "p_a_wins": result["p_a_wins"],
            "expected_outcome": result["expected_outcome"],
            "welfare": welfare,
            "converged": info["converged"],
            "iterations": info["iterations"],
        })

        print(f"\nVoter noise σ = {noise:.2f}:")
        print(f"  Equilibrium: τ_A={tau_a:.3f} (→g={g_a:.3f}), τ_B={tau_b:.3f} (→g={g_b:.3f})")
        print(f"  Distance from median: A={dist_a:.3f}, B={dist_b:.3f}")
        print(f"  P(A wins) = {result['p_a_wins']:.2%}")
        print(f"  Expected Gini = {result['expected_outcome']:.3f}")
        print(f"  Welfare (neg. dist from median) = {welfare:.4f}")
        print(f"  Converged in {info['iterations']} iterations")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Information and Policy Quality")
    print("=" * 70)

    low_noise = results[0]
    high_noise = results[-1]

    print(f"\nLow noise (σ={low_noise['noise']}):")
    print(f"  Candidates converge: τ_A={low_noise['tau_a']:.3f}, τ_B={low_noise['tau_b']:.3f}")
    print(f"  Expected outcome close to median: {low_noise['expected_outcome']:.3f}")

    print(f"\nHigh noise (σ={high_noise['noise']}):")
    print(f"  Candidates stay extreme: τ_A={high_noise['tau_a']:.3f}, τ_B={high_noise['tau_b']:.3f}")
    print(f"  Expected outcome further from median: {high_noise['expected_outcome']:.3f}")

    welfare_gain = low_noise["welfare"] - high_noise["welfare"]
    print(f"\nWelfare gain from better information: {welfare_gain:.4f}")
    print("(Measured as reduction in expected distance from median voter's ideal)")

    return results


def run_candidate_expertise_experiment():
    """
    What if candidates also have noisy beliefs about f(τ)?
    """
    np.random.seed(123)

    print("\n" + "=" * 70)
    print("CANDIDATE EXPERTISE EXPERIMENT")
    print("=" * 70)

    f = PolicyOutcomeFunction()
    voters = generate_strategic_voters(n=200, ideal_mean=0.40, noise_std=0.05)

    expertise_levels = [0.0, 0.05, 0.10]

    for expertise_noise in expertise_levels:
        candidate_a = StrategicCandidate("A", ideal_outcome=0.25, noise_std=expertise_noise)
        candidate_b = StrategicCandidate("B", ideal_outcome=0.55, noise_std=expertise_noise)

        result = run_strategic_election(
            voters, candidate_a, candidate_b, f, find_equilibrium_first=True
        )

        print(f"\nCandidate noise σ = {expertise_noise:.2f}:")
        print(f"  τ_A={result['tau_a']:.3f}, τ_B={result['tau_b']:.3f}")
        print(f"  Expected Gini = {result['expected_outcome']:.3f}")


if __name__ == "__main__":
    run_noise_experiment()
    run_candidate_expertise_experiment()
