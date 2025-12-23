---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Strategic Candidate Model

This document explores the general equilibrium model where candidates strategically choose policy positions to maximize expected outcomes.

## Model Setup

Candidates are **policy-motivated**: they care about the policy outcome, not winning per se. Their objective is:

$$
\max_{\tau_j} \mathbb{E}[U_j] = P(\text{win}|\tau_j, \tau_k) \cdot u_j(f(\tau_j)) + (1-P(\text{win})) \cdot u_j(f(\tau_k))
$$

Where:
- $\tau$ is the policy instrument (e.g., tax rate)
- $f(\tau)$ maps policy to outcome (e.g., Gini index)
- $u_j(g) = -(g - g_j^*)^2$ is quadratic loss from candidate's ideal outcome

```{code-cell} python
import numpy as np
np.random.seed(42)

from democrasim.core.strategic import (
    PolicyOutcomeFunction,
    StrategicCandidate,
    StrategicVoter,
    generate_strategic_voters,
    find_equilibrium,
    run_strategic_election,
)
```

## Policy-Outcome Mapping

We use a simple linear function: higher taxes → lower inequality (Gini).

```{code-cell} python
f = PolicyOutcomeFunction()

# Show the mapping
taus = np.linspace(0, 1, 5)
for tau in taus:
    print(f"τ = {tau:.2f} → Gini = {f(tau):.2f}")
```

## Candidates and Voters

Two candidates with different ideals:
- **Candidate A**: Wants low inequality (Gini = 0.25)
- **Candidate B**: Accepts higher inequality (Gini = 0.55)

Voters are distributed around moderate preferences (mean Gini = 0.40).

```{code-cell} python
candidate_a = StrategicCandidate("A", ideal_outcome=0.25)
candidate_b = StrategicCandidate("B", ideal_outcome=0.55)

voters = generate_strategic_voters(
    n=200,
    ideal_mean=0.40,
    ideal_std=0.10,
    noise_std=0.05,  # Moderate voter noise
)

print(f"Candidate A ideal: Gini = {candidate_a.ideal_outcome}")
print(f"Candidate B ideal: Gini = {candidate_b.ideal_outcome}")
print(f"Median voter ideal: Gini = {np.median([v.ideal_outcome for v in voters]):.2f}")
```

## Finding Equilibrium

Candidates iteratively best-respond until positions converge.

```{code-cell} python
tau_a, tau_b, info = find_equilibrium(candidate_a, candidate_b, voters, f)

print(f"Equilibrium positions:")
print(f"  A: τ = {tau_a:.3f} → Gini = {f(tau_a):.3f}")
print(f"  B: τ = {tau_b:.3f} → Gini = {f(tau_b):.3f}")
print(f"Converged in {info['iterations']} iterations")
```

## The Key Question: How Does Voter Noise Affect Outcomes?

```{code-cell} python
noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20]
results = []

for noise in noise_levels:
    voters = generate_strategic_voters(n=200, ideal_mean=0.40, noise_std=noise)
    result = run_strategic_election(voters, candidate_a, candidate_b, f)
    results.append({
        "noise": noise,
        "tau_a": result["tau_a"],
        "tau_b": result["tau_b"],
        "expected_gini": result["expected_outcome"],
        "p_a_wins": result["p_a_wins"],
    })

print("Noise | τ_A   | τ_B   | E[Gini] | P(A wins)")
print("-" * 50)
for r in results:
    print(f"{r['noise']:.2f}  | {r['tau_a']:.3f} | {r['tau_b']:.3f} | {r['expected_gini']:.3f}   | {r['p_a_wins']:.1%}")
```

## Interpretation

The relationship between voter noise and policy quality is **non-monotonic**:

1. **Very low noise** (σ=0.01): Voters precisely identify the better candidate → one dominates → no convergence pressure → winner may not match median
2. **Moderate noise** (σ=0.03-0.05): Competition at the margin → both converge toward median → best welfare
3. **High noise** (σ=0.20): Election is nearly random → candidates drift toward personal ideals

This suggests an optimal level of voter uncertainty that maximizes policy quality - a counterintuitive but important finding.
