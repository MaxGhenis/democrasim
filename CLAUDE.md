# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with optional dependencies
pip install -e ".[dev,examples,metrics]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_election.py

# Run a specific test
pytest tests/test_election.py::TestElection::test_run_election -v

# Format code (88 char line length)
black .

# Lint
flake8

# Type check
mypy democrasim

# Run example scripts
python examples/qaly_simulation.py
python examples/inequality_simulation.py

# Run Streamlit app (requires examples dependencies)
streamlit run examples/streamlit_app.py
```

## Architecture Overview

Democrasim simulates how voter interventions affect election outcomes and policy metrics. The simulation flow is:

```
Interventions → Voter Behavior → Election → Winner → Policy Outcome (QALYs, inequality, etc.)
```

### Core Components (`democrasim/core/`)

- **Voter** (`voter.py`): Central entity with multi-dimensional preferences (weights, accuracies, biases), turnout probability, and age. Voters perceive policy values with noise inversely proportional to their accuracy.

- **Election** (`election.py`): `run_election(voters, candidate_policies, candidate_baseline)` - Each voter evaluates candidates by summing weighted perceived policy values plus baseline utility. Plurality winner returned.

- **Policy** (`policy.py`): Named container for policy dimension values (e.g., `{"economic": 1.2, "environmental": 0.8}`).

### Models (`democrasim/models/`)

- **Candidate**: Wraps a Policy with an ID and baseline popularity score
- **Intervention**: Defines effects that modify voter attributes:
  - `accuracy_multipliers`: Improve/reduce voter accuracy per dimension
  - `weight_multipliers`: Change how much voters care about dimensions
  - `turnout_boost`/`turnout_mult`: Affect voting probability
- **Outcome**: Maps winner ID to outcome value (e.g., QALYs)

### Metrics (`democrasim/metrics/`)

Placeholder functions for measuring policy outcomes:
- `qaly.py`: QALY estimation from GDP growth and coverage expansion
- `economic.py`: GDP growth calculation
- `inequality.py`: Gini coefficient computation
- `uncertainty.py`: Confidence interval calculation

### Voter Preference Generation

Voters are randomly generated using `VoterParams`:
- Dimension weights: Beta distribution, normalized to sum to 1
- Accuracies: Lognormal distribution
- Biases: Normal(0, 0.2)
- Turnout: Beta distribution

Interventions are applied via `voter.apply_intervention(id, effect)` and stack multiplicatively.

### Strategic Model (`democrasim/core/strategic.py`)

General equilibrium model where candidates optimize policy positions:

```
Candidate objective:
E[U] = P(win) · u(outcome_if_win) + (1-P(win)) · u(outcome_if_lose)

Voter behavior:
Vote for candidate whose perceived outcome is closest to ideal
```

Key components:
- **PolicyOutcomeFunction**: Maps policy τ → outcome g (e.g., tax rate → Gini)
- **StrategicVoter**: Has ideal outcome g*, perceives f(τ) with noise
- **StrategicCandidate**: Optimizes τ to maximize expected policy utility
- **find_equilibrium()**: Solves Nash equilibrium via iterated best response

Run `python examples/strategic_equilibrium.py` to see how voter noise affects equilibrium.

## Key Dependencies

- **squigglepy**: Uncertainty quantification (listed but distributions currently use numpy directly)
- **streamlit**: Interactive demo app (optional)
- **lifetable, gini, scikit-learn**: Extended metrics (optional)
