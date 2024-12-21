"""
compare_intervention_app.py

A Streamlit app to compare baseline vs. intervention scenarios,
to highlight how an intervention (e.g. civics) affects election outcomes.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Imports from democrasim
from democrasim.core.voter import Voter, VoterParams
from democrasim.core.election import run_election
from democrasim.models.candidate import Candidate
from democrasim.core.policy import Policy
from democrasim.models.intervention import Intervention


def multi_year_simulation(voters, candidates, years=5, rng_seed=42):
    """
    Runs an election each year.
    Returns a DataFrame with columns: [year, winner, outcome].
    The 'outcome' is a naive sum-of-dimensions * 500 for the winner.
    """
    rng = np.random.default_rng(rng_seed)

    # Suppose each candidate has a base outcome = sum of policy dims * 500 + baseline_popularity * 100
    cand_outcomes = {}
    for c in candidates:
        sdim = sum(c.policy.dimensions.values())
        base_outcome = sdim * 500
        base_outcome += c.baseline_popularity * 100
        cand_outcomes[c.candidate_id] = base_outcome

    results = []
    for y in range(1, years + 1):
        # Build dicts for run_election
        candidate_policies = {}
        candidate_baseline = {}
        for c in candidates:
            candidate_policies[c.candidate_id] = c.get_policy_dimensions()
            candidate_baseline[c.candidate_id] = c.baseline_popularity

        winner = run_election(voters, candidate_policies, candidate_baseline)
        outcome = cand_outcomes[winner]

        results.append({"year": y, "winner": winner, "outcome": outcome})

        # Age all voters
        for v in voters:
            v.age_one_year()

    return pd.DataFrame(results)


def main():
    st.title("Democrasim: Compare Baseline vs. Intervention")

    st.markdown(
        """
    This app runs **two** multi-year simulations:
    1. **Baseline**: No intervention.
    2. **Intervention**: We apply a civics-like program to a fraction of youth voters.
    
    We then compare the outcomes side by side.
    """
    )

    # 1) Basic parameters
    st.subheader("Simulation Parameters")
    num_voters = st.slider(
        "Number of Voters", min_value=100, max_value=5000, value=1000, step=100
    )
    num_years = st.slider(
        "Number of Years", min_value=1, max_value=20, value=5
    )

    # 2) Intervention fraction
    st.subheader("Intervention Fraction")
    st.markdown(
        "We'll create a civics-style intervention that boosts accuracy (20% on each dimension) and turnout (+0.1) for some fraction of youth (<18)."
    )
    intervention_fraction = st.slider(
        "Fraction of youth receiving intervention", 0.0, 1.0, 0.5, 0.1
    )

    # Let's define the effect of the 'civics_program'
    civics_program = Intervention(
        name="CivicsProgram",
        effect={
            "accuracy_multipliers": {
                "economic": 1.2,
                "social": 1.2,
                "environmental": 1.2,
            },
            "turnout_boost": 0.1,
        },
    )

    # 3) Candidates
    st.subheader("Candidates Setup")
    candidate_count = st.selectbox("Number of Candidates", [2, 3], index=0)

    cand_data = []
    for i in range(candidate_count):
        cid = st.text_input(f"Candidate {i+1} ID", value=f"C{i+1}")
        eco_val = st.number_input(
            f"{cid} economic dimension", value=1.0, step=0.1
        )
        soc_val = st.number_input(
            f"{cid} social dimension", value=1.0, step=0.1
        )
        env_val = st.number_input(
            f"{cid} environmental dimension", value=1.0, step=0.1
        )
        base_pop = st.number_input(
            f"{cid} baseline popularity", value=0.0, step=0.1
        )
        cand_data.append(
            {
                "cid": cid,
                "policy_dims": {
                    "economic": eco_val,
                    "social": soc_val,
                    "environmental": env_val,
                },
                "baseline": base_pop,
            }
        )

    if st.button("Run Comparison"):
        # Build common population for baseline vs. intervention
        rng = np.random.default_rng(42)
        vparams = VoterParams()

        # Create baseline population
        baseline_voters = []
        for _ in range(num_voters):
            v = Voter(params=vparams)
            baseline_voters.append(v)

        # Create intervention population (deep copy so random draws match or differ?)
        # We'll just re-draw them for demonstration,
        # but if you want identical draws except for the intervention, you'd do a copy.
        # For clarity, let's do a second creation, so "all else equal" might differ slightly
        # due to random draws.
        intervention_voters = []
        for _ in range(num_voters):
            v = Voter(params=vparams)
            intervention_voters.append(v)

        # Identify youth in intervention_voters
        youth = [vv for vv in intervention_voters if vv.age < 18]
        rng.shuffle(youth)
        treat_count = int(len(youth) * intervention_fraction)
        for i in range(treat_count):
            youth[i].apply_intervention(
                civics_program.name, civics_program.effect
            )

        # Build candidate objects
        candidates_baseline = []
        candidates_intervention = []
        from democrasim.core.policy import Policy
        from democrasim.models.candidate import Candidate

        for cinfo in cand_data:
            policy_obj = Policy(
                name=f"{cinfo['cid']}_policy", dimensions=cinfo["policy_dims"]
            )
            # Baseline candidate
            c_base = Candidate(
                cinfo["cid"], policy_obj, baseline_popularity=cinfo["baseline"]
            )
            candidates_baseline.append(c_base)
            # Intervention candidate - same policy
            c_int = Candidate(
                cinfo["cid"], policy_obj, baseline_popularity=cinfo["baseline"]
            )
            candidates_intervention.append(c_int)

        # Run simulations
        df_baseline = multi_year_simulation(
            baseline_voters, candidates_baseline, years=num_years, rng_seed=123
        )
        df_interv = multi_year_simulation(
            intervention_voters,
            candidates_intervention,
            years=num_years,
            rng_seed=456,
        )

        # Show results side by side
        st.write("## Baseline Results")
        st.dataframe(df_baseline)
        total_base_outcome = df_baseline["outcome"].sum()
        st.write(f"Total outcome (Baseline): {total_base_outcome:,.2f}")

        st.write("## Intervention Results")
        st.dataframe(df_interv)
        total_int_outcome = df_interv["outcome"].sum()
        st.write(f"Total outcome (Intervention): {total_int_outcome:,.2f}")

        # Compare
        diff = total_int_outcome - total_base_outcome
        st.write(f"**Difference (Intervention - Baseline):** {diff:,.2f}")

        # Simple chart showing winners across time for baseline vs. intervention
        st.write("### Year-by-Year Winners Comparison")
        combined = pd.DataFrame(
            {
                "year": df_baseline["year"],
                "baseline_winner": df_baseline["winner"],
                "intervention_winner": df_interv["winner"],
            }
        )
        st.dataframe(combined)

        # Tally
        st.write("**Winner Tallies**")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Baseline**")
            st.bar_chart(df_baseline["winner"].value_counts())

        with col2:
            st.write("**Intervention**")
            st.bar_chart(df_interv["winner"].value_counts())

        st.markdown(
            """
        Here you can see how often each candidate wins with or without the intervention,
        as well as the total outcome difference.
        """
        )


if __name__ == "__main__":
    main()
