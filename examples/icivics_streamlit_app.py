"""
icivics_streamlit_app.py

A Streamlit app demonstrating how an iCivics-like intervention (civics education)
could affect election outcomes and total QALYs, using the democrasim structure.
"""

import streamlit as st
import numpy as np
import pandas as pd
import copy

# We assume these imports are from your democrasim package
# Adjust the import paths if your code is in different folders
from democrasim.core.voter import Voter, VoterParams
from democrasim.core.election import run_election
from democrasim.models.candidate import Candidate
from democrasim.core.policy import Policy
from democrasim.models.intervention import Intervention


def multi_year_simulation(voters, candidates, years=5, rng_seed=42):
    """
    Runs a multi-year election simulation. Each year:
      - We run run_election() to find the winner among these candidates
      - We interpret each candidate as having a certain QALY value
      - We sum up those QALYs across all years
      - We age all voters by 1

    Returns:
      df_results: DataFrame with columns [year, winner, qaly_award]
    """
    rng = np.random.default_rng(rng_seed)

    # For demonstration: each candidate has a "QALY award" we store in a dict
    # Suppose candidateA => 500 QALYs, candidateB => 600 QALYs, etc.
    # We'll define that below from the candidate's policy or baseline popularity.
    candidate_qaly_values = {}
    for c in candidates:
        # We'll do something super simple:
        # qaly = c.baseline_popularity * 100 + some fixed base, e.g. 500
        # or we can do random draws each year.
        # Let's just define a static "candidate_qaly" now:
        #   e.g. 500 + baseline_pop*200
        base = 500.0
        # TOTALLY ARBITRARY, adjust as you like:
        candidate_qaly_values[c.candidate_id] = base + (
            c.baseline_popularity * 200
        )

    results = []
    for y in range(1, years + 1):
        # Build election inputs
        candidate_policies = {}
        candidate_baseline = {}
        for c in candidates:
            # Even though we talk about "policy," we only use run_election to figure out who wins
            # and we won't interpret the policy dimensions for QALYs here.
            # In a real model, you'd combine candidate policy with a QALY calculation.
            candidate_policies[c.candidate_id] = {}  # dummy, not used
            candidate_baseline[c.candidate_id] = c.baseline_popularity

        winner = run_election(voters, candidate_policies, candidate_baseline)
        qaly_award = candidate_qaly_values[winner]

        results.append({"year": y, "winner": winner, "qaly_award": qaly_award})

        # Age voters
        for v in voters:
            v.age_one_year()

    df = pd.DataFrame(results)
    return df


def main():
    st.title("iCivics (Civics Education) Intervention Demo with QALYs")

    st.markdown(
        """
    This app simulates elections over multiple years. We create:
    1. A **baseline** scenario with no intervention.
    2. An **iCivics** scenario where a fraction of youth (<18) receive a civics education 
       intervention that boosts their accuracy and turnout.

    We compare total QALYs from the winning candidates across all years.
    """
    )

    # Simulation parameters
    num_voters = st.slider("Number of Voters", 100, 5000, 1000, step=100)
    num_years = st.slider("Number of Years", 1, 20, 5)
    st.write("---")

    st.subheader("Intervention Setup")
    st.markdown(
        """
    We'll define a 'civics' intervention that:
    - Boosts accuracy by 20% 
    - Increases turnout by +0.1
    We'll apply it to a fraction of youth in the population.
    """
    )
    fraction_intervention = st.slider(
        "Fraction of treatable youth receiving civics", 0.0, 1.0, 0.5, 0.1
    )

    # Define the iCivics intervention
    icivics_effect = {
        "accuracy_multipliers": {
            "economic": 1.2
        },  # we only define one dimension
        "turnout_boost": 0.1,
    }
    # Or you might define multiple dimensions if your real model uses them.
    # The baseline democrasim code might define "economic", "social", "environmental".
    # We'll keep it minimal for demonstration.

    civics_program = Intervention(name="CivicsEdu", effect=icivics_effect)

    st.write("---")
    st.subheader("Candidates")

    st.markdown(
        """
    For simplicity, let's define **2 candidates**. We'll store a 'baseline_popularity'
    to differentiate them. That baseline popularity influences the election.
    """
    )
    cand1_id = st.text_input("Candidate 1 ID", value="C1")
    cand1_pop = st.number_input(
        "Candidate 1 baseline popularity", value=0.1, step=0.1
    )

    cand2_id = st.text_input("Candidate 2 ID", value="C2")
    cand2_pop = st.number_input(
        "Candidate 2 baseline popularity", value=-0.05, step=0.1
    )

    if st.button("Run Simulation"):
        st.write("## Running Baseline vs. iCivics Intervention...")

        # 1) Create a single population
        rng = np.random.default_rng(42)
        vparams = VoterParams()
        all_voters = []
        for _ in range(num_voters):
            v = Voter(params=vparams)
            all_voters.append(v)

        # 2) Deep-copy them for the intervention scenario
        import copy

        int_voters = copy.deepcopy(all_voters)

        # 3) Identify youth in the intervention group, apply civics
        youth = [v for v in int_voters if v.age < 18]
        rng.shuffle(youth)
        treat_count = int(len(youth) * fraction_intervention)
        for i in range(treat_count):
            youth[i].apply_intervention(
                civics_program.name, civics_program.effect
            )

        # 4) Build the 2 candidate objects for each scenario
        #    We'll create them fresh for baseline vs. intervention so everything is the same
        c1_base = Candidate(
            candidate_id=cand1_id,
            policy=Policy("C1_policy", {}),
            baseline_popularity=cand1_pop,
        )
        c2_base = Candidate(
            candidate_id=cand2_id,
            policy=Policy("C2_policy", {}),
            baseline_popularity=cand2_pop,
        )
        cands_baseline = [c1_base, c2_base]

        c1_int = Candidate(
            candidate_id=cand1_id,
            policy=Policy("C1_policy", {}),
            baseline_popularity=cand1_pop,
        )
        c2_int = Candidate(
            candidate_id=cand2_id,
            policy=Policy("C2_policy", {}),
            baseline_popularity=cand2_pop,
        )
        cands_intervention = [c1_int, c2_int]

        # 5) Run multi-year simulation
        df_baseline = multi_year_simulation(
            all_voters, cands_baseline, years=num_years, rng_seed=123
        )
        df_interv = multi_year_simulation(
            int_voters, cands_intervention, years=num_years, rng_seed=999
        )

        st.write("### Baseline Results")
        st.dataframe(df_baseline)
        base_total_qalys = df_baseline["qaly_award"].sum()
        st.write(f"**Total QALYs (Baseline)**: {base_total_qalys:.2f}")

        st.write("### iCivics Intervention Results")
        st.dataframe(df_interv)
        interv_total_qalys = df_interv["qaly_award"].sum()
        st.write(f"**Total QALYs (Intervention)**: {interv_total_qalys:.2f}")

        diff = interv_total_qalys - base_total_qalys
        st.write(f"**Difference (Intervention - Baseline)**: {diff:.2f}")

        # Compare winners visually
        st.write("### Winner Comparison by Year")
        df_compare = pd.DataFrame(
            {
                "year": df_baseline["year"],
                "baseline_winner": df_baseline["winner"],
                "intervention_winner": df_interv["winner"],
            }
        )
        st.dataframe(df_compare)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Baseline winner tally**")
            st.bar_chart(df_baseline["winner"].value_counts())

        with col2:
            st.write("**Intervention winner tally**")
            st.bar_chart(df_interv["winner"].value_counts())

        st.markdown(
            """
        If the intervention is truly changing election outcomes, you should see differences in
        the winners each year (or at least in some years), which leads to a difference in total QALYs.
        """
        )


if __name__ == "__main__":
    main()
