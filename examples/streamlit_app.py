import streamlit as st
from democrasim.core.voter import Voter
from democrasim.core.election import run_election
from democrasim.models.candidate import Candidate
from democrasim.core.policy import Policy


def main():
    st.title("Democrasim Example App")

    num_voters = st.slider(
        "Number of Voters", min_value=100, max_value=5000, value=1000, step=100
    )
    if st.button("Run Election"):
        voters = [Voter() for _ in range(num_voters)]
        candA = Candidate("A", Policy("A_policy", {"economic": 1.0}))
        candB = Candidate("B", Policy("B_policy", {"economic": 0.8}))

        candidate_policies = {
            candA.candidate_id: candA.get_policy_dimensions(),
            candB.candidate_id: candB.get_policy_dimensions(),
        }
        winner = run_election(voters, candidate_policies)
        st.write(f"Election winner: {winner}")


if __name__ == "__main__":
    main()
