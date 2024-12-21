import matplotlib.pyplot as plt


def plot_election_results(results_dict):
    """
    Given a dict like {'A': 100, 'B': 150, 'C': 80}, plot a bar chart of votes.
    """
    candidates = list(results_dict.keys())
    votes = list(results_dict.values())

    plt.bar(candidates, votes, color="skyblue")
    plt.xlabel("Candidate")
    plt.ylabel("Votes")
    plt.title("Election Results")
    plt.show()
