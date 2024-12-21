def compute_outcome(winner_id: str, candidate_outcomes: dict) -> float:
    """
    Suppose each candidate, if they win, yields a certain outcome
    (which might be QALYs, or an economic index, etc.).
    e.g. candidate_outcomes = {
        'A': 1000,  # 1000 net QALYs
        'B': 900
    }
    Then, if winner_id = 'A', we return 1000.
    """
    return candidate_outcomes.get(winner_id, 0.0)
