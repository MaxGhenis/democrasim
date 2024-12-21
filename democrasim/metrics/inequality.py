def gini_coefficient(income_distribution) -> float:
    """
    Compute a Gini coefficient from an array of incomes.
    Placeholder or you can use a library.
    """
    sorted_inc = sorted(income_distribution)
    n = len(sorted_inc)
    cum = 0
    for i, val in enumerate(sorted_inc, 1):
        cum += i * val
    return (2 * cum) / (n * sum(sorted_inc)) - (n + 1) / n
