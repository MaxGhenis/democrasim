def confidence_interval(values, alpha=0.05):
    """
    Compute a basic confidence interval (e.g. 95%) for an array of values.
    """
    import numpy as np

    lower = np.percentile(values, 100 * (alpha / 2))
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return (lower, upper)
