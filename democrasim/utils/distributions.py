import numpy as np


def sample_beta(alpha: float, beta: float) -> float:
    return np.random.beta(alpha, beta)


def sample_lognormal(mu: float, sigma: float) -> float:
    return np.random.lognormal(mu, sigma)
