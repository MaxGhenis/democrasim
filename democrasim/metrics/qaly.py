def estimate_qalys(gdp_growth: float, coverage_expansion: float) -> float:
    """
    Very rough placeholder: e.g., if coverage_expansion is 0.1,
    that might mean +10% population covered by health insurance,
    which yields some approximate QALYs.
    """
    # TOTALLY FAKE EXAMPLE
    return gdp_growth * 10_000 + coverage_expansion * 50_000
