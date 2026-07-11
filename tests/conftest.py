import numpy as np
import pandas as pd
import pytest

from democrasim import Electorate


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def small_electorate() -> Electorate:
    """Four voters, two policies, hand-checkable numbers.

    Voters 0 and 1 are two adults of one household (id via demographics);
    the household gains $1,000 under policy 0 and loses $500 under policy 1.
    Voter 2 is a single adult gaining $2,000 under policy 1 only.
    Voter 3 is a single adult untouched by either policy.
    """
    return Electorate(
        deltas=np.array(
            [
                [1_000.0, -500.0],
                [1_000.0, -500.0],
                [0.0, 2_000.0],
                [0.0, 0.0],
            ]
        ),
        weights=np.array([10.0, 10.0, 30.0, 50.0]),
        base_income=np.array([40_000.0, 40_000.0, 90_000.0, 60_000.0]),
        hh_adults=np.array([2.0, 2.0, 1.0, 1.0]),
        policy_labels=("Policy A", "Policy B"),
        source="TEST fixture",
        demographics=pd.DataFrame(
            {"group": ["x", "x", "y", "y"], "household": [0, 0, 1, 2]}
        ),
    )
