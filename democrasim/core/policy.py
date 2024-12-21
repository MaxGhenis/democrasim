from typing import Dict, Any


class Policy:
    """
    Represents a policy or platform across multiple dimensions,
    e.g. {'economic': 1.2, 'environmental': 0.8, 'health': 0.9, ...}
    """

    def __init__(self, name: str, dimensions: Dict[str, float]):
        self.name = name
        self.dimensions = dimensions

    def get_dimension_value(self, dim: str) -> float:
        return self.dimensions.get(dim, 0.0)

    def __repr__(self):
        return f"Policy(name={self.name}, dimensions={self.dimensions})"
