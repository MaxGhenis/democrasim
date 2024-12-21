from typing import Dict


class Intervention:
    """
    Represents an intervention that can alter voter attributes.

    Example effect:
    {
      "accuracy_multipliers": {"economic": 1.1},
      "weight_multipliers": {"environmental": 1.2},
      "turnout_boost": 0.05
    }
    """

    def __init__(self, name: str, effect: Dict):
        self.name = name
        self.effect = effect

    def __repr__(self):
        return f"Intervention({self.name}, effect={self.effect})"
