import unittest
from democrasim.models.intervention import Intervention


class TestIntervention(unittest.TestCase):
    def test_intervention_init(self):
        effect = {
            "accuracy_multipliers": {"economic": 1.2},
            "turnout_boost": 0.1,
        }
        i = Intervention("CivicsProgram", effect)
        self.assertEqual(i.name, "CivicsProgram")
        self.assertIn("turnout_boost", i.effect)
