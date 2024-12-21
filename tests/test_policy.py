import unittest
from democrasim.core.policy import Policy


class TestPolicy(unittest.TestCase):
    def test_policy_init(self):
        p = Policy("TestPolicy", {"economic": 1.0, "social": 0.5})
        self.assertEqual(p.name, "TestPolicy")
        self.assertEqual(p.dimensions["economic"], 1.0)
