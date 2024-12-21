import unittest
from democrasim.metrics.qaly import estimate_qalys
from democrasim.metrics.economic import compute_gdp_growth
from democrasim.metrics.inequality import gini_coefficient


class TestMetrics(unittest.TestCase):
    def test_estimate_qalys(self):
        val = estimate_qalys(0.02, 0.1)
        self.assertIsInstance(val, float)

    def test_compute_gdp_growth(self):
        growth = compute_gdp_growth(1.1)
        self.assertAlmostEqual(growth, 0.1)

    def test_gini_coefficient(self):
        inc = [10, 10, 10, 10]
        gini = gini_coefficient(inc)
        # perfect equality => Gini ~ 0
        self.assertTrue(abs(gini) < 1e-6)
