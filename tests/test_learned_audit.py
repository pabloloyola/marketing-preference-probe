import unittest
import numpy as np
from marketing_probe.learned_audit import paired_interval, nonlinear_positive_control

class LearnedAuditTests(unittest.TestCase):
    def test_paired_interval(self):
        r=paired_interval([.2,.3,.4],[.1,.2,.3],draws=200,seed=12)
        self.assertAlmostEqual(r["difference"],.1,places=9)
        self.assertAlmostEqual(r["ci95"][0],.1,places=9)
    def test_nonlinear_positive_control_runs(self):
        r=nonlinear_positive_control(seed=11,n=250)
        self.assertIn("mlp",r["results"])
        self.assertIn("linear",r["results"])
        self.assertTrue(np.isfinite(r["mlp_improvement"]))
        self.assertLessEqual(r["results"]["mlp"]["metrics"]["brier"],1)
