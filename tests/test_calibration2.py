import tempfile
import unittest
from pathlib import Path

from marketing_probe.calibration2 import model_groups, synthetic_control, analyze
from marketing_probe.experiment2 import save

def pair(i, a=100, b=100, n=2000):
    return {"pair_id":str(i),"test_id":str(i),"headline_A":"Headline A",
            "headline_B":"Headline B","clicks_A":a,"clicks_B":b,
            "impressions_A":n,"impressions_B":n,
            "ctr_A":a/n,"ctr_B":b/n,"winner":"A" if a>b else "B"}

class CalibrationTwoTests(unittest.TestCase):
    def test_strata_and_ties(self):
        rows=[pair(1,160,100),pair(2,101,100),pair(3,80,130)]
        forecasts={"1":.95,"2":.5,"3":.9}
        post={"1":.99,"2":.51,"3":.01}
        groups=model_groups(rows,forecasts,post)
        self.assertEqual(groups["all_non_tied"]["n"],2)
        self.assertEqual(groups["low_ambiguity"]["n"],2)
        self.assertEqual(groups["low_ambiguity"]["mistakes"],1)
        self.assertNotIn("high_ambiguity",groups) # abstention case excluded
    def test_synthetic_known_truth(self):
        result=synthetic_control(n=120,draws=200,seed=3,
                                 impressions=(100,1000),effects=(.002,.03))
        self.assertEqual(result["n"],120)
        self.assertEqual(len(result["cells"]),4)
        for cell in result["cells"].values():
            self.assertTrue(0<=cell["ctr_difference_95ci_coverage"]<=1)
    def test_largest_negative_effect_stays_valid(self):
        # Regression: a 4 pp negative effect cannot be applied to a 1.5% baseline.
        for seed in (1, 3, 2027, 999):
            result=synthetic_control(n=40,draws=100,seed=seed,
                                     impressions=(30,),effects=(.04,))
            self.assertEqual(result["n"],40)
            self.assertGreater(result["baseline_ctr_range"][0],.04)
            self.assertEqual(len(result["cells"]),1)
        with self.assertRaises(ValueError):
            synthetic_control(n=40,draws=100,effects=(.15,))

    def test_offline_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            save(d/"test.jsonl",[pair(i,100+i,95) for i in range(1,7)])
            save(d/"pred.jsonl",[{"stage":"test","pair_id":str(i),
                                  "model_key":"jev","condition":"zero","p_A":.6}
                                  for i in range(1,7)])
            result=analyze(d,(d/"pred.jsonl",),d/"out",draws=100,
                           synthetic_n=40,synthetic_draws=100)
            self.assertIn("jev/zero",result["models"])
            self.assertTrue((d/"out"/"report.md").exists())
            self.assertTrue((d/"out"/"synthetic_coverage.csv").exists())
