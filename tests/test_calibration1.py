import json
import math
import tempfile
import unittest
from pathlib import Path
from marketing_probe.experiment2 import save
from marketing_probe.calibration1 import observations, calibration_bins, analyze, collect_predictions

def pair(i, clicks_a, clicks_b):
    n=2000
    return {"pair_id":str(i),"test_id":str(i),"headline_A":"A","headline_B":"B",
            "clicks_A":clicks_a,"clicks_B":clicks_b,"impressions_A":n,
            "impressions_B":n,"ctr_A":clicks_a/n,"ctr_B":clicks_b/n,
            "winner":"A" if clicks_a>clicks_b else "B"}

class CalibrationOneTests(unittest.TestCase):
    def test_posterior_ambiguity_and_direction(self):
        equal=observations(pair(1,100,100),draws=3000,seed=11)
        strong=observations(pair(2,200,50),draws=3000,seed=12)
        self.assertAlmostEqual(equal["p_latent_A_superior"],.5,delta=.07)
        self.assertGreater(strong["p_latent_A_superior"],.99)
        self.assertGreater(strong["p_repeat_A_wins_approx"],.99)
        flipped=observations(pair(3,50,200),draws=3000,seed=12)
        self.assertLess(flipped["p_latent_A_superior"],.01)
    def test_bins(self):
        rows=[{"p_A":.1,"winner":"B","posterior":{"p_latent_A_superior":.3}},
              {"p_A":.9,"winner":"A","posterior":{"p_latent_A_superior":.7}}]
        result=calibration_bins(rows,2)
        self.assertEqual(len(result),2)
    def test_file_integration_and_no_network(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); d=root/"splits"
            save(d/"cal.jsonl",[pair(i,100+i,100) for i in range(1,9)])
            save(d/"test.jsonl",[pair(i,105,100+i%3) for i in range(20,30)])
            rows=[]
            for stage,indices in (("cal",range(1,9)),("test",range(20,30))):
                for i in indices:
                    rows.append({"stage":stage,"pair_id":str(i),"model_key":"jev",
                                 "condition":"zero","p_A":.65})
            save(root/"pred.jsonl",rows)
            result=analyze(d,(root/"pred.jsonl",),root/"out",priors=(.5,2.),draws=200,seed=4)
            self.assertIn("jev/zero",result["systems"])
            self.assertTrue((root/"out"/"report.md").exists())
            self.assertTrue((root/"out"/"observation_uncertainty.csv").exists())
            self.assertEqual(result["systems"]["jev/zero"]["test_coverage"],"10/10")
    def test_duplicate_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"dupes.jsonl"
            save(file,[{"stage":"test","pair_id":"1","model_key":"jev","condition":"zero","p_A":.6}]*2)
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                collect_predictions((file,),{("test","1"):pair(1,101,100)})
