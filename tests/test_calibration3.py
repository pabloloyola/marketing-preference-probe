import tempfile
import unittest
from pathlib import Path
from marketing_probe.calibration3 import auc,metrics,fit_logistic_mapping,transformed,analyze
from marketing_probe.experiment2 import save

class CalibrationThreeTests(unittest.TestCase):
    def test_auc_ties_and_symmetry(self):
        self.assertAlmostEqual(auc([1,0,1,0],[.9,.1,.3,.3]),.875)
        self.assertIsNone(auc([1,1],[.2,.9]))
        probs=[.01,.13,.5,.7,.99]
        model=fit_logistic_mapping([.1,.2,.4,.6,.8,.9]*8,[0,0,0,1,1,1]*8,intercept=False)
        q=transformed(probs,model)
        for p,pred in zip(probs,q):
            self.assertAlmostEqual(pred+transformed([1-p],model)[0],1,places=8)
    def test_metrics_constant_is_not_classifier(self):
        m=metrics([0,1,0,1],[.5]*4)
        self.assertEqual(m["auc"],.5)
        self.assertEqual(m["non_tied_predictions"],0)
        self.assertIsNone(m["accuracy_non_tied"])
    def test_regularized_platt_learns_direction(self):
        x=[.1,.2,.3,.4,.6,.7,.8,.9]*6
        y=[0,0,0,0,1,1,1,1]*6
        model=fit_logistic_mapping(x,y,intercept=True)
        self.assertGreaterEqual(model["slope"],0)
        pred=transformed([.25,.75],model)
        self.assertLess(pred[0],pred[1])
    def test_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            cal=[{"pair_id":str(i),"test_id":str(i),"winner":"A" if i%2 else "B"} for i in range(1,41)]
            test=[{"pair_id":str(i),"test_id":str(i),"winner":"A" if i%2 else "B"} for i in range(101,121)]
            save(path/"cal.jsonl",cal);save(path/"test.jsonl",test)
            records=[]
            for stage,rows in (("cal",cal),("test",test)):
                for r in rows:
                    records.append({"stage":stage,"pair_id":r["pair_id"],"model_key":"jev",
                                    "condition":"zero","p_A":.75 if r["winner"]=="A" else .25})
            save(path/"pred.jsonl",records)
            result=analyze(path,(path/"pred.jsonl",),path/"out",bootstrap_draws=100)
            self.assertEqual(result["systems"]["jev/zero"]["status"],"complete")
            self.assertEqual(result["systems"]["jev/zero"]["metrics"]["raw"]["auc"],1)
            self.assertTrue((path/"out"/"report.md").exists())
