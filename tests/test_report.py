import json
import tempfile
import unittest
from pathlib import Path
from marketing_probe.report import write_report

class ReportTest(unittest.TestCase):
    def test_offline_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pairs=[]
            preds=[]
            for i in range(12):
                truth='A' if i%2 else 'B'
                pairs.append({"pair_id":str(i),"winner":truth,"ctr_A":.05 if i%2 else .04,
                              "ctr_B":.04 if i%2 else .05,"headline_A":"aaa","headline_B":"bbb"})
                for model in ("good","bad"):
                    p=.8 if truth=='A' else .2
                    preds.append({"pair_id":str(i),"model_key":model,
                                  "p_A":p if model=="good" else 1-p,
                                  "latency_s":.5,"reported_cost_usd":.00001})
            (root/"pairs.jsonl").write_text("".join(json.dumps(p)+"\n" for p in pairs))
            (root/"preds.jsonl").write_text("".join(json.dumps(p)+"\n" for p in preds))
            r=write_report(root/"pairs.jsonl",root/"preds.jsonl",root/"report")
            self.assertEqual(r["models"]["good"]["accuracy"],1)
            self.assertEqual(r["models"]["bad"]["accuracy"],0)
            for path in ("report.md","summary.json","accuracy.svg","brier.svg","confidence_bins.csv",
                         "ctr_margin_bins.csv","confident_errors.json"):
                self.assertTrue((root/"report"/path).exists())
