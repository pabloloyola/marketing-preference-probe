import json
import tempfile
import unittest
from pathlib import Path
from marketing_probe.final_diagnostic import ChoiceLogit, bootstrap, prepare_final, validated, reliability
from marketing_probe.experiment2 import save

def row(i, winner="A"):
    return {"pair_id":str(i),"test_id":str(i),
            "headline_A":f"How she survived a difficult year {i}?",
            "headline_B":f"A person found courage on a difficult day {i}",
            "winner":winner,"ctr_A":.08 if winner=="A" else .02,
            "ctr_B":.02 if winner=="A" else .08,
            "impressions_A":2000,"impressions_B":2000}

class FinalDiagnosticTests(unittest.TestCase):
    def test_choice_logit_inverts_under_swap(self):
        train=[row(i,"A" if i%2 else "B") for i in range(40)]
        predictor=ChoiceLogit(train,steps=20)
        r=row(999)
        inverted=dict(r,headline_A=r["headline_B"],headline_B=r["headline_A"])
        self.assertAlmostEqual(predictor.predict(r)+predictor.predict(inverted),1.0,places=8)
    def test_bootstrap_paired(self):
        pairs=[row(i,"A" if i%2 else "B") for i in range(12)]
        systems={"perfect":[1.0 if r["winner"]=="A" else 0.0 for r in pairs],
                 "wrong":[0.0 if r["winner"]=="A" else 1.0 for r in pairs]}
        result=bootstrap(pairs,systems,draws=100)
        self.assertAlmostEqual(result["perfect minus wrong"]["accuracy_diff"],1.0)
        self.assertGreater(reliability(pairs[0]),2)
    def test_rejects_overlap_and_modified_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pool=[row(i) for i in range(1,6)]
            save(root/"pool.jsonl",pool)
            save(root/"explore.jsonl",[row(10)])
            spl=root/"splits"
            for name,i in (("train",20),("cal",21),("test",22)):
                save(spl/(name+".jsonl"),[row(i)])
            hold=root/"hold"
            save(hold/"cal.jsonl",[row(40)])
            save(hold/"test.jsonl",[row(41)])
            import hashlib
            hashes={"retrieval_pool_sha256":hashlib.sha256((root/"pool.jsonl").read_bytes()).hexdigest(),
                    "cal_sha256":hashlib.sha256((hold/"cal.jsonl").read_bytes()).hexdigest(),
                    "test_sha256":hashlib.sha256((hold/"test.jsonl").read_bytes()).hexdigest()}
            (hold/"manifest.json").write_text(json.dumps(hashes))
            validated(root/"pool.jsonl",hold,spl,root/"explore.jsonl")
            save(hold/"test.jsonl",[row(20)])
            with self.assertRaises(ValueError):
                validated(root/"pool.jsonl",hold,spl,root/"explore.jsonl")
