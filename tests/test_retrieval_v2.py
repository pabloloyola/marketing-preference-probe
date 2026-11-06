import json
import tempfile
import unittest
from pathlib import Path

from marketing_probe.retrieval_v2 import RetrievalV2, signature, make_pool, preview, run, analyze
from marketing_probe.experiment2 import save

def pair(i, a, b, winner="A"):
    return {"pair_id":str(i),"test_id":str(i),
            "headline_A":a,"headline_B":b,"winner":winner,
            "ctr_A":.06 if winner=="A" else .04,
            "ctr_B":.04 if winner=="A" else .06}

class RetrievalV2Tests(unittest.TestCase):
    def test_differences_and_reversed_alignment(self):
        history=[pair(1,"How to save money?","A great day", "A"),
                 pair(2,"Boring report","An amazing discovery!", "B")]
        target=pair(99,"A great day","How to save money?", "B")
        retr=RetrievalV2(history,method="structure")
        examples=retr.find(target,k=2)
        self.assertEqual(len(examples),2)
        self.assertTrue(all("observed_higher_ctr" in e for e in examples))
        self.assertEqual(signature(target),signature(history[0]))
        self.assertEqual(examples[0]["observed_higher_ctr"],"B")

    def test_no_overlapping_pool(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            q=pair(1,"One example","Two examples")
            rows=[pair(2,"A fairly long headline","Another fairly long headline")]
            for name in ("train","cal","test"):
                save(d/(name+".jsonl"),[q])
            save(d/"pool.jsonl",rows)
            with self.assertRaises(ValueError):
                # Caller verifies overlaps; use same test id as protected.
                from marketing_probe.retrieval_v2 import load_verified
                save(d/"pool.jsonl",[q])
                load_verified(d/"pool.jsonl",d)

    def test_dry_run_no_api(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for stage,indices in (("train",range(1,6)),("cal",range(6,9)),("test",range(9,12))):
                save(root/(stage+".jsonl"),
                     [pair(i,"Long story about happiness","Brief facts of today") for i in indices])
            save(root/"pool.jsonl",
                 [pair(i,"How they found the answer?","One surprising answer!") for i in range(21,28)])
            run(root/"pool.jsonl",root,method="structure",limit=1,out=root/"paid.jsonl")
            self.assertFalse((root/"paid.jsonl").exists())
            result=preview(root/"pool.jsonl",root,method="structure",k=2)
            self.assertEqual(result["pool_size"],7)
