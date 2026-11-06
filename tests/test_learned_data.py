import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from marketing_probe.learned_data import prepare_split,load_split,infer,sha256,save_jsonl

def row(i):
    return {"pair_id":str(i),"test_id":str(i),
            "headline_A":f"Story A {i} one", "headline_B":f"Story B {i} two",
            "winner":"A" if i%2 else "B","impressions_A":2000,
            "impressions_B":2000,"clicks_A":100+i,"clicks_B":100}

class LearnedDataTests(unittest.TestCase):
    def test_split_reproducible_and_protected(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);pool=d/"pool.jsonl";save_jsonl(pool,[row(i) for i in range(12)])
            a=prepare_split(pool,d/"split",train=8,val=2,test=2)
            b=prepare_split(pool,d/"split",train=8,val=2,test=2)
            self.assertEqual(a,b)
            groups,_=load_split(d/"split")
            self.assertEqual(len(groups["train"]),8)
            with self.assertRaisesRegex(ValueError,"manifest differs"):
                prepare_split(pool,d/"split",train=8,val=2,test=2,seed=9)
    def test_dry_run_has_no_calls(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);pool=d/"pool.jsonl";save_jsonl(pool,[row(i) for i in range(12)])
            prepare_split(pool,d/"split",train=8,val=2,test=2)
            with patch("marketing_probe.learned_data.predict",side_effect=AssertionError("API called")):
                plan=infer(d/"split",d/"run.jsonl",models=("jev",),stages=("train",),limit=2)
            self.assertEqual(plan["requested"],2)
            self.assertFalse((d/"run.jsonl").exists())
    def test_resume_and_cumulative_guard(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);pool=d/"pool.jsonl";save_jsonl(pool,[row(i) for i in range(12)])
            prepare_split(pool,d/"split",train=8,val=2,test=2)
            def mock(model,a,b):
                return {"model_key":model,"p_A":.6,"reported_cost_usd":.001}
            with patch.dict(os.environ,{"OPENROUTER_API_KEY":"not-a-real-key"}), patch(
                "marketing_probe.learned_data.predict",side_effect=mock):
                first=infer(d/"split",d/"run.jsonl",models=("jev",),stages=("train",),
                            limit=2,budget=.1,max_calls=2,live=True)
                second=infer(d/"split",d/"run.jsonl",models=("jev",),stages=("train",),
                             limit=2,budget=.1,max_calls=2,live=True)
            self.assertEqual(first["new_successful"],2)
            self.assertEqual(second["new_successful"],0)
            self.assertEqual(len(list(__import__("marketing_probe.dataset",fromlist=["load_pairs"]).load_pairs(d/"run.jsonl"))),2)
