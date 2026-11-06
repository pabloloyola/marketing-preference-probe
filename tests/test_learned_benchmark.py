import os
import tempfile
import unittest
from pathlib import Path
from marketing_probe.learned_data import save_jsonl,prepare_split,load_split, infer
from marketing_probe.learned_benchmark import fit_linear,fit_mlp,predict_mlp,sigmoid,learning_curve
from unittest.mock import patch

def row(i):
    return {"pair_id":str(i),"test_id":str(i),
        "headline_A":f"An exciting story {i}?", "headline_B":f"Simple summary of day {i}",
        "winner":"A" if i%2 else "B","clicks_A":150 if i%2 else 90,
        "clicks_B":90 if i%2 else 150,
        "impressions_A":2000,"impressions_B":2000}
class BenchmarkTests(unittest.TestCase):
    def test_odd_mlp_and_linear(self):
        import numpy as np
        x=np.array([[-1.,-.2],[1.,.2],[-2.,-.5],[2.,.5]]*20)
        y=np.array([0,1,0,1]*20)
        w=fit_linear(x,y,steps=100)
        self.assertAlmostEqual(float(sigmoid(x[0]@w)+sigmoid(-x[0]@w)),1,places=9)
        head=fit_mlp(x,y,steps=60)
        self.assertAlmostEqual(float(predict_mlp(x[:1],head)[0]+predict_mlp(-x[:1],head)[0]),1,places=9)
    def test_incomplete_predictions_do_not_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);save_jsonl(d/"pool.jsonl",[row(i) for i in range(50)])
            prepare_split(d/"pool.jsonl",d/"data",train=30,val=10,test=10)
            save_jsonl(d/"empty.jsonl",[])
            with self.assertRaisesRegex(ValueError,"Missing model prediction"):
                learning_curve(d/"data",d/"empty.jsonl",d/"out",models=("jev",),sizes=(10,))
    def test_end_to_end_complete_mocked_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);save_jsonl(d/"pool.jsonl",[row(i) for i in range(50)])
            prepare_split(d/"pool.jsonl",d/"data",train=30,val=10,test=10)
            def fake(model,a,b):
                return {"model_key":model,"p_A":.7,"reported_cost_usd":.0001}
            with patch.dict(os.environ,{"OPENROUTER_API_KEY":"fake"}),patch(
                "marketing_probe.learned_data.predict",side_effect=fake):
                response=infer(d/"data",d/"pred.jsonl",models=("jev",),
                               budget=1,max_calls=50,live=True)
            self.assertEqual(response["new_successful"],50)
            report=learning_curve(d/"data",d/"pred.jsonl",d/"out",
                                  models=("jev",),sizes=(10,20),seed=5)
            self.assertEqual(report["sizes"],[10,20])
            self.assertTrue((d/"out"/"report.md").exists())
