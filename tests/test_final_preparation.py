import tempfile
import unittest
from pathlib import Path
from marketing_probe.dataset import prepare, load_pairs
from marketing_probe.final_diagnostic import prepare_final
from marketing_probe.experiment2 import save

class ExhaustivePreparationTests(unittest.TestCase):
    def test_partial_enumeration_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            path=base/"input.csv"
            path.write_text("clickability_test_id,headline,impressions,clicks,eyecatcher_id,lede,problem\n"+
                "".join(f"{i},Headline version alpha {i},2000,100,im{i},the same,0\n"
                        f"{i},Headline version beta {i},2000,110,im{i},the same,0\n"
                        for i in range(20)))
            with self.assertRaisesRegex(ValueError,"Only 20 eligible"):
                prepare(path,base/"fail.jsonl",n=100)
            n,_=prepare(path,base/"all.jsonl",n=100,allow_partial=True)
            self.assertEqual(n,20)
            self.assertEqual(len(list(load_pairs(base/"all.jsonl"))),20)
    def test_final_prepare_checks_size_after_protection(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            csv=base/"input.csv"
            csv.write_text("clickability_test_id,headline,impressions,clicks,eyecatcher_id,lede,problem\n"+
                "".join(f"{i},Headline version alpha {i},2000,100,im{i},the same,0\n"
                        f"{i},Headline version beta {i},2000,110,im{i},the same,0\n"
                        for i in range(50)))
            pairs=base/"all.jsonl"
            prepare(csv,pairs,n=50)
            rows=list(load_pairs(pairs))
            save(base/"pool.jsonl",rows[:15])
            save(base/"exploratory.jsonl",rows[15:20])
            splits=base/"split"
            for name,segment in (("train",rows[20:25]),("cal",rows[25:30]),("test",rows[30:35])):
                save(splits/(name+".jsonl"),segment)
            with self.assertRaisesRegex(ValueError,"Only 15 fresh distinct tests"):
                prepare_final(csv,base/"pool.jsonl",splits,base/"exploratory.jsonl",
                              base/"hold",cal_n=20,test_n=30)
