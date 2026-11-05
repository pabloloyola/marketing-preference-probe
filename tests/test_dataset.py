import csv
import tempfile
import unittest
from pathlib import Path
from marketing_probe.dataset import prepare, load_pairs

class DataTest(unittest.TestCase):
    def test_real_pair_filters_and_no_label_in_api(self):
        with tempfile.TemporaryDirectory() as td:
            csvpath = Path(td)/"in.csv"
            with csvpath.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["clickability_test_id","headline","impressions","clicks","eyecatcher_id","lede","problem"])
                w.writeheader()
                w.writerows([
                    {"clickability_test_id":"1","headline":"Headline A in our archive","impressions":2000,"clicks":100,"eyecatcher_id":"x","lede":"same","problem":0},
                    {"clickability_test_id":"1","headline":"Headline B in our archive","impressions":2000,"clicks":200,"eyecatcher_id":"x","lede":"same","problem":0},
                    {"clickability_test_id":"2","headline":"Headline C in our archive","impressions":2000,"clicks":100,"eyecatcher_id":"y","lede":"same","problem":1},
                    {"clickability_test_id":"2","headline":"Headline D in our archive","impressions":2000,"clicks":200,"eyecatcher_id":"y","lede":"same","problem":1}])
            out=Path(td)/"out.jsonl"
            num,_=prepare(csvpath,out,n=1)
            self.assertEqual(num,1)
            p=list(load_pairs(out))[0]
            self.assertIn(p["winner"],("A","B"))
            self.assertEqual(p["test_id"],"1")
