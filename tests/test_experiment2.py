import json, tempfile, unittest
from pathlib import Path
from marketing_probe.experiment2 import save, examples_from, shrinkage, train_logit, analyze2, run2

def pair(i):
    return {'pair_id':str(i),'test_id':str(i),'headline_A':'Why You Should Read This Headline',
            'headline_B':'Seven Facts About This Headline','winner':'A' if i%2 else 'B',
            'ctr_A':.06 if i%2 else .04,'ctr_B':.04 if i%2 else .06}

class Experiment2Test(unittest.TestCase):
    def test_calibration(self):
        self.assertEqual(shrinkage([.95]*10,[0]*10),0.)
        self.assertEqual(shrinkage([.7]*10,[1]*10),1.)
    def test_offline_and_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); d=root/'data'
            for name,ids in [('train',range(20)),('cal',range(20,30)),('test',range(30,40))]:
                save(d/(name+'.jsonl'),[pair(i) for i in ids])
            self.assertEqual(len(examples_from([pair(i) for i in range(20)],5)),5)
            self.assertEqual(len(train_logit([pair(i) for i in range(20)])),7)
            run2(directory=d,output=root/'pred.jsonl',limit=2)
            self.assertFalse((root/'pred.jsonl').exists())
            rows=[{'stage':stage,'pair_id':str(i),'model_key':'jev','condition':cond,
                   'p_A':.8 if i%2 else .2, 'reported_cost_usd':.00001}
                  for stage,start in [('cal',20),('test',30)]
                  for i in range(start,start+10) for cond in ('zero','fewshot')]
            save(root/'pred.jsonl',rows)
            report=analyze2(d,root/'pred.jsonl',root/'report.json')
            self.assertEqual(report['systems']['jev/fewshot']['raw_test']['accuracy'],1.)
            self.assertEqual(report['systems']['jev/zero']['cal_coverage'],'10/10')
            self.assertTrue((root/'report.json').exists())
