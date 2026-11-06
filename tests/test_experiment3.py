import json
import tempfile
import unittest
from pathlib import Path

from marketing_probe.experiment3 import Retriever, retrieval_baseline, run3, analyze3
from marketing_probe.experiment2 import save

class TestRetrieval(unittest.TestCase):
    def test_reversal_and_no_same_test(self):
        train=[
            {'pair_id':'1','test_id':'1','headline_A':'Blue cat sleeping now',
             'headline_B':'Red dog running now','winner':'A'},
            {'pair_id':'2','test_id':'2','headline_A':'Moon above the sky',
             'headline_B':'Stars above the sky','winner':'B'}]
        query={'pair_id':'3','test_id':'3','headline_A':'Red dog running now',
               'headline_B':'Blue cat sleeping now'}
        retr=Retriever(train)
        matches=retr.find(query,1)
        self.assertEqual(matches[0]['source_pair_id'],'1')
        self.assertEqual(matches[0]['observed_higher_ctr'],'B')
        self.assertLess(retrieval_baseline(matches),.5)
        self.assertEqual(retr.find(dict(query,test_id='1'),1)[0]['source_pair_id'],'2')

    def test_offline_analysis_and_dryrun(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            d=root/'data'
            def row(i):
                return {'pair_id':str(i),'test_id':str(i),
                        'headline_A':'Why coffee is interesting '+str(i),
                        'headline_B':'New facts about coffee '+str(i),
                        'winner':'A' if i%2 else 'B','ctr_A':.05,'ctr_B':.04}
            for stage,ids in (('train',range(20)),('cal',range(20,30)),('test',range(30,40))):
                save(d/(stage+'.jsonl'),[row(i) for i in ids])
            old=[];new=[]
            for stage,start in (('cal',20),('test',30)):
                for i in range(start,start+10):
                    for cond in ('zero','fewshot'):
                        old.append({'stage':stage,'pair_id':str(i),
                                    'model_key':'jev','condition':cond,'p_A':.6})
                    new.append({'stage':stage,'pair_id':str(i),
                                'model_key':'jev','condition':'retrieved','p_A':.7,'k':5})
            save(root/'previous.jsonl',old)
            save(root/'retrieved.jsonl',new)
            run3(directory=d,out=root/'dry.jsonl',limit=2)
            self.assertFalse((root/'dry.jsonl').exists())
            report=analyze3(d,root/'retrieved.jsonl',root/'previous.jsonl',root/'report.json')
            self.assertIn('retrieval_only',report['baselines'])
            self.assertEqual(report['models']['jev/retrieved']['coverage'],'10/10')
            self.assertTrue((root/'report.json').exists())
