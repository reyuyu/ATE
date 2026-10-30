from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from paper_rq1_early_joint_attention.statistics import bootstrap,mcnemar_exact
from paper_rq1_early_joint_attention.analyze import anomaly,summarize
from paper_rq1_early_joint_attention.protocol import CONDITIONS


class Tests(unittest.TestCase):
    def test_image_cluster_not_question_bootstrap(self):
        value=bootstrap([1,1,-1],['a','a','b'],[8],2000)[0]
        self.assertAlmostEqual(value['mean'],1/3)
        self.assertEqual(value['clusters'],2)
    def test_families_and_ni(self):
        result=bootstrap(np.array([[0,0],[0,0],[-1,-1],[1,1]]),list('abcd'),[1,60],4000)
        self.assertLessEqual(result[1]['family95_low'],result[0]['family95_low'])
        self.assertGreaterEqual(result[1]['family95_high'],result[0]['family95_high'])
    def test_seed_replay(self):
        first=bootstrap([0,-1,1],list('abc'),[8],1000)
        self.assertEqual(first,bootstrap([0,-1,1],list('abc'),[8],1000))
    def test_invalid_finite(self):
        with self.assertRaises(ValueError):bootstrap([float('nan')],['a'],[8],1000)
    def test_exact_mcnemar(self):
        self.assertEqual(mcnemar_exact(0,0),1)
        self.assertEqual(mcnemar_exact(3,0),.25)
        self.assertAlmostEqual(mcnemar_exact(3,1),.625)
    def test_anomaly_eos_not_repetition(self):
        row=anomaly([3,4,2],'blue',True)
        self.assertFalse(row['repetition_flag']);self.assertFalse(row['missing_eos'])
        self.assertTrue(anomaly([3]*8,'aaaa',False)['repetition_flag'])
    def test_degenerate_is_flagged_not_zero_risk(self):
        row=bootstrap([0,0,0],['a','b','c'],[8],1000)[0]
        self.assertTrue(row['degenerate_bootstrap'])
    def test_generation_eos_stops_without_post_eos_decode(self):
        from paper_rq1_early_joint_attention.engine import generate
        prompt=dict(input_ids=torch.ones(1,1,dtype=torch.long),attention_mask=torch.ones(1,1,dtype=torch.long))
        class Model:
            def __init__(self,first):self.calls=0;self.first=first
            def __call__(self,**kwargs):
                self.calls+=1
                logits=torch.zeros(1,1,5);logits[0,0,self.first if self.calls==1 else 2]=1
                return SimpleNamespace(logits=logits,past_key_values=SimpleNamespace(get_seq_length=lambda layer:self.calls))
        with patch('paper_rq1_early_joint_attention.engine.model_layers',lambda m:[None]*32):
            for first,expected in [(2,[2]),(4,[4,2])]:
                model=Model(first);result,_=generate(model,prompt,CONDITIONS['S0'])
                self.assertEqual(result['token_ids'],expected);self.assertEqual(model.calls,len(expected));self.assertTrue(result['eos'])
    def test_paired_summary_missing_tasks(self):
        selected=[dict(task='ChooseAttr',question_id=str(i),image_id=str(i//2)) for i in range(4)]
        data={}
        for i,s in enumerate(selected):
            values={}
            for c in CONDITIONS:
                score=[1,1,0,0][i] if c=='S0' else [0,1,1,0][i]
                values[c]=dict(score=score,text=str(score),prediction=str(score),token_ids=[4+score,2],
                    length=2,eos=True,truncated=False,repetition_flag=False,empty_answer=False)
            data[(s['task'],s['question_id'])]=values
        original=bootstrap
        def quick(values,ids,families,repetitions):return original(values,ids,families,1000)
        with tempfile.TemporaryDirectory() as d,patch('paper_rq1_early_joint_attention.analyze.OUT',Path(d)),patch('paper_rq1_early_joint_attention.analyze.bootstrap',quick):
            summary,taskwise,contrasts=summarize(dict(samples=selected),data)
            self.assertTrue((Path(d)/'taskwise_paired_comparisons.csv').exists())
        self.assertEqual(summary[1]['correct_to_wrong'],1);self.assertEqual(summary[1]['wrong_to_correct'],1)
        self.assertEqual(summary[1]['delta_accuracy_pp'],0);self.assertEqual(len(contrasts),6)
        missing=[r for r in taskwise if r['task']=='LogicalObj']
        self.assertTrue(all(r['n_questions']==0 for r in missing))


if __name__=='__main__':unittest.main()
