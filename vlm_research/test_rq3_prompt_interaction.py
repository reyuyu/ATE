import unittest

import numpy as np
import torch

from benchmarks.smoke import hook_footprint
from rq2_prompt_five.engine import generate
from rq3_prompt_interaction.engine import ALL,CONDITIONS,CHECKPOINTS,StateCapture,captured_generate,prefill,state_metrics
from rq3_prompt_interaction.analyze import contrasts,anomaly
from paper_rq1_early_joint_attention.statistics import bootstrap
from test_rq2_source_knockout import Tiny,inputs


class Tests(unittest.TestCase):
    def test_frozen_conditions_and_contrasts(self):
        self.assertEqual(ALL['E'],list(range(2,13)))
        self.assertEqual(ALL['M'],list(range(13,17)))
        self.assertEqual(CONDITIONS,{'E':list(range(2,13)),'EM':list(range(2,17)),'L':list(range(17,32))})
        self.assertEqual(CHECKPOINTS,[2,6,9,13,17,31])
        scores=np.array([[1,0,0,0,1],[1,1,0,0,0],[0,1,1,0,0]])
        values=contrasts(scores);np.testing.assert_array_equal(values[:,0],[100,0,-200])
        np.testing.assert_array_equal(values[:,0],values[:,-1]-values[:,2])
        a=bootstrap(values,['a','a','b'],[1,6,6,6,6,6,6],1000,13)[0]
        b=bootstrap(values[:,:1],['a','a','b'],[1],1000,13)[0]
        self.assertEqual(a,b)

    def test_read_only_capture_and_native_decode(self):
        torch.manual_seed(991);model=Tiny();data=inputs();before=hook_footprint(model)
        for blocked in [[],[0],[1,2]]:
            plain,first,trace,debug=generate(model,data,1,blocked,limit=4,audit=True,collect=True,expected_visual=4)
            g,f,t,d,h=captured_generate(model,data,1,blocked,capture_layers=[0,1,2],limit=4,audit=True,collect=True,expected_visual=4)
            self.assertEqual(plain,g);self.assertTrue(torch.equal(first,f));self.assertEqual(h.shape,(3,32))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(trace,t)))
            self.assertTrue(torch.equal(debug['prefill_logits'],d['prefill_logits']))
            pf,ph,_=prefill(model,data,1,blocked,capture_layers=[0,1,2],expected_visual=4)
            self.assertTrue(torch.equal(pf,f));self.assertTrue(torch.equal(ph,h))
        self.assertEqual(before,hook_footprint(model))

    def test_causal_layer_input_invariant(self):
        torch.manual_seed(321);model=Tiny();data=inputs()
        _,f,_=prefill(model,data,1,[],capture_layers=[0,1,2],expected_visual=4)
        _,changed,_=prefill(model,data,1,[1],capture_layers=[0,1,2],expected_visual=4)
        self.assertTrue(torch.equal(f[:2],changed[:2]));self.assertFalse(torch.equal(f[2],changed[2]))

    def test_dynamic_query_cleanup(self):
        model=Tiny();data=inputs();before=hook_footprint(model)
        data['input_ids']=torch.cat([data['input_ids'],torch.tensor([[14,15]])],1);data['attention_mask']=torch.ones_like(data['input_ids'])
        with self.assertRaises(RuntimeError):
            with StateCapture(model,data,[0,1,2]) as cap:
                self.assertEqual(cap.query,10);raise RuntimeError('cleanup')
        self.assertEqual(before,hook_footprint(model))

    def test_state_metrics_zero_and_fp32(self):
        a=torch.tensor([3.,4.],dtype=torch.float16);r=state_metrics(a,a)
        self.assertEqual(r['relative_l2_distance'],0);self.assertEqual(r['cosine_distance'],0);self.assertTrue(r['exact_full'])
        r=state_metrics(a,2*a);self.assertEqual(r['relative_l2_distance'],1);self.assertEqual(r['norm_change'],5)
        self.assertIsNone(state_metrics(a,torch.zeros_like(a))['cosine_distance'])
        with self.assertRaises(ValueError):state_metrics(torch.zeros(2),a)

    def test_anomaly_flags_not_scoring(self):
        s={'question':'What color is the shirt?'}
        g=dict(text='',token_ids=[2],truncated=False)
        self.assertTrue(anomaly(s,g)['empty'])
        g=dict(text=s['question'],token_ids=[4,4,4],truncated=True)
        flags=anomaly(s,g);self.assertTrue(flags['question_echo']);self.assertTrue(flags['repeated_text']);self.assertTrue(flags['truncated'])


if __name__=='__main__':unittest.main()
