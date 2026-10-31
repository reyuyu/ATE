import unittest

import numpy as np
import torch

from benchmarks.smoke import hook_footprint
from rq2_visual_sufficiency.engine import MultiImageKnockout, generate, CONDITIONS
from rq2_visual_sufficiency.run import assign, CONTRASTS
from rq2_source_knockout.engine import forward
from test_rq2_source_knockout import Tiny, inputs
from paper_rq1_early_joint_attention.statistics import bootstrap


class Tests(unittest.TestCase):
    def test_frozen_layers_and_family(self):
        self.assertEqual([len(CONDITIONS[k]) for k in 'ABCD'],[32,31,30,2])
        self.assertEqual(set(range(32))-set(CONDITIONS['C']),{14,19})
        self.assertEqual(CONDITIONS['D'],[14,19])
        self.assertEqual(len(CONTRASTS),6)

    def test_multilayer_noop_and_other_queries(self):
        torch.manual_seed(4)
        model=Tiny(); data=inputs(); before=hook_footprint(model)
        base,first,logits,debug=generate(model,data,1,[],limit=4,audit=True,collect=True,expected_visual=4)
        noop,no_first,no_logits,no_debug=generate(model,data,1,[0,1,2],noop=True,limit=4,audit=True,collect=True,expected_visual=4)
        self.assertEqual(base['token_ids'],noop['token_ids'])
        self.assertTrue(torch.equal(first,no_first))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(logits,no_logits)))
        self.assertTrue(torch.equal(debug['prefill_logits'],no_debug['prefill_logits']))
        changed,_,_,audit=generate(model,data,1,[0,2],limit=4,audit=True,expected_visual=4)
        self.assertEqual(changed['masked_layers'],[0,2])
        self.assertEqual(changed['mask_calls'],3)
        self.assertEqual(audit['observer'].states,debug['observer'].states)
        self.assertTrue(torch.equal(audit['prefill_logits'][0,:8],debug['prefill_logits'][0,:8]))
        self.assertEqual([r['Image_max'] for r in audit['observer'].rows if r['layer'] in [0,2]],[0.,0.])
        self.assertGreater(audit['observer'].rows[1]['Image'],0.)
        self.assertEqual(hook_footprint(model),before)

    def test_single_layer_matches_legacy(self):
        model=Tiny();data=inputs()
        _,first,_,_=generate(model,data,1,[1],limit=2,expected_visual=4)
        old,_,_=forward(model,data,1,1,'Image',expected_visual=4)
        self.assertTrue(torch.equal(first,old))

    def test_cleanup_error_and_validation(self):
        model=Tiny();data=inputs(); before=hook_footprint(model)
        with self.assertRaises(RuntimeError):
            with MultiImageKnockout(model,data,1,[0,1],expected_visual=4):
                raise RuntimeError('test')
        self.assertEqual(hook_footprint(model),before)
        for layers in [[0,0],[-1],[3],[True]]:
            with self.assertRaises(ValueError): MultiImageKnockout(model,data,1,layers,expected_visual=4)

    def test_dynamic_query_and_own_cache(self):
        model=Tiny();data=inputs()
        data['input_ids']=torch.cat([data['input_ids'],torch.tensor([[14,15]])],-1)
        data['attention_mask']=torch.ones_like(data['input_ids'])
        a,af,_,_=generate(model,data,1,[0,2],limit=3,expected_visual=4)
        generate(model,data,1,[1],limit=3,expected_visual=4)
        b,bf,_,_=generate(model,data,1,[0,2],limit=3,expected_visual=4)
        self.assertEqual(a,b);self.assertTrue(torch.equal(af,bf))

    def test_cached_mask_rejects_existing_history(self):
        model=Tiny();data=inputs();before=hook_footprint(model)
        with torch.inference_mode():
            cache=model(**data,use_cache=True).past_key_values
            with self.assertRaises(ValueError):
                with MultiImageKnockout(model,data,1,[0],expected_visual=4):
                    model(**data,past_key_values=cache,use_cache=True)
        self.assertEqual(hook_footprint(model),before)

    def test_assignment_keeps_images_together(self):
        ss=[dict(question_id=str(i),image_sha256=str(i//2)) for i in range(24)]
        a=assign(ss)
        self.assertEqual(a,assign(list(reversed(ss))))
        self.assertEqual(set(a.values()),{0,1,2})
        for i in range(0,24,2):self.assertEqual(a[str(i)],a[str(i+1)])

    def test_cluster_family_interval(self):
        x=np.array([[-1,0],[1,1],[0,1]],dtype=float)
        stats=bootstrap(x,['same','same','other'],[6,6],repetitions=1000,seed=5)
        self.assertEqual(stats[0]['clusters'],2)
        self.assertEqual(stats[0]['mean'],0.)
        for s in stats:
            self.assertLessEqual(s['family95_low'],s['ci95_low'])
            self.assertGreaterEqual(s['family95_high'],s['ci95_high'])


if __name__=='__main__':unittest.main()
