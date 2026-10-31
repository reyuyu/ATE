import unittest

import numpy as np
import torch

from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import step_inputs
from paper_rq1_early_joint_attention.statistics import bootstrap
from rq2_decode_visual.engine import generate, DecodeImageKnockout, BLOCKED
from rq2_decode_visual.run import select, TASKS, length_group
from rq2_decode_visual.analyze import differences
from rq2_visual_sufficiency.engine import generate as historical
from test_rq2_source_knockout import Tiny, inputs


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(76)

    def test_fixed_two_layer_definition(self):
        self.assertEqual(set(range(32))-set(BLOCKED),{14,19})
        self.assertEqual(len(BLOCKED),30)

    def test_true_incremental_masks_and_exact_prefill(self):
        model=Tiny();data=inputs();footprint=hook_footprint(model)
        b,bf,_,bd=historical(model,data,1,[0,2],limit=4,audit=True,expected_visual=4)
        c,cf,_,cd=generate(model,data,1,[0,2],limit=4,audit=True,expected_visual=4)
        self.assertTrue(torch.equal(bf,cf))
        self.assertTrue(torch.equal(bd['prefill_logits'],cd['prefill_logits']))
        self.assertEqual(b['token_ids'][0],c['token_ids'][0])
        self.assertGreater(c['decode_steps'],0)
        for step,a in enumerate(c['decode_mask_audit'],1):
            self.assertEqual(a['step'],step)
            self.assertEqual(a['consumed_token'],c['token_ids'][step-1])
            self.assertEqual(a['predicted_token'],c['token_ids'][step])
            self.assertEqual([r['image_probability_max'] for r in a['probabilities'] if r['blocked']],[0.,0.])
            self.assertGreater(a['probabilities'][1]['image_mass'],0.)
            for r in a['masks']:
                self.assertEqual(r['local_query'],0)
                self.assertEqual(r['absolute_query'],8+step)
                self.assertEqual(r['cache_before'],8+step)
                self.assertEqual(r['cache_after'],9+step)
                self.assertTrue(r['non_image_mask_unchanged'])
        self.assertEqual(footprint,hook_footprint(model))

    def test_noop_all_logits_and_generation_exact(self):
        model=Tiny();data=inputs()
        a,af,at,ad=historical(model,data,1,[],limit=4,audit=True,collect=True,expected_visual=4)
        b,bf,bt,bd=generate(model,data,1,[0,2],noop=True,limit=4,audit=True,collect=True,expected_visual=4)
        self.assertEqual(a['token_ids'],b['token_ids'])
        self.assertTrue(torch.equal(af,bf))
        self.assertTrue(torch.equal(ad['prefill_logits'],bd['prefill_logits']))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(at,bt)))
        self.assertEqual(len(at),len(bt))
        self.assertTrue(all(not r['applied'] for d in b['decode_mask_audit'] for r in d['masks']))

    def test_non_target_prefill_queries_unchanged(self):
        model=Tiny();data=inputs()
        _,_,_,full=historical(model,data,1,[],limit=2,audit=True,expected_visual=4)
        _,_,_,new=generate(model,data,1,[0,2],limit=2,audit=True,expected_visual=4)
        self.assertTrue(torch.equal(full['prefill_logits'][0,:8],new['prefill_logits'][0,:8]))
        self.assertEqual(full['observer'].states,new['observer'].states)
        self.assertEqual([r['other_query_attention_sha256'] for r in full['observer'].rows],
                         [r['other_query_attention_sha256'] for r in new['observer'].rows])

    def test_dynamic_mapping_and_no_cross_generation_cache(self):
        model=Tiny();data=inputs()
        data['input_ids']=torch.cat([data['input_ids'],torch.tensor([[10,11,12]])],-1)
        data['attention_mask']=torch.ones_like(data['input_ids'])
        a,af,_,_=generate(model,data,1,[0,2],limit=3,expected_visual=4)
        generate(model,data,1,[1],limit=3,expected_visual=4)
        b,bf,_,_=generate(model,data,1,[0,2],limit=3,expected_visual=4)
        self.assertEqual(a,b);self.assertTrue(torch.equal(af,bf))
        self.assertEqual(a['original_p_minus_one'],11)
        self.assertEqual(a['decode_mask_audit'][0]['masks'][0]['absolute_query'],12)

    def test_exception_cleanup_and_invalid_cache_step(self):
        model=Tiny();data=inputs();footprint=hook_footprint(model)
        with self.assertRaises(RuntimeError):
            with DecodeImageKnockout(model,data,[0,2],1,expected_visual=4):
                raise RuntimeError('intentional cleanup test')
        self.assertEqual(hook_footprint(model),footprint)
        with torch.inference_mode():
            cache=model(**data,use_cache=True).past_key_values
            with self.assertRaises(ValueError):
                with DecodeImageKnockout(model,data,[0,2],2,expected_visual=4):
                    model(**step_inputs(data,[10]),past_key_values=cache,use_cache=True)
        self.assertEqual(hook_footprint(model),footprint)

    def test_input_hash_selection_ignores_outputs(self):
        ss=[dict(task=t,question_id=t+str(i),image_sha256=f'{t}/{i}') for t in TASKS for i in range(80)]
        a=select(ss);b=select(list(reversed(ss)))
        self.assertEqual(a,b);self.assertEqual(len(a),240)
        self.assertEqual(len({s['image_sha256'] for s in a}),240)
        self.assertEqual(length_group([10,2]),'One-token')
        self.assertEqual(length_group([10,11,2]),'Multi-token')

    def test_eos_content_and_group_statistics(self):
        def g(tokens,text,score):
            return dict(token_ids=tokens,text=text,prediction=text,score=score,eos=tokens[-1]==2,length=len(tokens))
        a=g([10,2],'a',1);b=g([10,11,2],'a b',0)
        d=differences(a,b)
        self.assertTrue(d['first_token_equal']);self.assertTrue(d['first_difference_involves_eos'])
        self.assertEqual(d['first_difference_token_ordinal'],2)
        self.assertTrue(d['continuation_tokens_changed']);self.assertEqual(d['b_correct_c_wrong'],1)
        same=differences(a,a)
        self.assertFalse(same['token_sequence_changed']);self.assertEqual(same['first_difference_token_ordinal'],'')
        eos=differences(g([10],'a',1),a)
        self.assertFalse(eos['content_tokens_changed']);self.assertTrue(eos['eos_flag_changed'])
        self.assertTrue(eos['token_change_without_text_change'])
        stat=bootstrap(np.array([[0.],[2.],[10.]]),['a','a','b'],[1],repetitions=1000,seed=4)[0]
        self.assertEqual(stat['mean'],4.);self.assertEqual(stat['clusters'],2)


if __name__=='__main__':unittest.main()
