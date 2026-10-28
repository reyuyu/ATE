"""CPU synthetic correctness tests; random outputs are not research observations."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from test_knockout import tiny_model,tiny_inputs
from knockout_probe import model_layers,VisualAttentionKnockout,layout_from_inputs,knockout_mask
from directed_suffix import DirectedSuffixKnockout
from phase3_7.pathways import PathwayMask,observed_forward,spatial,js_cosine,drift,footprint,state_hash
from phase3_7.protocol import CONDITIONS,RULES,select,digest,atomic
from phase3_7.run import checkpoint


class PathwaysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.model=tiny_model()

    def setUp(self):
        self.x=tiny_inputs();self.q=[6,7,8];self.a=10

    def forward(self,condition='B0',**kwargs):
        return observed_forward(self.model,self.x,self.q,self.a,condition,expected_visual=4,**kwargs)

    def test_only_five_fixed_conditions_and_two_layers(self):
        self.assertEqual(CONDITIONS,['B0','B1','B2','B3','B4']);self.assertEqual(RULES['layers'],[0,1])
        for mode in CONDITIONS[1:]:
            out=self.forward(mode)
            self.assertEqual([r['applied'] for r in out['audit']['mask_records']],[i in (0,1) for i in range(32)])

    def test_exact_query_key_orientation_and_native_later_masks(self):
        native=[];handles=[]
        for l in model_layers(self.model):
            handles.append(l.self_attn.register_forward_pre_hook(lambda m,a,k:native.append(k['attention_mask'].clone()),with_kwargs=True))
        try:
            with torch.inference_mode():self.model(**self.x,use_cache=False)
        finally:
            for h in handles:h.remove()
        for mode in CONDITIONS[1:]:
            seen=[];handles=[]
            with PathwayMask(self.model,self.x,self.q,self.a,mode,4):
                for l in model_layers(self.model):
                    handles.append(l.self_attn.register_forward_pre_hook(lambda m,a,k:seen.append(k['attention_mask'].clone()),with_kwargs=True))
                try:
                    with torch.inference_mode():self.model(**self.x,use_cache=False)
                finally:
                    for h in handles:h.remove()
            qs={'B1':list(range(6,12)),'B2':[10],'B3':self.q,'B4':[10]}[mode]
            keys=self.q if mode=='B4' else list(range(2,6))
            for i,(a,b) in enumerate(zip(native,seen)):
                region=torch.zeros(12,12,dtype=torch.bool)
                if i<2:region[np.ix_(qs,keys)]=True
                self.assertTrue(torch.equal(a[0,0][~region],b[0,0][~region]))
                self.assertTrue(torch.isneginf(b[0,0][region]).all())
                self.assertTrue((b[0,0]==0).any(-1).all())
                self.assertTrue(torch.equal(a[:,:,2:6,:],b[:,:,2:6,:]))

    def test_actual_blocked_attention_zero_and_other_queries_local_exact(self):
        for mode in CONDITIONS[1:]:
            out=self.forward(mode)
            for r in out['audit']['local_redistribution']:
                self.assertEqual(r['masked_mass_after'],0.)
                self.assertTrue(r['non_target_queries_bitwise_exact'])
                self.assertGreater(np.mean(r['masked_mass_before']),0)
                self.assertLess(r['max_target_row_mass_error'],.002)
            self.assertTrue(all(r['future_attention_max']==0 for r in out['audit']['layer_attention']))

    def test_native_logits_and_decoder_block_hidden_match_observation(self):
        with torch.inference_mode():
            native=self.model(**self.x,use_cache=False,output_hidden_states=True)
        out=self.forward()
        self.assertTrue(torch.equal(native.logits[0,self.a],out['prediction_logits']))
        for i in (0,1):
            np.testing.assert_array_equal(out['arrays']['hidden'][i],native.hidden_states[i+1][0,self.q+[self.a]].numpy())

    def test_noop_effective_text_logits_bitwise_full(self):
        full=self.forward()
        for mode in CONDITIONS[1:]:
            value=self.forward(mode,noop=True)
            self.assertTrue(torch.equal(full['text_logits'],value['text_logits']))
            np.testing.assert_array_equal(full['arrays']['hidden'],value['arrays']['hidden'])

    def test_global_reuses_original_knockout(self):
        with torch.inference_mode(),VisualAttentionKnockout(model_layers(self.model),self.x,99,0,4,only_layers=[0,1]):
            reference=self.model(**self.x,use_cache=False).logits[0,self.a]
        self.assertTrue(torch.equal(reference,self.forward('B1')['prediction_logits']))

    def test_directed_reuses_original_class(self):
        for mode,q,a in [('B2',[],[10]),('B3',self.q,[])]:
            hook=DirectedSuffixKnockout(model_layers(self.model),self.x,99,0,q,a,4);hook.active={0,1}
            with torch.inference_mode(),hook:
                reference=self.model(**self.x,use_cache=False).logits[0,self.a]
            self.assertTrue(torch.equal(reference,self.forward(mode)['prediction_logits']))

    def test_prediction_only_causal_question_invariant(self):
        full=self.forward()
        for mode in ('B2','B4'):
            value=self.forward(mode)
            np.testing.assert_array_equal(full['arrays']['hidden'][:,:-1],value['arrays']['hidden'][:,:-1])
            np.testing.assert_array_equal(full['arrays']['final_hidden'][:-1],value['arrays']['final_hidden'][:-1])

    def test_softmax_redistributes_not_preserves_other_keys(self):
        out=self.forward('B2')
        for r in out['audit']['local_redistribution']:
            before=np.asarray(r['before_group_mass']);after=np.asarray(r['after_group_mass'])
            self.assertGreater(float(np.mean(after[:,:,1:]-before[:,:,1:])),0)
            np.testing.assert_allclose(after.sum(-1),1,atol=1e-6)

    def test_l0_same_position_prediction_visual_qk_invariant(self):
        first=self.forward(probe=True)
        self.x['input_ids'][0,self.q]=torch.tensor([31,32,33])
        second=self.forward(probe=True)
        np.testing.assert_array_equal(first['arrays']['l0_visual_scores'],second['arrays']['l0_visual_scores'])
        np.testing.assert_array_equal(first['arrays']['l0_visual_only_conditional'],second['arrays']['l0_visual_only_conditional'])
        self.assertEqual(first['audit']['l0_answer_input_sha256'],second['audit']['l0_answer_input_sha256'])

    def test_dynamic_shifted_mapping_not_single_case_indices(self):
        for key in ('input_ids','attention_mask'):
            prefix=torch.ones((1,1),dtype=self.x[key].dtype)
            self.x[key]=torch.cat([prefix,self.x[key]],1)
        self.q=[p+1 for p in self.q];self.a+=1
        out=self.forward('B4')
        self.assertEqual(out['audit']['local_redistribution'][0]['query_indices'],[11])
        self.assertEqual(out['audit']['local_redistribution'][0]['key_indices'],[7,8,9])

    def test_invalid_indices_modes_and_training_fail_closed(self):
        for q,a,mode in [([],10,'B1'),([-1],10,'B4'),([6,6],10,'B3'),([10],10,'B2'),([3],10,'B3'),([6],10,'extra')]:
            with self.assertRaises(ValueError):PathwayMask(self.model,self.x,q,a,mode,4)

    def test_all_keys_masked_is_rejected(self):
        layout=layout_from_inputs(self.x,99,4)
        layout=dict(layout,text_queries=torch.ones(12,dtype=torch.bool),visual=torch.ones(12,dtype=torch.bool))
        mask=torch.full((1,1,12,12),-torch.inf).triu(1)
        with self.assertRaises(ValueError):knockout_mask(mask,layout)

    def test_parameters_inputs_and_all_hooks_unchanged_on_exception(self):
        weights=state_hash(self.model);original=copy.deepcopy(self.x);before=footprint(self.model)
        def fail(*a):raise RuntimeError('injected failure')
        h=model_layers(self.model)[12].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected'):self.forward('B1')
        finally:h.remove()
        self.assertEqual(before,footprint(self.model));self.assertEqual(weights,state_hash(self.model))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in self.x.items()))
        self.forward('B0')

    def test_nonfinite_attention_fails_and_cleans_up(self):
        before=footprint(self.model)
        def corrupt(m,a,out):
            x=out[1].clone();x[0,0,0,0]=float('nan');return(out[0],x)+tuple(out[2:])
        h=model_layers(self.model)[0].self_attn.register_forward_hook(corrupt)
        try:
            with self.assertRaisesRegex(AssertionError,'Nonfinite'):self.forward('B2')
        finally:h.remove()
        self.assertEqual(before,footprint(self.model))


class DefinitionsTest(unittest.TestCase):
    def test_conditional_spatial_pooling_uniform_entropy_and_scale(self):
        x=np.ones((3,4,576));x[:,0]*=7
        a=spatial(x);b=spatial(x*2)
        np.testing.assert_allclose(a['distribution'],1/576)
        np.testing.assert_array_equal(a['distribution'],b['distribution'])
        self.assertAlmostEqual(a['normalized_entropy'],1)
        self.assertAlmostEqual(b['mass'],a['mass']*2)
        self.assertFalse(spatial(np.zeros((2,576)))['available'])

    def test_js_distance_natural_logs_and_cosine(self):
        self.assertEqual(js_cosine([1.,0.],[1.,0.])['js_distance'],0.)
        self.assertAlmostEqual(js_cosine([1.,0.],[0.,1.])['js_distance'],np.sqrt(np.log(2)))
        self.assertEqual(js_cosine([1.,0.],[0.,1.])['cosine'],0.)
        self.assertAlmostEqual(drift([1.,0.],[0.,1.])['change_norm_mean'],np.sqrt(2))

    def test_selection_four_per_historical_stratum_and_leading_zeros(self):
        rows=[]
        for label in ('left','right'):
            for correct in (False,True):
                for i in range(5):
                    q=f'{label}{correct}{i:03}'
                    rows.append(dict(question_id=q,img_id=q,answer=label,historical_correct=correct,
                                     historical_full_margin=(1 if correct else -1)*(i+1),pair=None,matched_pair=False))
        a=select(rows);b=select(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(len(a),16)
        for label in ('left','right'):
            for correct in (False,True):self.assertEqual(sum(s['answer']==label and s['historical_correct']==correct for s in a),4)
        with self.assertRaises(ValueError):select(rows[:2])

    def test_checkpoint_detects_binding_sample_and_condition_tampering(self):
        sample=dict(question_id='001',pair=None)
        body=dict(binding='binding',sample_sha256=digest(sample),passed=True,a=None,question_id='001',
                  b=dict(rows=[dict(condition=c,question_id='001') for c in CONDITIONS]),artifacts={})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'c.json';atomic(path,dict(body=body,sha256=digest(body)))
            self.assertEqual(checkpoint(path,'binding',sample,root),body)
            with self.assertRaises(ValueError):checkpoint(path,'other',sample,root)
            altered=copy.deepcopy(body);altered['b']['rows'].pop();atomic(path,dict(body=altered,sha256=digest(altered)))
            with self.assertRaises(ValueError):checkpoint(path,'binding',sample,root)

    def test_descriptive_flips_count_ties_as_wrong(self):
        from phase3_7.analyze import descriptive
        rows=[dict(question_id='001',gain=-1.,margin=0.,full_margin=1.,full_correct_to_wrong=True,
                   full_wrong_to_correct=False,absolute_correct_logp_gain=-.2),
              dict(question_id='002',gain=2.,margin=1.,full_margin=-1.,full_correct_to_wrong=False,
                   full_wrong_to_correct=True,absolute_correct_logp_gain=.1)]
        r=descriptive(rows)
        self.assertEqual(r['accuracy'],.5);self.assertEqual(r['harmed'],1);self.assertEqual(r['recovered'],1)
        self.assertEqual(r['ties'],1);self.assertEqual(r['mean_gain'],.5)

    def test_checkpoint_rejects_foreign_question_identity(self):
        sample=dict(question_id='001',pair=None)
        body=dict(binding='b',sample_sha256=digest(sample),passed=True,a=None,question_id='other',
                  b=dict(rows=[dict(condition=c,question_id='001') for c in CONDITIONS]),artifacts={})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'point.json';atomic(p,dict(body=body,sha256=digest(body)))
            with self.assertRaises(ValueError):checkpoint(p,'b',sample,Path(d))


if __name__=='__main__':unittest.main()
