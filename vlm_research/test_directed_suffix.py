"""CPU tests of actual mask orientation, causality and safe cleanup."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from test_knockout import tiny_model,tiny_inputs
from directed_suffix import (DirectedSuffixKnockout,MODES,CUTOFFS,query_mapping,
                             directed_forward,intervention)
from knockout_probe import model_layers
from physical_exit import reference_forward
from directed_suffix_probe import KEYS,checkpoint,read_checkpoint


class DirectedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.model=tiny_model()

    def setUp(self):
        self.inputs=tiny_inputs();self.q=[6,7,8];self.P=11
        self.records=[dict(query_position=10,target_token_position=11,token_id=17)]
        self.mapping=query_mapping(self.inputs,99,self.q,self.records,self.P,4)

    def forward(self,mode,k=16,audit=False):
        return directed_forward(self.model,self.inputs,mode,k,self.mapping,audit,4)

    def test_prediction_query_is_prompt_end_not_input_answer(self):
        self.assertEqual(self.mapping['answer_prediction_queries'],[10])
        self.assertEqual(self.mapping['answer_token_positions'],[11])
        self.assertFalse(set(self.mapping['question_queries'])&set(self.mapping['answer_prediction_queries']))
        for q,records in [(self.q,[dict(query_position=11,target_token_position=11)]),
                          ([6,10],self.records),([3,6],self.records),([6,6],self.records),([],self.records)]:
            with self.assertRaises(ValueError):query_mapping(self.inputs,99,q,records,11,4)

    def test_mask_orientation_exact_target_only_suffix_and_clone(self):
        originals=[];handles=[]
        for layer in model_layers(self.model):
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda m,a,k:originals.append(k['attention_mask'].clone()),with_kwargs=True))
        try:self.forward('full')
        finally:
            for h in handles:h.remove()
        for mode in MODES:
            for k in CUTOFFS:
                seen=[];handles=[]
                with intervention(self.model,self.inputs,mode,k,self.mapping,4) as hook:
                    for layer in model_layers(self.model):
                        handles.append(layer.self_attn.register_forward_pre_hook(
                            lambda m,a,kw:seen.append(kw['attention_mask'].clone()),with_kwargs=True))
                    try:
                        with torch.inference_mode():self.model(**self.inputs,use_cache=False)
                    finally:
                        for h in handles:h.remove()
                hook.assert_one_forward()
                qs={'question_only':self.q,'answer_only':[10],'question_answer':self.q+[10],'global':list(range(6,12))}[mode]
                target=torch.zeros(12,12,dtype=torch.bool);target[qs,2:6]=True
                for index,(a,b) in enumerate(zip(originals,seen)):
                    with self.subTest(mode=mode,k=k,layer=index):
                        if index<k:self.assertTrue(torch.equal(a,b))
                        else:
                            self.assertTrue(torch.equal(a[0,0][~target],b[0,0][~target]))
                            self.assertTrue(torch.isneginf(b[0,0][target]).all())
                            # Opposite orientation visual Query -> text Key is unchanged.
                            self.assertTrue(torch.equal(a[:,:,2:6,6:],b[:,:,2:6,6:]))
                self.assertEqual([r['applied'] for r in hook.records],[i>=k for i in range(32)])
                self.assertTrue(all(r['cloned_mask']==(r['layer']>=k) for r in hook.records))

    def test_actual_attention_zero_all_modes_and_cutoffs(self):
        for mode in MODES:
            for k in CUTOFFS:
                with self.subTest(mode=mode,k=k):
                    out=self.forward(mode,k,True)
                    self.assertEqual(len(out['attention_audit']),32)
                    self.assertTrue(all(r['target_attention_max']==0 for r in out['attention_audit'] if r['layer']>=k))
                    self.assertTrue(torch.isfinite(out['logits']).all())
                    self.assertTrue(torch.isfinite(out['hidden']).all())

    def test_k32_exact_native_and_attentions_flag_invariant(self):
        full=self.forward('full')
        for mode in MODES:
            other=self.forward(mode,32,True)
            self.assertTrue(torch.equal(full['logits'],other['logits']))
            self.assertTrue(torch.equal(full['hidden'],other['hidden']))

    def test_causal_question_hidden_invariants(self):
        full=self.forward('full')
        for k in CUTOFFS:
            q=self.forward('question_only',k);a=self.forward('answer_only',k);qa=self.forward('question_answer',k)
            self.assertTrue(torch.equal(full['hidden'][:,self.q],a['hidden'][:,self.q]))
            self.assertTrue(torch.equal(q['hidden'][:,self.q],qa['hidden'][:,self.q]))

    def test_global_reuses_original_phase3_mask_numerically(self):
        for k in CUTOFFS:
            original=reference_forward(self.model,self.inputs,k,4)
            out=self.forward('global',k)
            self.assertTrue(torch.equal(original['logits'],out['logits']))
            self.assertTrue(torch.equal(original['hidden'],out['hidden']))

    def test_input_answer_token_is_causally_irrelevant_to_prediction(self):
        baseline=self.forward('question_answer')
        old=self.inputs;self.inputs=dict(old,input_ids=old['input_ids'].clone())
        self.inputs['input_ids'][0,11]=18
        other=self.forward('question_answer')
        self.assertTrue(torch.equal(baseline['logits'][:,:11],other['logits'][:,:11]))

    def test_no_mutation_or_hook_leak_including_exception(self):
        model=self.model;full=self.forward('full');params=[p.clone() for p in model.parameters()]
        inputs={k:v.clone() for k,v in self.inputs.items()}
        before=[len(l.self_attn._forward_pre_hooks) for l in model_layers(model)]
        for mode in MODES:self.forward(mode)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(params,model.parameters())))
        self.assertTrue(all(torch.equal(v,self.inputs[k]) for k,v in inputs.items()))
        def fail(*args):raise RuntimeError('injected failure')
        handle=model_layers(model)[20].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected'):self.forward('question_answer')
        finally:handle.remove()
        self.assertEqual(before,[len(l.self_attn._forward_pre_hooks) for l in model_layers(model)])
        self.assertTrue(torch.equal(full['logits'],self.forward('full')['logits']))

    def test_invalid_targets_fail_closed(self):
        for q,a in [([6],[6]),([6,6],[]),([3],[]),([],[]),([30],[])]:
            with self.assertRaises(ValueError):DirectedSuffixKnockout(model_layers(self.model),self.inputs,99,16,q,a,4)

    def test_checkpoint_resume_rejects_tamper_and_incomplete_keys(self):
        sample=dict(question_id='0001',img_id='2',answer='left')
        data=dict(rows=[dict(mode=m,cutoff=k,**sample) for m,k in KEYS],
                  checks={key:True for key in ('k32_exact','causal_hidden_exact','global_history_within_tolerance','full_history_exact')})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json';checkpoint(path,'binding',sample,data)
            self.assertEqual(read_checkpoint(path,'binding',sample),data)
            with self.assertRaises(ValueError):checkpoint(path,'binding',sample,data)
            with self.assertRaises(ValueError):read_checkpoint(path,'changed',sample)
            envelope=json.loads(path.read_text());envelope['body']['data']['rows'].pop()
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError):read_checkpoint(path,'binding',sample)
        altered=copy.deepcopy(data);altered['rows'][-1]=altered['rows'][0]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):checkpoint(Path(d)/'x.json','binding',sample,altered)


if __name__=='__main__':unittest.main()
