"""CPU gates for confirmatory cohort, exact edit equivalence and paired statistics."""
import copy
import unittest
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from phase38b_c2_confirm import C2Trace, PLAN, HISTORY, PILOT, read, cohorts
from phase38_sink_hooks import SinkStageTrace
from summarize_phase38b_c2 import bias_row, paired, describe, statistics
from physical_exit_pilot import sequence_score
from roi_sink_probe import score


class ConfirmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(38)
        config=LlamaConfig(vocab_size=40,hidden_size=16,intermediate_size=24,num_hidden_layers=32,
                           num_attention_heads=4,num_key_value_heads=2,attn_implementation='eager')
        config._attn_implementation='eager'
        self.model=LlamaForCausalLM(config).half().eval()
        self.layers=self.model.model.layers
        self.ids=torch.tensor([[1,4,5,6,7,8,9]])

    def run_model(self, trace=None):
        with torch.inference_mode():
            if trace is None: return self.model(self.ids,use_cache=False).logits
            with trace:
                result=self.model(self.ids,use_cache=False).logits
                trace.assert_complete()
                return result

    def test_native_observer_and_restore(self):
        full=self.run_model()
        trace=C2Trace(self.layers,5,'C0')
        self.assertTrue(torch.equal(full,self.run_model(trace)))
        self.assertFalse(trace.edits)
        self.run_model(C2Trace(self.layers,5,'C2'))
        self.assertTrue(torch.equal(full,self.run_model()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_original_phase38_c2_bitwise_equivalence(self):
        new=C2Trace(self.layers,5,'C2')
        old=SinkStageTrace(self.layers,self.model.model.norm,5,0,[1,2],[3],[0,3,4],[],'C2')
        a,b=self.run_model(new),self.run_model(old)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(new.edits,old.edits)
        self.assertEqual([e['layer'] for e in new.edits],[2,3,4])

    def test_causal_prefix_and_l1(self):
        a,b=C2Trace(self.layers,5,'C0'),C2Trace(self.layers,5,'C2')
        full,changed=self.run_model(a),self.run_model(b)
        self.assertEqual(a.prefix,b.prefix)
        self.assertEqual(a.hidden[1],b.hidden[1])
        self.assertEqual(a.attention[2],b.attention[2])
        self.assertTrue(torch.equal(full[:,:5],changed[:,:5]))
        self.assertFalse(torch.equal(full[:,5],changed[:,5]))
        for edit in b.edits:
            self.assertTrue(edit['other_tokens_bitwise_exact'])
        # Future token 6 is deliberately not required to retain its hidden state.

    def test_exception_cleanup_and_weights(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with C2Trace(self.layers,5,'C2'):
                raise RuntimeError('intentional')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in self.model.state_dict().items()))

    def test_frozen_cohorts(self):
        original=read(HISTORY/'manifest.json')['samples']
        discovery=read(PILOT/'manifest.json')['config']['samples']
        before=copy.deepcopy(original)
        samples=cohorts(original,discovery)
        self.assertEqual(original,before)
        self.assertEqual(sum(s['cohort']=='Confirmation224' for s in samples),224)
        self.assertEqual(len({s['img_id'] for s in samples}),240)
        self.assertEqual(PLAN['smoke_ids'],['02474907','00784767'])
        with self.assertRaises(ValueError):cohorts(original[:-1]+[original[0]],discovery)
        with self.assertRaises(ValueError):cohorts(original,discovery[:-1])

    def test_bias_directions_and_ties(self):
        full=dict(question_id='x',img_id='i',gold='right',cohort='Confirmation224',left_logp=-2.,right_logp=-3.,margin=-1.,correct=False)
        row=bias_row(full)
        self.assertEqual(row['margin'],0.)
        self.assertFalse(row['correct'])
        full.update(gold='left',margin=1.,correct=True)
        row=bias_row(full)
        self.assertEqual(row['margin'],0.)
        self.assertTrue(row['correct_to_wrong'])
        self.assertNotIn('gold_logp',row)

    def test_exact_mcnemar(self):
        p=paired([False]*4,[True]*4)
        self.assertEqual(p['exact_mcnemar_two_sided_p'],.125)
        self.assertEqual(p['net_corrections'],4)
        self.assertEqual(paired([True,False],[False,True])['exact_mcnemar_two_sided_p'],1.)
        self.assertEqual(paired([True],[True])['exact_mcnemar_two_sided_p'],1.)

    def test_distribution_preserves_negative_extremes(self):
        d=describe([-100,0,1,2])
        self.assertEqual(d['minimum'],-100.)
        self.assertEqual(d['negative'],1)
        self.assertEqual(d['positive'],2)
        self.assertAlmostEqual(d['std'],np.std([-100,0,1,2],ddof=1))
        with self.assertRaises(ValueError):describe([float('nan')])

    def test_bootstrap_and_absolute_logp(self):
        items=[]
        for j in range(4):
            full=dict(question_id=str(j),img_id=str(j),question='original question',gold='right',
                      cohort='Confirmation224',left_logp=-2.,right_logp=-3.,gold_logp=-3.,wrong_logp=-2.,margin=-1.,correct=False)
            c2=dict(full,right_logp=-1.,left_logp=-1.5,gold_logp=-1.,wrong_logp=-1.5,margin=.5,correct=True,gain=1.5,
                    gold_logp_gain=2.,wrong_logp_gain=.5)
            items.append(dict(C0=full,C2=c2,B1=bias_row(full),delta_d=1.5))
        a,b=statistics(items),statistics(items)
        self.assertEqual(a,b)
        self.assertEqual(a['ci95']['mean_gain'],[1.5,1.5])
        self.assertEqual(a['ci95']['c2_minus_full_accuracy'],[1.,1.])
        self.assertEqual(a['both_candidates_absolute_logp_increase']['count'],4)
        self.assertEqual(a['c2_vs_full']['exact_mcnemar_two_sided_p'],.125)

    def test_original_full_vocabulary_rescoring(self):
        logits=torch.tensor([1.,2.,3.,4.],dtype=torch.float16)
        rec=[dict(token_id=2)]
        self.assertEqual(score(dict(prediction_logits=logits),rec),
                         sequence_score(logits[None],[r['token_id'] for r in rec]))

    def test_cleanup_after_actual_forward_failure(self):
        def fail(module,args,out): raise RuntimeError('actual forward failure')
        handle=self.layers[6].mlp.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'actual forward failure'):
                self.run_model(C2Trace(self.layers,5,'C2'))
        finally:handle.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))


if __name__=='__main__':unittest.main()
