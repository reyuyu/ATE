import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM,AutoProcessor
from benchmarks.smoke import hook_footprint
from early_source_delay.protocol import CONDITIONS,KEYS,select,true_mapping,old
from early_source_delay.engine import observed,SourceMask,key_groups
from early_source_delay.statistics import interaction,intersections,summarize
from knockout_probe import knockout_mask,layout_from_inputs


class Tests(unittest.TestCase):
    def test_conditions_and_budget(self):
        self.assertEqual(len(CONDITIONS),11)
        self.assertEqual(240*len(CONDITIONS),2640)
        self.assertEqual([k for k in CONDITIONS if k.startswith('A')],['A1','A2','A3','A4','A5','A6'])

    def test_hash_sampling_outcome_blind(self):
        pool=old.cohort(); a=select(pool)
        b=select([dict(s,answer='not used',score=-999,margin=999) for s in reversed(pool)])
        self.assertEqual([(s['task'],s['question_id']) for s in a],[(s['task'],s['question_id']) for s in b])
        self.assertEqual(len(a),240)
        self.assertTrue(all(sum(s['task']==t for s in a)==40 for t in old.TASKS))

    def test_true_question_excludes_instruction(self):
        s=old.cohort()[0]
        p=AutoProcessor.from_pretrained(old.MODEL_PATH,local_files_only=True)
        m=true_mapping(p,s); ids=s['input_mapping']['prompt_ids']
        self.assertEqual(p.tokenizer.decode([ids[i] for i in m['question']],clean_up_tokenization_spaces=False),s['question'])
        self.assertLess(len(m['question']),len(s['input_mapping']['question_and_instruction_positions']))
        self.assertFalse(set(m['question'])&set(m['template']))
        self.assertEqual(sum(len(x) for x in key_groups(m).values()),len(ids))

    def test_interaction_sign(self):
        v=np.ones((4,12)); v[0,KEYS.index('EarlyOFF_L2Image')]=0
        np.testing.assert_array_equal(interaction(v,'Image'),[1,0,0,0])
        v[0,KEYS.index('L2Image')]=0
        np.testing.assert_array_equal(interaction(v,'Image'),[0,0,0,0])

    def test_intersections(self):
        rows=[{k:dict(score=int(k not in ['A1','A2'])) for k in KEYS}]
        r=next(r for r in intersections(rows) if r['condition_a']=='A1' and r['condition_b']=='A2')
        self.assertEqual(r['full_correct_joint_harm'],1)

    def test_summary_layout_and_comparison_families(self):
        rows=[{k:dict(score=1,prediction='yes',truncated=False,first_top1_flip=False,first_logp_loss=0.) for k in KEYS} for _ in range(4)]
        rows[0]['EarlyOFF_L2Image']['score']=0
        def fake(values,identities,families,seed):
            self.assertEqual(values.shape,(4,69));self.assertEqual(len(families),69)
            self.assertEqual(families[12:18],[6]*6);self.assertEqual(families[-2:],[2,2])
            return [dict(mean=float(v),ci95_low=float(v),ci95_high=float(v),family95_low=float(v),
                family95_high=float(v),family=f,repetitions=200000,degenerate_bootstrap=True)
                for v,f in zip(values.mean(0),families)]
        with patch('early_source_delay.statistics.bootstrap',fake):
            summary,inter=summarize(rows,['a','b','c','d'])
        self.assertEqual(len(summary),12);self.assertEqual(inter[0]['D_pp'],25.)
        self.assertEqual(inter[1]['D_pp'],0.)

    def setup_model(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        config=LlamaConfig(vocab_size=48,hidden_size=16,intermediate_size=24,num_hidden_layers=32,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,attn_implementation='eager',eos_token_id=2)
        config.image_token_index=39
        model=LlamaForCausalLM(config).eval()
        p=dict(input_ids=torch.tensor([[1,39,39,4,5,6]]),attention_mask=torch.ones(1,6,dtype=torch.long))
        m=dict(bos=[0],image=[1,2],question=[3,4],template=[5],query=5)
        return model,p,m

    def patches(self,stack):
        for module in ['early_source_delay.engine','l0l1_bos.capture','paper_rq1_early_joint_attention.engine']:
            stack.enter_context(patch(module+'.model_layers',lambda m:m.model.layers))

    def test_local_masks_keep_causal_and_padding(self):
        model,p,m=self.setup_model()
        n=6; native=torch.zeros(1,1,n,n).masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf)
        layout=layout_from_inputs(p,39,2); layout=dict(layout,text_queries=torch.tensor([False]*5+[True]))
        new,count=knockout_mask(native,layout)
        self.assertEqual(count,2); self.assertTrue(torch.equal(native[0,0,0],new[0,0,0]))
        self.assertTrue(torch.isneginf(new[0,0,5,1:3]).all())
        self.assertTrue((new==0).any(-1).all())
        bad=dict(layout,text_queries=torch.ones(6,dtype=torch.bool),visual=torch.ones(6,dtype=torch.bool))
        with self.assertRaises(ValueError):knockout_mask(native,bad)

    def test_real_tiny_cached_noop_scope_and_rows(self):
        model,p,m=self.setup_model(); fp=hook_footprint(model)
        with ExitStack() as stack:
            self.patches(stack)
            g,first,base,stats,audit,trace=observed(model,p,m,{},audit=True,collect=True,expected_visual=2)
            for name,c in CONDITIONS.items():
                noop,nf,na,ns,au,nt=observed(model,p,m,c,noop=True,audit=True,collect=True,expected_visual=2)
                self.assertEqual(g['token_ids'],noop['token_ids'],name)
                self.assertEqual(g['initial_cache_sha256'],noop['initial_cache_sha256'],name)
                self.assertTrue(torch.equal(first,nf),name)
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(trace,nt)),name)
                changed,cf,ca,cs,au,ct=observed(model,p,m,c,audit=True,collect=True,expected_visual=2)
                self.assertEqual(int(cf.argmax()),changed['token_ids'][0])
                self.assertEqual(au['decode_mask_edits'],0)
                self.assertEqual(changed['edits'],2 if c.get('early_off') else 0)
                self.assertEqual(au['mask_prefill_edits'],int('layer'in c))
                if 'layer'in c:
                    self.assertEqual(len(au['local_checks']),1)
                    if c['query']=='last':
                        keys=m[c['source']]
                        self.assertEqual(np.count_nonzero(ca[f'attention_row_{c["layer"]}'][:,keys]),0)
                for layer in range(2,6):
                    for kind in ['key','value']:
                        np.testing.assert_array_equal(base[f'bos_{kind}_{layer}'],ca[f'bos_{kind}_{layer}'])
                self.assertEqual(hook_footprint(model),fp)

    def test_exception_hook_cleanup(self):
        model,p,m=self.setup_model();fp=hook_footprint(model)
        with ExitStack() as stack:
            self.patches(stack)
            with patch.object(model.model.layers[1].self_attn.q_proj,'forward',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    observed(model,p,m,CONDITIONS['A3'],audit=True,expected_visual=2)
        self.assertEqual(hook_footprint(model),fp)

    def test_signature_guard_remains_enabled(self):
        model,p,m=self.setup_model();fp=hook_footprint(model)
        with ExitStack() as stack:
            self.patches(stack)
            with patch.object(model.model.layers[1].self_attn,'forward',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(ValueError,'Unsupported LlamaAttention.forward signature'):
                    observed(model,p,m,CONDITIONS['A3'],audit=True,expected_visual=2)
        self.assertEqual(hook_footprint(model),fp)


if __name__=='__main__':unittest.main()
