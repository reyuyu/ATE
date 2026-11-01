import unittest

import numpy as np
import torch

from benchmarks.smoke import hook_footprint
from rq2_prompt_five.engine import S,CONDITIONS,MultiPromptKnockout,generate,native_reference
from rq2_prompt_five.run import selection,assign,pool,TASKS,read,discovery
from rq2_prompt_five.analyze import contrast_values,transitions
from rq2_source_knockout.engine import forward
from test_rq2_source_knockout import Tiny,inputs


class Tests(unittest.TestCase):
    def test_frozen_conditions(self):
        self.assertEqual(S,[1,13,14,15,16])
        self.assertEqual(len(CONDITIONS),5)
        self.assertEqual(set(CONDITIONS['D']),set(range(32))-set(S))
        self.assertEqual(CONDITIONS['E'],list(range(32)))

    def test_exact_partition_dynamic_query(self):
        model=Tiny()
        for extra in [0,2]:
            data=inputs()
            if extra:
                data['input_ids']=torch.cat([data['input_ids'],torch.tensor([[14,15]])],1)
                data['attention_mask']=torch.ones_like(data['input_ids'])
            mask=MultiPromptKnockout(model,data,1,[1],expected_visual=4)
            self.assertEqual(mask.query,8+extra)
            self.assertEqual(int(mask.groups['Prompt'].sum()),3+extra)
            self.assertFalse(mask.layout['visual'][mask.query])
            self.assertFalse((mask.layout['visual']&(mask.groups['Image']|mask.groups['BOS'])).any())

    def test_noop_native_generation_and_cleanup(self):
        torch.manual_seed(317);model=Tiny();data=inputs();footprint=hook_footprint(model)
        full,first,trace,debug=generate(model,data,1,[],limit=4,audit=True,collect=True,expected_visual=4)
        noop,logits,nt,nd=generate(model,data,1,[0,1,2],limit=4,noop=True,audit=True,collect=True,expected_visual=4)
        self.assertEqual(full['token_ids'],noop['token_ids']);self.assertTrue(torch.equal(first,logits))
        self.assertTrue(torch.equal(debug['prefill_logits'],nd['prefill_logits']))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(trace,nt)))
        self.assertEqual(native_reference(model,data,1,[],4,4),full['token_ids'])
        self.assertEqual(footprint,hook_footprint(model))

    def test_legacy_single_mask_and_native_decode(self):
        torch.manual_seed(111);model=Tiny();data=inputs();before=hook_footprint(model)
        full,first,_,native=generate(model,data,1,[],limit=4,audit=True,expected_visual=4)
        for blocked in [[0],[1],[0,1,2]]:
            g,changed,_,audit=generate(model,data,1,blocked,limit=4,audit=True,expected_visual=4)
            self.assertEqual(g['token_ids'],native_reference(model,data,1,blocked,4,4))
            self.assertTrue(torch.equal(audit['prefill_logits'][:,:-1],native['prefill_logits'][:,:-1]))
            self.assertEqual(audit['observer'].states,native['observer'].states)
            for k in blocked:self.assertEqual(audit['rows'][k]['Prompt_max'],0)
            self.assertEqual(footprint:=hook_footprint(model),before)
            if len(blocked)==1:
                old,_,_=forward(model,data,1,blocked[0],'Prompt',expected_visual=4)
                self.assertTrue(torch.equal(changed,old))
        with self.assertRaises(RuntimeError):
            with MultiPromptKnockout(model,data,1,[0,1,2],expected_visual=4):raise RuntimeError('exception cleanup')
        self.assertEqual(footprint,hook_footprint(model))

    def test_selection_disjoint_and_whole_image_shards(self):
        samples=selection();self.assertEqual(len(samples),240)
        self.assertEqual(len(pool()),2656)
        self.assertEqual({t:sum(s['task']==t for s in samples) for t in TASKS},{t:40 for t in TASKS})
        original=read(discovery.OUT/'sample_manifest.json')['samples']
        self.assertFalse({s['image_sha256'] for s in samples}&{s['image_sha256'] for s in original})
        for count in [1,2,3]:
            ranks=assign(samples,count);self.assertEqual(ranks,assign(list(reversed(samples)),count))
            images={}
            for s in samples:self.assertEqual(images.setdefault(s['image_sha256'],ranks[s['question_id']]),ranks[s['question_id']])
            self.assertLessEqual(max(list(ranks.values()).count(k) for k in range(count))-min(list(ranks.values()).count(k) for k in range(count)),2)

    def test_paired_contrasts_and_error_lists(self):
        scores=np.array([[1,1,1,0,1,0],[0,1,0,1,0,0]],dtype=float)
        values=contrast_values(scores)
        self.assertEqual(values.shape,(2,8))
        np.testing.assert_array_equal(values[:,-1],[-100,0])
        self.assertEqual(transitions(scores[:,0],scores[:,3],['a','b']),dict(harm=['a'],fix=['b']))


if __name__=='__main__':unittest.main()
