import unittest
from unittest.mock import patch
import torch

from .common import *
from ..accuracy_autoresearch.tests import Tests as Inherited
from ..accuracy_autoresearch.engine import ResearchWriter
from ..closedqa_transfer.engine import DenseReference


class Tests(Inherited):
    def test_seven_predeclared_conditions(self):
        self.assertEqual(len(LABELS),7)
        self.assertEqual(CFG,config('all','both',head=1))

    def test_writer_prefix_and_suffix(self):
        m=dict(R=list(range(10,24)),Pminus1=24,
            parts=dict(Q=[10,11],Hint=[12],Options=[13],Instruction=[14,15,16],ASSISTANT=[20,21,22,23]))
        for name,n in (('C3',14),('E1',11),('E2-P1',10),('E2-P2',9),('E2-P3',8),('E2-P4',7),('E2-S2',9)):
            w=writer_set(m,name)
            self.assertEqual(len(w),n)
            self.assertTrue(set([10,11,12,13])<=set(w))
            self.assertNotIn(24,w)
        self.assertNotEqual(writer_set(m,'E2-P2'),writer_set(m,'E2-S2'))

    def test_real_instruction_tokenization(self):
        from transformers import AutoProcessor
        p=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
        closed=read(OLD/'sample_manifest.json')
        for s in (next(s for s in closed.values() if s['dataset']=='scienceqa_img' and s['input_mapping']['parts']['Hint']),
                  next(s for s in closed.values() if s['dataset']=='pope')):
            m=localize(p,s)
            self.assertEqual(m['prompt_ids'],s['input_mapping']['prompt_ids'])
            self.assertTrue(set(m['parts']['Hint']+m['parts']['Options']+m['parts']['Q'])<=set(writer_set(m,'E2-P4')))
            self.assertEqual([m['prompt_ids'][i] for i in m['parts']['ASSISTANT']],ASSISTANT_IDS)

    def test_boundary_crossing_is_rejected(self):
        with self.assertRaises(AssertionError):
            offsets_for(dict(input_ids=[10],offset_mapping=[(0,10)]),3,10,'abcAnswer!')

    def test_exact_mask_only_relay_image(self):
        full=ResearchWriter(self.a,self.layout,CFG,list(self.layout.relay),4)
        writers=list(self.layout.relay)[:-2]
        actual=ResearchWriter(self.a,self.layout,CFG,writers,4)
        expected=full.prefill_early.clone()
        for q in set(self.layout.relay)-set(writers):
            expected[0,0,q,list(self.layout.image)]=-torch.inf
        self.assertTrue(torch.equal(actual.prefill_early,expected))
        self.assertTrue(torch.equal(actual.active_mask,expected[...,:-1,:]))
        self.assertTrue(torch.equal(actual.prefill_late,full.prefill_late))
        self.assertTrue((expected==0).any(-1).all())

    def test_pruning_dense_reference(self):
        writers=list(self.layout.relay)[:-2]
        actual=ResearchWriter(self.a,self.layout,CFG,writers,4)
        dense=DenseReference(self.a,self.layout,dict(CFG,writer='qsa'),writers,4,True)
        with torch.inference_mode(),actual.context():
            x=actual.forward(inputs_embeds=self.h).last_hidden_state
            y=actual.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state
        with torch.inference_mode(),dense.context():
            u=dense.forward(inputs_embeds=self.h).last_hidden_state
            v=dense.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state
        torch.testing.assert_close(x,u[:,list(self.layout.text)],atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(y,v,atol=3e-7,rtol=3e-6)

    def test_scoring_unchanged(self):
        self.assertEqual(scored(dict(dataset='GQA',answer='red'),'red.')['score'],1)
        s=dict(dataset='VQAv2',answers=['yes']*3+['no']*7)
        self.assertAlmostEqual(scored(s,'yes')['score'],.9)
        s=dict(dataset='textvqa',answer=['yes']*3+['no']*7)
        self.assertAlmostEqual(scored(s,'yes')['score'],.9)
        self.assertTrue(scored(dict(dataset='scienceqa_img',answer='A',candidates=['A','B'],gold_index=0),'A.')['parse_failed'])


if __name__=='__main__':
    unittest.main(verbosity=2)
