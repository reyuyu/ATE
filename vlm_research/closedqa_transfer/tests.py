import unittest
import torch

from .common import *
from .engine import DenseReference
from ..accuracy_autoresearch import tests as inherited


class Tests(inherited.Tests):
    def test_fixed_five_conditions(self):
        self.assertEqual(len(CONFIGS),5)
        self.assertEqual(CONFIGS['Full-QSA']['answer_attention'],'decode')
        self.assertEqual(CONFIGS['Historical C4-V2']['writer'],'conditional15')
        self.assertTrue(all(c['logits_to_keep']==1 and c['cutoff']==20 for c in CONFIGS.values()))

    def test_dense_reference_keeps_visual_rows(self):
        from .engine import ResearchWriter
        cfg=config('qsa','decode'); writers=[6,7,8,9]
        actual=ResearchWriter(self.a,self.layout,cfg,writers,4)
        dense=DenseReference(self.a,self.layout,cfg,writers,4,True)
        with torch.inference_mode(),actual.context():
            a=actual.forward(inputs_embeds=self.h).last_hidden_state
            b=actual.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state
        with torch.inference_mode(),dense.context():
            c=dense.forward(inputs_embeds=self.h).last_hidden_state
            d=dense.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state
        torch.testing.assert_close(a,c[:,list(self.layout.text)],atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(b,d,atol=3e-7,rtol=3e-6)
        self.assertTrue(all(r['used_tokens']==15 for r in dense.cache.audit()['layers']))

    def test_mapping_uses_actual_stem_not_hint_or_choices(self):
        from transformers import AutoProcessor
        p=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
        manifest=json.loads(git_bytes(PHASE,'vlm_research/results/phase4_0_dataset_suite/manifests/scienceqa_img.json'))
        s=next(s for s in manifest['samples'] if s['metadata']['raw_row'].get('hint'))
        m=mapping(p,s)
        self.assertTrue(m['parts']['Hint'] and m['parts']['Options'])
        self.assertTrue(set(m['parts']['Q']).isdisjoint(m['parts']['Hint']+m['parts']['Options']))
        self.assertEqual(m['prompt_ids'],s['metadata']['input_mapping']['prompt_ids'])

    def test_frozen_scoring_not_repaired(self):
        s=dict(dataset='scienceqa_img',answer='A',candidates=['A','B'],gold_index=0)
        self.assertEqual(scored(s,'A')['score'],1)
        self.assertTrue(scored(s,'A.')['parse_failed'])
        p=dict(dataset='pope',answer='yes',candidates=['no','yes'],gold_index=None)
        self.assertEqual(scored(p,'')['parsed'],'yes')


if __name__=='__main__': unittest.main(verbosity=2)
