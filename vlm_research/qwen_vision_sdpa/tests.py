"""Small meaningful CPU gates; never initialize CUDA or load model weights."""
import copy
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from .common import *


class CPU(unittest.TestCase):
    def test_official_modality_configuration(self):
        from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
        c=Qwen3VLConfig.from_pretrained(MODELS['qwen'],local_files_only=True)
        self.assertEqual(c.text_config.num_hidden_layers,36)
        self.assertEqual(c.vision_config.depth,27)
        c.vision_config.depth=2;c.vision_config.hidden_size=128;c.vision_config.num_heads=4
        c.vision_config.intermediate_size=256;c.vision_config.out_hidden_size=128
        c.vision_config.deepstack_visual_indexes=[0,1]
        c.text_config.num_hidden_layers=2;c.text_config.hidden_size=128
        c.text_config.intermediate_size=256;c.text_config.num_attention_heads=1
        c.text_config.num_key_value_heads=1;c.text_config.head_dim=128;c.text_config.vocab_size=256
        m=Qwen3VLForConditionalGeneration._from_config(c,attn_implementation=BACKENDS)
        self.assertTrue(all(b.attn.config._attn_implementation=='sdpa' for b in m.model.visual.blocks))
        self.assertTrue(all(b.self_attn.config._attn_implementation=='eager' for b in m.model.language_model.layers))
        m.set_attn_implementation({'vision_config':'eager','text_config':'eager'})
        self.assertTrue(all(b.attn.config._attn_implementation=='eager' for b in m.model.visual.blocks))
        m.set_attn_implementation(BACKENDS)
        self.assertEqual(m.config.text_config._attn_implementation,'eager')
        self.assertEqual(m.config.vision_config._attn_implementation,'sdpa')

    def test_checkpoint_resume_and_corruption(self):
        tag={k:'frozen' for k in ('source_manifest_sha256','protocol_sha256','sample_manifest_sha256','model_manifest_sha256')}
        s={'shard':0,'key':'synthetic'}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'Native.json'
            self.assertFalse(checkpoint_complete(p,tag,s))
            r=dict(**tag,sample_sha256=digest(s),uuid=UUIDS[0],result={'correct':True})
            write_checkpoint(p,r);self.assertTrue(checkpoint_complete(p,tag,s))
            p.write_text('{}')
            with self.assertRaises(AssertionError):checkpoint_complete(p,tag,s)

    def test_frozen_coverage_and_shards(self):
        rows=read(OUT/'sample_manifest.json');b=read(OUT/'qwen_input_bindings.json.gz')
        self.assertEqual(len(rows),16607);self.assertEqual(len(b),16607)
        shards={}
        for s in rows:
            self.assertEqual(shards.setdefault(s['image_sha256'],s['shard']),s['shard'])
            x=b[s['key']];self.assertLessEqual(x['image_tokens'],2048)
            ids=x['input_ids'];im=[i for i,x in enumerate(ids) if x==151655]
            self.assertEqual(len(im),x['image_tokens']);self.assertEqual(im,list(range(im[0],im[-1]+1)))
            if s['benchmark']=='SEED':self.assertEqual(len(x['candidate_token_ids']),4)
        ids=[s for s in rows if s['benchmark']=='SEED' and str(s['source_question_id'])=='49914']
        self.assertEqual(len(ids),2)

    def test_numerical_and_backend_gates_frozen(self):
        p=read(OUT/'protocol.json')
        self.assertEqual(p['backend'],BACKENDS)
        self.assertEqual(p['preserved_dense_physical_strict_gate']['mean_abs'],0.02)
        self.assertEqual(p['preserved_dense_physical_strict_gate']['max_abs'],0.5)
        self.assertTrue(p['no_timing'] and p['no_search'] and p['no_profiler'])


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPU))
    atomic(OUT/'cpu_validation.json',dict(status='PASSED' if result.wasSuccessful() else 'FAILED',
        tests=result.testsRun,failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],
        cuda_used=False,at=now(),source_manifest_sha256=sha(OUT/'source_manifest.json')))
    assert result.wasSuccessful()


if __name__=='__main__':main()
