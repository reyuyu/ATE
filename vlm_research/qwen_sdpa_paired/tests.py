"""Small meaningful CPU gates; never initialize CUDA or load model weights."""
import copy
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from .common import *


class CPU(unittest.TestCase):
    def test_compact_kv_content_positions_and_independence(self):
        import torch
        from ..qwen3vl_c3v2_physical.engine import CompactKV,SKIP
        self.assertEqual(SKIP,tuple(range(1,7)))
        pos=torch.tensor([0,3,4,5]);x=torch.arange(24,dtype=torch.float32).reshape(1,1,6,4)
        c=CompactKV(6,pos,8,absent=SKIP)
        k,v=c.update(x,x+100,0)
        self.assertTrue(torch.equal(k,x));self.assertEqual(c.used[0],4)
        self.assertTrue(torch.equal(c.entries[0][0][...,:4,:],x.index_select(-2,pos)))
        d=copy.deepcopy(c);c.prefill=False
        z=torch.ones(1,1,1,4);c.update(z,z,0)
        self.assertEqual(c.used[0],5);self.assertEqual(d.used[0],4)
        self.assertNotEqual(c.entries[0][0].data_ptr(),d.entries[0][0].data_ptr())
        with self.assertRaises(AssertionError):c.update(z,z,1)

    def test_mask_scope_all_layers(self):
        from ..qwen3vl_keep_l0_21.tests import Tests
        Tests('test_all_layer_scopes').test_all_layer_scopes()

    def test_official_mme_and_seed_score_reconstruction(self):
        from ..mme_seed_quality.analyze import mme_boot,seed_boot
        tasks=sum(TASKS.values(),[]);rows=[];n=[];c=[]
        for task in tasks:
            for i in range(4):
                rows.append(dict(task=task,pair_id=task+str(i//2),image_sha256=task+str(i//2)))
                n.append([True,False,True,True][i]);c.append([True,True,False,False][i])
        table,totals,_=mme_boot(rows,n,c)
        self.assertTrue(all(x['native_score']==125 and x['c3_score']==100 for x in table))
        self.assertEqual(totals[-1]['delta_score_points'],-350)
        sr=[dict(task='toy',image_sha256=x) for x in ['a','a','b']]
        stats,_=seed_boot(sr,[True,False,False],[False,False,True])
        self.assertEqual(stats[0]['delta_pp'],0)
        for text,pred in [('Yes.','yes'),('No.','no'),('The answer is yes','other')]:self.assertEqual(mme_parse(text),pred)

    def test_inference_sdpa_composite_observation(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        class Observe(TorchDispatchMode):
            def __init__(self):super().__init__();self.names=[]
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                name=str(func)
                if 'scaled_dot_product' in name:self.names.append(name)
                if name=='aten.scaled_dot_product_attention.default':
                    with self:return func.decompose(*args,**(kwargs or {}))
                return func(*args,**(kwargs or {}))
        with torch.inference_mode():
            x=torch.arange(384,dtype=torch.float32).reshape(1,4,12,8)/384
            a=torch.nn.functional.scaled_dot_product_attention(x,x,x)
            observer=Observe()
            with observer:b=torch.nn.functional.scaled_dot_product_attention(x,x,x)
            self.assertTrue(torch.equal(a,b))
            self.assertIn('aten.scaled_dot_product_attention.default',observer.names)
            self.assertIn('aten._scaled_dot_product_flash_attention_for_cpu.default',observer.names)

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
        self.assertTrue(p['cross_backend_numeric_thresholds_are_diagnostic_only'])


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPU))
    atomic(OUT/'cpu_validation.json',dict(status='PASSED' if result.wasSuccessful() else 'FAILED',
        tests=result.testsRun,failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],
        cuda_used=False,at=now(),source_manifest_sha256=sha(OUT/'source_manifest.json')))
    assert result.wasSuccessful()


if __name__=='__main__':main()
