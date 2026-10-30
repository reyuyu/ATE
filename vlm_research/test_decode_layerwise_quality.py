import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from paper_rq1.engine import clone_cache
from paper_rq1_decode_layerwise_quality.engine import DecodeLayer,is_decode
from paper_rq1_decode_layerwise_quality.data import unique_input_order,rgb_sha
from paper_rq1_decode_layerwise_quality.protocol import CONDITIONS,condition_layer
from paper_rq1_decode_layerwise_quality.statistics import bootstrap,cluster_deltas,interval_summary


class Attention(nn.Module):
    def __init__(self):super().__init__();self.o_proj=nn.Identity()
    def forward(self,hidden_states,cache_position=None,past_key_value=None):return self.o_proj(hidden_states*2)


class Tests(unittest.TestCase):
    def model(self):return SimpleNamespace(layers=[SimpleNamespace(self_attn=Attention()) for _ in range(32)])
    def invoke(self,m,k,n=5,start=5,past=5):
        cache=SimpleNamespace(get_seq_length=lambda layer:past)
        return m.layers[k].self_attn(torch.ones(1,1,3),cache_position=torch.tensor([start]),past_key_value=cache)
    def test_all_indices(self):
        self.assertEqual([condition_layer(c) for c in CONDITIONS],[None]+list(range(32)))
    def test_bad_layers(self):
        for k in [-1,32,1.5]:
            with self.assertRaises(ValueError):DecodeLayer(None,k,5)
    def test_decode_not_sequence_only(self):
        self.assertFalse(is_decode(torch.tensor([0]),1,0,1));self.assertTrue(is_decode(torch.tensor([5]),5,5,1))
    def test_cache_misalignment(self):
        for p,n,past,q in [(None,5,5,1),(torch.tensor([5]),5,0,1),(torch.tensor([2]),5,0,1)]:
            with self.assertRaises(AssertionError):is_decode(p,n,past,q)
    def test_prefill_zero_not_applied(self):
        m=self.model()
        with patch('paper_rq1_decode_layerwise_quality.engine.model_layers',lambda m:m.layers):
            with DecodeLayer(m,0,1) as h:
                out=m.layers[0].self_attn(torch.ones(1,1,3),cache_position=torch.tensor([0]),past_key_value=None)
            self.assertTrue(torch.equal(out,torch.ones_like(out)*2));self.assertEqual(h.edits,0)
    def test_all32_target_only(self):
        for k in range(32):
            m=self.model()
            with patch('paper_rq1_decode_layerwise_quality.engine.model_layers',lambda m:m.layers):
                with DecodeLayer(m,k,5):
                    self.assertEqual(self.invoke(m,k).sum(),0)
                    self.assertEqual(self.invoke(m,(k+1)%32).sum(),6)
    def test_noop_exact(self):
        m=self.model()
        with patch('paper_rq1_decode_layerwise_quality.engine.model_layers',lambda m:m.layers):
            with DecodeLayer(m,5,5,True):self.assertEqual(self.invoke(m,5).sum(),6)
    def test_hook_cleanup_exception(self):
        m=self.model()
        with patch('paper_rq1_decode_layerwise_quality.engine.model_layers',lambda m:m.layers):
            with self.assertRaises(RuntimeError):
                with DecodeLayer(m,5,5):raise RuntimeError('intentional')
        self.assertFalse(m.layers[5].self_attn._forward_pre_hooks);self.assertFalse(m.layers[5].self_attn.o_proj._forward_hooks)
    def test_cloned_cache_independent(self):
        c=DynamicCache();c.update(torch.ones(1,1,5,2),torch.ones(1,1,5,2),0)
        copied=clone_cache(c);copied.key_cache[0].zero_();self.assertEqual(c.key_cache[0].sum(),10)
    def test_sampling_no_duplicate_image(self):
        rows=[dict(image_identity=str(i//2),question_id=str(i)) for i in range(20)]
        self.assertEqual(unique_input_order(rows),unique_input_order(list(reversed(rows))));self.assertEqual(len(unique_input_order(rows)),10)
    def test_clusters_not_questions(self):
        sums,counts=cluster_deltas([1,3,4],['a','a','b']);np.testing.assert_equal(sums[:,0],[4,4]);np.testing.assert_equal(counts,[2,1])
    def test_constant_bootstrap(self):
        b=bootstrap(np.ones((4,2)),list('abcd'),1000);np.testing.assert_equal(b['corrected'],np.ones((2,2)))
    def test_paired_and_family_ci(self):
        b=bootstrap(np.array([[-1,0],[0,0],[1,0],[2,0]]),list('abcd'),5000)
        self.assertLessEqual(b['corrected'][0,0],b['ordinary'][0,0]);self.assertGreaterEqual(b['corrected'][1,0],b['ordinary'][1,0])
        self.assertEqual(b['ordinary'][0,1],0)
    def test_ni_not_zero_in_ci(self):
        b=interval_summary(0,[-.03,.03],[-.04,.04],.02);self.assertFalse(b['ordinary_noninferior'])
    def test_checkpoint_rejects_changed_data(self):
        from paper_rq1_decode_layerwise_quality.run import checkpoint_valid
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text(json.dumps(dict(binding='old')))
            with self.assertRaises(ValueError):checkpoint_valid(p,'new',{},'Full')
    def test_generation_eos_and_max_limit(self):
        from paper_rq1_decode_layerwise_quality.engine import generate
        # Immediate EOS must never perform Decode, and first token stays untouched.
        cache=DynamicCache();cache.update(torch.ones(1,1,5,2),torch.ones(1,1,5,2),0)
        prompt={'input_ids':torch.ones(1,5,dtype=torch.long)};first=torch.tensor([0.,0.,1.])
        result,_=generate(None,prompt,32,parent=cache,first=first)
        self.assertTrue(result['eos']);self.assertEqual(result['decode_steps'],0)
        result,_=generate(None,prompt,1,parent=cache,first=torch.tensor([0.,1.,0.]))
        self.assertTrue(result['truncated']);self.assertEqual(result['token_ids'],[1])


if __name__=='__main__':unittest.main()
