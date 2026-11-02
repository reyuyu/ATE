import unittest
from unittest.mock import patch
import torch
from query_role_attention.tests import prompt,blocks
from .engine import RoleBypass,roles,target_positions
from .common import LAYERS

class JointTests(unittest.TestCase):
    def test_joint_scope_and_simultaneous_prefill(self):
        p=prompt();p['attention_mask']=torch.ones_like(p['input_ids']);r=roles(p)
        self.assertEqual(target_positions('I+A-Bypass',r,False),r['I']+r['A'])
        self.assertEqual(target_positions('I+A-Bypass',r,True),[0])
        b=blocks();native=torch.randn(1,r['prompt_length'],3)
        with patch('vlm_research.query_role_ia_joint.engine.model_layers',return_value=b):
            with RoleBypass(None,r,'I+A-Bypass') as ctl:
                for k in range(32):
                    out=b[k].self_attn(native,cache_position=torch.arange(r['prompt_length']),past_key_value=None)
                    expected=native.clone()
                    if k in LAYERS:expected[:,r['I']+r['A']]=0
                    self.assertTrue(torch.equal(out,expected))
                    self.assertTrue(torch.equal(out[:,r['R']],native[:,r['R']]))
            self.assertEqual(len(ctl.events),4)
        self.assertTrue(all(not x.self_attn.o_proj._forward_hooks for x in b))

    def test_decode_eos_positions_noop_exception(self):
        p=prompt();p['attention_mask']=torch.ones_like(p['input_ids']);r=roles(p);b=blocks()
        cache=type('Cache',(),{'get_seq_length':lambda self,k:r['prompt_length']})()
        native=torch.randn(1,1,4)
        with patch('vlm_research.query_role_ia_joint.engine.model_layers',return_value=b):
            with RoleBypass(None,r,'I+A-Bypass'):
                for k in LAYERS:
                    self.assertEqual(int(torch.count_nonzero(b[k].self_attn(native,cache_position=torch.tensor([r['prompt_length']]),past_key_value=cache))),0)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with RoleBypass(None,r,'I+A-Bypass',noop=True):
                    self.assertTrue(torch.equal(b[2].self_attn(native,cache_position=torch.tensor([r['prompt_length']]),past_key_value=cache),native))
                    raise RuntimeError('intentional')
        self.assertTrue(all(not x.self_attn._forward_pre_hooks and not x.self_attn.o_proj._forward_hooks for x in b))

    def test_interaction_and_unique_harm(self):
        import numpy as np
        from .analyze import contrasts,unique_harm
        scores=np.array([[1,1,1,0],[0,1,0,1],[1,0,1,1]],float)
        self.assertEqual(unique_harm(scores).tolist(),[True,False,False])
        self.assertTrue(np.array_equal(contrasts(scores)[:,-1],scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]))
