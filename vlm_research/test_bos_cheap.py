import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaConfig
from bos_cheap.engine import cheap_project,rotate_selected,Features,selected_positions
from bos_cheap.data import split
from bos_cheap.predict import logistic,raw_score,sigmoid,danger,select_threshold,should_skip_attention,SPECS
from bos_cheap.analyze import intervals,risk_values


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)

    def test_selected_projection_real_rope_nonzero_bos(self):
        config=LlamaConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=4,intermediate_size=64)
        module=llama.LlamaAttention(config,0).half();hidden=torch.randn(1,11,32).half();phase=torch.randn(1,11,8)
        cos,sin=phase.cos().half(),phase.sin().half();q,bos,extra=9,3,[1,6,8]
        fullq=module.q_proj(hidden).view(1,11,4,8).transpose(1,2);fullk=module.k_proj(hidden).view(1,11,4,8).transpose(1,2)
        qr,kr=llama.apply_rotary_pos_emb(fullq,fullk,cos,sin)
        with patch('torch.nn.functional.softmax',side_effect=AssertionError('Forbidden full Softmax')):
            cq,ck,cv=cheap_project(module,hidden,cos,sin,q,bos,extra)
        torch.testing.assert_close(cq,qr[:,:,q:q+1],atol=.02,rtol=.005)
        torch.testing.assert_close(ck,kr[:,:,[bos]+extra],atol=.02,rtol=.005)
        torch.testing.assert_close(cv,module.v_proj(hidden).view(1,11,4,8).transpose(1,2)[:,:,bos:bos+1],atol=.02,rtol=.005)
        self.assertEqual(cq.shape[2],1);self.assertEqual(ck.shape[2],4)

    def test_projection_gram_matches_actual_head_projection(self):
        w=torch.randn(9,4);v=torch.randn(4);self.assertAlmostEqual(float((v@(w.T@w)@v).sqrt()),float((w@v).norm()),places=5)
        matrix=torch.randn(12,12);heads=torch.randn(3,4)
        pieces=[matrix[:,i*4:(i+1)*4]@heads[i] for i in range(3)]
        torch.testing.assert_close(sum(pieces),matrix@heads.flatten())

    def test_real_eager_observer_native_unchanged_and_cleanup(self):
        config=LlamaConfig(hidden_size=128,num_attention_heads=32,num_key_value_heads=32,intermediate_size=128)
        layers=torch.nn.ModuleList([llama.LlamaDecoderLayer(config,i).half() for i in range(32)])
        model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
        layer=layers[0];hidden=torch.randn(1,8,128).half();phase=torch.randn(1,8,4);pe=(phase.cos().half(),phase.sin().half());mask=torch.triu(torch.full((1,1,8,8),torch.finfo(torch.float16).min),diagonal=1)
        w=layer.self_attn.o_proj.weight.float().reshape(128,32,4).permute(1,2,0);g=[w@w.transpose(1,2)]*32
        original=llama.eager_attention_forward;kwargs=dict(position_embeddings=pe,attention_mask=mask)
        with torch.inference_mode():native=layer(hidden,**kwargs)[0]
        with patch('bos_cheap.engine.selected_positions',return_value=(6,2,[1,3])):
            with torch.inference_mode(),Features(model,{},g,True) as trace:observed=layer(hidden,**kwargs)[0]
        self.assertTrue(torch.equal(native,observed));self.assertEqual(len(trace.cheap_checks),1);self.assertIs(llama.eager_attention_forward,original)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in layers.modules()))
        with self.assertRaises(ValueError):trace.arrays()
        with patch('bos_cheap.engine.selected_positions',return_value=(6,2,[1,3])):
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with Features(model,{},g):raise RuntimeError('fixture')
        self.assertIs(llama.eager_attention_forward,original)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in layers.modules()))

    def test_dynamic_positions_and_extra_key_limit(self):
        q,b,extra=selected_positions(dict(p_minus_one=650,bos_positions=[4],visual_positions=list(range(10,586)),question_and_instruction_positions=[600,601,602]))
        self.assertEqual((q,b),(650,4));self.assertEqual(extra,[10,201,393,585,600,602]);self.assertLessEqual(len(extra),6)

    def test_image_split_stable_and_task_independent(self):
        self.assertEqual(split('same-image-sha'),split('same-image-sha'));self.assertTrue({split(str(i)) for i in range(100)}=={'train','calibration'})

    def test_no_post_attention_features_in_online_models(self):
        for cols,layer in SPECS.values():
            self.assertFalse(set(cols)&{'oracle_bos_mass','attention_ratio','mlp_ratio','norm_attention','norm_mlp','delta_logp','js'})
        with self.assertRaises(ValueError):should_skip_attention(1,{},dict(name='BOS-Mass-Oracle'))

    def test_danger_sign_and_flip_not_abs_delta(self):
        rows=[dict(top1_flip=False,delta_logp=.5),dict(top1_flip=False,delta_logp=-.04),dict(top1_flip=True,delta_logp=0.)]
        np.testing.assert_array_equal(danger(rows,.03),[0,1,1])

    def test_tiny_logistic_and_inference(self):
        x=np.linspace(-2,2,80)[:,None];y=(x[:,0]>0).astype(float);m=logistic(x,y);p=sigmoid(raw_score(x,m))
        self.assertGreater(p[-1],p[0]);self.assertTrue(np.isfinite(p).all());self.assertEqual(m['parameters'],2)

    def test_threshold_uses_calibration_and_per_layer_rule(self):
        rows=[dict(bos_cos_mean=.9,top1_flip=False,delta_logp=0.,layer=0),dict(bos_cos_mean=.1,top1_flip=True,delta_logp=0.,layer=1)]
        rules=[dict(name='BOS-cos',kind='scalar',column='bos_cos_mean',threshold=t,direction='high') for t in [0,.5,1]]
        selected=select_threshold(rows,rules,.03);self.assertEqual(selected['threshold'],.5)
        self.assertTrue(should_skip_attention(0,dict(bos_cos_mean=.9),selected));self.assertFalse(should_skip_attention(0,dict(bos_cos_mean=.1),selected))

    def test_cluster_interval_zero_events_not_certified_zero_risk(self):
        rows=[dict(image_sha256=str(i),top1_flip=False,delta_logp=0.,js=0.) for i in range(20) for _ in range(32)]
        s=np.ones(len(rows),bool);ci=intervals(rows,[s],.03,10)[0]
        self.assertEqual(ci['n_image_clusters'],20);self.assertEqual(ci['dangerous_among_skipped_ci95_high'],0)
        self.assertGreater(ci['simultaneous_hoeffding_risk_upper95'],.01)
        self.assertIsNone(risk_values(rows,np.zeros(len(rows),bool),.03)['dangerous_among_skipped'])


if __name__=='__main__':unittest.main()
