"""CPU correctness tests; random tiny-model outputs are not research results."""
import unittest
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
from vlm_common import default_model
from test_knockout import tiny_model,tiny_inputs
from knockout_probe import model_layers
from roi_sink_common import geometry,map_box,sink_candidates,matched_background,PatchKnockout,forward,WINDOWS

class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ip=AutoProcessor.from_pretrained(default_model(),local_files_only=True).image_processor
    def test_actual_processor_reconstruction_and_odd_crop(self):
        rng=np.random.default_rng(1)
        for w,h in [(501,333),(333,501),(336,336),(11,7)]:
            image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8));g,c=geometry(image,self.ip)
            self.assertTrue(g['pixel_values_exact']);self.assertEqual(c.shape,(336,336,3))
            self.assertEqual(g['crop_left'],(g['resized_size'][0]-336)//2)
    def test_half_open_boundary_area_tiling_crop_and_patch_order(self):
        g=dict(scale_x=1,scale_y=1,crop_left=0,crop_top=0)
        self.assertEqual(map_box([14,28,14,14],g)['patch_indices'],[49])
        self.assertEqual(map_box([0,0,14,14],g)['patch_indices'],[0])
        self.assertEqual(map_box([336,0,5,5],g)['patch_count'],0)
        b=map_box([-7,0,14,14],g);self.assertEqual(b['visible_fraction'],.5);self.assertEqual(b['patch_indices'],[0])
        self.assertEqual(map_box([0,0,336,336],g)['patch_indices'],list(range(576)))
    def test_real_clip_embedding_flatten_row_major(self):
        c=CLIPVisionConfig(hidden_size=8,image_size=336,patch_size=14,num_channels=3)
        embedding=CLIPVisionEmbeddings(c).eval();x=torch.arange(24*24).reshape(24,24).repeat_interleave(14,0).repeat_interleave(14,1).float()[None,None].repeat(1,3,1,1)
        with torch.no_grad():
            embedding.patch_embedding.weight.zero_();embedding.patch_embedding.weight[:,0,0,0]=1
            out=embedding(x);position=embedding.position_embedding(embedding.position_ids)
            patch=(out-position)[0,1:,0]
        self.assertTrue(torch.allclose(patch,torch.arange(576).float(),atol=1e-4))

class DefinitionTest(unittest.TestCase):
    def test_neutral_persistence_holdout_and_no_depth_preference(self):
        a=np.ones((3,32,2,576));a[:,:,:,17]=100
        r=sink_candidates(a);self.assertEqual(r['indices'],[17]);self.assertEqual(r['holdout_jaccard'],1)
        # High only 7 layers is not persistent; no forced top-k candidates.
        a=np.ones((3,32,2,576));a[:,:7,:,17]=100
        self.assertEqual(sink_candidates(a)['indices'],[])
        a=np.ones((3,32,2,576));a[:,-8:,:,17]=100
        self.assertEqual(sink_candidates(a)['indices'],[17])
        a=np.ones((3,32,2,576));a[:,:8,:,17]=100
        self.assertEqual(sink_candidates(a)['indices'],[17])
    def test_random_exact_count_no_target_overlap_and_reproducibility(self):
        target=[0,1,24,25];pool=sorted(set(range(576))-set(target))
        a=matched_background(target,pool,7);b=matched_background(target,pool,7)
        self.assertEqual(a,b);self.assertEqual(len(a['indices']),4);self.assertFalse(set(target)&set(a['indices']))
        self.assertEqual(a['mean_bin_manhattan'],0)
        self.assertFalse(matched_background(target,[2],7)['available'])
        self.assertEqual(matched_background([],pool,7)['indices'],[])

class MaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);cls.model=tiny_model();cls.inputs=tiny_inputs();cls.q=[6,7,8];cls.a=10
    def run_forward(self,**kw):return forward(self.model,self.inputs,self.q,self.a,expected_visual=4,**kw)
    def test_stream_matches_native_attention_and_prediction_exact(self):
        with torch.inference_mode():native=self.model(**self.inputs,output_attentions=True,use_cache=False,output_hidden_states=True)
        streamed=self.run_forward(observe=True);plain=self.run_forward()
        self.assertTrue(torch.equal(native.logits[0,self.a],streamed['prediction_logits']))
        self.assertTrue(torch.equal(plain['prediction_logits'],streamed['prediction_logits']))
        self.assertTrue(torch.equal(native.hidden_states[-1][:,self.q],streamed['question_hidden']))
        for l,attn in enumerate(native.attentions):
            expected=torch.stack([attn[0,:,self.q,2:6].mean(1),attn[0,:,self.a,2:6]],1).numpy().astype(np.float16)
            np.testing.assert_array_equal(streamed['rows'][l],expected)
    def test_exact_window_orientation_only_target_mask_changed(self):
        original=[];handles=[]
        for layer in model_layers(self.model):handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k:original.append(k['attention_mask'].clone()),with_kwargs=True))
        try:self.run_forward()
        finally:
            for h in handles:h.remove()
        for window in WINDOWS.values():
            changed=[];handles=[]
            with PatchKnockout(model_layers(self.model),self.inputs,99,self.q,[1,3],window,4) as hook:
                for layer in model_layers(self.model):handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k:changed.append(k['attention_mask'].clone()),with_kwargs=True))
                try:
                    with torch.inference_mode():self.model(**self.inputs,use_cache=False)
                finally:
                    for h in handles:h.remove()
            for i,(old,new) in enumerate(zip(original,changed)):
                target=torch.zeros(12,12,dtype=torch.bool)
                if i in window:target[self.q,3]=True;target[self.q,5]=True
                self.assertTrue(torch.equal(old[0,0][~target],new[0,0][~target]));self.assertTrue(torch.isneginf(new[0,0][target]).all())
    def test_actual_zero_causality_and_cleanup(self):
        baseline=self.run_forward()
        for window in WINDOWS.values():
            a=self.run_forward(queries=[10],patches=[0,2],active=window,observe=True)
            self.assertTrue(torch.equal(a['question_hidden'],baseline['question_hidden']))
            self.assertTrue(all(r['target_attention_max']==0 for r in a['attention_audit']))
        self.assertTrue(torch.equal(baseline['prediction_logits'],self.run_forward()['prediction_logits']))
    def test_empty_keys_sham_and_noop_window(self):
        full=self.run_forward()
        for patches,active in [([],WINDOWS['deep']),([1,2],[])]:
            out=self.run_forward(queries=self.q,patches=patches,active=active)
            self.assertTrue(torch.equal(full['prediction_logits'],out['prediction_logits']))
    def test_failure_cleanup_and_parameters_inputs_unchanged(self):
        params=[p.clone() for p in self.model.parameters()];inputs={k:v.clone() for k,v in self.inputs.items()}
        def fail(*a):raise RuntimeError('injected')
        h=model_layers(self.model)[12].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'injected'):self.run_forward(queries=self.q,patches=[1],active=WINDOWS['middle'],observe=True)
        finally:h.remove()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(params,self.model.parameters())))
        self.assertTrue(all(torch.equal(v,self.inputs[k]) for k,v in inputs.items()))
        self.run_forward()
    def test_invalid_indices_rejected(self):
        for q,p in [([2],[0]),([],[0]),([6],[4]),([6],[1,1])]:
            with self.assertRaises(ValueError):PatchKnockout(model_layers(self.model),self.inputs,99,q,p,WINDOWS['deep'],4)

class StatisticsTest(unittest.TestCase):
    def test_uniform_attention_enrichment_entropy_and_roi_overlap(self):
        from summarize_roi_sink import regional_metrics
        a=np.full((32,32,2,576),.2/576)
        entry=dict(roi_indices=[0,1,2,3],sink_indices=[3,4],sink_nonroi_indices=[4],roi_sink_overlap_indices=[3])
        sample=dict(objects=[dict(center_patch_indices=[0,1]),dict(center_patch_indices=[2,3])])
        metrics,_=regional_metrics(a,entry,sample)
        np.testing.assert_allclose(metrics['roi_enrichment'],1)
        np.testing.assert_allclose(metrics['sink_enrichment'],1)
        np.testing.assert_allclose(metrics['normalized_entropy'],1)
        np.testing.assert_allclose(metrics['roi_mass']+metrics['background_mass'],.2)
        np.testing.assert_allclose(metrics['sink_mass'],metrics['sink_nonroi_mass']+metrics['roi_sink_overlap_mass'])

    def test_paired_controls_exact_count_and_cases_excluded_from_main(self):
        from summarize_roi_sink import paired_diagnostics
        from roi_sink_common import RANDOM_SEEDS
        rows=[];metrics=[]
        for q,is_main,value in [('main',True,1.),('case',False,100.)]:
            for group in ['question','answer']:
                for window in WINDOWS:
                    metrics.append(dict(question_id=q,query_group=group,window=window,roi_mass=.1,sink_nonroi_mass=.05))
                    for name in ['roi','sink_nonroi']+[f'random_{t}_s{s}' for t in ['roi','sink'] for s in RANDOM_SEEDS]:
                        rows.append(dict(question_id=q,main=is_main,query_group=group,window=window,condition=name,
                            available=True,blocked_patches=2,answer_margin_drop=value if name in ['roi','sink_nonroi'] else .25,
                            full_margin=1.,weak_margin=False))
        pairs,summary=paired_diagnostics(rows,metrics)
        self.assertEqual(len(pairs),16)
        self.assertTrue(all(r['n']==1 and r['paired_excess_drop_mean']==.75 for r in summary))
        rows[-1]['blocked_patches']=1
        with self.assertRaises(AssertionError):paired_diagnostics(rows,metrics)

if __name__=='__main__':unittest.main()
