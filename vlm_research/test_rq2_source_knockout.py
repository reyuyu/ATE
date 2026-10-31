import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.smoke import hook_footprint
from knockout_probe import knockout_mask, model_layers
from rq2_source_knockout.engine import partition, SourceKnockout, forward, SOURCES
from rq2_source_knockout.run import assign, selected_samples
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.statistics import bootstrap


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=48,
            num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=4,
            max_position_embeddings=64,attention_dropout=0.)
        config._attn_implementation='eager'
        config.image_token_index=60
        self.config=config
        self.language_model=LlamaForCausalLM(config).eval()

    def forward(self,**kwargs):
        return self.language_model(**kwargs)


def inputs():
    ids=torch.tensor([[1,10,60,60,60,60,11,12,13]])
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids))


class Tests(unittest.TestCase):
    def test_partition_excludes_self(self):
        query,groups=partition(inputs(),60,1,4)
        self.assertEqual(query,8)
        self.assertEqual({k:int(v.sum()) for k,v in groups.items()},dict(Image=4,BOS=1,Self=1,Prompt=3))
        self.assertFalse(groups['Prompt'][query])
        self.assertTrue(groups['Prompt'][1])

    def test_invalid_partition(self):
        bad=inputs();bad['input_ids'][0,1]=1
        with self.assertRaises(ValueError): partition(bad,60,1,4)
        with self.assertRaises(ValueError): partition(inputs(),60,1,576)

    def test_hard_mask_only_target_legal_edges(self):
        model=Tiny();data=inputs();n=9
        original=torch.full((1,1,n,n),torch.finfo(torch.float32).min)
        original[0,0].masked_fill_(torch.ones(n,n,dtype=torch.bool).tril(),0)
        for source in SOURCES:
            context=SourceKnockout(model_layers(model),data,60,1,1,source,expected_visual=4)
            changed,count=knockout_mask(original,context.layout)
            self.assertEqual(count,int(context.groups[source].sum()))
            self.assertTrue(torch.equal(changed[:,:,:8],original[:,:,:8]))
            self.assertEqual(float(changed[0,0,8,8]),0.)
            self.assertTrue(torch.isneginf(changed[0,0,8,context.groups[source]]).all())
            self.assertTrue((changed[0,0]==0).any(-1).all())
            self.assertFalse(torch.isneginf(original).any())

    def test_noop_scope_and_cleanup(self):
        torch.manual_seed(123)
        model=Tiny();data=inputs();before=hook_footprint(model)
        native,obs,_=forward(model,data,1,observe=True,audit=True,expected_visual=4)
        noop,_,_=forward(model,data,1,layer=1,source='Prompt',noop=True,audit=True,expected_visual=4)
        self.assertTrue(torch.equal(native,noop))
        for source in SOURCES:
            changed,trace,mask=forward(model,data,1,layer=1,source=source,observe=True,audit=True,expected_visual=4)
            self.assertTrue(torch.equal(native[0,:8],changed[0,:8]))
            self.assertEqual(obs.states,trace.states)
            self.assertEqual(trace.rows[1][source+'_max'],0.)
            self.assertEqual([r['layer'] for r in mask.records if r['applied']],[1])
            self.assertEqual(hook_footprint(model),before)
        with self.assertRaises(RuntimeError):
            with SourceKnockout(model_layers(model),data,60,1,0,'BOS',expected_visual=4):
                raise RuntimeError('cleanup')
        self.assertEqual(hook_footprint(model),before)

    def test_metric_signed_loss_and_js(self):
        full=torch.tensor([2.,0.,-1.]);changed=torch.tensor([[2.,0.,-1.],[3.,0.,-1.],[0.,3.,-1.]])
        a,b,c=metric_rows(full,changed)
        self.assertEqual(a['delta_logp'],0.)
        self.assertAlmostEqual(a['js'],0.,places=7)
        self.assertGreater(b['delta_logp'],0.)
        self.assertTrue(c['top1_flip'])
        self.assertGreater(c['js'],0.)

    def test_sample_and_group_assignment(self):
        samples=selected_samples()
        self.assertEqual(len(samples),240)
        ranks=assign(samples)
        self.assertEqual(ranks,assign(list(reversed(samples))))
        images={}
        for sample in samples:
            image=sample['image_sha256'];rank=ranks[sample['question_id']]
            self.assertEqual(images.setdefault(image,rank),rank)
        self.assertLessEqual(abs(list(ranks.values()).count(0)-list(ranks.values()).count(1)),2)

    def test_bootstrap_question_weighted_whole_images(self):
        result=bootstrap(np.array([[0.],[2.],[10.]]),['a','a','b'],[1],repetitions=1000,seed=7)[0]
        self.assertEqual(result['mean'],4.)
        self.assertEqual(result['clusters'],2)
        self.assertLessEqual(result['ci95_low'],4.)
        self.assertGreaterEqual(result['ci95_high'],4.)


if __name__=='__main__': unittest.main()
