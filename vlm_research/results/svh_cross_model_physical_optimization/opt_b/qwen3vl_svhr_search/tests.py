import unittest
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from .engine import restore_answer_access, Functional, SKIP
from .freeze import CANDIDATES, choose
from .physical import SelectedKV
from ..qwen3vl_overnight.metrics import metric, paired, complete_keys


class Tests(unittest.TestCase):
    def test_stage_handoff_and_disjoint_freeze(self):
        from unittest.mock import patch
        from . import freeze as module
        from ..qwen3vl_overnight.common import write,digest,TASKS
        parent=Path('/data/lk/ATE-resources/experiments/qwen3vl_overnight_cpu')
        parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for task in TASKS:
            for i in range(900 if task=='MME' else 120):
                group=i//2 if task=='MME' else i//4 if task=='MMBench' else i
                key=f'SYNTHETIC_CPU_TEST:{task}:{i}'
                rows.append(dict(key=key,benchmark=task,image_sha256=digest([task,group]),
                    pair_id=f'{task}:{group}',circular_group=f'{task}:{group}',
                    task=f'category{group%14}',strategy=['random','popular','adversarial'][i%3]))
        mapping={r['key']:dict(cluster=r['image_sha256'],shard=0) for r in rows}
        records=[dict(key=r['key'],condition=c,status='COMPLETE',input_binding_sha256=digest(r['key']))
            for r in rows for c in ['Native','SVH']]
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp)
            first=root/'first'
            out=root/'search'
            write(first/'completion.json',dict(status='TERMINAL',stage2_allowed=True))
            write(first/'protocol.json',dict(data_kind='SYNTHETIC_CPU_TEST'))
            write(first/'shards.json',mapping)
            with patch.multiple(module,FIRST=first,OUT=out,check_first=lambda:None,
                    inventory=lambda:rows,records=lambda:(records,{})):
                module.freeze()
                protocol=module.check()
            expected=subprocess.check_output(['git','rev-parse','codex/qwen-internvl-sparse-visual-3000'],text=True).strip()
            self.assertEqual(protocol['history_commit'],expected)
            self.assertEqual(protocol['ranking_tasks'],list(TASKS))
            cohorts=[json.loads((out/(s+'_manifest.json')).read_text()) for s in ['Discovery','Validation','Held-out']]
            images=[{r['image_sha256'] for r in pool} for pool in cohorts]
            self.assertFalse(images[0]&images[1] or images[0]&images[2] or images[1]&images[2])

    def test_six_pinned_scorers_cpu(self):
        from ..qwen3vl_overnight.common import inventory,scorers,TASKS
        rows=inventory()
        score=scorers()
        for task,text in zip(TASKS,['yes','A','test','yes','yes','A']):
            row=next(r for r in rows if r['benchmark']==task)
            actual=score.score(row,text)
            self.assertEqual(actual,score.score(row,text))
            self.assertGreaterEqual(actual['score'],0)
            self.assertLessEqual(actual['score'],1)

    def test_cpu_plot_without_model_environment_change(self):
        from ..qwen3vl_overnight.common import table,write,TASKS
        from PIL import Image,ImageStat
        parent=Path('/data/lk/ATE-resources/experiments/qwen3vl_overnight_cpu')
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp)
            # Synthetic software-test fixture only; never a research result.
            rows=[dict(stage='Discovery',task=task,condition=name,score=10+i,delta_K0=i,
                delta_K0_low=i-.1,delta_K0_high=i+.1,data_kind='SYNTHETIC_CPU_TEST')
                for task in TASKS for i,name in enumerate(['K0','K1','K8','Middle8'])]
            table(root/'all_stage_accuracy.csv',rows)
            write(root/'candidate_grid.json',{n:dict(k=len(CANDIDATES[n])) for n in ['K0','K1','K8','Middle8']})
            subprocess.run(['/data/lk/深度学习/.envs/ate-vlm-research/bin/python',str(Path(__file__).with_name('plot.py')),str(root)],
                check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(root/'mpl')))
            for stem in ['discovery_quality_vs_layers','textvqa_reaccess_comparison']:
                self.assertGreater((root/'figures'/(stem+'.pdf')).stat().st_size,1000)
                with Image.open(root/'figures'/(stem+'.png')) as image:
                    self.assertGreater(image.width,1000)
                    self.assertGreater(max(ImageStat.Stat(image.convert('RGB')).var),1)

    def test_physical_graph_against_dense_cpu(self):
        from torch import nn
        from .physical import Physical
        from .engine import Functional
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
        torch.set_num_threads(1)
        torch.manual_seed(802)
        class Attention(nn.Module):
            def __init__(self,i):
                super().__init__()
                self.i=i
                self.layer_idx=i
                self.num_key_value_groups=1
                for name in ['q_proj','k_proj','v_proj','o_proj']:
                    setattr(self,name,nn.Linear(8,8,bias=False))
                self.seen=None
            def forward(self,hidden_states,attention_mask,past_key_values=None,**kwargs):
                q,k,v=[getattr(self,name)(hidden_states).unsqueeze(1) for name in ['q_proj','k_proj','v_proj']]
                self.seen=(k.detach().clone(),v.detach().clone())
                if past_key_values is not None:
                    k,v=past_key_values.update(k,v,self.i)
                out,weights=qm.eager_attention_forward(self,q,k,v,attention_mask,scaling=8**-.5,dropout=0.0)
                return self.o_proj(out.reshape(1,hidden_states.shape[1],8)),weights
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj=nn.Linear(8,8,bias=False)
                self.up_proj=nn.Linear(8,8,bias=False)
                self.down_proj=nn.Linear(8,8,bias=False)
            def forward(self,h):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(h))*self.up_proj(h))
        class Block(nn.Module):
            def __init__(self,i):
                super().__init__()
                self.self_attn=Attention(i)
                self.input_layernorm=nn.LayerNorm(8)
                self.post_attention_layernorm=nn.LayerNorm(8)
                self.mlp=MLP()
            def forward(self,h,**kw):
                a,_=self.self_attn(hidden_states=self.input_layernorm(h),**kw)
                h=h+a
                return h+self.mlp(self.post_attention_layernorm(h))
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=nn.ModuleList([Block(i) for i in range(36)])
                self.norm=nn.LayerNorm(8)
                self.config=Qwen3VLTextConfig(hidden_size=8,intermediate_size=8,num_hidden_layers=36,
                    num_attention_heads=1,num_key_value_heads=1,head_dim=8,vocab_size=16)
                self.config._attn_implementation='eager'
            def rotary_emb(self,h,pos):
                return torch.ones(1,h.shape[1],8),torch.zeros(1,h.shape[1],8)
        model=nn.Module()
        model.model=nn.Module()
        model.model.language_model=Decoder()
        d=model.model.language_model
        x=torch.randn(1,6,8)
        layout=dict(input_length=6,image_positions=[1,2,3])
        # A toy independent dense oracle; no call to Physical.forward or its mask builder.
        cache={}
        def dense(h,step):
            for i,layer in enumerate(d.layers):
                q=layer.input_layernorm(h)
                n=h.shape[1]
                k,v=[getattr(layer.self_attn,name)(q).unsqueeze(1) for name in ['k_proj','v_proj']]
                if step:
                    k=torch.cat((cache[i][0],k),dim=-2)
                    v=torch.cat((cache[i][1],v),dim=-2)
                cache[i]=(k.detach().clone(),v.detach().clone())
                query=layer.self_attn.q_proj(q).unsqueeze(1)
                mask=torch.zeros(1,1,n,k.shape[-2])
                if not step:
                    mask.masked_fill_(torch.ones(n,n,dtype=torch.bool).triu(1),torch.finfo(h.dtype).min)
                    if i>=22:
                        mask[... ,4,[1,2,3]]=-torch.inf
                if i!=26:
                    mask[...,-1,[1,2,3]]=-torch.inf
                weights=torch.softmax(query@k.transpose(-1,-2)/8**.5+mask,dim=-1)
                a=layer.self_attn.o_proj((weights@v).squeeze(1))
                if i in SKIP:
                    a=a.clone()
                    a[:,-1]=0
                h=h+a
                h=h+layer.mlp(layer.post_attention_layernorm(h))
            return d.norm(h)
        expected=dense(x.clone(),0)
        positions=torch.arange(6).view(1,1,6).expand(4,1,6)
        fingerprints=[]
        for layers in [[],[26]]:
            observed=Functional(model,layout,'F',max_tokens=4,layers=layers,audit=True)
            with observed.context():
                got=observed.forward(inputs_embeds=x.clone(),position_ids=positions,
                    cache_position=torch.arange(6),attention_mask=torch.ones(1,6,dtype=torch.long))
            fingerprints.append(observed.prefill_hashes)
            self.assertEqual(len(observed.permission_audit),36)
            if layers:
                torch.testing.assert_close(got.last_hidden_state,expected,atol=2e-5,rtol=1e-5)
        self.assertEqual(fingerprints[0],fingerprints[1])
        executor=Physical(model,layout,'C',max_tokens=4,layers=[26])
        with executor.context():
            out=executor.forward(inputs_embeds=x.clone(),position_ids=positions,cache_position=torch.arange(6))
            torch.testing.assert_close(out.last_hidden_state,expected[:,[0,4,5]],atol=2e-5,rtol=1e-5)
            for step in [1,2]:
                token=torch.randn(1,1,8)
                reference=dense(token.clone(),step)
                out=executor.forward(inputs_embeds=token.clone(),position_ids=torch.zeros(4,1,1,dtype=torch.long),cache_position=torch.tensor([5+step]))
                torch.testing.assert_close(out.last_hidden_state,reference,atol=2e-5,rtol=1e-5)

    def test_mask_only_answer_image_row(self):
        x=torch.zeros(1,1,8,8)
        x[..., -1, [1,2,3]]=-torch.inf
        frozen=x.clone()
        y=restore_answer_access(x,[1,2,3],True)
        self.assertTrue(torch.equal(x,frozen))
        self.assertTrue(torch.equal(y[...,:-1,:],x[...,:-1,:]))
        self.assertTrue(torch.equal(y[...,-1,[0,4,5,6,7]],x[...,-1,[0,4,5,6,7]]))
        self.assertEqual(y[...,-1,[1,2,3]].tolist(),[[[0.,0.,0.]]])
        self.assertIs(restore_answer_access(x,[1,2,3],False),x)

    def test_decode_mask(self):
        x=torch.zeros(1,1,1,12)
        x[...,0,[2,3]]=-torch.inf
        self.assertEqual(float(restore_answer_access(x,[2,3],True).sum()),0)

    def test_grid_preserves_frozen_skip(self):
        self.assertEqual(len(CANDIDATES),16)
        invalid={name for name,layers in CANDIDATES.items() if set(layers)&SKIP}
        self.assertEqual(invalid,{'Early8','Middle24'})
        self.assertEqual(CANDIDATES['K8'],[10,12,21,23,24,25,26,27])

    def test_deterministic_group_selection(self):
        rows=[dict(key=str(i),pair_id=str(i//2)) for i in range(12)]
        a=choose(rows,6,'fixed','pair_id')
        b=choose(list(reversed(rows)),6,'fixed','pair_id')
        self.assertEqual({r['key'] for r in a},{r['key'] for r in b})
        for p in {r['pair_id'] for r in a}:
            self.assertEqual(sum(r['pair_id']==p for r in a),2)

    def test_incomplete_group_exclusion(self):
        rows=[dict(key=str(i),pair_id=str(i//2)) for i in range(6)]
        self.assertEqual(complete_keys('MME',rows,{'0','1','2','4','5'}),{'0','1','4','5'})

    def test_image_clustered_pairing(self):
        mapping={str(i):dict(cluster=str(i//2)) for i in range(8)}
        a=[dict(key=str(i),scoring=dict(score=i%2)) for i in range(8)]
        b=[dict(key=str(i),scoring=dict(score=1-i%2)) for i in range(8)]
        result=paired('GQA',a,b,mapping,repeats=2000)
        self.assertEqual(result['delta'],0)
        self.assertEqual((result['low'],result['high']),(0,0))
        self.assertEqual(result['clusters'],4)

    def test_mme_official_sum(self):
        rows=[]
        for j in range(14):
            for q in range(2):
                rows.append(dict(key=f'{j}:{q}',mme_task=str(j),pair_id=str(j),scoring=dict(score=1)))
        mapping={r['key']:dict(cluster=r['pair_id']) for r in rows}
        self.assertEqual(metric('MME',rows,mapping),2800)
        # An absent category is not silently scored as a fourteen-category total.
        self.assertIsNone(metric('MME',rows[:-2],mapping))

    def test_circular_accuracy(self):
        rows=[dict(key=str(i),circular_group=i//2,scoring=dict(score=float(i!=1))) for i in range(4)]
        mapping={r['key']:dict(cluster=str(r['circular_group'])) for r in rows}
        self.assertEqual(metric('MMBench',rows,mapping),50)

    def test_physical_selected_cache_original_positions(self):
        n=6
        text=torch.tensor([0,4,5])
        image=torch.tensor([1,2,3])
        k=torch.arange(1*8*n*128,dtype=torch.float32).reshape(1,8,n,128).to(torch.bfloat16)
        v=k/2
        late={26:(k.index_select(-2,image).clone(),v.index_select(-2,image).clone())}
        cache=SelectedKV(n,text,3,[20,26],late)
        early=cache.update(k,v,0)
        self.assertTrue(torch.equal(early[0],k))
        self.assertEqual(cache.used[0],len(text))
        cache.update(k,v,20)
        restored=cache.update(k.index_select(-2,text),v.index_select(-2,text),26)
        self.assertTrue(torch.equal(restored[0],k))
        self.assertTrue(torch.equal(restored[1],v))
        self.assertFalse(late)
        cache.prefill=False
        extra=torch.ones(1,8,1,128,dtype=torch.bfloat16)
        for i in [0,20,26]:
            cache.update(extra,extra,i)
        self.assertEqual([cache.used[i] for i in [0,20,26]],[4,7,7])
        self.assertTrue(torch.equal(cache.entries[26][0][...,:n,:],k))
        self.assertEqual(cache.positions[0],[0,4,5])
        self.assertEqual(cache.positions[26],list(range(n)))

    def test_bad_skip_cache_rejected(self):
        with self.assertRaises(AssertionError):
            SelectedKV(6,torch.tensor([0,4,5]),3,[1],{})


if __name__=='__main__':
    unittest.main()
