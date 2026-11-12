"""Independent SVH-R port; selected image KV only, with charged construction.

Late visual keys are causally independent of text after the image. An auxiliary
causal image-prefix path constructs them from the actual L22 input; its full cost
is charged. Original SVH never calls this module. Strict status stays separate.
"""
import time
from unittest.mock import patch

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_c3v2_physical import engine as original
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest

FrozenExecution = original.Execution


class SelectedKV(original.CompactKV):
    def __init__(self, n, text, capacity_extra, layers, late, text_cpu=None):
        super().__init__(n, text, len(text)+capacity_extra, original.SKIP)
        self.selected=frozenset(layers)
        assert not self.selected & set(original.SKIP)
        self.late=late
        self.positions=[]
        self.capacities=[]
        text_cpu=text.tolist() if text_cpu is None else text_cpu
        text_set=set(text_cpu)
        self.image=[p for p in range(n) if p not in text_set]
        for i in range(36):
            positions=list(range(n)) if i in self.selected else list(text_cpu)
            self.positions.append(positions)
            self.capacities.append(len(positions)+capacity_extra)

    def update(self,k,v,layer_idx,cache_kwargs=None):
        i=layer_idx
        assert i not in self.absent and k.shape==v.shape
        self.updates[i]+=1
        if self.prefill:
            if i in self.selected and i>=22:
                assert k.shape[-2]==len(self.text)
                ik,iv=self.late.pop(i)
                merged=[]
                for now,visual in zip((k,v),(ik,iv)):
                    out=now.new_empty((*now.shape[:-2],self.length,now.shape[-1]))
                    out.index_copy_(-2,self.text,now)
                    out.index_copy_(-2,torch.tensor(self.image,device=now.device),visual)
                    merged.append(out)
                returned=tuple(merged)
            else:
                returned=(k,v)
            if i in self.selected:
                stored=returned
                assert stored[0].shape[-2]==self.length
            else:
                stored=tuple(t.index_select(-2,self.text) if t.shape[-2]==self.length else t for t in returned)
            n=stored[0].shape[-2]
            assert n==len(self.positions[i])
            pair=[t.new_empty((*t.shape[:-2],self.capacities[i],t.shape[-1])) for t in stored]
            for destination,source in zip(pair,stored):
                destination[...,:n,:].copy_(source)
            self.entries[i]=pair
            self.used[i]=n
            return returned
        assert k.shape[-2]==1 and self.entries[i] is not None
        n=self.used[i]
        assert n+1<=self.capacities[i]
        for destination,source in zip(self.entries[i],(k,v)):
            destination[...,n:n+1,:].copy_(source)
        self.used[i]+=1
        return tuple(t[...,:n+1,:] for t in self.entries[i])

    def audit(self):
        out=super().audit()
        for row in out['layers']:
            i=row['layer']
            visual=len(self.image) if i in self.selected else 0
            row['persistent_image_tokens']=visual
            row['persistent_image_bytes']=visual*2*8*128*2
            row['persistent_text_tokens']=row['live_tokens']-visual
            assert row['persistent_text_tokens']>=0
        out.update(persistent_image_bytes=sum(r['persistent_image_bytes'] for r in out['layers']),
            selected_layers=sorted(self.selected),late_KV_not_consumed=sorted(self.late),
            historical_append_copies=0)
        return out


class Physical(FrozenExecution):
    def __init__(self,*args,layers=(),**kwargs):
        super().__init__(*args,**kwargs)
        assert self.c=='C' and self.skip and self.physical
        self.read_layers=frozenset(layers)
        assert not self.read_layers & set(original.SKIP)
        self.cache=SelectedKV(self.n,self.text,self.cache.capacity-len(self.text),layers,{},text_cpu=self._text_cpu)
        self.builder_events=[]
        self.builder_layers=[]
        self._masks={}
        self._decode_masks={}

    def construct_late_KV(self,h,rotary,textpos,cp):
        wanted={i for i in self.read_layers if i>=22}
        if not wanted:
            return
        assert not self.cache.late
        end=self.I[-1]+1
        if h.device.type=='cuda':
            start,finish=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
        prefix=h[:,:end].clone()
        positions=cp[:end]
        rope=tuple(v[:,:end] for v in rotary)
        mask=h.new_full((1,1,end,end),torch.finfo(h.dtype).min).triu(1)
        eager=qm.eager_attention_forward
        def capture(module,q,k,v,attention_mask=None,**kwargs):
            i=module.layer_idx
            assert k.shape[-2]==v.shape[-2]==end
            if i in wanted:
                self.cache.late[i]=tuple(x.index_select(-2,self.image).clone() for x in (k,v))
            if i==max(wanted):
                raise VisualKeysReady()
            return eager(module,q,k,v,attention_mask,**kwargs)
        with patch.object(qm,'eager_attention_forward',capture):
            for i in range(22,max(wanted)+1):
                ev,sh=len(self.events),len(self.shapes)
                try:
                    prefix=self.d.layers[i](prefix,attention_mask=mask,position_ids=textpos[:,:end],
                        cache_position=positions,position_embeddings=rope,past_key_values=None,use_cache=False)
                except VisualKeysReady:
                    assert i==max(wanted)
                    break
                finally:
                    self.builder_layers.append(dict(layer=i,hidden_rows=end,image_rows=len(self.I),persistent_KV=i in wanted))
                    for row in self.events[ev:]+self.shapes[sh:]:
                        row['role']='VISUAL_KV_BUILDER'
        assert set(self.cache.late)==wanted
        if h.device.type=='cuda':
            finish.record()
            self.builder_events.append((start,finish))

    def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,
                attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None,**kwargs):
        h=inputs_embeds if inputs_embeds is not None else self.d.embed_tokens(input_ids)
        pre=self.step==0;self.cache.prefill=pre;cp=cache_position
        if self.audit_enabled:assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])
        textpos=position_ids[0]
        if position_ids.shape[0]==4:position_ids=position_ids[1:]
        rotary=self.d.rotary_emb(h,position_ids)
        for i,layer in enumerate(self.d.layers):
            if pre and i==22:
                self.construct_late_KV(h,rotary,textpos,cp)
                h=h.index_select(1,self.text);cp=cp.index_select(0,self.text)
                textpos=textpos.index_select(-1,self.text)
                rotary=tuple(v.index_select(1,self.text) for v in rotary)
            mask=self.masks_for(h,pre,i)
            if self.audit_enabled:
                qpos=cp
                kpos=torch.arange(self.n,device=h.device) if pre and (i<22 or i in self.read_layers) else self.text if pre else torch.tensor(self.cache.positions[i]+list(range(self.n,self.n+self.step)),device=h.device)
                reference=h.new_zeros((1,1,len(qpos),len(kpos)))
                reference.masked_fill_((kpos[None,:]>qpos[:,None])[None,None],torch.finfo(h.dtype).min)
                image_keys=torch.isin(kpos,self.image)
                if image_keys.any():
                    if i not in self.read_layers:reference[0,0,(qpos>=self.n-1)[:,None]&image_keys[None,:]]=-torch.inf
                    if pre and i>=22:
                        relay=(qpos>self.I[-1])&(qpos<self.n-1)
                        reference[0,0,relay[:,None]&image_keys[None,:]]=-torch.inf
                if mask is not None:assert torch.equal(mask,reference),'MASK_CHANGED'
                self.layers.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],logical_positions=cp.tolist(),persistent_image_keys=len(self.I) if i in self.read_layers else 0))
            self.current=(i,pre,mask)
            h=layer(h,attention_mask=mask,position_ids=textpos,cache_position=cp,
                    position_embeddings=rotary,past_key_values=self.cache,use_cache=True)
            if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
                assert pre and i<3 and h.shape[1]==self.n
                h=self.d._deepstack_process(h,visual_pos_masks,deepstack_visual_embeds[i])
                self.deepstack.append(i)
        self.cache.logical_length=self.n+self.step;self.step+=1
        return BaseModelOutputWithPast(last_hidden_state=self.d.norm(h),past_key_values=self.cache)

    def masks_for(self,h,pre,i):
        selected=i in self.read_layers
        if not pre:
            if i in original.SKIP:
                return None
            if selected not in self._decode_masks:
                length=(self.n if selected else len(self._text_cpu))+31
                self._decode_masks[selected]=h.new_zeros((1,1,1,length))
            return self._decode_masks[selected][...,:len(self.cache.positions[i])+self.step]
        key=(i>=22,selected)
        if key not in self._masks:
            q_cpu=self._text_cpu if i>=22 else list(range(self.n))
            k_cpu=list(range(self.n)) if i<22 or selected else self._text_cpu
            q=torch.tensor(q_cpu,dtype=torch.long)
            k=torch.tensor(k_cpu,dtype=torch.long)
            cpu=torch.zeros((1,1,len(q_cpu),len(k_cpu)),dtype=h.dtype)
            cpu.masked_fill_((k[None,:]>q[:,None])[None,None],torch.finfo(h.dtype).min)
            image_keys=torch.tensor([p in set(self.I) for p in k_cpu])
            if image_keys.any():
                if not selected:cpu[0,0,(q>=self.n-1)[:,None]&image_keys[None,:]]=-torch.inf
                if i>=22:
                    relay=(q>self.I[-1])&(q<self.n-1)
                    cpu[0,0,relay[:,None]&image_keys[None,:]]=-torch.inf
            assert bool(torch.isfinite(cpu).any(-1).all())
            self._masks[key]=cpu.to(h.device)
        return self._masks[key]


class VisualKeysReady(Exception):
    pass


def generate(model,processor,inputs,layout,layers,audit=False,capture=False,cap=128):
    assert layers and not set(layers)&set(original.SKIP)
    hooks=hook_counts(model)
    versions=[(p.data_ptr(),p._version) for p in model.parameters()]
    start=time.perf_counter()
    try:
        holder=[]
        def execution(*args,**kwargs):
            e=Physical(*args,layers=layers,**kwargs)
            holder.append(e)
            return e
        with patch.object(original,'Execution',execution):
            result,vectors=original.generate(model,processor,inputs,layout,'C',audit=audit,capture=capture,max_tokens=cap)
        torch.cuda.synchronize()
        result['e2e_ms']=1000*(time.perf_counter()-start)
        e=holder[0]
        result['late_visual_KV_construction']=dict(extra_prefill_ms=sum(a.elapsed_time(b) for a,b in e.builder_events),
            layers=e.builder_layers,temporary_image_hidden_released=True,second_vision_or_full_prefill=False,
            already_included_in_TTFT_and_E2E=True,
            construction='Actual L22 input causal prefix through image-end; R/P1 are absent; only selected visual KV persists')
        result['physical_svhr']=True
        kv=result['audit']['kv']
        assert not kv['late_KV_not_consumed']
        assert kv['persistent_image_bytes']==len(layers)*layout['image_tokens']*2*8*128*2
        for row in kv['layers']:
            expected=0 if row['layer'] in original.SKIP else layout['input_length']-layout['image_tokens']+len(result['token_ids'])-1
            assert row['persistent_text_tokens']==expected
        if audit:
            for row in result['audit']['layers']:
                if row['step']==0:
                    assert row['hidden_rows']==layout['input_length']-(layout['image_tokens'] if row['layer']>=22 else 0)
        assert versions==[(p.data_ptr(),p._version) for p in model.parameters()]
        assert {k:tensor_digest(v) for k,v in inputs.items()}==layout['input_tensor_hashes']
        assert result['audit']['deepstack']==[0,1,2]
        return result,vectors
    finally:
        assert hooks==hook_counts(model)
