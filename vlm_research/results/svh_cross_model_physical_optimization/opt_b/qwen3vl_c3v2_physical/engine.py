"""Qwen3-VL physical C3: native blocks, original mRoPE, compact real KV.

No learned parameters, alternate attention kernel, or mask fallback.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import time
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qmodel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

CONDITIONS = ['N','F','P','B','C']
SKIP = tuple(range(1,7))

class CompactKV:
    def __init__(self, length, text, capacity, absent=()):
        self.logical_length=0; self.length=length; self.text=text
        self.capacity=capacity; self.absent=absent; self.prefill=True
        self.entries=[None]*36; self.used=[0]*36; self.updates=[0]*36
    def get_seq_length(self, layer_idx=0):
        return self.logical_length
    def update(self,k,v,layer_idx,cache_kwargs=None):
        i=layer_idx
        assert i not in self.absent
        self.updates[i]+=1
        if self.prefill:
            stored=[t.index_select(-2,self.text) if t.shape[-2]==self.length else t for t in (k,v)]
            n=stored[0].shape[-2]
            pair=[t.new_empty((*t.shape[:-2],self.capacity,t.shape[-1])) for t in stored]
            for out,t in zip(pair,stored):out[...,:n,:].copy_(t)
            self.entries[i]=pair;self.used[i]=n
            return k,v
        pair=self.entries[i]; assert pair is not None
        n=self.used[i];m=k.shape[-2];assert n+m<=self.capacity
        for out,t in zip(pair,(k,v)):out[...,n:n+m,:].copy_(t)
        self.used[i]+=m
        return tuple(t[...,:n+m,:] for t in pair)
    def audit(self):
        rows=[]
        for i,pair in enumerate(self.entries):
            if pair is None:
                assert i in self.absent
                rows.append(dict(layer=i,live_tokens=0,capacity_tokens=0,live_bytes=0,capacity_bytes=0,updates=self.updates[i]));continue
            rows.append(dict(layer=i,live_tokens=self.used[i],capacity_tokens=pair[0].shape[-2],
                live_bytes=sum(t.shape[0]*t.shape[1]*self.used[i]*t.shape[-1]*t.element_size() for t in pair),
                capacity_bytes=sum(t.untyped_storage().nbytes() for t in pair),updates=self.updates[i]))
        return dict(layers=rows,live_bytes=sum(x['live_bytes'] for x in rows),capacity_bytes=sum(x['capacity_bytes'] for x in rows),logical_length=self.logical_length,historical_append_copies=0)

class Execution:
    def __init__(self,model,layout,condition,max_tokens=32,audit=False,noop=False):
        self.model=model;self.d=model.model.language_model;self.l=layout;self.c=condition
        self.audit_enabled=audit;self.noop=noop;self.step=0;self.events=[];self.shapes=[];self.probability=[]
        self.device=next(model.parameters()).device
        self.n=layout['input_length']; self.I=layout['image_positions']
        image_set=set(self.I); self._text_cpu=[i for i in range(self.n) if i not in image_set]
        self.text=torch.tensor(self._text_cpu,device=self.device)
        self.image=torch.tensor(self.I,device=self.device)
        self.physical=condition in ('P','C');self.bypass=condition in ('F','B','C','A16-functional')
        self.skip=condition in ('B','C')
        self.late=condition in ('F','P','B','C','W22-functional')
        self.cache=CompactKV(self.n,self.text,len(self.text)+max_tokens-1,SKIP if self.skip else ()) if condition in ('P','B','C') else DynamicCache(config=self.d.config)
        self.layers=[];self.deepstack=[];self.position_hashes=[]
        self.current=None
    def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,
                attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None,**kwargs):
        h=inputs_embeds if inputs_embeds is not None else self.d.embed_tokens(input_ids)
        pre=self.step==0
        if isinstance(self.cache,CompactKV): self.cache.prefill=pre
        cp=cache_position
        if self.audit_enabled: assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])
        textpos=position_ids[0]
        if position_ids.shape[0]==4:position_ids=position_ids[1:]
        rotary=self.d.rotary_emb(h,position_ids)
        if not pre and isinstance(self.cache,CompactKV):
            # Every persistent key is text, in its original causal order. HF's
            # mask builder cannot infer our physical length from a Cache=None.
            early=late=None
        else:
            base=create_causal_mask(config=self.d.config,input_embeds=h,attention_mask=attention_mask,
                cache_position=cp,past_key_values=self.cache if not isinstance(self.cache,CompactKV) else None,
                position_ids=textpos)
            if base is None:
                if pre:base=torch.full((1,1,self.n,self.n),torch.finfo(h.dtype).min,device=h.device,dtype=h.dtype).triu(1)
                else:base=h.new_zeros((1,1,1,self.n+self.step))
            assert base.shape[-1]>=self.n
            early=base.clone()
            early[..., -1,self.image]=-torch.inf
            late=early.clone()
            if pre and self.late:
                qi=torch.arange(self.I[-1]+1,self.n-1,device=h.device)
                late[0,0,qi[:,None],self.image[None,:]]=-torch.inf
        for i,layer in enumerate(self.d.layers):
            if pre and self.physical and i==22:
                h=h.index_select(1,self.text);cp=cp.index_select(0,self.text)
                textpos=textpos.index_select(-1,self.text)
                rotary=tuple(v.index_select(1,self.text) for v in rotary)
            mask=late if self.late and i>=22 else early
            if pre and self.physical and i>=22:
                if i==22: compact_late=mask.index_select(-2,self.text).index_select(-1,self.text)
                mask=compact_late
            if not pre and isinstance(self.cache,CompactKV):mask=None
            self.current=(i,pre,mask)
            if self.audit_enabled:self.layers.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],logical_positions=cp.tolist()))
            h=layer(h,attention_mask=mask,position_ids=textpos,cache_position=cp,
                    position_embeddings=rotary,past_key_values=self.cache,use_cache=True)
            if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
                assert pre and i<3 and h.shape[1]==self.n
                h=self.d._deepstack_process(h,visual_pos_masks,deepstack_visual_embeds[i])
                self.deepstack.append(i)
        h=self.d.norm(h)
        if isinstance(self.cache,CompactKV):self.cache.logical_length=self.n+self.step
        self.step+=1
        return BaseModelOutputWithPast(last_hidden_state=h,past_key_values=self.cache)
    @contextmanager
    def context(self):
        with ExitStack() as st:
            if self.audit_enabled:
                eager=qmodel.eager_attention_forward
                def observed_eager(module,q,k,v,attention_mask=None,**kw):
                    if not hasattr(module,'layer_idx'):
                        return eager(module,q,k,v,attention_mask,**kw)
                    assert q.shape[-1]==k.shape[-1] and k.shape[-2]==v.shape[-2]
                    flops=2*q.shape[0]*q.shape[1]*q.shape[-2]*k.shape[-2]*q.shape[-1]
                    for op in ['QK','AV']:
                        self.shapes.append(dict(layer=module.layer_idx,step=self.step,op=op,
                            query=list(q.shape),key=list(k.shape),value=list(v.shape),flops=flops))
                    return eager(module,q,k,v,attention_mask,**kw)
                st.enter_context(patch.object(qmodel,'eager_attention_forward',observed_eager))
            if self.c!='N':st.enter_context(patch.object(self.d,'forward',self.forward))
            for i,layer in enumerate(self.d.layers):
                a=layer.self_attn;original=a.forward
                def wrapped(*args,original=original,i=i,**kw):
                    x=kw.get('hidden_states',args[0] if args else None)
                    pre=x.shape[1]>1
                    if self.c=='N':return original(*args,**kw)
                    shape=list(x.shape)
                    if self.skip and i in SKIP:
                        if not pre:
                            self.events.append(dict(layer=i,step=self.step,skipped_rows=1,qkv_calls=0,o_proj_calls=0))
                            return torch.zeros_like(x),None
                        assert x.shape[1]==self.n
                        # P-1 KV can never be consumed by a future query at these layers.
                        kw['hidden_states']=x[:,:-1];args=()
                        kw['attention_mask']=kw['attention_mask'][...,:-1,:-1]
                        kw['position_embeddings']=tuple(v[:,:-1] for v in kw['position_embeddings'])
                        kw['cache_position']=kw['cache_position'][:-1]
                        # Temporary prefix KV only; all later queries are skipped.
                        kw['past_key_values']=None
                        out,w=original(**kw)
                        result=torch.cat((out,torch.zeros_like(x[:,-1:])),dim=1)
                        self.events.append(dict(layer=i,step=self.step,skipped_rows=1,qkv_calls=1,o_proj_calls=1,computed_rows=self.n-1))
                    else:
                        result,w=original(*args,**kw)
                        if self.bypass and i in SKIP:
                            result=result.clone();result[:,-1,:]=0
                    if self.audit_enabled and w is not None:
                        mask=kw.get('attention_mask')
                        assert bool(torch.isfinite(w).all()) and float(w.min())>=0
                        rowerr=float((w.float().sum(-1)-1).abs().max());assert rowerr<.01
                        if mask is not None:
                            blocked=torch.isneginf(mask[0,0])
                            if blocked.any(): assert float(w[0,:,blocked].abs().max())==0
                        self.probability.append(dict(layer=i,step=self.step,row_sum_error=rowerr,forbidden_probability=0))
                    return result,None # Native eager computes weights transiently; never retain matrices.
                if self.c!='N' or self.noop:st.enter_context(patch.object(a,'forward',wrapped))
                if self.audit_enabled:
                    trace={}
                    def start(m,args,t=trace):t['input']=args[0].detach()
                    def attn_out(m,args,out,t=trace):t['attention']=out[0].detach()
                    def mid(m,args,t=trace):
                        t['mid']=args[0].detach()
                        assert torch.equal(t['input']+t['attention'],t['mid'])
                    def mlp_out(m,args,out,t=trace):t['mlp']=out.detach()
                    def end(m,args,out,t=trace,i=i):
                        assert torch.equal(t['mid']+t['mlp'],out)
                        self.events.append(dict(layer=i,step=self.step,residual_exact=True,mlp_and_norm_executed=True))
                        t.clear()
                    for handle in [layer.register_forward_pre_hook(start),a.register_forward_hook(attn_out),
                            layer.post_attention_layernorm.register_forward_pre_hook(mid),layer.mlp.register_forward_hook(mlp_out),layer.register_forward_hook(end)]:st.callback(handle.remove)
                    for name in ['q_proj','k_proj','v_proj','o_proj']:
                        mod=getattr(a,name)
                        def record(m,ar,out,i=i,name=name):
                            self.shapes.append(dict(layer=i,step=self.step,op=name,input=list(ar[0].shape),output=list(out.shape),flops=2*ar[0].numel()*out.shape[-1]))
                        handle=mod.register_forward_hook(record);st.callback(handle.remove)
                    for name in ['gate_proj','up_proj','down_proj']:
                        mod=getattr(layer.mlp,name)
                        def record(m,ar,out,i=i,name=name):
                            self.shapes.append(dict(layer=i,step=self.step,op=name,input=list(ar[0].shape),output=list(out.shape),flops=2*ar[0].numel()*out.shape[-1]))
                        handle=mod.register_forward_hook(record);st.callback(handle.remove)
            yield self
    def audit(self):
        if isinstance(self.cache,CompactKV):kv=self.cache.audit()
        else:
            rows=[]
            for i,l in enumerate(self.cache.layers):
                pair=[l.keys,l.values]
                rows.append(dict(layer=i,live_tokens=pair[0].shape[-2],capacity_tokens=pair[0].shape[-2],live_bytes=sum(t.numel()*t.element_size() for t in pair),capacity_bytes=sum(t.untyped_storage().nbytes() for t in pair)))
            kv=dict(layers=rows,live_bytes=sum(x['live_bytes'] for x in rows),capacity_bytes=sum(x['capacity_bytes'] for x in rows))
        return dict(condition=self.c,physical_fallback=False,layers=self.layers,deepstack=self.deepstack,
            attention_events=self.events,projection_shapes=self.shapes,probability=self.probability,kv=kv)

def generate(model,processor,inputs,layout,condition,audit=False,capture=False,noop=False,max_tokens=32):
    device=next(model.parameters()).device
    x={k:v.to(device) for k,v in inputs.items()}
    e=Execution(model,layout,condition,max_tokens,audit,noop)
    cache=None;ids=[];scores=[];events=[];vision=[]
    original=model.model.get_image_features
    def vision_call(*a,**kw):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();v=original(*a,**kw);end.record();vision.append((start,end));return v
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall=time.perf_counter()
    with torch.inference_mode(),e.context(),patch.object(model.model,'get_image_features',vision_call):
        for t in range(max_tokens):
            kw=dict(x) if t==0 else dict(input_ids=token.reshape(1,1),attention_mask=torch.ones((1,layout['input_length']+t),device=device,dtype=torch.long))
            kw.update(use_cache=True,past_key_values=cache,cache_position=torch.arange(layout['input_length'],device=device) if t==0 else torch.tensor([layout['input_length']+t-1],device=device),logits_to_keep=1)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();out=model(**kw);end.record();events.append((start,end))
            assert out.logits.shape[1]==1
            vector=out.logits[0,-1]
            assert bool(torch.isfinite(vector).all())
            token=vector.argmax();tid=int(token);ids.append(tid)
            if capture:scores.append(vector.detach().float().cpu())
            cache=out.past_key_values
            if tid in eos:break
        torch.cuda.synchronize();total=(time.perf_counter()-wall)*1000
    # Native uses its own DynamicCache, not the initially unused Execution cache.
    if condition=='N':e.cache=cache
    timing=[a.elapsed_time(b) for a,b in events];vm=sum(a.elapsed_time(b) for a,b in vision)
    result=dict(condition=condition,token_ids=ids,text=processor.tokenizer.decode(ids,skip_special_tokens=True),
        eos=ids[-1] in eos,truncated=ids[-1] not in eos,empty=not processor.tokenizer.decode(ids,skip_special_tokens=True).strip(),
        generated_tokens=len(ids),physical_fallback=False,vision_ms=vm,prefill_ms=timing[0]-vm,ttft_ms=timing[0],
        decode_ms=sum(timing[1:]),decode_ms_per_token=sum(timing[1:])/max(1,len(ids)-1),e2e_ms=total,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),audit=e.audit())
    return result,scores
