"""Frozen physical prefix, independent decode caches, selected real visual KV.

The search uses full visual KV as a functional reference. Physical evaluation
retains only requested layers. Neither path changes the main-view SVH prefix.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import time

from common import BASE
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from vlm_research.qwen3vl_c3v2_physical import engine as original
from vlm_research.qwen3vl_svh_sqa7908_repair.engine import EarlyAlignedExecution
from vlm_research.qwen3vl_early_windows.experiment import hook_counts, tensor_digest

PINNED_EAGER = qm.eager_attention_forward
SKIP = frozenset(original.SKIP)
ACTIVE = frozenset(range(36))-SKIP


def permissions(mask, images, allow):
    if not allow:
        return mask
    out = mask.clone()
    out[..., -1, images] = 0
    return out


class PrefixKV(original.CompactKV):
    def __init__(self, *args, selected=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.selected = frozenset(selected)
        self.visual = {}
        self.images = torch.tensor([i for i in range(self.length) if i not in set(self.text.tolist())],device=self.text.device)

    def update(self,k,v,layer_idx,cache_kwargs=None):
        if self.prefill and layer_idx in self.selected and layer_idx<22:
            self.visual[layer_idx] = tuple(t.index_select(-2,self.images).clone() for t in (k,v))
        return super().update(k,v,layer_idx,cache_kwargs)


class DecodeKV(original.CompactKV):
    def __init__(self, source, selected, extra):
        super().__init__(source.length,source.text,len(source.text)+extra,source.absent)
        self.selected = frozenset(selected)
        self.images = source.images
        self.logical_length = source.logical_length
        self.prefill = False
        self.positions = {}
        for i in ACTIVE:
            keep = i in self.selected
            positions = list(range(self.length)) if keep else self.text.tolist()
            self.positions[i] = positions
            pair = []
            for j in [0,1]:
                base = source.entries[i][j][...,:source.used[i],:]
                target = base.new_empty((*base.shape[:-2],len(positions)+extra,base.shape[-1]))
                if keep:
                    target.index_copy_(-2,self.text,base)
                    target.index_copy_(-2,self.images,source.visual[i][j])
                else:
                    target[...,:len(positions),:].copy_(base)
                pair.append(target)
            self.entries[i] = pair
            self.used[i] = len(positions)

    def update(self,k,v,layer_idx,cache_kwargs=None):
        assert not self.prefill and layer_idx not in self.absent
        pair=self.entries[layer_idx]
        n,m=self.used[layer_idx],k.shape[-2]
        assert k.shape==v.shape and n+m<=pair[0].shape[-2]
        for out,value in zip(pair,(k,v)):
            out[...,n:n+m,:].copy_(value)
        self.used[layer_idx]+=m
        self.updates[layer_idx]+=1
        return tuple(t[...,:n+m,:] for t in pair)

    def audit(self):
        out = super().audit()
        for r in out['layers']:
            visual = len(self.images) if r['layer'] in self.selected else 0
            r['image_tokens'] = visual
            r['image_bytes'] = visual*2*self.entries[r['layer']][0].shape[1]*self.entries[r['layer']][0].shape[-1]*2 if visual else 0
        out['persistent_image_bytes'] = sum(r['image_bytes'] for r in out['layers'])
        return out


class Prefix(EarlyAlignedExecution):
    def __init__(self,*args,collect=(),prefill_layers=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.collect = frozenset(collect)
        self.prefill_layers = frozenset(prefill_layers)
        assert not (self.collect|self.prefill_layers)&SKIP
        self.cache = PrefixKV(self.n,self.text,len(self.text)+self.cache.capacity-len(self.text),SKIP,selected=collect)
        self.sidecar = False
        self.builder = []
        self.decode_layers = frozenset()
        self.decoding = False
        self.permissions = []
        self.matrix_shapes = []
        self.prefill_non_target = []

    def build_late(self,h,kw):
        wanted = {i for i in self.collect if i>=22}
        if not wanted:
            return
        assert not self.sidecar
        end = self.I[-1]+1
        prefix = h[:,:end].clone()
        cp = kw['cache_position'][:end]
        rope = tuple(t[:,:end] for t in kw['position_embeddings'])
        pos = kw['position_ids'][...,:end]
        mask = h.new_full((1,1,end,end),torch.finfo(h.dtype).min).triu(1)
        previous_physical = self.physical
        self.sidecar,self.physical = True,False
        try:
            for i in range(22,max(wanted)+1):
                self.builder.append(dict(layer=i,rows=end,image_rows=len(self.I),persistent=i in wanted))
                if i==max(wanted):
                    # Last layer only needs projected/rotated K,V, not an unused output.
                    layer = self.d.layers[i]
                    norm = layer.input_layernorm(prefix)
                    a = layer.self_attn
                    shape = (*norm.shape[:-1],-1,a.head_dim)
                    k = a.k_norm(a.k_proj(norm).view(shape)).transpose(1,2)
                    v = a.v_proj(norm).view(shape).transpose(1,2)
                    q = a.q_norm(a.q_proj(norm).view(shape)).transpose(1,2)
                    q,k = qm.apply_rotary_pos_emb(q,k,*rope)
                    self.cache.visual[i] = tuple(t.index_select(-2,self.image).clone() for t in (k,v))
                    break
                prefix = self.d.layers[i](prefix,attention_mask=mask,position_ids=pos,
                    cache_position=cp,position_embeddings=rope,past_key_values=None,use_cache=False)
        finally:
            self.sidecar,self.physical = False,previous_physical

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            for i,block in enumerate(self.d.layers):
                def allow_prefill(module,args,kw,i=i):
                    if self.step==0 and not self.sidecar and i<22 and i in self.prefill_layers:
                        kw['attention_mask']=permissions(kw['attention_mask'],self.I,True)
                        return args,kw
                stack.callback(block.self_attn.register_forward_pre_hook(allow_prefill,with_kwargs=True).remove)
            def at21(module,args,kw,out):
                if self.step==0 and not self.sidecar:
                    h = out[0] if isinstance(out,tuple) else out
                    self.build_late(h,kw)
            stack.callback(self.d.layers[21].register_forward_hook(at21,with_kwargs=True).remove)
            if self.audit_enabled:
                for i,block in enumerate(self.d.layers):
                    def non_target(module,args,out,i=i):
                        if self.step==0 and not self.sidecar:
                            h=out[0] if isinstance(out,tuple) else out
                            self.prefill_non_target.append(dict(layer=i,sha256=tensor_digest(h[:,:-1])))
                    stack.callback(block.register_forward_hook(non_target).remove)
            stack.enter_context(super().context())
            inherited = qm.eager_attention_forward

            def eager(module,q,k,v,attention_mask=None,**kw):
                i = getattr(module,'layer_idx',-1)
                if i<0:
                    return inherited(module,q,k,v,attention_mask,**kw)
                if self.sidecar:
                    if i in self.collect:
                        self.cache.visual[i] = tuple(t.index_select(-2,self.image).clone() for t in (k,v))
                    return PINNED_EAGER(module,q,k,v,attention_mask,**kw)
                if self.decoding:
                    assert q.shape[-2]==1 and attention_mask is None
                    positions = self.cache.positions[i]+list(range(self.n,self.n+self.step))
                    assert len(positions)==k.shape[-2]
                    if i not in self.cache.selected:
                        from vlm_research.qwen3vl_c3v2_numerical_repair2.repair import restore_dead_image_columns
                        k,v,mask,ix = restore_dead_image_columns(k,v,self.text.tolist(),self.n,self.step)
                    else:
                        mask = k.new_zeros((1,1,1,k.shape[-2]))
                        ix = None
                    allow = i in self.decode_layers
                    if not allow:
                        mask[...,self.image] = -torch.inf
                    out,weights = PINNED_EAGER(module,q,k,v,mask,**kw)
                    if self.audit_enabled:
                        mass = weights[...,self.image].float().sum(-1).mean()
                        assert allow or float(mass)==0
                        assert torch.isfinite(weights).all()
                        self.permissions.append(dict(layer=i,step=self.step,allowed=allow,image_mass=float(mass)))
                        self.matrix_shapes.append(dict(layer=i,step=self.step,query=list(q.shape),key=list(k.shape),
                            flops=4*q.shape[0]*q.shape[1]*q.shape[-2]*k.shape[-2]*q.shape[-1]))
                    return out,weights if ix is None else weights.index_select(-1,ix)
                if self.step==0 and i in self.prefill_layers:
                    mask = permissions(attention_mask,self.image.tolist(),True) if i<22 else attention_mask
                    if i>=22:
                        # Functional auxiliary prefix: restore only real visual K/V at this layer.
                        from vlm_research.qwen3vl_svh_numerical_repair.engine import restore_columns
                        k,v,mask,ix = restore_columns(k,v,mask,self.text.tolist(),self.n)
                        k.index_copy_(-2,self.image,self.cache.visual[i][0])
                        v.index_copy_(-2,self.image,self.cache.visual[i][1])
                        mask[..., -1,self.image]=0
                        fullq = q.new_zeros((*q.shape[:-2],self.n,q.shape[-1]))
                        fullq.index_copy_(-2,self.text,q)
                        fullmask = mask.new_full((1,1,self.n,self.n),-torch.inf)
                        fullmask.index_copy_(-2,self.text,mask)
                        # Dead rows must have one legal key to avoid NaN.
                        fullmask[...,self.image,0]=0
                        out,w=PINNED_EAGER(module,fullq,k,v,fullmask,**kw)
                        if self.audit_enabled:
                            assert torch.isfinite(w).all()
                            assert (w.float().sum(-1)-1).abs().max()<.01
                            relay=(self.text>self.I[-1])&(self.text<self.n-1)
                            assert float(w.index_select(-2,self.text[relay])[...,self.image].abs().max())==0
                            self.permissions.append(dict(layer=i,step=0,allowed=True,
                                image_mass=float(w[...,self.n-1,self.image].float().sum(-1).mean()),relay_mask_zero=True))
                            self.matrix_shapes.append(dict(layer=i,step=0,query=list(fullq.shape),key=list(k.shape),
                                flops=4*fullq.shape[0]*fullq.shape[1]*self.n*self.n*fullq.shape[-1]))
                        # The inherited observer expects compact-only row sums. Full weights
                        # are audited here; do not misrepresent a visual-trimmed row as normalized.
                        return out.index_select(1,self.text),None
                    return inherited(module,q,k,v,mask,**kw)
                return inherited(module,q,k,v,attention_mask,**kw)
            stack.enter_context(patch.object(qm,'eager_attention_forward',eager))
            yield self


class Request:
    def __init__(self,model,processor,inputs,layout,collect=ACTIVE,prefill_layers=(),audit=False,cap=128):
        self.model,self.processor,self.layout = model,processor,layout
        self.hooks = hook_counts(model)
        self.versions = [(p.data_ptr(),p._version) for p in model.parameters()]
        self.inputs,self.cap = inputs,cap
        self.e = Prefix(model,layout,'C',max_tokens=cap,audit=audit,collect=collect,prefill_layers=prefill_layers)
        self.context = self.e.context()
        self.scope = torch.inference_mode()

    def __enter__(self):
        self.scope.__enter__()
        self.context.__enter__()
        try:
            device = next(self.model.parameters()).device
            x = {k:v.to(device) for k,v in self.inputs.items()}
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.first = self.model(**x,use_cache=True,past_key_values=None,
                cache_position=torch.arange(self.layout['input_length'],device=device),logits_to_keep=1)
            torch.cuda.synchronize()
            self.prefill_ms = (time.perf_counter()-start)*1000
            self.source = self.e.cache
            self.rope_delta = self.model.model.rope_deltas.clone()
            assert self.e.step==1 and torch.isfinite(self.first.logits).all()
            assert self.e.deepstack==[0,1,2]
            assert set(self.source.visual)==self.e.collect
            self.prefix_hash = tensor_digest(self.first.logits)
            return self
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise

    def generate(self,layers,physical=False,cap=None,fixed=False,capture=False,consume=False):
        layers = frozenset(layers)
        assert layers<=self.e.collect and not layers&SKIP
        limit = cap or self.cap
        self.e.cache = DecodeKV(self.source,layers if physical else self.e.collect,limit-1)
        if consume:
            assert physical
            # Single-request physical timing does not retain a second prefix cache.
            self.first.past_key_values=None
            self.source.entries=[]
            self.source.visual.clear()
            self.source=None
        self.e.step,self.e.decoding,self.e.decode_layers = 1,True,layers
        self.model.model.rope_deltas = self.rope_delta.clone()
        self.e.permissions = []
        self.e.matrix_shapes = []
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos,int) else set(eos)
        vector = self.first.logits[0,-1]
        token = vector.argmax()
        ids = [int(token)]
        vectors = [vector.detach().float().cpu()] if capture else []
        times = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(1,limit):
            if not fixed and ids[-1] in eos:
                break
            device = token.device
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            out = self.model(input_ids=token.reshape(1,1),attention_mask=torch.ones((1,self.layout['input_length']+t),device=device,dtype=torch.long),
                use_cache=True,past_key_values=self.e.cache,cache_position=torch.tensor([self.layout['input_length']+t-1],device=device),logits_to_keep=1)
            b.record()
            times.append((a,b))
            vector = out.logits[0,-1]
            assert torch.isfinite(vector).all()
            token = vector.argmax()
            ids.append(int(token))
            if capture:
                vectors.append(vector.detach().float().cpu())
        torch.cuda.synchronize()
        decode_wall = (time.perf_counter()-start)*1000
        text = self.processor.tokenizer.decode(ids,skip_special_tokens=True)
        kv = self.e.cache.audit()
        result = dict(token_ids=ids,text=text,generated_tokens=len(ids),eos=ids[-1] in eos,truncated=ids[-1] not in eos,
            empty=not text.strip(),prefix_logits_sha256=self.prefix_hash,prefill_ms=self.prefill_ms,
            decode_ms=sum(a.elapsed_time(b) for a,b in times),decode_wall_ms=decode_wall,
            request_ms=self.prefill_ms+decode_wall,kv=kv,physical_svhr=physical,
            permissions=self.e.permissions,matrix_shapes=self.e.matrix_shapes,
            temporary_sidecar=self.e.builder,cache_independent=True,
            prefill_layers=sorted(self.e.prefill_layers),decode_layers=sorted(layers),
            prefill_non_target_hashes=self.e.prefill_non_target,
            strict_numerical_status='FAIL_RETAINED')
        if self.e.audit_enabled:
            result['execution_audit']=self.e.audit()
        if self.source is not None:
            self.e.cache = self.source
        self.e.decoding = False
        return result,vectors

    def __exit__(self,*args):
        try:
            self.context.__exit__(*args)
        finally:
            self.scope.__exit__(*args)
        assert self.hooks==hook_counts(self.model)
        assert self.versions==[(p.data_ptr(),p._version) for p in self.model.parameters()]
        assert {k:tensor_digest(v) for k,v in self.inputs.items()}==self.layout['input_tensor_hashes']
