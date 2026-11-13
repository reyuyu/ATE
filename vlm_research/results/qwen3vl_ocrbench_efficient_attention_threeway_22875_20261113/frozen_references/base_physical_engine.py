"""Isolated memory-efficient execution of the frozen Original Qwen Physical C3.

The frozen Execution.context and CompactKV implement skipping, native blocks,
residuals, MLPs and physical KV. Only additive bias construction/storage changes:
one aligned bias instead of base/early/late clones. No dense score tensor is used.
"""
import hashlib
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from vlm_research.qwen3vl_c3v2_physical.engine import Execution,CompactKV,SKIP

def causal_bias(n,dtype,device):
    width=(n+7)//8*8
    storage=torch.full((1,1,n,width),torch.finfo(dtype).min,dtype=dtype,device=device)
    storage.triu_(1)
    return storage[...,:n]

class EfficientExecution(Execution):
    def __init__(self,model,layout,condition,max_tokens=128,audit=False):
        assert condition in ('C','F')
        super().__init__(model,layout,condition,max_tokens,audit)
        self.structure=[];self.mask_checks=[];self.rotary_checks=[]
        self.text_list=self.text.tolist();self.Iset=set(self.I)
        self.max_tokens=max_tokens

    def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,
                attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None,**kwargs):
        h=inputs_embeds if inputs_embeds is not None else self.d.embed_tokens(input_ids)
        pre=self.step==0
        if isinstance(self.cache,CompactKV):self.cache.prefill=pre
        cp=cache_position
        assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])
        assert attention_mask is None or bool((attention_mask==1).all()),'Only frozen unpadded B1 input is supported'
        textpos=position_ids[0]
        if position_ids.shape[0]==4:position_ids=position_ids[1:]
        rotary=self.d.rotary_emb(h,position_ids)
        if pre:
            mask=causal_bias(self.n,h.dtype,h.device)
            mask[...,-1,self.image]=-torch.inf
        elif isinstance(self.cache,CompactKV):mask=None
        else:
            width=(self.n+self.step+7)//8*8
            mask=h.new_zeros((1,1,1,width))[...,:self.n+self.step]
            mask[...,-1,self.image]=-torch.inf
        for i,layer in enumerate(self.d.layers):
            if pre and i==22:
                if self.physical:
                    old_cp=cp;old_rotary=rotary
                    h=h.index_select(1,self.text);cp=cp.index_select(0,self.text)
                    textpos=textpos.index_select(-1,self.text)
                    rotary=tuple(v.index_select(1,self.text) for v in rotary)
                    assert cp.tolist()==self.text_list
                    assert all(torch.equal(a,b.index_select(1,self.text)) for a,b in zip(rotary,old_rotary))
                    self.rotary_checks.append(dict(step=self.step,layer=22,original_absolute_positions=True,rotary_gather_exact=True))
                    mask=causal_bias(len(self.text_list),h.dtype,h.device)
                    del old_cp,old_rotary
                else:
                    qi=torch.arange(self.I[-1]+1,self.n-1,device=h.device)
                    mask[0,0,qi[:,None],self.image[None,:]]=-torch.inf
            self.current=(i,pre,mask)
            expected=len(self.text_list) if pre and self.physical and i>=22 else self.n if pre else 1
            assert h.shape[1]==expected
            self.structure.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],visual_hidden_rows=(len(self.I) if pre and (not self.physical or i<22) else 0),cache_position_count=cp.numel(),cache_position_first=int(cp[0]),cache_position_last=int(cp[-1]),mask_shape=list(mask.shape) if mask is not None else None))
            if self.audit_enabled:self.layers.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],logical_positions=cp.tolist()))
            h=layer(h,attention_mask=mask,position_ids=textpos,cache_position=cp,
                    position_embeddings=rotary,past_key_values=self.cache,use_cache=True)
            if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
                assert pre and i<3 and h.shape[1]==self.n
                h=self.d._deepstack_process(h,visual_pos_masks,deepstack_visual_embeds[i]);self.deepstack.append(i)
        h=self.d.norm(h)
        if isinstance(self.cache,CompactKV):self.cache.logical_length=self.n+self.step
        self.step+=1
        return BaseModelOutputWithPast(last_hidden_state=h,past_key_values=self.cache)

    def check_attention(self,module,q,k,v,mask,event):
        """Check actual keys, masks and compact storage; never allocate full scores."""
        i,pre,_=self.current
        assert i==module.layer_idx==event['layer'] and event['step']==self.step
        if pre:
            kp=self.text_list if self.physical and i>=22 else list(range(self.n-(self.skip and i in SKIP)))
            qp=kp
        else:
            qp=[self.n+self.step-1]
            kp=self.text_list+list(range(self.n,self.n+self.step)) if self.physical else list(range(self.n+self.step))
        assert q.shape[-2]==len(qp) and k.shape[-2]==v.shape[-2]==len(kp)
        if self.physical:
            assert not (not pre and i in SKIP),'Skipped decode attention was executed'
            if i not in SKIP:
                pair=self.cache.entries[i];used=self.cache.used[i]
                assert pair is not None and used==len(self.text_list)+(0 if pre else self.step)
                assert pair[0].shape[-2]==len(self.text_list)+self.max_tokens-1
                # Check the real persisted K/V against the expected text-only
                # selection of the actual attention arguments, not a shape guess.
                if pre:
                    sel=self.text if i<22 else torch.arange(len(self.text_list),device=q.device)
                    assert all(torch.equal(p[...,:used,:],t.index_select(-2,sel)) for p,t in zip(pair,(k,v)))
                else:assert all(torch.equal(p[...,:used,:],t) for p,t in zip(pair,(k,v)))
        if mask is not None:
            assert mask.dtype==q.dtype and mask.shape[-2:]==(len(qp),len(kp))
            assert mask.stride(-2)%8==0
        # Full text-query coverage in gates, bounded deterministic coverage in
        # formal generation. Image queries also checked to preserve causality.
        select=[j for j,p in enumerate(qp) if p not in self.Iset]
        if not self.audit_enabled and len(select)>8:select=select[:3]+select[-5:]
        visual=[j for j,p in enumerate(qp) if p in self.Iset]
        if visual:select+= [visual[0],visual[len(visual)//2],visual[-1]]
        select=sorted(set(select));sel=torch.tensor(select,device=q.device)
        qpos=torch.tensor([qp[j] for j in select],device=q.device);kpos=torch.tensor(kp,device=q.device)
        expected=kpos[None,:]<=qpos[:,None]
        imagekey=torch.isin(kpos,self.image)
        recipient=(qpos==self.n-1)|(qpos>=self.n)
        if i>=22:recipient|=(qpos>self.I[-1])&(qpos<self.n-1)
        expected &= ~(recipient[:,None]&imagekey[None,:])
        actual=torch.ones_like(expected) if mask is None else mask[0,0].index_select(0,sel)==0
        assert torch.equal(actual,expected),('VISIBLE_KEY_SET_MISMATCH',i,self.step)
        forbidden_probability=None
        if self.audit_enabled and bool((recipient & (qpos>=self.n-1)).any()) and bool(imagekey.any()):
            # Only the answer query: at most 32 x 1 x K, not N x N.
            answer=sel[(qpos>=self.n-1).nonzero().flatten()]
            from attention_backend import repeat_kv
            logits=torch.matmul(q.index_select(-2,answer),repeat_kv(k,module.num_key_value_groups).transpose(-1,-2))*module.scaling
            logits+=mask.index_select(-2,answer)
            probabilities=torch.softmax(logits.float(),dim=-1)
            forbidden_probability=float(probabilities[...,imagekey].abs().max())
            assert forbidden_probability==0.0
        self.mask_checks.append(dict(step=self.step,layer=i,query_rows=len(qp),key_rows=len(kp),sampled_queries=len(select),visible_key_set_exact=True,forbidden_image_probability=forbidden_probability,zero_padded_KV=False,cache_text_selection_exact=self.physical and i not in SKIP))

    def audit(self):
        a=super().audit()
        a.update(structure=self.structure,mask_checks=self.mask_checks,rotary_checks=self.rotary_checks,
                 mask_storage='single stride-aligned BF16 additive bias; compact text bias after L22',
                 dense_scores_materialized=False,original_visual_KV_live=0 if self.physical else len(self.I),
                 original_visual_KV_capacity=0 if self.physical else len(self.I),
                 original_frozen_context_and_compactKV_reused=True)
        assert self.deepstack==[0,1,2]
        if self.physical:
            assert all(self.cache.entries[i] is None for i in SKIP)
            assert len(self.mask_checks)==30*self.step+6
            assert len(self.structure)==36*self.step
            assert len(self.rotary_checks)==1
        return a
