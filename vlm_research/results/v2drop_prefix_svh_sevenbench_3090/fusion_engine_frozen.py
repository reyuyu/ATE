"""Isolated V2Drop L3/L17 prefix + frozen SVH execution, HF 4.49 eager.

The original official selector and ATE attention/MLP/norm/residual implementations
are imported unchanged. DenseOracle is solely a short numerical smoke reference.
"""
from dataclasses import replace
import hashlib,importlib.util
from pathlib import Path
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter,SelectiveKV
from vlm_research.ate_fulljoint_v2.engine import AnswerWriter,LAYERS

R=Path('/home/featurize/work/ATE');V=R/'vlm_research/results/v2drop_quick_reproduction'
spec=importlib.util.spec_from_file_location('frozen_v2drop_adapter',V/'hf_adapter.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
SELECT,FIXED_SELECT,SELECTOR_PROVENANCE=v2.official_selector(V/'official_sources/V2Drop.py')

class PrefixKV(SelectiveKV):
    """Reuse frozen in-place text gather for the current, shorter Prefill layout.

    Keys returned to current attention remain transient and contain retained visual
    rows; only text is copied into the final persistent buffers. Global positions
    and the frozen audit layout are restored before returning from update().
    """
    def update(self,keys,values,layer_idx,cache_kwargs=None):
        if not self.prefill:return super().update(keys,values,layer_idx,cache_kwargs)
        original_layout,original_index=self.layout,self.text_index
        assert keys.shape[-2]==self.active_positions.numel()
        try:
            self.layout=replace(original_layout,length=keys.shape[-2],text=tuple(self.local_text.tolist()))
            self.text_index=self.local_text
            return super().update(keys,values,layer_idx,cache_kwargs)
        finally:self.layout,self.text_index=original_layout,original_index

class PrefixSVH(MiddleWriter):
    def __init__(self,adapter,layout,max_tokens,enabled=True,audit=False):
        assert layout.cutoff==20 and len(layout.image)==576
        super().__init__(adapter,layout,(),max_tokens,audit,dense=False)
        self.enabled=enabled
        self.cache=PrefixKV(len(adapter.layers),layout,'Fast-P2',max_tokens,self.device,keep=())
        self.active=torch.arange(layout.length,device=self.device)
        self.image_tensor=torch.tensor(layout.image,device=self.device)
        self.selection_events=[];self.layer_events=[];self.visual_events=[]
        self.refresh()

    def refresh(self):
        image=torch.isin(self.active,self.image_tensor)
        self.local_image=torch.where(image)[0]
        self.local_text=torch.where(~image)[0]
        assert torch.equal(self.active.index_select(0,self.local_text),self.text_positions),'Text row lost or reordered'
        self.cache.active_positions=self.active;self.cache.local_text=self.local_text

    def prune(self,h,previous,layer):
        amount=self.local_image.numel();start=int(self.local_image[0])
        assert torch.equal(self.local_image,torch.arange(start,start+amount,device=self.device))
        keep,selected,scores=SELECT(h,previous,start,amount,layer)
        assert selected.numel()==amount//2
        # The unchanged official fixed-35 kernel independently verifies adaptation.
        if self.audit_enabled:
            pad=h.new_zeros((1,35,h.shape[-1]))
            _,rs,rv=FIXED_SELECT(torch.cat([pad,h[:,start:]],1),torch.cat([pad,previous[:,start:]],1),35,amount,layer)
            assert torch.equal(rs-35,selected-start) and torch.equal(rv,scores)
        h=h.index_select(1,keep);self.active=self.active.index_select(0,keep);self.refresh()
        self.selection_events.append(dict(layer=layer,visual_before=amount,visual_after=self.local_image.numel(),
            retained_original_image_positions=self.active.index_select(0,self.local_image).tolist(),
            official_fixed35_selection_exact=bool(self.audit_enabled),text_preserved=True))
        return h

    def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,**kwargs):
        h=inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        pre=self.step==0;n=self.layout.length
        original=self.positions[:n] if pre else self.positions[n+self.step-1:n+self.step]
        full_ids=original.unsqueeze(0) if position_ids is None else position_ids
        assert torch.equal(full_ids[0],original),'Unexpected RoPE input positions'
        full_rotary=self.decoder.rotary_emb(h,full_ids)
        if self.audit_enabled:self.rotary.append(dict(position_ids=full_ids.detach().cpu().clone(),cos=full_rotary[0].detach().cpu().clone(),sin=full_rotary[1].detach().cpu().clone()))
        self.cache.prefill=pre;previous=None
        for i,layer in enumerate(self.a.layers):
            if pre:
                if self.enabled and i in (3,17):h=self.prune(h,previous,i)
                if i==20:
                    h=h.index_select(1,self.local_text);self.active=self.active.index_select(0,self.local_text);self.refresh()
                positions=self.active
                if i>=20:mask=self.prefill_late
                elif self.active.numel()==n:mask=self.prefill_early
                else:mask=self.prefill_early.index_select(-2,self.active).index_select(-1,self.active)
                ids=full_ids if self.active.numel()==n else full_ids.index_select(-1,self.active)
                rotary=full_rotary if self.active.numel()==n else tuple(t.index_select(1,self.active) for t in full_rotary)
                self.active_mask=mask[...,:-1,:]
                key_count=h.shape[1]
            else:
                positions=original;ids=full_ids;rotary=full_rotary;mask=None;key_count=self.cache.used[i]+1
            self.current=(i,pre,mask,positions,key_count)
            if self.audit_enabled:
                assert h.shape[1]==positions.numel() and torch.equal(ids[0],positions)
                if pre:
                    assert torch.equal(rotary[0],full_rotary[0].index_select(1,self.active))
                    assert torch.equal(rotary[1],full_rotary[1].index_select(1,self.active))
                    assert (mask==0).any(-1).all()
                self.layer_events.append(dict(layer=i,prefill=pre,step=self.step,hidden_rows=h.shape[1],
                    visual_hidden=self.local_image.numel() if pre else 0,
                    original_positions_sha256=hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
                    RoPE_original_positions_exact=True,text_preserved=True))
            previous=h
            h=layer(h,attention_mask=mask,position_ids=ids,cache_position=positions,
                position_embeddings=rotary,past_key_value=self.cache,use_cache=True)[0]
            if self.audit_enabled:
                assert torch.isfinite(h).all(),'Nonfinite hidden state'
                if pre:self.hidden.append(h.index_select(1,self.local_text).detach().float().cpu())
        h=self.decoder.norm(h);self.cache.logical_length=n+self.step;self.step+=1
        return BaseModelOutputWithPast(last_hidden_state=h,past_key_values=self.cache)

    def record(self,out,weights,positions,shape,mask):
        if not self.collect:return
        i,pre,_,global_query,_=self.current
        local_query=list(positions)
        query_original=global_query[local_query].tolist() if local_query else []
        AnswerWriter.record(self,out,weights,query_original,shape,mask)
        if weights is None:
            assert not pre and i in LAYERS
            self.visual_events.append(dict(layer=i,prefill=False,image_keys=0,answer_image_mass=0.,relay_image_mass=0.,answer_attention_skipped=True));return
        image=self.local_image if pre else torch.empty(0,dtype=torch.long,device=self.device)
        relay=[j for j,p in enumerate(query_original) if p in self.layout.relay]
        answer=[j for j,p in enumerate(query_original) if p>=self.layout.length-1]
        mass=writer_mass=0.
        if image.numel():
            assert pre and i<20
            if answer:
                answer_visual=weights[0,:,answer,:].index_select(-1,image)
                assert torch.count_nonzero(answer_visual)==0
                mass=float(answer_visual.float().sum())
            assert relay,'No SVH receiving queries'
            allowed=mask[0,0,relay,:].index_select(-1,image);assert torch.count_nonzero(allowed)==0
            writer_mass=float(weights[0,:,relay,:].index_select(-1,image).float().sum(-1).mean())
            assert writer_mass>0,'Retained visual keys not actually read by SVH writers'
        if i in LAYERS and pre:assert weights.shape[-2]==shape[1]-1 and not answer
        self.visual_events.append(dict(layer=i,prefill=pre,image_keys=int(image.numel()),answer_image_mass=mass,
            relay_image_mass=writer_mass,answer_attention_skipped=i in LAYERS,retained_visual_keys_read=bool(image.numel())))

class DenseOracle(MiddleWriter):
    """Full-shape, masked short reference with the physical run's frozen selections.

    Never used for quality generation. This isolates physical compaction from
    selector branch instability; official selector equality is checked separately.
    """
    def __init__(self,adapter,layout,max_tokens,selections):
        super().__init__(adapter,layout,(),max_tokens,audit=True,dense=True)
        self.selected={s['layer']:s['retained_original_image_positions'] for s in selections}
        self.deleted=[]
    def prefill_mask(self,layer):
        if layer in self.selected:self.deleted=sorted(set(self.layout.image)-set(self.selected[layer]))
        mask=(self.prefill_early if layer<20 else self.full_late).clone()
        if self.deleted:mask[...,self.deleted]=-torch.inf
        return mask
    def record(self,out,weights,positions,shape,mask):
        # Dense is a numerical oracle; structural checks apply to PrefixSVH.
        AnswerWriter.record(self,out,weights,positions,shape,mask)

@torch.inference_mode()
def greedy(adapter,layout,control,max_tokens,capture=False,fixed=False):
    """Frozen greedy/EOS semantics, without timing/FLOPs/profiling hooks."""
    n=layout.length;device=adapter.inputs['input_ids'].device
    positions=torch.arange(n+max_tokens-1,device=device)
    attention=torch.ones((1,n+max_tokens-1),dtype=torch.long,device=device)
    eos=adapter.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    tokens=[];logits=[];cache=None;gpu_token=None
    with control.context():
        for step in range(max_tokens):
            kw=dict(adapter.inputs) if step==0 else dict(input_ids=gpu_token.reshape(1,1),attention_mask=attention[:,:n+step])
            kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],logits_to_keep=1)
            output=adapter.model(**kw);last=output.logits[:,-1]
            assert torch.isfinite(last).all(),'Nonfinite effective logits'
            gpu_token=last.argmax(-1);tokens.append(int(gpu_token));cache=output.past_key_values
            if capture:logits.append(last.detach().float().cpu().clone())
            del output
            if not fixed and tokens[-1] in eos:break
    raw=adapter.processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
    return dict(text=raw,token_ids=tokens,eos=tokens[-1] in eos,empty=not bool(raw),
        truncated=not fixed and len(tokens)==max_tokens and tokens[-1] not in eos,
        all_effective_logits_finite=True),logits
