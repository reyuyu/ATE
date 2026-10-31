"""Adapt the Phase3.7 directed additive-mask primitive to cached Prefill only."""
from contextlib import AbstractContextManager, ExitStack
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from directed_suffix import DirectedSuffixKnockout
from knockout_probe import knockout_mask, model_layers, ROW_ATOL, tensor_hash
from l0l1_bos.capture import EarlyRows, head_mass, vector_change
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from paper_rq1_decode_layerwise_quality.engine import is_decode


def key_groups(mapping):
    return {k:mapping[k] for k in ('bos','image','question','template')}


def local_weights(module, hidden, embeddings, mask):
    shape=(*hidden.shape[:-1],-1,module.head_dim)
    q=F.linear(hidden,module.q_proj.weight,module.q_proj.bias).view(shape).transpose(1,2)
    k=F.linear(hidden,module.k_proj.weight,module.k_proj.bias).view(shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*embeddings)
    k=repeat_kv(k,module.num_key_value_groups)
    scores=torch.matmul(q,k.transpose(2,3))*module.scaling
    scores=scores+mask[:,:,:,:k.shape[-2]]
    return F.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)


class SourceMask(DirectedSuffixKnockout):
    def __init__(self,model,prompt,mapping,condition,noop=False,audit=False,expected_visual=576):
        self.n=prompt['input_ids'].shape[1]
        self.target_layer=condition.get('layer')
        self.noop,self.audit=noop,audit
        self.phase={}; self.local={}; self.applied=0; self.decode_calls=0; self.local_checks=[]
        queries=mapping['question'] if condition.get('query')=='question' else [self.n-1]
        super().__init__(model_layers(model),prompt,model.config.image_token_index,0,queries,[],expected_visual)
        self.active=set() if noop or self.target_layer is None else {self.target_layer}
        if condition.get('source')=='question':
            selected=torch.zeros_like(self.layout['visual']); selected[mapping['question']]=True
            self.layout=dict(self.layout,visual=selected)
        self.queries=queries
        self.keys=mapping.get(condition.get('source'),[])

    def __enter__(self):
        try:
            super().__enter__()
            if self.audit and self.target_layer is not None:
                self.handles.append(self.layers[self.target_layer].self_attn.register_forward_hook(self._after))
            return self
        except BaseException:
            self.__exit__(None,None,None); raise

    def _hook(self,index):
        def hook(module,args,kwargs):
            hidden=kwargs['hidden_states']; cache=kwargs.get('past_key_value')
            past=cache.get_seq_length(index) if cache is not None else 0
            decode=is_decode(kwargs.get('cache_position'),self.n,past,hidden.shape[1])
            self.phase[index]=decode
            if decode:
                self.decode_calls+=1
                return args,kwargs
            assert hidden.shape[:2]==(1,self.n)
            if index not in self.active: return args,kwargs
            native=kwargs['attention_mask']; before=tensor_hash(native) if self.audit else None
            changed,count=knockout_mask(native,self.layout)
            assert count==len(self.queries)*len(self.keys) and count>0
            self.applied+=1
            if self.audit:
                assert tensor_hash(native)==before
                self.local[index]=(hidden,kwargs['position_embeddings'],native,changed)
            return args,dict(kwargs,attention_mask=changed)
        return hook

    def _after(self,module,args,output):
        index=self.target_layer
        if self.phase[index] or index not in self.active: return
        hidden,emb,native,changed=self.local.pop(index)
        a=output[1]
        assert a is not None and torch.isfinite(a).all()
        target=a[0,:,self.queries][:,:,self.keys]
        assert torch.count_nonzero(target)==0
        assert float((a.float().sum(-1)-1).abs().max())<=ROW_ATOL
        expected=local_weights(module,hidden,emb,changed)
        assert torch.equal(a,expected),'Masked local native eager attention mismatch'
        original=local_weights(module,hidden,emb,native)
        other=torch.ones(self.n,dtype=torch.bool,device=a.device); other[self.queries]=False
        assert torch.equal(a[:,:,other],original[:,:,other]),'Other local Query attention changed'
        self.local_checks.append(dict(layer=index,target_queries=self.queries,target_keys=self.keys,
            target_probability_exact_zero=True,other_query_local_weights_exact=True,
            native_mask_unchanged=True,masked_native_eager_exact=True))


class PrefillRows(AbstractContextManager):
    """Use the prior six-row observer only during the first actual cached Prefill."""
    def __init__(self,model,prompt,mapping):
        self.model,self.query,self.keys=model,prompt['input_ids'].shape[1]-1,key_groups(mapping)
        self.handles=[]; self.row_handles=[]; self.rows=None; self.arrays=None; self.logits=None; self.stats=None

    def __enter__(self):
        def before(module,args,kwargs):
            if self.arrays is None:
                assert self.rows is None
                self.rows=EarlyRows(self.model,self.query,self.keys)
                for k,block in enumerate(model_layers(self.model)[:2]):
                    def question_rows(module,args,out,k=k):
                        a=out[1][0,:,self.keys['question']]
                        self.rows.arrays[f'question_visual_row_{k}']=a[:,:,self.keys['image']].mean(1).detach().cpu().clone().numpy()
                    self.row_handles.append(block.self_attn.register_forward_hook(question_rows))
                self.rows.__enter__()
        def after(module,args,output):
            if self.rows is not None:
                try:
                    assert output.past_key_values is not None
                    self.logits=output.logits[0,-1].detach().float().cpu().clone()
                    self.arrays=self.rows.arrays; self.stats=self.rows.stats
                    assert len(self.stats)==6
                    for k in range(6):
                        assert output.past_key_values.get_seq_length(k)==self.query+1
                        self.arrays[f'bos_key_{k}']=output.past_key_values.key_cache[k][0,:,0].detach().cpu().clone().numpy()
                        self.arrays[f'bos_value_{k}']=output.past_key_values.value_cache[k][0,:,0].detach().cpu().clone().numpy()
                    assert all(np.isfinite(a).all() for a in self.arrays.values())
                finally:
                    self.rows.__exit__(None,None,None); self.rows=None
                    for h in self.row_handles:h.remove()
                    self.row_handles=[]
        self.handles=[self.model.register_forward_pre_hook(before,with_kwargs=True),self.model.register_forward_hook(after)]
        return self

    def __exit__(self,*exc):
        if self.rows is not None: self.rows.__exit__(*exc); self.rows=None
        for h in self.row_handles:h.remove()
        self.row_handles=[]
        for h in self.handles: h.remove()
        self.handles=[]


@torch.inference_mode()
def observed(model,prompt,mapping,condition,generate_answer=True,noop=False,audit=False,collect=False,expected_visual=576):
    with ExitStack() as stack:
        mask=stack.enter_context(SourceMask(model,prompt,mapping,condition,noop,audit,expected_visual))
        early=[0,1] if condition.get('early_off') else []
        observer=stack.enter_context(PrefillRows(model,prompt,mapping))
        if generate_answer:
            generation,logits=generate(model,prompt,dict(layers=early,mode='P-1-only' if early else 'Full'),
                noop=noop,audit=audit,collect=collect)
            assert mask.decode_calls==32*generation['decode_steps']
        else:
            stack.enter_context(JointAttention(model,early,'P-1-only' if early else 'Full',mapping['query']+1,noop,audit))
            output=model(**prompt,past_key_values=None,use_cache=True,return_dict=True,output_attentions=False)
            del output
            generation,logits=None,[]
        assert mask.applied==(1 if 'layer' in condition and not noop else 0)
        if audit and mask.applied: assert len(mask.local_checks)==1
        return generation,observer.logits,observer.arrays,observer.stats,dict(mask_prefill_edits=mask.applied,
            decode_mask_edits=0,decode_layer_calls=mask.decode_calls,local_checks=mask.local_checks),logits


def measurements(arrays,stats,mapping,full):
    masses=[]; changes=[]
    for layer in range(6):
        row=arrays[f'attention_row_{layer}']
        assert np.max(np.abs(row.astype(float).sum(-1)-1))<=ROW_ATOL
        masses.append(dict(layer=layer,**{k+'_mass':float(v.mean()) for k,v in head_mass(row,key_groups(mapping)).items()},
            attention_output_norm=stats[layer]['attention_output_norm']))
        if layer>=2:
            for name in ['hidden_in','query_rope','hidden_out','bos_key','bos_value']:
                change=vector_change(full[f'{name}_{layer}'],arrays[f'{name}_{layer}'])
                changes.append(dict(layer=layer,measurement=name,**change))
                if name.startswith('bos_'): assert change['exact']
    return dict(layer_mass=masses,vector_changes=changes,bos_kv_exact=True)
