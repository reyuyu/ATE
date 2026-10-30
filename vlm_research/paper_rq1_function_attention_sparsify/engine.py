from contextlib import AbstractContextManager
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama
from knockout_probe import model_layers
from paper_rq1_token_role_decode.engine import owned_clone, signature, gamma, metrics
from paper_rq1_late_attention_sources.engine import partition, validate_partition, group_head
from .protocol import LAYERS, MODES


def retained(groups, mode):
    names = {'No-Image':['BOS','PROMPT_TEXT','GENERATED_HISTORY'],
             'BOS+History':['BOS','GENERATED_HISTORY'], 'History-only':['GENERATED_HISTORY']}[mode]
    return sorted(i for name in names for i in groups[name])


def sparse_attention(module, query, key, value, mask, scaling, positions):
    """Gather before repeat_kv/QK/AV: excluded keys never enter these operators."""
    index = torch.as_tensor(positions, device=key.device)
    k = llama.repeat_kv(key.index_select(-2,index), module.num_key_value_groups)
    v = llama.repeat_kv(value.index_select(-2,index), module.num_key_value_groups)
    scores = (query @ k.transpose(2,3))*scaling
    if mask is not None: scores = scores + mask[:,:,:,:key.shape[-2]].index_select(-1,index)
    weights = F.softmax(scores,dim=-1,dtype=torch.float32).to(query.dtype)
    out = (weights @ v).transpose(1,2).contiguous()
    return out,weights


def independent_reference(module,q,k,v,mask,scale,positions):
    k=llama.repeat_kv(k,module.num_key_value_groups);v=llama.repeat_kv(v,module.num_key_value_groups)
    logits=(q@k.transpose(2,3))*scale
    if mask is not None:logits=logits+mask[:,:,:,:k.shape[-2]]
    excluded=torch.ones(k.shape[-2],dtype=torch.bool,device=q.device);excluded[positions]=False
    logits=logits.masked_fill(excluded, float('-inf'))
    weights=logits.float().softmax(-1).to(q.dtype)
    return (weights@v).transpose(1,2).contiguous(),weights


class Trace(AbstractContextManager):
    def __init__(self,model,groups=None,algorithm=None,mode=None,layer=31,noop=False,hidden=False,audit=False):
        if algorithm not in [None,'A1','A2','zero']:raise ValueError('Unknown algorithm')
        if algorithm is not None and layer not in LAYERS:raise ValueError('Unfrozen layer')
        if algorithm in ['A1','A2'] and mode not in MODES:raise ValueError('Unfrozen source retention')
        self.model,self.groups,self.algorithm,self.mode,self.layer=model,groups,algorithm,mode,layer
        self.noop,self.capture,self.audit=noop,hidden,audit
        self.handles=[];self.original=llama.eager_attention_forward;self.edits=[];self.checks=[];self.head=None;self.hidden=None

    def __enter__(self):
        try:
            layers=model_layers(self.model)
            if self.capture:
                def capture(module,args,kwargs):
                    x=args[0] if args else kwargs['hidden_states']
                    self.hidden=x[0,-1].detach().cpu().clone()
                self.handles.append(layers[31].register_forward_pre_hook(capture,with_kwargs=True))
            if self.algorithm is not None:
                self.handles.append(layers[self.layer].self_attn.o_proj.register_forward_hook(self.output))
                if self.algorithm in ['A1','A2']:
                    def attention(module,q,k,v,mask,scaling,**kwargs):
                        if module.layer_idx!=self.layer:return self.original(module,q,k,v,mask,scaling,**kwargs)
                        if q.shape[:1]!=(1,) or q.shape[-2]!=1:raise AssertionError('Only one real Decode query')
                        validate_partition(self.groups,k.shape[-2])
                        keep=retained(self.groups,self.mode)
                        if self.algorithm=='A1' or self.noop:
                            out,w=self.original(module,q,k,v,mask,scaling,**kwargs)
                            removed=sorted(set(range(k.shape[-2]))-set(keep))
                            self.head=group_head(w[0,:,0],llama.repeat_kv(v,module.num_key_value_groups)[0],removed)
                            if self.audit:
                                self.native=(w[0,:,0],llama.repeat_kv(v,module.num_key_value_groups)[0],out.reshape(1,1,-1))
                        else:
                            out,w=sparse_attention(module,q,k,v,mask,scaling,keep)
                            if self.audit:
                                reference,rw=independent_reference(module,q,k,v,mask,scaling,keep)
                                index=torch.as_tensor(keep,device=q.device)
                                delta=(w.float()-rw.index_select(-1,index).float()).abs()
                                # The unchanged FP16 rounding model, not a data-fitted tolerance.
                                wb=2.**-11*(w.float().abs()+rw.index_select(-1,index).float().abs())+2.**-24
                                if not torch.all(delta<=wb):raise AssertionError('Sparse/reference softmax round bound')
                                vv=llama.repeat_kv(v,module.num_key_value_groups).index_select(-2,index).float()
                                magnitude=w.float().abs()@vv.abs()
                                bound=(delta@vv.abs())+(gamma(len(keep))+2.**-10)*magnitude+2.**-24
                                err=(out.float()-reference.float()).abs().transpose(1,2)
                                if not torch.all(err<=bound):raise AssertionError('Sparse/reference AV bound')
                                self.checks.append(dict(reference_max_error=float(err.max()),reference_bound_ratio=float((err/bound.clamp_min(2.**-24)).max()),
                                    full_keys=k.shape[-2],retained_keys=len(keep),qk_columns=len(keep),av_rows=len(keep)))
                        return out,w
                    llama.eager_attention_forward=attention
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def output(self,module,args,out):
        if out.shape[:2]!=(1,1):raise AssertionError('Intervention applied outside real Decode')
        self.edits.append(dict(layer=self.layer,query=0,algorithm=self.algorithm,mode=self.mode,noop=self.noop))
        if self.noop:return out
        if self.algorithm=='zero':return torch.zeros_like(out)
        if self.algorithm=='A1':
            if self.audit:
                from paper_rq1_late_attention_sources.engine import contributions
                row,v,_=self.native
                _,_,check=contributions(module,row,v,self.groups,args[0],out,torch.ones_like(out).float(),self.layer)
                self.checks.append(check)
            return (out.float()-F.linear(self.head,module.weight.float(),bias=None)).to(out.dtype)
        return out

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]
        llama.eager_attention_forward=self.original


def step(model,inputs,cache,**kwargs):
    before=signature(cache);clone=owned_clone(cache)
    with torch.inference_mode(),Trace(model,**kwargs) as trace:
        output=model(**inputs,past_key_values=clone,use_cache=True,output_attentions=False,return_dict=True)
    logits=output.logits[0,-1].detach().float().cpu()
    if not torch.isfinite(logits).all() or signature(cache)!=before:raise AssertionError('Invalid output/parent mutation')
    if kwargs.get('algorithm') is not None and len(trace.edits)!=1:raise AssertionError('Wrong number of edits')
    return logits,output.past_key_values,trace
