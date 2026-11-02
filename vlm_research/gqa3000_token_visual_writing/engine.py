"""Native dense attention with receiver masks; both metrics use independent caches."""
from contextlib import AbstractContextManager
import numpy as np
import torch
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from query_role_formal.common import repeated_trigram
from semantic_visual_writers.engine import blocked_edges as frozen_edges,apply_mask as frozen_apply
from .common import assert_no_stop,gold_losses

def cpu_mask(layout,writers,past=0):
    q=1 if past else layout['prompt_length'];n=past+1 if past else q
    native=torch.zeros(1,1,q,n,dtype=torch.float16)
    if not past:native.masked_fill_(torch.ones(q,n,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
    changed,blocked=frozen_apply(native,layout,writers,past)
    return native,changed,blocked

class ReceiverMask(AbstractContextManager):
    """Keep validation reductions on device until the condition completes.

    This changes audit synchronization only, not model kernels or arithmetic.
    Every layer's original native mask is compared to the first native mask of
    that same forward; every forbidden probability is checked without sampling.
    """
    def __init__(self,model,inputs,writers,noop=False):
        self.model=model;self.layout=roles(inputs,model.config.image_token_index);self.writers=list(writers);self.noop=noop
        assert self.writers==sorted(set(self.writers)) and set(writers)<=set(self.layout['R'])
        assert int(inputs['input_ids'][0,0])==model.config.text_config.bos_token_id==1
        self.handles=[];self.metrics=[];self.meta=[];self.state={};self.cache={};self.forward_masks={}
    def __enter__(self):
        try:
            for k,layer in enumerate(model_layers(self.model)):
                def before(module,args,kw,k=k):
                    h=kw['hidden_states'];cp=kw['cache_position'];kv=kw.get('past_key_value');past=kv.get_seq_length(k) if kv is not None else 0
                    n=self.layout['prompt_length'];assert h.shape[:2]==(1,1 if past else n) and (past==0 or past>=n)
                    old=kw['attention_mask'];assert old.shape==(1,1,h.shape[1],past+h.shape[1])
                    if past not in self.cache:
                        # Exact frozen primitive on CPU, same Q->K edge set.
                        blocked=frozen_edges(self.layout,self.writers,past,'cpu')
                        self.cache[past]=(blocked.to(old.device),int(blocked.sum()))
                    blocked,count=self.cache[past]
                    if k==0:
                        assert_no_stop();native=old.detach().clone();changed=old if self.noop else old.clone().masked_fill(blocked[None,None],-torch.inf)
                        self.forward_masks[past]=(native,changed)
                    native,changed=self.forward_masks[past]
                    native_equal=(old==native).all();non_target=(changed[0,0][~blocked]==old[0,0][~blocked]).all()
                    legal=(old[0,0][blocked]==0).all()
                    bos=(changed[0,0,:,0]==0).all()
                    self_legal=(changed[0,0,0,past]==0) if past else (changed[0,0].diagonal()==0).all()
                    position=(cp==torch.arange(past,past+h.shape[1],device=cp.device)).all()
                    self.state[k]=(past,blocked,count,torch.stack([native_equal,non_target,legal,bos&self_legal,position]).float())
                    return args,dict(kw,attention_mask=changed,output_attentions=True)
                def after(module,args,out,k=k):
                    past,blocked,count,checks=self.state[k];w=out[1];assert w is not None and w.shape[2:]==blocked.shape
                    idx=[0] if past else self.layout['R']+[self.layout['prompt_length']-1];rows=w[0,:,idx,:].float()
                    maximum=w[0,:,blocked].float().max();error=(rows.sum(-1)-1).abs().max();finite=torch.isfinite(rows).all().float();minimum=rows.min()
                    self.metrics.append(torch.cat([torch.stack([maximum,error,finite,minimum]),checks]));self.meta.append((k,past,w.shape[2],w.shape[3],count))
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*_):
        for handle in self.handles:handle.remove()
        self.handles=[]
    def finalize(self,steps):
        assert not self.handles and len(self.metrics)==32*steps
        values=torch.stack(self.metrics).detach().float().cpu().numpy();assert np.isfinite(values).all()
        if not self.noop:assert (values[:,0]==0).all(),'A forbidden actual probability is nonzero'
        assert (values[:,1]<=ROW_ATOL).all() and (values[:,2]==1).all() and (values[:,3]>=0).all() and (values[:,4:]==1).all()
        for t in range(steps):
            block=self.meta[32*t:32*t+32];assert [v[0] for v in block]==list(range(32))
            assert len({v[1:] for v in block})==1
        return dict(passed=True,steps=steps,calls=32*steps,noop=self.noop,all_32_layers=True,
            forbidden_max_by_layer=[float(values[k::32,0].max()) for k in range(32)],
            row_sum_error_by_layer=[float(values[k::32,1].max()) for k in range(32)],
            all_non_target_masks_exact=True,all_native_masks_equal=True,BOS_self_legal=True,absolute_positions_exact=True,
            prefill_blocked_per_head=0 if self.noop else self.meta[0][4],decode_blocked_per_head=0 if self.noop else 576),values

def forward(model,inputs,prefix,cache,t):
    n=inputs['input_ids'].shape[1];before=n+t-1 if t else 0
    if cache is not None:assert all(cache.get_seq_length(k)==before for k in range(32))
    result=model(**step_inputs(inputs,prefix),past_key_values=cache,use_cache=True,output_attentions=False,return_dict=True,logits_to_keep=1)
    cache=result.past_key_values;assert all(cache.get_seq_length(k)==n+t for k in range(32))
    assert result.logits.shape[:2]==(1,1)
    v=result.logits[0,-1].detach().cpu().numpy().copy();assert np.isfinite(v).all()
    return v,cache,dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=prefix[-1] if prefix else None)

@torch.inference_mode()
def generate(model,tokenizer,inputs,writers,noop=False,capture=False,retain_cache=False):
    ids=[];trace=[];cache=None;lengths=[]
    with ReceiverMask(model,inputs,writers,noop) as mask:
        while len(ids)<32 and (not ids or ids[-1]!=2):
            assert_no_stop();v,cache,step=forward(model,inputs,ids,cache,len(ids));lengths.append(step)
            if not ids:first=v.copy()
            if capture:trace.append(v)
            ids.append(int(v.argmax()))
    audit,values=mask.finalize(len(ids));text=tokenizer.decode(ids,skip_special_tokens=True).strip()
    g=dict(token_ids=ids,text=text,prediction=gqa_prediction(text),length=len(ids),eos=ids[-1]==2,empty=not bool(text),
        truncated=len(ids)==32 and ids[-1]!=2,repeated_trigram=repeated_trigram(ids),cache_lengths=lengths,
        own_cache=True,own_prefix=True,writer_positions=list(writers),query_layout=mask.layout,logits_to_keep=1,audit=audit)
    return g,first,np.stack(trace) if trace else None,values,cache if retain_cache else None

@torch.inference_mode()
def teacher_force(model,inputs,writers,gold_ids,noop=False,retain_cache=False):
    cache=None;logits=[];lengths=[]
    with ReceiverMask(model,inputs,writers,noop) as mask:
        for t,tid in enumerate(gold_ids):
            assert_no_stop();v,cache,step=forward(model,inputs,gold_ids[:t],cache,t);logits.append(v);step['gold_target_id']=tid;lengths.append(step)
    audit,values=mask.finalize(len(gold_ids));logits=np.stack(logits);losses,nll=gold_losses(logits,gold_ids)
    result=dict(gold_token_ids=list(gold_ids),per_token_nll=losses,nll=nll,cache_lengths=lengths,
        own_cache=True,own_gold_prefix=True,logits_to_keep=1,eos_in_primary_nll=False,audit=audit)
    return result,logits,values,cache if retain_cache else None

@torch.inference_mode()
def native_teacher(model,inputs,gold_ids):
    cache=None;values=[]
    for t in range(len(gold_ids)):
        v,cache,_=forward(model,inputs,gold_ids[:t],cache,t);values.append(v)
    return np.stack(values)
