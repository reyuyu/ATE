"""R-only pre-softmax source masks. Audit actual returned eager probabilities."""
from contextlib import AbstractContextManager
import torch
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from benchmarks.dataset_registry import digest
from query_role_formal.common import repeated_trigram
from .common import NEW,LAYERS

def source_edges(layout,bos,source,device):
    n=layout['prompt_length'];q=torch.arange(n,device=device)
    image=torch.zeros(n,dtype=torch.bool,device=device);image[layout['I']]=True
    relay=torch.zeros_like(image);relay[layout['R']]=True
    causal=q[None,:]<q[:,None]
    if source=='R-Image-OFF':blocked=relay[:,None]&image[None,:]&causal
    elif source=='R-Text-OFF':blocked=relay[:,None]&(~image)[None,:]&causal&(q[None,:]!=bos)
    elif source=='Full':blocked=torch.zeros((n,n),dtype=torch.bool,device=device)
    else:raise ValueError('Unfrozen source condition')
    assert not blocked[:,bos].any() and not blocked.diagonal().any()
    assert not blocked[n-1].any() and not blocked[~relay].any()
    return blocked

def apply_mask(old,layout,bos,source,noop=False):
    n=layout['prompt_length']
    assert old.shape==(1,1,n,n) and old.is_floating_point()
    assert bool(((old==0)|(old==torch.finfo(old.dtype).min)|torch.isneginf(old)).all())
    legal=old[0,0]==0
    future=torch.ones((n,n),device=old.device,dtype=torch.bool).triu(1)
    assert not legal[future].any()
    blocked=source_edges(layout,bos,source,old.device)
    assert not (blocked&~legal).any()
    changed=old
    if blocked.any() and not noop:
        changed=old.clone();changed[0,0].masked_fill_(blocked,-torch.inf)
        assert torch.equal(changed[0,0][~blocked],old[0,0][~blocked])
    assert (changed[0,0]==0).any(-1).all()
    assert (changed[0,0,layout['R'],bos]==0).all()
    assert (changed[0,0].diagonal()[layout['R']]==0).all()
    return changed,blocked

class SourceMask(AbstractContextManager):
    def __init__(self,model,inputs,condition,noop=False):
        assert condition in ['Full']+NEW
        self.model,self.condition,self.noop=model,condition,noop
        self.layout=roles(inputs,model.config.image_token_index)
        ids=inputs['input_ids'][0];bos=torch.where(ids==model.config.text_config.bos_token_id)[0]
        assert len(bos)==1;self.bos=int(bos[0]);self.handles=[];self.events=[];self.state={}

    def __enter__(self):
        try:
            for k in LAYERS:
                attn=model_layers(self.model)[k].self_attn
                def before(module,args,kw,k=k):
                    h=kw['hidden_states'];cp=kw['cache_position'];cache=kw.get('past_key_value')
                    past=cache.get_seq_length(k) if cache is not None else 0;n=self.layout['prompt_length']
                    assert torch.equal(cp,torch.arange(past,past+h.shape[1],device=h.device))
                    old=kw['attention_mask'];decode=past>=n
                    assert h.shape[:2]==(1,1 if decode else n)
                    if decode:
                        assert old.shape==(1,1,1,past+1) and (old==0).all()
                        changed=old;blocked=None
                    else:
                        assert past==0;changed,blocked=apply_mask(old,self.layout,self.bos,self.condition,self.noop)
                    self.state[k]=(decode,past,blocked)
                    return args,dict(kw,attention_mask=changed,output_attentions=True)
                def after(module,args,out,k=k):
                    decode,past,blocked=self.state[k];w=out[1]
                    assert w is not None and torch.isfinite(w).all() and (w>=0).all()
                    assert w.shape[0]==1 and w.shape[2:]==((1,past+1) if decode else (self.layout['prompt_length'],)*2)
                    idx=[0] if decode else self.layout['R'];rows=w[0,:,idx,:].float()
                    error=float((rows.sum(-1)-1).abs().max());assert error<=ROW_ATOL
                    applied=not decode and self.condition in NEW and not self.noop
                    maximum=float(w[0,:,blocked].max()) if blocked is not None and blocked.any() else 0.
                    if applied:assert maximum==0.
                    row=dict(layer=k,phase='decode' if decode else 'prefill',absolute_start=past,query_length=w.shape[2],key_length=w.shape[3],
                        applied=applied,blocked_connections_per_head=int(blocked.sum()) if applied else 0,
                        proposed_blocked_sha256=digest(blocked.nonzero().cpu().tolist()) if blocked is not None else None,
                        blocked_probability_max=maximum,row_sum_max_error=error,non_target_mask_unchanged=True,BOS_Self_legal=True,
                        P_minus_one_mask_native=True,decode_mask_native=True)
                    if not decode:
                        masses=[];n=self.layout['prompt_length']
                        for j,q in enumerate(idx):
                            history=[x for x in range(q) if x not in set(self.layout['I']) and x!=self.bos]
                            masses.append(dict(query=q,image=float(rows[:,j,self.layout['I']].sum(-1).mean()),
                                history=float(rows[:,j,history].sum(-1).mean()),BOS=float(rows[:,j,self.bos].mean()),Self=float(rows[:,j,q].mean()),
                                blocked_keys=int(blocked[q].sum()) if applied else 0))
                        row['source_masses']=masses
                    self.events.append(row)
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(attn.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[]

@torch.inference_mode()
def generate(model,tokenizer,prompt,condition='Full',noop=False,capture=False,retain_cache=False):
    layout=roles(prompt,model.config.image_token_index);n=layout['prompt_length'];tokens=[];cache=None;trace=[];lengths=[]
    with SourceMask(model,prompt,condition,noop) as ctl:
        while len(tokens)<32 and (not tokens or tokens[-1]!=2):
            t=len(tokens);before=n+t-1 if t else 0
            if cache is not None:assert all(cache.get_seq_length(k)==before for k in range(32))
            out=model(**step_inputs(prompt,tokens),past_key_values=cache,use_cache=True,output_attentions=False,return_dict=True,logits_to_keep=1)
            cache=out.past_key_values;assert all(cache.get_seq_length(k)==n+t for k in range(32))
            assert out.logits.shape[1]==1 and torch.isfinite(out.logits).all()
            current=out.logits[0,-1].detach().float().cpu().clone()
            if not tokens:first=current
            if capture:trace.append(current)
            lengths.append(dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=tokens[-1] if tokens else None))
            tokens.append(int(current.argmax()))
    assert not ctl.handles and len(ctl.events)==4*len(tokens)
    text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
    g=dict(token_ids=tokens,text=text,prediction=gqa_prediction(text),length=len(tokens),eos=tokens[-1]==2,truncated=len(tokens)==32 and tokens[-1]!=2,
        empty=not bool(text),repeated_trigram=repeated_trigram(tokens),decode_steps=len(tokens)-1,own_cache=True,own_prefix=True,
        query_layout=layout,cache_lengths=lengths,source_events=ctl.events,condition=condition)
    g['intervened_query_layer_updates']=4*len(layout['R']) if condition in NEW and not noop else 0
    return g,first,trace,cache if retain_cache else None

@torch.inference_mode()
def text_generate(model,tokenizer,prompt,bypass=False):
    """Question-only diagnostic: all non-BOS, pre-P-1 queries; no VQA scoring."""
    n=prompt['input_ids'].shape[1];targets=[q for q in range(n-1) if int(prompt['input_ids'][0,q])!=model.config.text_config.bos_token_id]
    handles=[];tokens=[];cache=None;calls=[]
    try:
        if bypass:
            for k in LAYERS:
                def zero(module,args,out,k=k):
                    if out.shape[1]==n:
                        changed=out.clone();changed[:,targets]=0
                        keep=[x for x in range(n) if x not in targets];assert torch.equal(changed[:,keep],out[:,keep]);calls.append(dict(layer=k,target_count=len(targets)))
                        return changed
                    return out
                handles.append(model_layers(model)[k].self_attn.o_proj.register_forward_hook(zero))
        while len(tokens)<32 and (not tokens or tokens[-1]!=2):
            out=model.language_model(**step_inputs(prompt,tokens),past_key_values=cache,use_cache=True,return_dict=True,logits_to_keep=1)
            cache=out.past_key_values;logits=out.logits[0,-1].detach().float().cpu()
            assert torch.isfinite(logits).all()
            if not tokens:first=logits.clone()
            tokens.append(int(logits.argmax()))
    finally:
        for h in handles:h.remove()
    if bypass:assert len(calls)==4
    return dict(token_ids=tokens,text=tokenizer.decode(tokens,skip_special_tokens=True).strip(),length=len(tokens),eos=tokens[-1]==2,
        bypass=bypass,scope='All original non-BOS text queries strictly before P-1; P-1/decode native',target_positions=targets,events=calls,VQA_accuracy_valid=False),first
