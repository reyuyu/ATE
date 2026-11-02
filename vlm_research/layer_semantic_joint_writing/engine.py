"""Compose frozen semantic masks with frozen layer permissions; inspect real probabilities."""
from contextlib import AbstractContextManager
import torch
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from query_role_formal.common import repeated_trigram
from semantic_visual_writers.engine import apply_mask as semantic_apply,blocked_edges as semantic_edges
from .common import CONDITIONS,KEEP,selected,writers,digest

def blocked_edges(layout,layer,condition,spec,past,device):
    return semantic_edges(layout,writers(condition,layer,layout,spec),past,device)
def apply_mask(old,layout,layer,condition,spec,past,noop=False):
    return semantic_apply(old,layout,writers(condition,layer,layout,spec),past,noop or condition=='Native')

class JointMask(AbstractContextManager):
    def __init__(self,model,inputs,condition,spec,noop=False):
        assert condition in CONDITIONS+['Native']
        self.model,self.condition,self.spec,self.noop=model,condition,spec,noop
        self.layout=roles(inputs,model.config.image_token_index)
        self.bos=torch.where(inputs['input_ids'][0]==model.config.text_config.bos_token_id)[0].tolist()
        assert len(self.bos)==1 and self.bos[0]<self.layout['I'][0]
        assert self.layout['R']==spec['relay_positions']
        self.handles=[];self.events=[];self.state={}
    def __enter__(self):
        try:
            for k,layer in enumerate(model_layers(self.model)):
                def before(module,args,kw,k=k):
                    h=kw['hidden_states'];cp=kw['cache_position'];cache=kw.get('past_key_value');past=cache.get_seq_length(k) if cache is not None else 0
                    n=self.layout['prompt_length'];assert torch.equal(cp,torch.arange(past,past+h.shape[1],device=h.device))
                    assert h.shape[:2]==(1,1 if past else n)
                    changed,blocked=apply_mask(kw['attention_mask'],self.layout,k,self.condition,self.spec,past,self.noop)
                    assert (changed[0,0,:,self.bos[0]]==0).all()
                    assert (changed[0,0,0,past]==0) if past else (changed[0,0].diagonal()==0).all()
                    self.state[k]=(past,blocked)
                    return args,dict(kw,attention_mask=changed,output_attentions=True)
                def after(module,args,out,k=k):
                    past,blocked=self.state[k];w=out[1];n=self.layout['prompt_length']
                    assert w is not None and w.shape[2:]==blocked.shape
                    idx=[0] if past else self.layout['R']+[n-1]
                    rows=w[0,:,idx,:].float();assert torch.isfinite(rows).all() and (rows>=0).all()
                    error=float((rows.sum(-1)-1).abs().max());assert error<=ROW_ATOL
                    maximum=float(w[0,:,blocked].max());applied=self.condition!='Native' and not self.noop
                    if applied:assert maximum==0.
                    permitted=writers(self.condition,k,self.layout,self.spec)
                    self.events.append(dict(layer=k,phase='decode' if past else 'prefill',absolute_start=past,query_length=w.shape[2],key_length=w.shape[3],
                        permitted_writer_queries=len(permitted),answer_image_masked=applied,
                        blocked_connections_per_head=int(blocked.sum()) if applied else 0,blocked_probability_max=maximum,row_sum_max_error=error,
                        non_target_mask_unchanged=True,BOS_Self_legal=True,
                        answer_image_mass_mean=float(rows[:,-1,self.layout['I']].sum(-1).mean())))
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[]

@torch.inference_mode()
def generate(model,tokenizer,prompt,condition,spec,noop=False,capture=False,retain_cache=False):
    n=prompt['input_ids'].shape[1];tokens=[];cache=None;trace=[];lengths=[]
    with JointMask(model,prompt,condition,spec,noop) as ctl:
        while len(tokens)<32 and (not tokens or tokens[-1]!=2):
            t=len(tokens);before=n+t-1 if t else 0
            if cache is not None:assert all(cache.get_seq_length(k)==before for k in range(32))
            out=model(**step_inputs(prompt,tokens),past_key_values=cache,use_cache=True,output_attentions=False,return_dict=True,logits_to_keep=1)
            cache=out.past_key_values;assert all(cache.get_seq_length(k)==n+t for k in range(32))
            assert out.logits.shape[:2]==(1,1) and torch.isfinite(out.logits).all()
            v=out.logits[0,-1].detach().float().cpu().clone()
            if not tokens:first=v
            if capture:trace.append(v)
            lengths.append(dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=tokens[-1] if tokens else None))
            tokens.append(int(v.argmax()))
    assert len(ctl.events)==32*len(tokens) and not ctl.handles
    text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
    g=dict(condition=condition,token_ids=tokens,text=text,prediction=gqa_prediction(text),length=len(tokens),eos=tokens[-1]==2,
        truncated=len(tokens)==32 and tokens[-1]!=2,empty=not bool(text),repeated_trigram=repeated_trigram(tokens),
        own_cache=True,own_prefix=True,logits_to_keep=1,query_layout=ctl.layout,cache_lengths=lengths,source_events=ctl.events,
        writer_positions=selected(condition,ctl.layout,spec),writer_layers=KEEP[condition] if condition!='Native' else list(range(32)),
        selection_binding=spec['selection_binding'],answer_image_blocked=condition!='Native' and not noop,
        noop=noop,all_logits_finite=True)
    return g,first,trace,cache if retain_cache else None
