"""Historical writing-mask semantics, with single-layer drops and isolated Gold TF."""
from contextlib import AbstractContextManager

import torch
from transformers.cache_utils import DynamicCache
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers, ROW_ATOL
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from query_role_attention.engine import roles
from query_role_formal.common import repeated_trigram
from .common import CONDITIONS, writer_layers, digest


def apply_mask(old,layout,layer,condition,past,noop=False):
    n=layout['prompt_length'];length=1 if past else n
    assert old.shape==(1,1,length,past+length) and old.is_floating_point()
    assert past==0 or past>=n
    blocked=torch.zeros((length,past+length),dtype=torch.bool,device=old.device)
    if condition!='Native':
        assert condition in CONDITIONS
        blocked[-1,layout['I']]=True
        if not past and layer not in writer_layers(condition):
            q=torch.tensor(layout['R'],device=old.device);k=torch.tensor(layout['I'],device=old.device)
            blocked[q[:,None],k[None,:]]=True
    legal=old[0,0]==0
    assert not (blocked&~legal).any()
    if not past:
        assert not legal.triu(1).any()
        keep=[q for q in range(n) if q not in layout['R'] and q!=n-1]
        assert not blocked[keep].any()
    else:assert legal.all()
    changed=old
    if blocked.any() and not noop:
        changed=old.clone();changed[0,0].masked_fill_(blocked,-torch.inf)
        assert torch.equal(changed[0,0][~blocked],old[0,0][~blocked])
    assert (changed[0,0]==0).any(-1).all()
    return changed,blocked


class WritingMask(AbstractContextManager):
    def __init__(self,model,prompt,condition,noop=False):
        assert condition in CONDITIONS+['Native']
        self.model,self.condition,self.noop=model,condition,noop
        self.layout=roles(prompt,model.config.image_token_index)
        ids=prompt['input_ids'][0];bos=torch.where(ids==model.config.text_config.bos_token_id)[0]
        assert len(bos)==1 and int(bos[0])<self.layout['I'][0]
        self.bos=int(bos[0]);self.handles=[];self.state={};self.events=[];self.path='free'

    def __enter__(self):
        try:
            for layer,block in enumerate(model_layers(self.model)):
                def before(module,args,kw,layer=layer):
                    h=kw['hidden_states'];cp=kw['cache_position'];cache=kw.get('past_key_value')
                    past=cache.get_seq_length(layer) if cache is not None else 0
                    assert h.shape[:2]==(1,1 if past else self.layout['prompt_length'])
                    assert torch.equal(cp,torch.arange(past,past+h.shape[1],device=h.device))
                    changed,blocked=apply_mask(kw['attention_mask'],self.layout,layer,self.condition,past,self.noop)
                    assert (changed[0,0,:,self.bos]==0).all()
                    assert (changed[0,0,0,past]==0) if past else (changed[0,0].diagonal()==0).all()
                    self.state[layer]=(past,blocked)
                    return args,dict(kw,attention_mask=changed,output_attentions=True)
                def after(module,args,out,layer=layer):
                    past,blocked=self.state[layer];w=out[1]
                    assert w is not None and w.shape[2:]==blocked.shape
                    idx=[0] if past else self.layout['R']+[self.layout['prompt_length']-1]
                    rows=w[0,:,idx,:].float()
                    assert torch.isfinite(rows).all() and (rows>=0).all()
                    error=float((rows.sum(-1)-1).abs().max());assert error<=ROW_ATOL
                    maximum=float(w[0,:,blocked].max()) if blocked.any() else 0.
                    active=self.condition!='Native' and not self.noop
                    if active:assert maximum==0.
                    self.events.append(dict(layer=layer,path=self.path,past=past,query_length=w.shape[2],key_length=w.shape[3],
                        blocked_keys=int(blocked.sum()) if active else 0,blocked_probability_max=maximum,
                        row_sum_max_error=error,non_target_mask_exact=True,Image_Query_mask_native=True,BOS_Self_legal=True))
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(block.self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(block.self_attn.register_forward_hook(after))
            return self
        except BaseException:self.__exit__(None,None,None);raise

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[]


def clone_prefix(cache):
    old=cache.to_legacy_cache()
    copied=tuple(tuple(t.clone() for t in pair) for pair in old)
    assert all(a.data_ptr()!=b.data_ptr() and torch.equal(a,b) for x,y in zip(old,copied) for a,b in zip(x,y))
    return DynamicCache.from_legacy_cache(copied)


def gold_logp(logits,token):
    assert logits.ndim==1 and torch.isfinite(logits).all()
    return float(logits.detach().float().cpu().log_softmax(-1)[token])


def compact_audit(events,layout,condition,steps):
    assert len(events)==32*steps
    expected=[]
    for t in range(steps):
        rows=events[t*32:(t+1)*32];assert [e['layer'] for e in rows]==list(range(32))
        for e in rows:
            masks=0 if condition=='Native' else 576*(1+(len(layout['R']) if e['past']==0 and e['layer'] not in writer_layers(condition) else 0))
            assert e['blocked_keys'] in (0,masks)
        expected.append(dict(past=rows[0]['past'],query_length=rows[0]['query_length'],key_length=rows[0]['key_length'],
            blocked_connections_per_layer=[r['blocked_keys'] for r in rows]))
    return dict(steps=steps,layer_events=len(events),actual_positions=expected,
        blocked_probability_max=max((e['blocked_probability_max'] for e in events),default=0.),
        row_sum_max_error=max((e['row_sum_max_error'] for e in events),default=0.),
        non_target_mask_exact=True,Image_Query_mask_native=True,BOS_Self_legal=True)


@torch.inference_mode()
def evaluate(model,tokenizer,prompt,condition,gold,noop=False,capture=False):
    n=prompt['input_ids'].shape[1];tokens=[];cache=None;trace=[];tf_cache=None;footprint=hook_footprint(model)
    assert gold and 1 not in gold and 2 not in gold
    with WritingMask(model,prompt,condition,noop) as control:
        while len(tokens)<32 and (not tokens or tokens[-1]!=2):
            t=len(tokens);past=n+t-1 if t else 0
            if cache is not None:assert all(cache.get_seq_length(k)==past for k in range(32))
            out=model(**step_inputs(prompt,tokens),past_key_values=cache,use_cache=True,return_dict=True,
                output_attentions=False,logits_to_keep=1)
            cache=out.past_key_values
            assert out.logits.shape[1]==1 and torch.isfinite(out.logits).all()
            assert all(cache.get_seq_length(k)==n+t for k in range(32))
            vector=out.logits[0,-1].detach().cpu().clone()
            if not tokens:
                first=vector;gold_scores=[gold_logp(first,gold[0])]
                if len(gold)>1:tf_cache=clone_prefix(cache)
            if capture:trace.append(vector)
            tokens.append(int(vector.argmax()))
        free_events=list(control.events)
        free_cache_lengths=[cache.get_seq_length(k) for k in range(32)]
        if tf_cache is not None:
            assert all(a.data_ptr()!=b.data_ptr() for x,y in zip(cache.to_legacy_cache(),tf_cache.to_legacy_cache()) for a,b in zip(x,y))
            control.path='gold'
            for t in range(1,len(gold)):
                assert all(tf_cache.get_seq_length(k)==n+t-1 for k in range(32))
                out=model(**step_inputs(prompt,gold[:t]),past_key_values=tf_cache,use_cache=True,
                    return_dict=True,output_attentions=False,logits_to_keep=1)
                tf_cache=out.past_key_values
                assert all(tf_cache.get_seq_length(k)==n+t for k in range(32))
                gold_scores.append(gold_logp(out.logits[0,-1],gold[t]))
            assert free_cache_lengths==[cache.get_seq_length(k) for k in range(32)]
        text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
        value=dict(text=text,prediction=gqa_prediction(text),token_ids=tokens,eos=tokens[-1]==2,
            truncated=len(tokens)==32 and tokens[-1]!=2,empty=not bool(text),repeated_trigram=repeated_trigram(tokens),
            first_top1=int(first.argmax()),first_logits_sha256=digest(first.tolist()),gold_token_ids=gold,
            gold_token_logp=gold_scores,gold_nll=-sum(gold_scores)/len(gold_scores),gold_EOS_excluded=True,
            writer_layers=writer_layers(condition) if condition!='Native' else list(range(32)),
            P1_Decode_image_off=condition!='Native' and not noop,independent_free_gold_caches=True,
            free_audit=compact_audit(free_events,control.layout,condition,len(tokens)),
            gold_audit=compact_audit(control.events[len(free_events):],control.layout,condition,len(gold)-1),
            layout=control.layout)
    assert not control.handles and hook_footprint(model)==footprint
    return value,first,trace


@torch.inference_mode()
def teacher_reference(model,prompt,condition,gold):
    cache=None;values=[]
    with WritingMask(model,prompt,condition):
        for t,target in enumerate(gold):
            out=model(**step_inputs(prompt,gold[:t]),past_key_values=cache,use_cache=True,
                return_dict=True,output_attentions=False,logits_to_keep=1)
            cache=out.past_key_values;values.append(gold_logp(out.logits[0,-1],target))
    return values
