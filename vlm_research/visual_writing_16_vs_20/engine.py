"""Compose audited All/No-Writer masks by layer; generation stays native."""
import torch
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers,ROW_ATOL
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from query_role_formal.common import repeated_trigram
from visual_writing_necessity.engine import WritingMask,apply_mask as historical_mask
from .common import writers,digest


def apply_mask(old,layout,layer,condition,past,noop=False):
    alias='Native' if condition=='Native' else ('All-Writer' if layer in writers(condition) else 'No-Writer')
    return historical_mask(old,layout,layer,alias,past,noop)


class KeepMask(WritingMask):
    def __init__(self,model,prompt,condition,noop=False):
        super().__init__(model,prompt,'Native',noop)
        writers(condition);self.condition=condition

    def __enter__(self):
        try:
            for layer,block in enumerate(model_layers(self.model)):
                def before(module,args,kw,layer=layer):
                    h=kw['hidden_states'];cache=kw.get('past_key_value');past=cache.get_seq_length(layer) if cache is not None else 0
                    assert h.shape[:2]==(1,1 if past else self.layout['prompt_length'])
                    assert torch.equal(kw['cache_position'],torch.arange(past,past+h.shape[1],device=h.device))
                    old=kw['attention_mask'];changed,b=apply_mask(old,self.layout,layer,self.condition,past,self.noop)
                    assert (changed[0,0,:,self.bos]==0).all()
                    assert (changed[0,0,0,past]==0) if past else (changed[0,0].diagonal()==0).all()
                    self.state[layer]=(past,b)
                    return args,dict(kw,attention_mask=changed,output_attentions=True)
                def after(module,args,out,layer=layer):
                    past,b=self.state[layer];w=out[1];assert w is not None and w.shape[2:]==b.shape
                    idx=[0] if past else self.layout['R']+[self.layout['prompt_length']-1]
                    rows=w[0,:,idx,:].float();assert torch.isfinite(rows).all() and (rows>=0).all()
                    error=float((rows.sum(-1)-1).abs().max());assert error<=ROW_ATOL
                    active=self.condition!='Native' and not self.noop
                    maximum=float(w[0,:,b].max()) if active else 0.
                    if active:assert maximum==0.
                    self.events.append(dict(layer=layer,past=past,query_length=w.shape[2],key_length=w.shape[3],
                        blocked_connections=int(b.sum()) if active else 0,blocked_probability_max=maximum,row_sum_max_error=error))
                    return (out[0],None)+tuple(out[2:])
                self.handles.append(block.self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(block.self_attn.register_forward_hook(after))
            return self
        except BaseException:self.__exit__(None,None,None);raise


@torch.inference_mode()
def generate(model,tokenizer,prompt,condition,noop=False,capture=False):
    n=prompt['input_ids'].shape[1];tokens=[];cache=None;trace=[];footprint=hook_footprint(model)
    with KeepMask(model,prompt,condition,noop) as control:
        while len(tokens)<32 and (not tokens or tokens[-1]!=2):
            t=len(tokens);past=n+t-1 if t else 0
            if cache is not None:assert all(cache.get_seq_length(k)==past for k in range(32))
            out=model(**step_inputs(prompt,tokens),past_key_values=cache,use_cache=True,
                return_dict=True,output_attentions=False,logits_to_keep=1)
            cache=out.past_key_values;assert out.logits.shape[1]==1 and torch.isfinite(out.logits).all()
            assert all(cache.get_seq_length(k)==n+t for k in range(32))
            vector=out.logits[0,-1].detach().cpu().clone()
            if not tokens:first=vector
            if capture:trace.append(vector)
            tokens.append(int(vector.argmax()))
        assert len(control.events)==32*len(tokens)
        active=condition!='Native' and not noop
        positions=[]
        for t in range(len(tokens)):
            events=control.events[t*32:(t+1)*32];assert [e['layer'] for e in events]==list(range(32))
            expected=[576*(1+(len(control.layout['R']) if not t and layer not in writers(condition) else 0)) if active else 0 for layer in range(32)]
            assert [e['blocked_connections'] for e in events]==expected
            assert all(e['past']==(n+t-1 if t else 0) and e['key_length']==n+t for e in events)
            positions.append(dict(past=events[0]['past'],query_length=events[0]['query_length'],key_length=n+t,
                blocked_connections_per_layer=expected))
        text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
        value=dict(text=text,prediction=gqa_prediction(text),token_ids=tokens,eos=tokens[-1]==2,
            truncated=len(tokens)==32 and tokens[-1]!=2,empty=not bool(text),repeated_trigram=repeated_trigram(tokens),
            first_top1=int(first.argmax()),first_logits_sha256=digest(first.tolist()),writer_layers=writers(condition),
            P1_Decode_image_off=active,independent_cache=True,independent_prefix=True,layout=control.layout,
            audit=dict(steps=len(tokens),layer_events=len(control.events),actual_positions=positions,
                blocked_probability_max=max(e['blocked_probability_max'] for e in control.events),
                row_sum_max_error=max(e['row_sum_max_error'] for e in control.events),
                non_target_mask_exact=True,Image_Query_mask_native=True,BOS_Self_legal=True))
    assert not control.handles and hook_footprint(model)==footprint
    return value,first,trace
