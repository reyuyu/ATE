"""Simultaneous multi-layer hooks and genuinely condition-owned cached generation."""
from contextlib import AbstractContextManager
import torch
from knockout_probe import model_layers
from paper_rq1.engine import step_inputs
from paper_rq1_decode_layerwise_quality.engine import is_decode


class JointAttention(AbstractContextManager):
    def __init__(self,model,layers,mode,prompt_length,noop=False,audit=False):
        if len(set(layers))!=len(layers) or any(type(k) is not int or not 0<=k<32 for k in layers):
            raise ValueError('Invalid layer set')
        if mode not in ('Full','P-1-only','Prefill-all-only','Decode-only','Combined'):
            raise ValueError('Invalid temporal scope')
        if prompt_length<1 or (mode=='Full' and layers):raise ValueError('Invalid Full/prompt')
        self.model,self.layers,self.mode,self.n=model,list(layers),mode,prompt_length
        self.noop,self.audit=noop,audit
        self.handles=[];self.phases={};self.events=[];self.edits=0

    def __enter__(self):
        try:
            blocks=model_layers(self.model)
            if len(blocks)!=32:raise AssertionError('Require exactly32 frozen Decoder layers')
            for k in self.layers:
                def before(module,args,kwargs,k=k):
                    hidden=args[0] if args else kwargs['hidden_states']
                    cache=kwargs.get('past_key_value')
                    past=cache.get_seq_length(k) if cache is not None else 0
                    self.phases[k]=is_decode(kwargs.get('cache_position'),self.n,past,hidden.shape[1])
                def output(module,args,out,k=k):
                    decode=self.phases[k]
                    if out.shape[0]!=1 or out.shape[1]!=(1 if decode else self.n):
                        raise AssertionError('Wrong actual Query layout')
                    active=(self.mode=='Combined' or self.mode=='Decode-only' and decode
                        or self.mode in ('P-1-only','Prefill-all-only') and not decode)
                    changed=out
                    query=self.n-1 if self.mode=='P-1-only' and not decode else None
                    if active:
                        self.edits+=1
                        if not self.noop:
                            if query is None:changed=torch.zeros_like(out)
                            else:changed=out.clone();changed[0,query]=0
                    if self.audit:
                        if not torch.isfinite(changed).all():raise AssertionError('Nonfinite hook output')
                        if active and not self.noop:
                            target=changed if query is None else changed[:,query]
                            if torch.count_nonzero(target):raise AssertionError('Target update not exactly zero')
                            if query is not None and not torch.equal(changed[:,:query],out[:,:query]):
                                raise AssertionError('Non-target Query modified')
                        elif not torch.equal(changed,out):raise AssertionError('Inactive/noop modified output')
                        self.events.append(dict(layer=k,decode=decode,active=bool(active),queries=out.shape[1],
                            target='original_P-1' if query is not None else 'all_current_queries',noop=self.noop))
                    return changed
                self.handles.append(blocks[k].self_attn.register_forward_pre_hook(before,with_kwargs=True))
                self.handles.append(blocks[k].self_attn.o_proj.register_forward_hook(output))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]


@torch.inference_mode()
def generate(model,prompt,condition,limit=32,noop=False,collect=False,audit=False):
    if limit<1:raise ValueError('Need positive generation limit')
    n=prompt['input_ids'].shape[1]
    if not bool(prompt['attention_mask'].all()):raise ValueError('Unpadded single-sample prompt required')
    tokens=[];history=None;logits_trace=[];initial_cache_sha=None
    with JointAttention(model,condition['layers'],condition['mode'],n,noop,audit) as trace:
        while len(tokens)<limit and (not tokens or tokens[-1]!=2):
            if history is not None and any(history.get_seq_length(k)!=n+len(tokens)-1 for k in range(32)):
                raise AssertionError('Condition-owned history length mismatch')
            output=model(**step_inputs(prompt,tokens),past_key_values=history,use_cache=True,
                output_attentions=False,return_dict=True)
            history=output.past_key_values;logits=output.logits[0,-1]
            if not torch.isfinite(logits).all():raise AssertionError('Nonfinite effective logits')
            if collect:logits_trace.append(logits.detach().float().cpu().clone())
            if audit and not tokens:
                from paper_rq1.engine import cache_fingerprint
                initial_cache_sha=cache_fingerprint(history)
            tokens.append(int(logits.argmax()))
    decode_steps=len(tokens)-1
    per_layer=(1 if condition['mode'] in ('P-1-only','Prefill-all-only') else decode_steps
        if condition['mode']=='Decode-only' else len(tokens) if condition['mode']=='Combined' else 0)
    if trace.edits!=len(condition['layers'])*per_layer:raise AssertionError('Wrong simultaneous edit count')
    return dict(token_ids=tokens,length=len(tokens),eos=tokens[-1]==2,
        truncated=len(tokens)==limit and tokens[-1]!=2,decode_steps=decode_steps,edits=trace.edits,
        own_prefill=True,own_prefix=True,initial_cache_sha256=initial_cache_sha,events=trace.events),logits_trace
