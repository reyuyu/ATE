"""Single-layer output ablation, independent caches, genuine own-prefix generation."""
from contextlib import AbstractContextManager
import torch
from knockout_probe import model_layers
from paper_rq1.engine import clone_cache, step_inputs


def is_decode(cache_position,prompt_length,past_length,query_length):
    if cache_position is None or cache_position.numel()!=query_length:
        raise AssertionError('Explicit real cache positions required')
    start=int(cache_position.flatten()[0]);end=int(cache_position.flatten()[-1])
    if start>=prompt_length:
        if query_length!=1 or past_length!=start:raise AssertionError('Wrong Decode/cache alignment')
        return True
    if start!=0 or end!=prompt_length-1 or past_length!=0:raise AssertionError('Partial/contaminated Prefill')
    return False


class DecodeLayer(AbstractContextManager):
    def __init__(self,model,layer,prompt_length,noop=False):
        if layer is not None and (not isinstance(layer,int) or not 0<=layer<32):raise ValueError('Invalid single layer')
        self.model,self.layer,self.n,self.noop=model,layer,prompt_length,noop
        self.handles=[];self.active=False;self.calls=0;self.edits=0;self.prefill_calls=0

    def __enter__(self):
        if self.layer is None:return self
        try:
            layer=model_layers(self.model)[self.layer]
            def before(module,args,kwargs):
                hidden=args[0] if args else kwargs['hidden_states']
                position=kwargs.get('cache_position');past=kwargs.get('past_key_value')
                length=past.get_seq_length(self.layer) if past is not None else 0
                self.active=is_decode(position,self.n,length,hidden.shape[1])
                self.calls+=1;self.prefill_calls+=not self.active
            def zero(module,args,output):
                if self.active:
                    if output.shape[:2]!=(1,1):raise AssertionError('Only current Decode query')
                    self.edits+=1
                    if not self.noop:return torch.zeros_like(output)
                return output
            self.handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(zero))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[]


@torch.inference_mode()
def prefill(model,prompt):
    output=model(**prompt,use_cache=True,output_attentions=False,return_dict=True)
    logits=output.logits[0,-1].detach().clone()
    if not torch.isfinite(logits).all():raise AssertionError('Nonfinite native Prefill logits')
    return logits,output.past_key_values


@torch.inference_mode()
def generate(model,prompt,limit,layer=None,parent=None,first=None,noop=False,collect=False):
    n=prompt['input_ids'].shape[1]
    if parent is None:
        with DecodeLayer(model,layer,n,noop):first,parent=prefill(model,prompt)
    if first is None or parent.get_seq_length()!=n:raise AssertionError('Bad shared native prefill')
    history=clone_cache(parent);tokens=[int(first.argmax())];trajectory=[first.detach().float().cpu()] if collect else []
    for original,copied in zip(parent.to_legacy_cache(),history.to_legacy_cache()):
        if any(a.data_ptr()==b.data_ptr() for a,b in zip(original,copied)):raise AssertionError('Aliased condition cache')
    with DecodeLayer(model,layer,n,noop) as trace:
        while len(tokens)<limit and tokens[-1]!=2:
            inputs=step_inputs(prompt,tokens)
            if history.get_seq_length()!=n+len(tokens)-1:raise AssertionError('Own-prefix cache length mismatch')
            output=model(**inputs,past_key_values=history,use_cache=True,output_attentions=False,return_dict=True)
            history=output.past_key_values;logits=output.logits[0,-1]
            if not torch.isfinite(logits).all():raise AssertionError('Nonfinite Decode logits')
            tokens.append(int(logits.argmax()))
            if collect:trajectory.append(logits.detach().float().cpu())
    if layer is not None and (trace.edits!=len(tokens)-1 or trace.prefill_calls):raise AssertionError('Missing or Prefill edit')
    return dict(token_ids=tokens,length=len(tokens),eos=tokens[-1]==2,truncated=len(tokens)==limit and tokens[-1]!=2,
        decode_steps=len(tokens)-1,edited_steps=trace.edits,layer=layer,prefill_native=True,own_prefix=True),trajectory
