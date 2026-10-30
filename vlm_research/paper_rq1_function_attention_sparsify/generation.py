"""Free-running policies; decisions see only the policy's consumed prefix/h31_in."""
import time
from contextlib import AbstractContextManager
import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama
from knockout_probe import model_layers
from paper_rq1.engine import step_inputs
from paper_rq1_late_attention_sources.engine import partition,group_head
from .engine import sparse_attention,retained


class Policy(AbstractContextManager):
    def __init__(self,model,prompt,strategy,predictor=None,implementation='A2',collect=False,profile=False):
        self.model,self.prompt,self.strategy,self.predictor=model,prompt,strategy,predictor
        self.implementation,self.collect,self.profile=implementation,collect,profile
        self.prefix=[];self.groups=None;self.keep=None;self.prune=False;self.hidden=None;self.events=[];self.steps=[];self.host_profile=[]
        self.handles=[];self.original=llama.eager_attention_forward;self.head=None;self.last_probability=None

    def __enter__(self):
        try:
            if self.collect or self.strategy in ['S3','S4']:
                def before(module,args,kwargs):
                    hidden=args[0] if args else kwargs['hidden_states']
                    if self.collect:self.hidden=hidden[0,-1].detach().cpu().clone()
                    if self.prefix and self.strategy in ['S3','S4']:
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        if self.profile:begin.record()
                        started=time.perf_counter()
                        probability=self.predictor.probability(self.prefix,hidden[0,-1])
                        self.last_probability=float(probability);self.prune=self.last_probability>=self.predictor.threshold
                        if self.profile:
                            end.record();self.events.append(('classifier',begin,end))
                            self.host_profile.append(dict(operator='classifier_host_including_sync',milliseconds=1000*(time.perf_counter()-started)))
                self.handles.append(model_layers(self.model)[31].register_forward_pre_hook(before,with_kwargs=True))
            if self.strategy!='S0' or self.profile:
                def attention(module,q,k,v,mask,scaling,**kwargs):
                    if module.layer_idx!=31 or not self.prefix:return self.original(module,q,k,v,mask,scaling,**kwargs)
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    if self.profile:begin.record()
                    if self.prune:
                        if q.shape[-2]!=1:raise AssertionError('Only real Decode query may be pruned')
                        mode='No-Image' if self.strategy in ['S1','S4'] else 'BOS+History'
                        self.keep=retained(self.groups,mode)
                        if self.implementation=='A2':result=sparse_attention(module,q,k,v,mask,scaling,self.keep)
                        else:
                            result=self.original(module,q,k,v,mask,scaling,**kwargs)
                            removed=sorted(set(range(k.shape[-2]))-set(self.keep))
                            self.head=group_head(result[1][0,:,0],llama.repeat_kv(v,module.num_key_value_groups)[0],removed)
                    else:result=self.original(module,q,k,v,mask,scaling,**kwargs)
                    if self.profile:end.record();self.events.append(('l31_attention',begin,end))
                    return result
                llama.eager_attention_forward=attention
            if self.implementation=='A1' and self.strategy!='S0':
                def subtract(module,args,out):
                    if self.prefix and self.prune:return (out.float()-F.linear(self.head,module.weight.float(),bias=None)).to(out.dtype)
                    return out
                self.handles.append(model_layers(self.model)[31].self_attn.o_proj.register_forward_hook(subtract))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def next(self,prefix):
        self.prefix=list(prefix);self.head=None;self.hidden=None;self.keep=None;self.last_probability=None
        self.groups=partition(self.prompt,len(prefix)) if prefix else None
        self.prune=bool(prefix) and self.strategy in ['S1','S2']

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles=[];llama.eager_attention_forward=self.original


def generate_policy(model,prompt,strategy,predictor=None,implementation='A2',collect=False,profile=False):
    cache=None;tokens=[];rows=[];logits=[];hidden=[];decode_events=[]
    torch.cuda.synchronize();start=time.perf_counter()
    total_start,total_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);total_start.record()
    with torch.inference_mode(),Policy(model,prompt,strategy,predictor,implementation,collect,profile) as policy:
        for t in range(64):
            policy.next(tokens);inputs=step_inputs(prompt,tokens)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            output=model(**inputs,past_key_values=cache,use_cache=True,output_attentions=False,return_dict=True)
            b.record();decode_events.append((a,b));cache=output.past_key_values
            current=output.logits[0,-1]
            if not torch.isfinite(current).all():raise AssertionError('Nonfinite free-generation logits')
            token=int(current.argmax())
            if collect:logits.append(current.detach().float().cpu().numpy());hidden.append(policy.hidden.numpy())
            n=prompt['input_ids'].shape[1]+t
            kept=len(policy.keep) if policy.keep is not None else n
            rows.append(dict(t=t,consumed_token_id=tokens[-1] if tokens else None,predicted_token_id=token,
                pruned=policy.prune,probability=policy.last_probability,full_keys=n,retained_keys=kept,
                qk_columns=kept if policy.implementation=='A2' else n,
                av_rows=kept if policy.implementation=='A2' else n,cache_keeps_all_keys=True))
            tokens.append(token)
            if token==2:break
    total_end.record();torch.cuda.synchronize();seconds=time.perf_counter()-start
    profile_rows=[dict(operator=name,milliseconds=a.elapsed_time(b)) for name,a,b in policy.events]+policy.host_profile
    return dict(token_ids=tokens,steps=rows,length=len(tokens),eos=tokens[-1]==2,truncated=len(tokens)==64 and tokens[-1]!=2,
        wall_seconds=seconds,cuda_timeline_ms=total_start.elapsed_time(total_end),prefill_cuda_ms=decode_events[0][0].elapsed_time(decode_events[0][1]),
        decode_cuda_ms=sum(a.elapsed_time(b) for a,b in decode_events[1:]),profile=profile_rows,
        actual_free_generation=True,self_prefix_only=True,implementation=implementation),dict(
            full_logits=np.stack(logits),h31_in=np.stack(hidden)) if collect else {}
