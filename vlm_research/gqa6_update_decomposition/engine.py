from contextlib import AbstractContextManager

import numpy as np
import torch

from knockout_probe import model_layers
from .protocol import NAMES, PROTOCOL


class NativeUpdates(AbstractContextManager):
    """Every hook returns None: observe actual rows without replacing computation."""
    def __init__(self, model, query):
        if not isinstance(query,int) or query<0:raise ValueError('Invalid actual query')
        self.model,self.query=model,query
        self.handles=[];self.values={};self.calls={}

    def capture(self, layer, name, value):
        if value.ndim!=3 or value.shape[0]!=1 or not self.query<value.shape[1]:
            raise ValueError('Expected one real unbatched Prompt query')
        key=(layer,name)
        if key in self.values:raise ValueError('A native node was visited more than once')
        self.values[key]=value[0,self.query].detach().clone()
        self.calls[key]=1

    def before(self,i,name):
        def hook(module,args,kwargs):
            self.capture(i,name,args[0] if args else kwargs['hidden_states'])
        return hook

    def after(self,i,name):
        def hook(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            self.capture(i,name,value)
        return hook

    def __enter__(self):
        try:
            layers=model_layers(self.model)
            if len(layers)!=32:raise ValueError('Expected32 decoder layers')
            for i,layer in enumerate(layers):
                self.handles.append(layer.register_forward_pre_hook(self.before(i,'h_in'),with_kwargs=True))
                self.handles.append(layer.self_attn.o_proj.register_forward_hook(self.after(i,'attention')))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.before(i,'h_mid'),with_kwargs=True))
                self.handles.append(layer.mlp.register_forward_hook(self.after(i,'mlp')))
                self.handles.append(layer.register_forward_hook(self.after(i,'h_out')))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def vectors(self):
        if set(self.values)!={(i,n) for i in range(32) for n in NAMES}:
            raise ValueError('Incomplete32-layer five-node capture')
        return torch.stack([torch.stack([self.values[i,n] for n in NAMES]) for i in range(32)]).cpu()


def measurements(vectors):
    raw=torch.as_tensor(vectors)
    if raw.ndim!=3 or raw.shape[:2]!=(32,5) or raw.dtype!=torch.float16 or not torch.isfinite(raw).all():
        raise ValueError('Require finite actual FP16 five-vector captures for32 layers')
    rows=[]
    for i in range(32):
        hin,a,mid,m,out=raw[i].float()
        na,nm,nr,no,nh=[torch.linalg.vector_norm(x) for x in (a,m,hin,out,mid)]
        if float(nr)<=PROTOCOL['denominator_floor']:raise ValueError('Near-zero input residual denominator')
        summed=a+m; net=torch.linalg.vector_norm(summed);den=na+nm
        def cosine(x,y,nx,ny):
            if min(float(nx),float(ny))<=PROTOCOL['denominator_floor']:return None
            value=torch.dot(x,y)/(nx*ny)
            if not torch.isfinite(value) or abs(float(value))>1+PROTOCOL['cosine_roundoff_bound']:
                raise ValueError('Invalid FP32 cosine')
            return float(value.clamp(-1,1))
        expected_mid=(hin+a).half();expected_out=(mid+m).half()
        if not torch.equal(expected_mid,raw[i,2]) or not torch.equal(expected_out,raw[i,4]):
            raise ValueError('Actual residual addition differs from exact FP16-rounded sum')
        err1=(mid-(hin+a)).abs();err2=(out-(mid+m)).abs()
        if not torch.all(err1<=PROTOCOL['fp16_sum_atol']+PROTOCOL['fp16_sum_rtol']*(hin+a).abs()):
            raise ValueError('First residual rounding exceeds fixed FP16 bound')
        if not torch.all(err2<=PROTOCOL['fp16_sum_atol']+PROTOCOL['fp16_sum_rtol']*(mid+m).abs()):
            raise ValueError('Second residual rounding exceeds fixed FP16 bound')
        cancellation=None if float(den)<=PROTOCOL['denominator_floor'] else float(net/den)
        if cancellation is not None and not 0<=cancellation<=1+PROTOCOL['cosine_roundoff_bound']:
            raise ValueError('Invalid cancellation ratio')
        row=dict(layer=i,norm_residual=float(nr),norm_attention=float(na),norm_mlp=float(nm),
            norm_output=float(no),norm_mid=float(nh),attention_ratio=float(na/nr),mlp_ratio=float(nm/nr),
            net_update_ratio=float(net/nr),cancellation_ratio=None if cancellation is None else min(1.,cancellation),
            cos_attention_residual=cosine(a,hin,na,nr),cos_mlp_residual=cosine(m,hin,nm,nr),
            cos_attention_mlp=cosine(a,m,na,nm),cos_input_output=cosine(hin,out,nr,no),
            actual_net_update_ratio=float(torch.linalg.vector_norm(out-hin)/nr),
            dot_attention_residual=float(torch.dot(a,hin)),dot_mlp_residual=float(torch.dot(m,hin)),
            dot_attention_mlp=float(torch.dot(a,m)),sum1_fp32_max_error=float(err1.max()),
            sum2_fp32_max_error=float(err2.max()),rounded_sums_exact=True)
        if any(v is not None and not np.isfinite(v) for v in row.values()):raise ValueError('Nonfinite statistic')
        rows.append(row)
    return rows


def collect(model,inputs,query):
    with torch.inference_mode(),NativeUpdates(model,query) as trace:
        output=model(**inputs,use_cache=False,output_attentions=False,return_dict=True)
    logits=output.logits[0,query].detach().float().cpu()
    vectors=trace.vectors()
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite native logits')
    return measurements(vectors),vectors.numpy(),logits.numpy()
