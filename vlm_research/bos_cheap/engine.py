from contextlib import AbstractContextManager
import time
import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama
from knockout_probe import model_layers
from .protocol import HEAD_FEATURES,PROTOCOL


def selected_positions(mapping):
    image=mapping['visual_positions'];question=mapping['question_and_instruction_positions']
    if len(image)!=576 or not question or len(mapping['bos_positions'])!=1:raise ValueError('Invalid real token mapping')
    extra=list(dict.fromkeys([image[i] for i in [0,191,383,575]]+[question[0],question[-1]]))
    return mapping['p_minus_one'],mapping['bos_positions'][0],extra


def rotate_selected(x,cos,sin,indices):
    c=cos[:,indices].unsqueeze(1);s=sin[:,indices].unsqueeze(1)
    return x*c+llama.rotate_half(x)*s


def cheap_project(module,normalized,cos,sin,query,bos,extra):
    """Only1 Query and<=7 Keys: no all-token QK or Softmax invocation."""
    indices=[bos]+extra;dim=module.head_dim;heads=module.config.num_attention_heads
    q=F.linear(normalized[:,query:query+1],module.q_proj.weight,module.q_proj.bias).view(1,1,heads,dim).transpose(1,2)
    rows=normalized[:,indices]
    k=F.linear(rows,module.k_proj.weight,module.k_proj.bias).view(1,len(indices),heads,dim).transpose(1,2)
    v=F.linear(rows[:,:1],module.v_proj.weight,module.v_proj.bias).view(1,1,heads,dim).transpose(1,2)
    return rotate_selected(q,cos,sin,[query]),rotate_selected(k,cos,sin,indices),v


def grams(model):
    result=[]
    with torch.inference_mode():
        for layer in model_layers(model):
            attn=layer.self_attn;h=attn.config.num_attention_heads;d=attn.head_dim
            w=attn.o_proj.weight.detach().float().reshape(-1,h,d).permute(1,2,0)
            result.append(w@w.transpose(1,2))
    return result


class Features(AbstractContextManager):
    def __init__(self,model,mapping,gram,check=False):
        self.model=model;self.mapping=mapping;self.gram=gram;self.check=check
        self.query,self.bos,self.extra=selected_positions(mapping)
        self.handles=[];self.hidden={};self.raw={};self.mass={};self.projected={};self.cheap_checks=[];self.bench=None

    def layer_pre(self,i):
        def hook(module,args,kwargs):
            value=args[0] if args else kwargs['hidden_states']
            if i in self.hidden:raise ValueError('Repeated native layer')
            self.hidden[i]=value[0,self.query].detach().clone()
            if self.check:self.selected_input=value[:,[self.query,self.bos]+self.extra].detach().clone()
        return hook

    def attention_pre(self,i):
        def hook(module,args,kwargs):
            self.normalized=kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            self.cos,self.sin=kwargs['position_embeddings']
            self.current=i
            if self.check:
                # Independent row-only RMSNorm, before any all-token QK/Softmax.
                self.reconstructed=model_layers(self.model)[i].input_layernorm(self.selected_input)
                torch.testing.assert_close(self.reconstructed,self.normalized[:,[self.query,self.bos]+self.extra],**PROTOCOL['cheap_tolerances'])
        return hook

    def observe(self,module,query,key,value,attention_mask,scaling,**kwargs):
        i=module.layer_idx
        if i!=self.current or key.shape[1]!=32 or query.shape[1]!=32 or i in self.raw:raise ValueError('Wrong native head/layer mapping')
        q=query[0,:,self.query];k=key[0,:,self.bos];v=value[0,:,self.bos]
        extra=key[0,:,self.extra]
        self.raw[i]=torch.stack([q,k,v]).detach().clone(),extra.detach().clone()
        self.projected[i]=torch.einsum('hd,hde,he->h',v.float(),self.gram[i],v.float()).clamp_min(0).sqrt()
        if self.check:
            positions=[self.query,self.bos]+self.extra
            cq,ck,cv=cheap_project(module,self.reconstructed,self.cos[:,positions],self.sin[:,positions],0,1,list(range(2,len(positions))))
            tol=PROTOCOL['cheap_tolerances'];errors=[]
            for actual,reproduced in [(query[:,:,self.query:self.query+1],cq),(key[:,:,[self.bos]+self.extra],ck),(value[:,:,self.bos:self.bos+1],cv)]:
                torch.testing.assert_close(actual,reproduced,**tol);errors.append(float((actual-reproduced).abs().max()))
            actual_score=(q.float()*k.float()).sum(-1)*scaling
            reproduced_score=(cq[0,:,0].float()*ck[0,:,0].float()).sum(-1)*scaling
            torch.testing.assert_close(actual_score,reproduced_score,**tol)
            native_cos=actual_score/scaling/(q.float().norm(dim=-1)*k.float().norm(dim=-1))
            cheap_cos=reproduced_score/scaling/(cq[0,:,0].float().norm(dim=-1)*ck[0,:,0].float().norm(dim=-1))
            torch.testing.assert_close(native_cos,cheap_cos,**tol)
            torch.testing.assert_close(v.float().norm(dim=-1),cv[0,:,0].float().norm(dim=-1),**tol)
            self.cheap_checks.append(dict(layer=i,max_qkv_error=max(errors),max_score_error=float((actual_score-reproduced_score).abs().max()),max_cosine_error=float((native_cos-cheap_cos).abs().max()),row_only_rmsnorm=True,only_query_rows=1,only_key_rows=1+len(self.extra),full_softmax=False))
            if i==0:
                self.bench=(query[:,:,self.query:self.query+1].detach(),key.detach(),value.detach(),self.bos,self.extra)
                self.projection_bench=(module,self.normalized.detach(),self.cos,self.sin,self.query,self.bos,self.extra)
        output=self.original(module,query,key,value,attention_mask,scaling,**kwargs)
        row=output[1][0,:,self.query]
        if not torch.isfinite(row).all() or float((row.float().sum(-1)-1).abs().max())>.002:raise ValueError('Invalid native probability row')
        self.mass[i]=row[:,self.bos].float().detach().clone()
        return output

    def __enter__(self):
        self.original=llama.eager_attention_forward
        try:
            for i,layer in enumerate(model_layers(self.model)):
                self.handles.append(layer.register_forward_pre_hook(self.layer_pre(i),with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_pre_hook(self.attention_pre(i),with_kwargs=True))
            llama.eager_attention_forward=self.observe
            return self
        except BaseException:self.__exit__(None,None,None);raise

    def __exit__(self,*args):
        llama.eager_attention_forward=self.original
        for h in self.handles:h.remove()
        self.handles=[]

    def arrays(self):
        expected=set(range(32))
        if any(set(x)!=expected for x in [self.raw,self.hidden,self.mass,self.projected]):raise ValueError('Incomplete32-layer observation')
        return dict(qkv=torch.stack([self.raw[i][0] for i in range(32)]).cpu().numpy(),
            extra_keys=torch.stack([self.raw[i][1] for i in range(32)]).cpu().numpy(),
            residual=torch.stack([self.hidden[i] for i in range(32)]).cpu().numpy(),
            bos_mass=torch.stack([self.mass[i] for i in range(32)]).cpu().numpy(),
            projected_v_norm=torch.stack([self.projected[i] for i in range(32)]).cpu().numpy())


def reduce_features(arrays):
    a={k:torch.as_tensor(v).float() for k,v in arrays.items()}
    if any(not torch.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite native feature capture')
    q,k,v=a['qkv'].unbind(1);qn=q.norm(dim=-1);kn=k.norm(dim=-1)
    if min(float(qn.min()),float(kn.min()))<=1e-12:raise ValueError('Undefined QK cosine')
    score=(q*k).sum(-1)/(q.shape[-1]**.5)
    sampled=(q[:,:,None]*a['extra_keys']).sum(-1)/(q.shape[-1]**.5)
    hf=torch.stack([((q*k).sum(-1)/(qn*kn)).clamp(-1,1),score,qn,kn,v.norm(dim=-1),v.mean(-1),v.std(-1,unbiased=False),v.min(-1).values,v.max(-1).values,a['projected_v_norm'],score-sampled.max(-1).values,sampled.mean(-1)],-1).numpy()
    residual=a['residual'];T=None
    base=torch.stack([residual.norm(dim=-1),residual.mean(-1),residual.std(-1,unbiased=False),residual.abs().max(-1).values],-1).numpy()
    return hf,base


def collect(model,inputs,mapping,gram,check=False):
    with torch.inference_mode(),Features(model,mapping,gram,check) as trace:
        output=model(**inputs,use_cache=False,output_attentions=False,return_dict=True)
    arrays=trace.arrays();arrays['full_logits']=output.logits[0,trace.query].detach().float().cpu().numpy()
    hf,base=reduce_features({k:v for k,v in arrays.items() if k!='full_logits'})
    return arrays,hf,base,trace


def benchmark(q,k,v,bos,extra):
    # Actual captured tensors; these are PyTorch microbenchmarks, not E2E speedups.
    qb=q[0,:,0];kb=k[0,:,bos];vb=v[0,:,bos];sample=k[0,:,extra]
    w=torch.ones(86,device=q.device,dtype=torch.float32)/86
    def functions(a,b,c,s,weight):
        return dict(bos_qk=lambda:(a.float()*b.float()).sum(-1)/(a.shape[-1]**.5),
            read_bos_kv=lambda:(b.clone(),c.clone()),
            prototype_logistic86=lambda:torch.sigmoid(a.float().flatten()[:86]@weight),
            sampled6_qk=lambda:(a[:,None].float()*s.float()).sum(-1),
            full_query_attention=lambda:torch.softmax((a[None,:,None].float()@k.float().transpose(-1,-2))/(a.shape[-1]**.5),-1)@v.float())
    result=[]
    for device in ['cuda','cpu']:
        a,b,c,s,weight=[x.to(device) for x in [qb,kb,vb,sample,w]]
        funcs=functions(a,b,c,s,weight)
        if device=='cpu':
            kc,vc=k.cpu().float(),v.cpu().float()
            funcs['full_query_attention']=lambda:torch.softmax((a[None,:,None].float()@kc.transpose(-1,-2))/(a.shape[-1]**.5),-1)@vc
        for name,fn in funcs.items():
            for _ in range(10):fn()
            if device=='cuda':
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):fn()
                end.record();torch.cuda.synchronize();us=begin.elapsed_time(end)*1000/100
            else:
                start=time.perf_counter()
                for _ in range(100):fn()
                us=(time.perf_counter()-start)*1e6/100
            result.append(dict(device=device,component=name,microseconds=us,iterations=100,heads=32,head_dim=128,length=k.shape[-2],scope='PyTorch dispatch+operations on actual tensors; not a custom kernel or E2E speedup'))
    return result


def projection_benchmark(args):
    rows=[]
    for name,extra in [('bos_only_qkv_projection',[]),('sampled6_qkv_projection',args[-1])]:
        params=(*args[:-1],extra)
        with torch.inference_mode():
            for _ in range(10):cheap_project(*params)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):cheap_project(*params)
            b.record();torch.cuda.synchronize()
        rows.append(dict(device='cuda',component=name,microseconds=a.elapsed_time(b)*10,iterations=100,scope='Actual FP16 projection weights and selected normalized rows plus RoPE; excludes already available RMSNorm and predictor.'))
    return rows


def fitted_predictor_benchmark(rule,features):
    from .predict import design
    model=rule['model'];g=model['groups']['0'];x=design([features],model['columns'],model['layer'])[0];c=g['calibration'];result=[]
    for device in ['cpu','cuda']:
        value,mean,scale,w=[torch.tensor(v,dtype=torch.float64 if device=='cpu' else torch.float32,device=device) for v in [x,g['mean'],g['scale'],g['weights']]]
        def fn():
            raw=((value-mean)/scale)@w+g['intercept']
            calibrated=(raw-c['mean'][0])/c['scale'][0]*c['weights'][0]+c['intercept']
            return torch.sigmoid(calibrated)<=rule['threshold']
        for _ in range(10):fn()
        if device=='cuda':
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):fn()
            b.record();torch.cuda.synchronize();us=a.elapsed_time(b)*10
        else:
            a=time.perf_counter()
            for _ in range(100):fn()
            us=(time.perf_counter()-a)*1e4
        result.append(dict(device=device,component='actual_frozen_BOS-QK+Residual_predictor',parameters=g['parameters']+c['parameters'],microseconds=us,iterations=100,scope='Actual fitted standardization, logistic, Platt calibration, threshold; already assembled feature vector, no E2E speedup.'))
    return result
