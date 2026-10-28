"""Reuse directed additive masks; stream early-layer observations and local audits."""
from contextlib import nullcontext
import hashlib
import time

import numpy as np
import torch
from directed_suffix import DirectedSuffixKnockout
from knockout_probe import layout_from_inputs, model_layers, ROW_ATOL, tensor_hash
from vlm_common import gpu_memory


class PathwayMask(DirectedSuffixKnockout):
    def __init__(self, model, inputs, qpos, apos, condition, expected_visual=576, noop=False):
        real = layout_from_inputs(inputs, model.config.image_token_index, expected_visual)
        if condition not in ('B1','B2','B3','B4') or not qpos or any(type(i)is not int or not 0<=i<apos for i in qpos):
            raise ValueError('Invalid early pathway condition or Question/P-1 mapping')
        if len(set(qpos))!=len(qpos) or apos in qpos:
            raise ValueError('Duplicate/overlapping query mapping')
        queries = torch.where(real['text_queries'])[0].tolist() if condition=='B1' else (qpos if condition=='B3' else [apos])
        super().__init__(model_layers(model),inputs,model.config.image_token_index,0,queries,[],expected_visual)
        self.active=set() if noop else {0,1}
        if condition=='B4':
            keys=torch.zeros_like(real['visual']);keys[qpos]=True
            if (keys & ~real['text_queries']).any():
                raise ValueError('Question keys are not real valid post-image text')
            self.layout=dict(self.layout,visual=keys)
        self.real_layout=real

    def _hook(self, index):
        original=super()._hook(index)
        def hook(module,args,kwargs):
            before=kwargs['attention_mask'];h=tensor_hash(before)
            result=original(module,args,kwargs)
            after=result[1]['attention_mask']
            if tensor_hash(before)!=h or (index>=2 and after is not before):
                raise AssertionError('Original/shared or later-layer mask changed')
            self.records[-1].update(native_mask_sha256=h,effective_mask_sha256=tensor_hash(after),
                                   native_mask_unmodified=True,visual_tokens_retained=len(self.real_layout['visual_indices']))
            return result
        return hook


def input_hash(inputs):
    return [(k,str(v.dtype),tuple(v.shape),tensor_hash(v)) for k,v in sorted(inputs.items())]


def parameter_stamp(model):
    return [(n,p.data_ptr(),p._version,tuple(p.shape)) for n,p in model.named_parameters()]


def state_hash(model):
    h=hashlib.sha256()
    for name,value in list(model.named_parameters())+list(model.named_buffers()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def footprint(model):
    return [(name,len(m._forward_pre_hooks),len(m._forward_hooks)) for name,m in model.named_modules()]


def qk_probe(module, kwargs, visual, apos):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
    hidden=kwargs['hidden_states'];shape=(*hidden.shape[:-1],-1,module.head_dim)
    q=module.q_proj(hidden).view(shape).transpose(1,2)
    k=module.k_proj(hidden).view(shape).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
    k=repeat_kv(k,module.num_key_value_groups)
    scores=(q@k.transpose(2,3))*module.scaling
    v=scores[0,:,apos].index_select(-1,visual)
    return dict(l0_visual_scores=v.detach().cpu().numpy(),
                l0_visual_only_conditional=torch.softmax(v.float(),-1).cpu().numpy(),
                l0_answer_input_sha256=tensor_hash(hidden[0,apos]),
                l0_visual_input_sha256=tensor_hash(hidden[0].index_select(0,visual)))


def observed_forward(model, inputs, qpos, apos, condition='B0', expected_visual=576, noop=False, probe=False):
    if model.training or not qpos or len(set(qpos))!=len(qpos) or any(type(i)is not int or not 0<=i<apos for i in qpos):
        raise ValueError('Requires eval mode and earlier real Question queries')
    real=layout_from_inputs(inputs,model.config.image_token_index,expected_visual)
    if not bool(real['text_queries'][apos]) or any(not bool(real['text_queries'][i]) for i in qpos):
        raise ValueError('Non-text Question/P-1 query')
    layers=model_layers(model);n=real['sequence_length'];visual=real['visual_indices']
    qmask=torch.zeros(n,dtype=torch.bool,device=visual.device);qmask[qpos]=True
    other=real['valid'] & ~real['visual'] & ~qmask
    groups=[real['visual'],qmask,other]
    selected=qpos+[apos]
    before_hooks=footprint(model);before_input=input_hash(inputs);before_params=parameter_stamp(model)
    handles=[];saved={};attention=[];hidden=[];mass=[];audits=[];local=[];probe_result={};final=[]
    patch=None if condition=='B0' else PathwayMask(model,inputs,qpos,apos,condition,expected_visual,noop)
    cuda=inputs['input_ids'].is_cuda
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    def pre(index):
        def hook(module,args,kwargs):
            if index<2:
                saved[index]=(args,dict(kwargs))
        return hook
    def post(index):
        def hook(module,args,out):
            a=out[1]
            if a is None or tuple(a.shape)!=(1,module.config.num_attention_heads,n,n):
                raise AssertionError('Unexpected eager attention shape/token retention')
            if not torch.isfinite(a).all() or (a<0).any():
                raise AssertionError('Nonfinite/negative attention')
            error=float((a.float().sum(-1)-1).abs().max())
            future=torch.ones(n,n,dtype=torch.bool,device=a.device).triu(1)
            if error>ROW_ATOL or torch.count_nonzero(a[0,:,future]) or torch.count_nonzero(a[0,:,:,~real['valid']]):
                raise AssertionError('Row/causal/padding attention audit failed')
            record=dict(layer=index,sequence_length=n,visual_tokens=expected_visual,row_sum_error=error,
                        future_attention_max=0.,padding_attention_max=0.)
            if index<2:
                aa,kw=saved.pop(index)
                active=patch is not None and index in patch.active
                if active:
                    # Local counterfactual holds this layer's actual hidden input fixed.
                    # It is not the global Full L1 after an earlier intervention.
                    reference=module.forward(*aa,**kw)[1]
                    targets=patch.layout['text_queries'];keys=patch.layout['visual']
                    if not torch.equal(a[:,:,~targets,:],reference[:,:,~targets,:]):
                        raise AssertionError('A non-target query attention row changed locally')
                    blocked=a[0,:,targets,:][:,:,keys]
                    if torch.count_nonzero(blocked):
                        raise AssertionError('Nonzero blocked target attention')
                    before=reference[0,:,targets,:];after=a[0,:,targets,:]
                    bm=before[:,:,keys].float().sum(-1)
                    def grouped(t):return torch.stack([t[:,:,g].float().sum(-1) for g in groups],-1)
                    before_mass,after_mass=grouped(before),grouped(after)
                    local.append(dict(layer=index,query_indices=torch.where(targets)[0].cpu().tolist(),
                        key_indices=torch.where(keys)[0].cpu().tolist(),masked_mass_before=bm.cpu().tolist(),
                        masked_mass_after=0.,before_group_mass=before_mass.cpu().tolist(),after_group_mass=after_mass.cpu().tolist(),
                        group_order=['image','question','other_text'],non_target_queries_bitwise_exact=True,
                        max_target_row_mass_error=float((after.float().sum(-1)-1).abs().max()),
                        reference='Native attention with identical actual local hidden inputs and original causal mask'))
                    del reference
                else:
                    local.append(dict(layer=index,query_indices=[],key_indices=[],masked_mass_after=0.,
                                      non_target_queries_bitwise_exact=True,reference='No local mask applied'))
                row=a[0,:,selected,:]
                attention.append(row.index_select(-1,visual).detach().cpu().numpy())
                mass.append(torch.stack([row[:,:,g].float().sum(-1) for g in groups],-1).cpu().numpy())
                if index==0 and probe:
                    probe_result.update(qk_probe(module,kw,visual,apos))
            audits.append(record)
            return (out[0],None)+tuple(out[2:])
        return hook
    def block_output(module,args,out):
        values=out[0][0,selected].detach().cpu()
        if not torch.isfinite(values).all():raise AssertionError('Nonfinite early hidden')
        hidden.append(values.numpy())
    try:
        # Register native-input capture before the mask hooks.
        for i,layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_pre_hook(pre(i),with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(post(i)))
            if i<2:handles.append(layer.register_forward_hook(block_output))
        handles.append(model.language_model.model.norm.register_forward_hook(lambda m,a,o:final.append(o[0,selected].detach().cpu())))
        with torch.inference_mode(), (patch if patch is not None else nullcontext()):
            output=model(**inputs,output_attentions=True,use_cache=False,return_dict=True)
        if patch:patch.assert_one_forward()
        if output.past_key_values is not None or not torch.isfinite(output.logits).all() or not torch.isfinite(final[0]).all():
            raise AssertionError('Nonfinite logits/final hidden or unexpected cache')
        if len(audits)!=len(layers) or len(hidden)!=2 or len(attention)!=2:
            raise AssertionError('Incomplete observation')
        text=torch.where(real['valid'] & ~real['visual'])[0]
        if cuda:torch.cuda.synchronize()
        return dict(prediction_logits=output.logits[0,apos].detach().cpu(),
                    text_logits=output.logits[0].index_select(0,text).detach().cpu(),
                    arrays=dict(attention=np.stack(attention),hidden=np.stack(hidden),mass=np.stack(mass),
                                final_hidden=final[0].numpy(),**{k:v for k,v in probe_result.items() if isinstance(v,np.ndarray)}),
                    audit=dict(layer_attention=audits,local_redistribution=local,mask_records=patch.records if patch else [],
                               input_unchanged=True,weights_version_unchanged=True,hooks_removed=True,
                               **{k:v for k,v in probe_result.items() if not isinstance(v,np.ndarray)}),
                    measurement=dict(seconds=time.perf_counter()-started,gpu_memory=gpu_memory() if cuda else None))
    finally:
        for handle in handles:handle.remove()
        if footprint(model)!=before_hooks:
            raise AssertionError('Observer/mask hook leak')
        if input_hash(inputs)!=before_input or parameter_stamp(model)!=before_params:
            raise AssertionError('Input or model parameter mutation')


def drift(reference,current):
    a,b=np.asarray(reference,dtype=float),np.asarray(current,dtype=float)
    distance=1-np.sum(a*b,-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-300)
    change=np.linalg.norm(b-a,axis=-1)
    return dict(cosine_distance_mean=float(distance.mean()),change_norm_mean=float(change.mean()),
                reference_norm_mean=float(np.linalg.norm(a,axis=-1).mean()))


def spatial(rows):
    x=np.asarray(rows,dtype=float);m=x.sum(-1)
    if (m<=0).any():
        return dict(available=False,reason='Zero visual Mass; no conditional map defined',mass=float(m.mean()))
    conditional=x/m[...,None]
    pooled=conditional.mean(axis=tuple(range(conditional.ndim-1)))
    entropy=-(np.where(conditional>0,conditional*np.log(np.maximum(conditional,1e-300)),0).sum(-1))/np.log(x.shape[-1])
    return dict(available=True,mass=float(m.mean()),distribution=pooled,normalized_entropy=float(entropy.mean()))


def js_cosine(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);m=(a+b)/2
    def kl(x):return np.sum(np.where(x>0,x*np.log(np.maximum(x,1e-300)/np.maximum(m,1e-300)),0))
    return dict(js_distance=float(np.sqrt(max(0.,(kl(a)+kl(b))/2))),
                cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
