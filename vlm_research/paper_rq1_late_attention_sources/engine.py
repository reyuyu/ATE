"""Original-softmax Value-source subtraction at a single native Decode Query."""
from contextlib import AbstractContextManager
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama

from knockout_probe import model_layers, ROW_ATOL
from paper_rq1_token_role_decode.engine import owned_clone, signature, gamma, decompose, metrics, generate
from paper_rq1.engine import step_inputs, cache_fingerprint
from .protocol import LAYERS, GROUPS, SOURCE, SPECIAL


def partition(prompt, t, image_token_id=32000, bos_token_id=1):
    if t < 1 or prompt['input_ids'].shape[0] != 1:
        raise ValueError('Sources require one native Decode position t>=1')
    ids = prompt['input_ids'][0].tolist()
    if not torch.all(prompt['attention_mask']==1) or prompt['attention_mask'].numel()!=len(ids):
        raise ValueError('Padding is not part of this frozen unbatched protocol')
    bos = [i for i,x in enumerate(ids) if x==bos_token_id]
    image = [i for i,x in enumerate(ids) if x==image_token_id]
    if len(bos)!=1 or len(image)!=576 or set(bos)&set(image):
        raise ValueError('Invalid dynamically located original BOS/image positions')
    excluded=set(bos+image)
    groups=dict(BOS=bos,IMAGE=image,PROMPT_TEXT=[i for i in range(len(ids)) if i not in excluded],
                GENERATED_HISTORY=list(range(len(ids),len(ids)+t)))
    groups.update(RECENT=[len(ids)+t-1],OLDER_HISTORY=list(range(len(ids),len(ids)+t-1)))
    validate_partition(groups,len(ids)+t)
    return groups


def validate_partition(groups, visible):
    keys=[k for name in GROUPS for k in groups[name]]
    if len(keys)!=visible or len(set(keys))!=visible or sorted(keys)!=list(range(visible)):
        raise ValueError('Sources must exactly partition all valid visible keys')
    if len(groups['RECENT'])!=1 or groups['RECENT'][0]!=visible-1:
        raise ValueError('Recent must be the currently consumed generated token')
    recent=groups['RECENT']; older=groups['OLDER_HISTORY']; history=groups['GENERATED_HISTORY']
    if set(recent)&set(older) or sorted(recent+older)!=history:
        raise ValueError('Recent/Older must exactly partition generated history')


def group_head(row, values, keys):
    if not keys:
        return torch.zeros((1,1,values.shape[0]*values.shape[-1]),device=values.device,dtype=torch.float32)
    positions=torch.as_tensor(keys,device=row.device)
    return (row.float().index_select(-1,positions).unsqueeze(-2) @
            values.float().index_select(-2,positions)).squeeze(-2).reshape(1,1,-1)


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return float((a*b).sum()/denominator) if float(denominator)>0 else None


def contributions(projection,row,values,groups,native_input,native_output,residual,layer):
    validate_partition(groups,row.shape[-1])
    if native_input.dtype!=torch.float16 or native_output.dtype!=torch.float16:
        raise ValueError('Native precision must remain FP16')
    names=GROUPS+(['RECENT','OLDER_HISTORY'] if layer in SPECIAL else [])
    heads={name:group_head(row,values,groups[name]) for name in names}
    weight=projection.weight.float()
    outputs={name:F.linear(heads[name],weight,bias=None) for name in names}
    if not all(torch.isfinite(v).all() for v in outputs.values()):
        raise AssertionError('Nonfinite projected source')
    # The previous frozen Image/non-Image test must still pass unchanged.
    _, old = decompose(projection,row,values,groups['IMAGE'],native_input,native_output,True)
    p,v=row.float(),values.float()
    magnitude=(p.abs().unsqueeze(-2) @ v.abs()).squeeze(-2).reshape_as(native_input)
    head_bound=(gamma(p.shape[-1])+2.**-11)*magnitude+2.**-25
    head_sum=sum(heads[n] for n in GROUPS)
    # Only the extra FP32 group summation error extends the original bound.
    head_bound=head_bound+gamma(len(GROUPS))*sum(heads[n].abs() for n in GROUPS)
    head_error=(native_input.float()-head_sum).abs()
    if not torch.all(head_error<=head_bound):raise AssertionError('Four-source native PV reconstruction failed')
    projected=sum(outputs[n] for n in GROUPS)
    bias=projection.bias.float() if projection.bias is not None else 0.
    reconstruction=projected+bias
    bound=F.linear(head_bound,weight.abs(),None)
    bound+=gamma(weight.shape[-1])*F.linear(native_input.float().abs()+sum(heads[n].abs() for n in GROUPS),weight.abs(),None)
    bound+=gamma(len(GROUPS))*sum(outputs[n].abs() for n in GROUPS)
    bound+=2.**-11*native_output.float().abs()+2.**-25
    if projection.bias is not None:bound+=2.**-24*projection.bias.float().abs()
    error=(native_output.float()-reconstruction).abs()
    if not torch.all(error<=bound):raise AssertionError('Four-source output reconstruction failed')
    audit=dict(layer=layer,**old,partition_exact=True,
        group_head_max_abs_error=float(head_error.max()),group_head_max_error_to_bound=float((head_error/head_bound).max()),
        group_projection_max_abs_error=float(error.max()),group_projection_max_error_to_bound=float((error/bound).max()))
    if layer in SPECIAL:
        history_error=(outputs['GENERATED_HISTORY']-outputs['RECENT']-outputs['OLDER_HISTORY']).abs()
        hmag=(p[:,groups['GENERATED_HISTORY']].abs().unsqueeze(-2) @ v[:,groups['GENERATED_HISTORY']].abs()).squeeze(-2).reshape_as(native_input)
        hbound=(2*gamma(len(groups['GENERATED_HISTORY']))+gamma(2))*hmag
        hbound=F.linear(hbound,weight.abs(),None)+gamma(weight.shape[-1])*F.linear(
            heads['GENERATED_HISTORY'].abs()+heads['RECENT'].abs()+heads['OLDER_HISTORY'].abs(),weight.abs(),None)
        hbound+=gamma(2)*(outputs['GENERATED_HISTORY'].abs()+outputs['RECENT'].abs()+outputs['OLDER_HISTORY'].abs())
        if not torch.all(history_error<=hbound):raise AssertionError('History Recent/Older projected reconstruction failed')
        ratio=torch.where(hbound>0,history_error/hbound,torch.zeros_like(hbound))
        audit.update(history_split_max_abs_error=float(history_error.max()),history_split_max_error_to_bound=float(ratio.max()))
    rn=float(residual.norm()); an=float(native_output.float().norm())
    if rn<=0 or not torch.isfinite(residual).all():raise AssertionError('Invalid actual layer-input residual')
    cancellation=float(projected.norm()/sum(outputs[n].norm() for n in GROUPS))
    rows=[]
    for name in names:
        output=outputs[name]; norm=float(output.norm())
        rows.append(dict(layer=layer,source=name,n_keys=len(groups[name]),
            mass=float(p[:,groups[name]].sum(-1).mean()) if groups[name] else 0.,
            contribution_norm=norm,relative_residual_ratio=norm/rn,residual_norm=rn,attention_output_norm=an,
            cosine_with_full_attention=cosine(output,native_output.float()),
            cosine_with_residual=cosine(output,residual),
            attention_residual_cosine=cosine(native_output.float(),residual),
            image_history_cosine=cosine(outputs['IMAGE'],outputs['GENERATED_HISTORY']),
            group_cancellation_ratio=cancellation,
            **{'cosine_with_'+n:cosine(output,outputs[n]) for n in GROUPS}))
    return outputs,rows,audit


class SourceTrace(AbstractContextManager):
    def __init__(self,model,groups,mode=None,layer=None,observe=False,noop=False):
        if mode not in [None,'Whole',*SOURCE]:raise ValueError('Unknown source intervention')
        if mode is not None and layer not in LAYERS:raise ValueError('Only frozen ten layers allowed')
        if mode in ['BOS','Recent','Older'] and layer not in SPECIAL:raise ValueError('Extra interventions only26/31')
        self.model,self.groups,self.mode,self.layer=model,groups,mode,layer
        self.observe,self.noop=observe,noop
        self.active=LAYERS if observe else [layer] if mode is not None else []
        self.handles=[];self.edits=[];self.rows=[];self.audits=[];self.pv={};self.residual={}
        self.original=llama.eager_attention_forward

    def __enter__(self):
        try:
            layers=model_layers(self.model)
            if len(layers)!=32:raise ValueError('Need frozen32-layer model')
            for index in self.active:
                if self.observe:
                    def capture(module,args,kwargs,index=index):
                        hidden=args[0] if args else kwargs['hidden_states']
                        if hidden.shape[:2]!=(1,1):raise AssertionError('Observation must be current Decode Query')
                        self.residual[index]=hidden[0,0].detach().float()
                    self.handles.append(layers[index].register_forward_pre_hook(capture,with_kwargs=True))
                self.handles.append(layers[index].self_attn.o_proj.register_forward_hook(self.output_hook(index)))
            if self.observe or self.mode in SOURCE:
                def tap(module,query,key,value,mask,scaling,**kwargs):
                    output,weights=self.original(module,query,key,value,mask,scaling,**kwargs)
                    if module.layer_idx in self.active:
                        if query.shape[:1]!=(1,) or query.shape[-2]!=1:raise AssertionError('Sources only at actual Decode Query')
                        row=weights[0,:,0]
                        validate_partition(self.groups,row.shape[-1])
                        if not torch.isfinite(row).all() or float((row.float().sum(-1)-1).abs().max())>ROW_ATOL:
                            raise AssertionError('Invalid original post-softmax weights')
                        self.pv[module.layer_idx]=(row,llama.repeat_kv(value,module.num_key_value_groups)[0])
                    return output,weights
                llama.eager_attention_forward=tap
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def output_hook(self,index):
        def hook(module,args,output):
            if output.shape[:2]!=(1,1):raise AssertionError('Sources only apply to one Decode Query')
            outputs=None
            if self.observe:
                row,values=self.pv.pop(index)
                outputs,rows,audit=contributions(module,row,values,self.groups,args[0],output,self.residual.pop(index),index)
                self.rows.extend(rows);self.audits.append(audit)
            if self.mode is None:return output
            self.edits.append(dict(layer=index,query=0,mode=self.mode,noop=self.noop))
            if self.mode=='Whole':return output if self.noop else torch.zeros_like(output)
            if outputs is None:
                row,values=self.pv.pop(index)
                head=group_head(row,values,self.groups[SOURCE[self.mode]])
                contribution=F.linear(head,module.weight.float(),bias=None)
            else:contribution=outputs[SOURCE[self.mode]]
            if not torch.isfinite(contribution).all():raise AssertionError('Nonfinite source contribution')
            return output if self.noop else (output.float()-contribution).to(output.dtype)
        return hook

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if self.observe or self.mode in SOURCE:llama.eager_attention_forward=self.original
        self.pv.clear();self.residual.clear()


def step(model,inputs,history,groups=None,mode=None,layer=None,observe=False,noop=False):
    before=signature(history);cloned=owned_clone(history)
    query=inputs['input_ids'].shape[1]-1 if history is None else 0
    if (observe or mode is not None) and (history is None or inputs['input_ids'].shape[1]!=1):
        raise ValueError('Intervention/observation requires real cached Decode')
    with torch.inference_mode(),SourceTrace(model,groups,mode,layer,observe,noop) as trace:
        output=model(**inputs,past_key_values=cloned,use_cache=True,output_attentions=False,return_dict=True)
    logits=output.logits[0,query].detach().float().cpu()
    if not torch.isfinite(logits).all() or signature(history)!=before:
        raise AssertionError('Nonfinite logits or parent cache mutation')
    expected=[layer] if mode is not None else []
    if [r['layer'] for r in trace.edits]!=expected:raise AssertionError('Wrong intervention layers')
    if observe and sorted({r['layer'] for r in trace.rows})!=LAYERS:raise AssertionError('Incomplete native layer observations')
    return logits,output.past_key_values,dict(edits=trace.edits,rows=trace.rows,audits=trace.audits)
