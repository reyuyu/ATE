from contextlib import ExitStack
from unittest.mock import patch
import importlib

import torch

from ..physical_visual_handoff.engine import run, equivalence
from .common import OUT, read, atomic, key, write_csv


def error(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    a,b=a.float(),b.float()
    finite=torch.isfinite(a)&torch.isfinite(b)
    d=(a[finite]-b[finite]).abs()
    if not len(d):return dict(exact=torch.equal(a,b),max_abs=0.,relative_rmse=0.)
    rmse=(a[finite]-b[finite]).square().mean().sqrt()/a[finite].square().mean().sqrt().clamp_min(1e-12)
    return dict(exact=torch.equal(a,b),max_abs=float(d.max()),mean_abs=float(d.mean()),relative_rmse=float(rmse),
                finite_pattern_equal=bool(torch.equal(torch.isfinite(a),torch.isfinite(b))))


def compare_runs(kind,a,b,al,bl,ah,bh):
    common=0
    for x,y in zip(a['token_ids'],b['token_ids']):
        common+=1
        if x!=y:break
    proof=equivalence(kind,al[:common],bl[:common],ah,bh)
    proof.update(common_prefix_prediction_steps=common,tokens_equal=a['token_ids']==b['token_ids'],
                 numerical_pass=proof['passed'],greedy_pass=a['token_ids']==b['token_ids'],
                 dense_text=a['text'],physical_text=b['text'],dense_tokens=a['token_ids'],physical_tokens=b['token_ids'])
    proof['overall_pass']=proof['numerical_pass'] and proof['greedy_pass']
    for prefix,logits in (('dense',al),('physical',bl)):
        top=logits[0].flatten().topk(2)
        proof[prefix+'_top1_top2_margin']=float(top.values[0]-top.values[1])
        proof[prefix+'_top2_ids']=top.indices.tolist()
    if len(al)>common and len(bl)>common:
        proof['different_prefix_logits_max_abs_descriptive_only']=max(float((x-y).abs().max()) for x,y in zip(al[common:],bl[common:]))
    return proof


def capture_boundary(adapter,sample,spec,condition='D1'):
    layer=adapter.layers[20 if adapter.kind=='llava' else 22]
    captured={}
    def before(m,args,kwargs):
        if len(kwargs['cache_position'])>1 and not captured:
            h=args[0] if args else kwargs['hidden_states']
            captured.update(hidden=h.detach().clone(),mask=kwargs['attention_mask'].detach().clone(),
                            positions=kwargs['cache_position'].detach().clone(),
                            position_ids=kwargs['position_ids'].detach().clone(),
                            rotary=tuple(v.detach().clone() for v in kwargs['position_embeddings']))
    handle=layer.register_forward_pre_hook(before,with_kwargs=True)
    try:
        result,_,_=run(adapter,sample,spec,condition,max_tokens=1)
    finally:handle.remove()
    assert captured
    captured['cache']=result['cache']
    return captured


@torch.inference_mode()
def operator_trace(adapter,boundary,fp32_attention=False,fp32_linears=False):
    kind=adapter.kind;cutoff=20 if kind=='llava' else 22
    layer=adapter.layers[cutoff]
    source=importlib.import_module(layer.self_attn.__class__.__module__)
    image_id=adapter.model.config.image_token_index if kind=='llava' else adapter.model.config.image_token_id
    image=adapter.inputs['input_ids'][0]==image_id
    keep=torch.where(~image)[0]
    traces=[];saved_outputs=[];visual_probabilities=[]
    for physical in (False,True):
        values={}
        h=boundary['hidden'].index_select(1,keep) if physical else boundary['hidden']
        mask=boundary['mask']
        if physical:mask=mask.index_select(2,keep).index_select(3,keep)
        rotary=tuple(v.index_select(1,keep) for v in boundary['rotary']) if physical else boundary['rotary']
        pos=boundary['positions'].index_select(0,keep) if physical else boundary['positions']
        pids=boundary['position_ids'].index_select(-1,keep) if physical else boundary['position_ids']
        def save(name,x,axis=1):
            if not physical:x=x.index_select(axis,keep)
            values[name]=x.detach().float().cpu()
        save('layer_input',h)
        native=source.eager_attention_forward
        def attention(module,q,k,v,attention_mask,scaling,dropout=0.,**kwargs):
            actual,weights=native(module,q,k,v,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
            save('rope_query',q,2);save('rope_key',k,2)
            kr=source.repeat_kv(k,module.num_key_value_groups);vr=source.repeat_kv(v,module.num_key_value_groups)
            raw=torch.matmul(q,kr.transpose(2,3))*scaling
            masked=raw+attention_mask
            probs=torch.nn.functional.softmax(masked,dim=-1,dtype=torch.float32).to(q.dtype)
            av=torch.matmul(probs,vr).transpose(1,2).contiguous()
            assert torch.equal(probs,weights) and torch.equal(av,actual), 'Local reconstruction changed native operator'
            if not physical:
                visual_probabilities.append(float(probs.index_select(2,keep).index_select(3,torch.where(image)[0]).abs().max()))
            def matrix(name,x):
                if not physical:x=x.index_select(2,keep).index_select(3,keep)
                values[name]=x.detach().float().cpu()
            matrix('attention_scores',raw);matrix('masked_scores',masked);matrix('softmax',probs)
            save('attention_times_value',actual)
            if fp32_attention:
                # Preserve native low-precision activation boundaries; change only GEMM accumulation.
                score=(q.float()@kr.float().transpose(2,3)).to(q.dtype)*scaling
                score=score+attention_mask
                prob=score.float().softmax(-1).to(q.dtype)
                output=(prob.float()@vr.float()).to(q.dtype).transpose(1,2).contiguous()
                matrix('fp32acc_softmax',prob);save('fp32acc_attention_times_value',output)
                return output,prob
            return actual,weights
        with ExitStack() as stack:
            if fp32_linears:
                for name,module in layer.named_modules():
                    if isinstance(module,torch.nn.Linear):
                        def linear(x,module=module):
                            return torch.nn.functional.linear(x.float(),module.weight.float(),None if module.bias is None else module.bias.float()).to(x.dtype)
                        stack.enter_context(patch.object(module,'forward',linear))
            stack.enter_context(patch.object(source,'eager_attention_forward',attention))
            modules=[('input_rmsnorm',layer.input_layernorm),('q_projection',layer.self_attn.q_proj),
                     ('k_projection',layer.self_attn.k_proj),('v_projection',layer.self_attn.v_proj),
                     ('o_projection',layer.self_attn.o_proj),('mlp_input',layer.post_attention_layernorm),
                     ('mlp_output',layer.mlp)]
            if kind=='qwen':modules += [('q_norm',layer.self_attn.q_norm),('k_norm',layer.self_attn.k_norm)]
            for name,module in modules:
                def observe(m,args,output,name=name):save(name,output)
                handle=module.register_forward_hook(observe);stack.callback(handle.remove)
            def residual(m,args):save('attention_residual',args[0])
            handle=layer.post_attention_layernorm.register_forward_pre_hook(residual);stack.callback(handle.remove)
            kw=dict(attention_mask=mask,position_ids=pids,cache_position=pos,position_embeddings=rotary,use_cache=False)
            kw['past_key_values' if kind=='qwen' else 'past_key_value']=None
            output=layer(h,**kw);output=output if kind=='qwen' else output[0]
            save('layer_output',output);saved_outputs.append(output.detach())
        traces.append(values)
    ordered=['layer_input','input_rmsnorm','q_projection','q_norm','k_projection','k_norm','v_projection',
             'rope_query','rope_key','attention_scores','masked_scores','softmax','attention_times_value',
             'fp32acc_softmax','fp32acc_attention_times_value','o_projection','attention_residual','mlp_input','mlp_output','layer_output']
    rows=[dict(stage=stage,fp32_attention=fp32_attention,fp32_linears=fp32_linears,**error(traces[0][stage],traces[1][stage]))
          for stage in ordered if stage in traces[0]]
    denselegal=boundary['mask'][0,0].index_select(0,keep).index_select(1,keep)==0
    compactlegal=boundary['mask'].index_select(2,keep).index_select(3,keep)[0,0]==0
    mapping=dict(legal_edges_exact=torch.equal(denselegal,compactlegal),original_text_positions=keep.tolist(),
                 image_positions=torch.where(image)[0].tolist(),
                 forbidden_visual_probability_max=max(visual_probabilities),
                 original_cache_positions_preserved=torch.equal(boundary['positions'].index_select(0,keep),keep),
                 GQA_query_heads=layer.self_attn.config.num_attention_heads,
                 GQA_kv_heads=layer.self_attn.config.num_key_value_heads,
                 text_query_key_edges=int(denselegal.sum()),first_layer=cutoff,
                 original_dtype=str(h.dtype),same_exact_layer_input=True)
    return rows,mapping


def diagnose(adapter,manifest,specs):
    kind=adapter.kind;folder=OUT/kind
    sample=next(s for s in manifest['all_samples'] if s['question_id']==('03567478' if kind=='llava' else '001025435'))
    adapter.prepare(sample,specs[key(sample)])
    repeats=[]
    for repeat in range(3):
        a,al,ah=run(adapter,sample,specs[key(sample)],'D1',audit=True,capture=True)
        b,bl,bh=run(adapter,sample,specs[key(sample)],'P1',audit=True,capture=True)
        proof=compare_runs(kind,a,b,al,bl,ah,bh);proof['repeat']=repeat
        if kind=='llava':
            proof.update(dense_yes_minus_no=float(al[0][0,3869]-al[0][0,1939]),
                         physical_yes_minus_no=float(bl[0][0,3869]-bl[0][0,1939]))
        repeats.append(proof)
    atomic(folder/'failure_repeats.json',repeats)
    boundary=capture_boundary(adapter,sample,specs[key(sample)])
    physical_boundary=capture_boundary(adapter,sample,specs[key(sample)],'P1')
    native,mapping=operator_trace(adapter,boundary)
    keep=torch.tensor(mapping['original_text_positions'],device=boundary['hidden'].device)
    mapping.update(actual_physical_input_exact=torch.equal(boundary['hidden'].index_select(1,keep),physical_boundary['hidden']),
                   actual_physical_mask_exact=torch.equal(boundary['mask'].index_select(2,keep).index_select(3,keep),physical_boundary['mask']),
                   actual_physical_position_ids_exact=torch.equal(boundary['position_ids'].index_select(-1,keep),physical_boundary['position_ids']),
                   actual_physical_cache_positions_exact=torch.equal(boundary['positions'].index_select(0,keep),physical_boundary['positions']),
                   cos_sin_exact=all(torch.equal(a.index_select(1,keep),b) for a,b in zip(boundary['rotary'],physical_boundary['rotary'])))
    assert all(mapping[k] for k in ('actual_physical_input_exact','actual_physical_mask_exact',
                                   'actual_physical_position_ids_exact','actual_physical_cache_positions_exact','cos_sin_exact'))
    assert mapping['forbidden_visual_probability_max']==0.
    write_csv(folder/'operator_trace.csv',native)
    controls=[]
    for linear,attn in ((False,True),(True,True)):
        rows,_=operator_trace(adapter,boundary,fp32_attention=attn,fp32_linears=linear)
        controls+=rows
    write_csv(folder/'local_fp32_operator_controls.csv',controls)
    atomic(folder/'attention_mask_equivalence.json',mapping)
    atomic(folder/'position_and_cache_audit.json',dict(**mapping,
           text_at_boundary_bit_exact=mapping['actual_physical_input_exact'],
           old_persistent_cache_rows=boundary['cache']['layers'],
           native_DeepStack_before_boundary=True,mrope_and_cache_position_unchanged=True))
    # Small summaries only; no model weights or vocabulary archives are written.
    write_csv(folder/'layerwise_error.csv',[dict(layer=i,**error(x,y)) for i,(x,y) in enumerate(zip(ah,bh))])
    atomic(folder/'diagnosis_complete.json',dict(status='COMPLETE',first_nonexact_operator=next((r['stage'] for r in native if not r['exact']),None)))
    adapter.verify()
