"""Two image-disjoint workers, quality only, immutable atomic checkpoints."""
import argparse
import gc
import traceback
import shutil
from contextlib import contextmanager
from .common import *


@contextmanager
def qwen_last_attention_shape(adapter):
    """Read-only shape trace, including ViT, so an OOM preserves actual Q/K/V."""
    if adapter.kind!='qwen':yield;return
    from unittest.mock import patch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
    original=qm.eager_attention_forward
    def observed(module,q,k,v,*args,**kwargs):
        adapter.last_attention_shape=dict(module=type(module).__name__,layer=getattr(module,'layer_idx',None),
            query=list(q.shape),key=list(k.shape),value=list(v.shape),dtype=str(q.dtype))
        return original(module,q,k,v,*args,**kwargs)
    with patch.object(qm,'eager_attention_forward',observed):yield


@contextmanager
def native_reference(adapter):
    import torch
    refs=[];handles=[];counter=[0];nlayer=len(adapter.layers)
    def observe(module,args,kw):
        step=counter[0]//nlayer
        values=dict(mask=kw.get('attention_mask'),rotary=kw['position_embeddings'],cp=kw['cache_position'])
        if counter[0]%nlayer==0:
            refs.append(dict(mask=values['mask'].cpu().clone() if values['mask'] is not None else None,
                rotary=[x.cpu().clone() for x in values['rotary']],cp=values['cp'].cpu().clone()))
        else:
            ref=refs[step]
            assert (values['mask'] is None)==(ref['mask'] is None)
            if ref['mask'] is not None:assert torch.equal(values['mask'].cpu(),ref['mask'])
            for x,y in zip(values['rotary'],ref['rotary']):assert torch.equal(x.cpu(),y)
            assert torch.equal(values['cp'].cpu(),ref['cp'])
        counter[0]+=1
    try:
        for layer in adapter.layers:handles.append(layer.self_attn.register_forward_pre_hook(observe,with_kwargs=True))
        yield refs
    finally:
        for h in handles:h.remove()


@contextmanager
def block_audit(adapter,layout,native_refs):
    import torch
    from contextlib import ExitStack
    records=[];state={};handles=[]
    before=hook_counts(adapter.model)
    try:
        for i,layer in enumerate(adapter.layers):
            t={};state[i]=t
            def start(m,args,i=i,t=t):
                t['input']=args[0];t['qkv']={};t['norms']=[];t['mlp_calls']=0
            def attn(m,args,out,t=t):t['attn']=out[0]
            def mid(m,args,t=t):
                assert torch.equal(t['input']+t['attn'],args[0]);t['mid']=args[0]
            def mlp(m,args,out,t=t):t['mlp']=out;t['mlp_calls']+=1
            def attn_input(m,args,kw,i=i,t=t):
                pre=len(records)<len(adapter.layers)
                cp=kw['cache_position'].cpu();mask=kw.get('attention_mask')
                n=layout.length if adapter.kind=='llava' else layout['input_length']
                image=list(layout.image) if adapter.kind=='llava' else layout['image_positions']
                text=list(layout.text) if adapter.kind=='llava' else [j for j in range(n) if j not in set(image)]
                cutoff=20 if adapter.kind=='llava' else 22
                step=len(records)//len(adapter.layers)
                expected_pos=(list(range(n)) if i<cutoff else text) if pre else [n+step-1]
                assert cp.tolist()==expected_pos
                if pre:
                    expected=native_refs[0]['mask'].clone()
                    expected[0,0,n-1,image]=-torch.inf
                    if i>=cutoff:
                        r=list(range(image[-1]+1,n-1))
                        expected[0,0,torch.tensor(r)[:,None],torch.tensor(image)[None,:]]=-torch.inf
                        expected=expected[:,:,text,:][:,:,:,text]
                    assert mask is not None and torch.equal(mask.cpu(),expected),'Non-target mask or causal values changed'
                    refrot=native_refs[0]['rotary']
                    for x,y in zip(kw['position_embeddings'],refrot):
                        expected_rope=y if i<cutoff else y[:,text]
                        assert torch.equal(x.cpu(),expected_rope),'Original rotary positions changed'
                else:
                    # Every persistent physical key is a past text key; an absent
                    # mask is exactly the all-zero native text-only submatrix.
                    assert mask is None
                    if step<len(native_refs):
                        for x,y in zip(kw['position_embeddings'],native_refs[step]['rotary']):assert torch.equal(x.cpu(),y)
                t['scope_exact']=True
            def end(m,args,out,i=i,t=t):
                h=out[0] if isinstance(out,tuple) else out
                assert torch.equal(t['mid']+t['mlp'],h)
                assert t['mlp_calls']==1 and len(t['norms'])==2
                records.append(dict(layer=i,hidden_rows=h.shape[1],qkv=t['qkv'],
                    mlp_norm_residual_exact=True,norm_calls=2,mlp_calls=1,
                    native_non_target_mask_and_original_rotary_exact=t['scope_exact']))
                t.clear()
            handles += [layer.register_forward_pre_hook(start),layer.self_attn.register_forward_hook(attn),
                layer.self_attn.register_forward_pre_hook(attn_input,with_kwargs=True),
                layer.post_attention_layernorm.register_forward_pre_hook(mid),
                layer.mlp.register_forward_hook(mlp),layer.register_forward_hook(end)]
            for name in ('input_layernorm','post_attention_layernorm'):
                handles.append(getattr(layer,name).register_forward_hook(lambda m,a,o,t=t:t['norms'].append(1)))
            for name in ('q_proj','k_proj','v_proj','o_proj'):
                handles.append(getattr(layer.self_attn,name).register_forward_hook(
                    lambda m,a,o,t=t,name=name:t['qkv'].__setitem__(name,list(a[0].shape))))
        yield records
    finally:
        for h in handles:h.remove()
        assert hook_counts(adapter.model)==before


def assert_vectors(a,b):
    import torch
    assert len(a)==len(b)
    for x,y in zip(a,b):assert torch.equal(x,y),'Effective logits must be bit-identical'


def smoke(adapter,rank,rows,bindings,tag):
    import torch
    from .engine import prepare,greedy,candidate_tokens,rank_choices
    protocol=read(OUT/'protocol.json');lookup={s['key']:s for s in rows}
    kind=adapter.kind
    specs=protocol['smoke'] if kind=='llava' else read(OUT/'qwen_gpu_protocol.json')['smoke']
    selected=[lookup[x['key']] for x in specs if x['rank']==rank]
    records=[]
    for s in selected:
        layout,b=prepare(adapter,s);expected=bindings[s['key']]
        assert b['input_ids']==expected['input_ids'] and b['prompt']==expected['prompt']
        if kind=='qwen':assert b['input_hashes']['pixel_values']==expected['pixel_values_sha256']
        atomic(OUT/f'smoke/{kind}_rank{rank}/{s["key"]}_input.json',dict(**tag,key=s['key'],binding=b,
            actual_input_shapes={k:list(v.shape) for k,v in adapter.inputs.items()},at=now()))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        inputs_hash=digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})
        with native_reference(adapter) as native_refs:
            native,nv=greedy(adapter,layout,'Native',capture=True)
        noop,noopv=greedy(adapter,layout,'Native',capture=True,noop=True)
        assert native['token_ids']==noop['token_ids'];assert_vectors(nv,noopv)
        hf=adapter.model.generate(**adapter.inputs,do_sample=False,num_beams=1,max_new_tokens=32,
            use_cache=True,return_dict_in_generate=True,output_scores=True,logits_to_keep=1)
        assert hf.sequences[0,len(b['input_ids']):].tolist()==native['token_ids']
        assert_vectors(nv,[x[0].cpu() for x in hf.scores]);del hf
        with block_audit(adapter,layout,native_refs) as blocks:
            physical,pv=greedy(adapter,layout,'Physical-C3',capture=True,audit=True)
        nlayer=32 if kind=='llava' else 36
        cutoff=20 if kind=='llava' else 22
        skip=(2,3,4,5) if kind=='llava' else tuple(range(1,7))
        text_length=len(layout.text) if kind=='llava' else b['input_length']-b['image_tokens']
        text_positions=list(layout.text) if kind=='llava' else [i for i in range(b['input_length']) if i not in set(b['image_positions'])]
        assert len(blocks)==nlayer*len(physical['token_ids'])
        for j,rec in enumerate(blocks):
            step=j//nlayer;i=rec['layer'];expected_rows=len(b['input_ids']) if step==0 and i<cutoff else text_length if step==0 else 1
            assert rec['hidden_rows']==expected_rows
            if i in skip:
                assert ('q_proj' not in rec['qkv']) if step else rec['qkv']['q_proj'][1]==len(b['input_ids'])-1
        kv=physical['audit']['kv']
        for layer in kv['layers']:
            if layer['layer'] in skip:assert layer.get('used_tokens',layer.get('live_tokens'))==0
            elif kind=='llava':assert layer['original_positions'][:len(layout.text)]==list(layout.text)
            else:assert layer['live_tokens']==text_length+len(physical['token_ids'])-1
        if kind=='qwen':
            assert physical['audit']['deepstack']==[0,1,2]
            for r in physical['audit']['layers']:
                want=(list(range(b['input_length'])) if r['layer']<22 else text_positions) if r['step']==0 else [b['input_length']+r['step']-1]
                assert r['logical_positions']==want
        dense,dv=greedy(adapter,layout,'Dense-F',capture=True,audit=True)
        # Preserve observed physical / functional arithmetic discrepancies.
        common=min(len(pv),len(dv))
        diff=[dict(step=t,mean_abs=float((pv[t].float()-dv[t].float()).abs().mean()),
            max_abs=float((pv[t].float()-dv[t].float()).abs().max()),same_top1=int(pv[t].argmax())==int(dv[t].argmax())) for t in range(common)]
        candidate_check=None
        if s['benchmark']=='SEED':
            candidates=candidate_tokens(adapter,b,s['choices'])
            assert candidates==expected['candidate_token_ids']
            tests=[]
            for c in ('Native','Physical-C3'):
                fork,fv=rank_choices(adapter,layout,c,candidates,capture=True)
                cold,cv=rank_choices(adapter,layout,c,candidates,capture=True,fork=False)
                assert fork['mean_nll']==cold['mean_nll'] and fork['token_logps']==cold['token_logps']
                for av,bv in zip(fv,cv):assert_vectors(av,bv)
                tests.append(dict(condition=c,all_candidate_logits_strict_equal=True,
                    nll_equal=True,prediction=fork['prediction'],candidate_token_ids=candidates))
            candidate_check=tests
        assert digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})==inputs_hash
        memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_bytes=torch.cuda.get_device_properties(0).total_memory,safety_margin_bytes=512*2**20)
        assert memory['total_bytes']-memory['peak_reserved_bytes']>=memory['safety_margin_bytes'],'Smoke memory safety margin <512 MiB'
        torch.save(dict(native=nv,physical=pv,dense=dv),PRIVATE/f'smoke_{kind}_rank{rank}_{s["key"]}.pt')
        entry=dict(key=s['key'],native_noop_torch_equal=True,native_hf_all_effective_logits_equal=True,
            binding=b,native=native,physical=physical,dense=dense,dense_physical_observed_diff=diff,
            block_audit=blocks,candidate_fork_test=candidate_check,input_tensors_unchanged=True,
            hook_cleanup=True,all_effective_logits_finite=True,memory_audit=memory)
        records.append(entry)
        atomic(OUT/f'smoke/{kind}_rank{rank}/{s["key"]}.json',dict(**tag,**entry))
        print('SMOKE',rank,s['key'],'PASSED',flush=True)
        del nv,pv,dv;gc.collect();torch.cuda.empty_cache()
    atomic(OUT/f'smoke/{kind}_rank{rank}/completion.json',dict(**tag,status='PASSED',cases=[x['key'] for x in records],at=now()))


def run(adapter,rank,rows,bindings,tag):
    import torch
    from .engine import prepare,greedy,candidate_tokens,rank_choices
    kind=adapter.kind
    assert all(read(OUT/f'smoke/{kind}_rank{i}/completion.json')['status']=='PASSED' for i in (0,1))
    completed=0
    for s in (s for s in rows if s['shard']==rank):
        assert not (OUT/f'STOP_{kind}.json').exists(),'Another worker failed'
        assert shutil.disk_usage('/data').free>=20*2**30,'20 GiB disk safety margin'
        paths={c:OUT/'checkpoints'/kind/s['key']/(c+'.json') for c in ('Native','Physical-C3')}
        missing=[c for c,p in paths.items() if not p.exists()]
        for c in paths:
            if c not in missing:
                prior=read(paths[c]);assert prior['sample_sha256']==digest(s)
                assert prior['source_manifest_sha256']==tag['source_manifest_sha256']
                assert prior['protocol_sha256']==tag['protocol_sha256']
        if missing:
            layout,b=prepare(adapter,s);expected=bindings[s['key']]
            assert b['input_ids']==expected['input_ids'] and b['prompt']==expected['prompt']
            if kind=='qwen':assert b['input_hashes']['pixel_values']==expected['pixel_values_sha256']
            baseline_hash=digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})
            for condition in missing:
                if s['benchmark']=='MME':
                    result,_=greedy(adapter,layout,condition)
                    result['prediction']=mme_parse(result['text']);result['correct']=result['prediction']==s['answer']
                else:
                    candidates=candidate_tokens(adapter,b,s['choices'])
                    assert candidates==expected['candidate_token_ids']
                    result,_=rank_choices(adapter,layout,condition,candidates)
                    result['correct']=result['prediction']==s['answer']
                assert digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})==baseline_hash
                atomic(paths[condition],dict(**tag,at=now(),key=s['key'],condition=condition,
                    sample_sha256=digest(s),image_sha256=s['image_sha256'],input_binding=b,
                    answer=s['answer'],result=result))
        completed+=1
        if completed%20==0:
            atomic(OUT/f'progress/{kind}_rank{rank}.json',dict(**tag,status='RUNNING',completed_questions=completed,
                last_key=s['key'],at=now()))
            print('PROGRESS',rank,completed,s['key'],flush=True)
    atomic(OUT/f'progress/{kind}_rank{rank}.json',dict(**tag,status='COMPLETE',completed_questions=completed,at=now()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[0,1],required=True)
    p.add_argument('--kind',choices=['llava','qwen'],required=True)
    p.add_argument('--stage',choices=['smoke','run'],required=True);args=p.parse_args();rank=args.rank;kind=args.kind
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    manifest=read(OUT/'source_manifest.json');assert manifest['files']==source_hashes(),'Frozen source changed'
    import transformers
    assert transformers.__version__==('4.49.0' if kind=='llava' else '4.57.6')
    for name,h in read(OUT/'model_manifests.json')[kind]['files'].items():assert sha(Path(MODELS[kind])/name)==h
    rows=read(OUT/'sample_manifest.json');bindings=read(OUT/f'{kind}_input_bindings.json.gz')
    tag=dict(kind=kind,rank=rank,uuid=UUIDS[rank],pid=os.getpid(),stage=args.stage,source_commit=git_head(),
        source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'),
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),model_manifest_sha256=sha(OUT/'model_manifests.json'))
    if kind=='qwen':
        q=read(OUT/'qwen_gpu_protocol.json')
        assert q['bindings_sha256']==sha(OUT/'qwen_input_bindings.json.gz')
        assert q['processor_validation_sha256']==sha(OUT/'qwen_processor_validation.json')
        tag['qwen_gpu_protocol_sha256']=sha(OUT/'qwen_gpu_protocol.json')
    atomic(OUT/f'environment/{kind}_rank{rank}_{args.stage}.json',dict(**tag,at=now()))
    from .engine import load
    try:
        adapter=load(kind)
        with qwen_last_attention_shape(adapter):
            (smoke if args.stage=='smoke' else run)(adapter,rank,rows,bindings,tag)
        del adapter
        import torch
        gc.collect();torch.cuda.empty_cache()
        atomic(OUT/f'environment/{kind}_rank{rank}_{args.stage}_exit.json',dict(**tag,status='SUCCESS',at=now()))
    except BaseException:
        err=dict(**tag,status='FAILED_STOP',at=now(),traceback=traceback.format_exc())
        try:
            import torch
            err['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            err['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            err['memory_summary']=torch.cuda.memory_summary()
            err['actual_input_shapes']={k:list(v.shape) for k,v in adapter.inputs.items()}
            err['last_actual_attention_shape']=getattr(adapter,'last_attention_shape',None)
        except Exception as diagnostic_error:err['diagnostic_error']=repr(diagnostic_error)
        atomic(OUT/f'environment/{kind}_rank{rank}_{args.stage}_failure.json',err);atomic(OUT/f'STOP_{kind}.json',err)
        raise

if __name__=='__main__':main()
