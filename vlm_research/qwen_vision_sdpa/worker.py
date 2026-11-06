"""Qwen-only, untimed quality requests with pre-frozen backend/resource gates."""
import argparse
import gc
import shutil
import traceback
from contextlib import contextmanager,nullcontext
import torch
from .common import *
from .backend import load,BackendAudit,vision_backend,efficient_sdpa_only
from ..mme_seed_quality.engine import prepare,greedy,rank_choices,candidate_tokens,call,control
from ..mme_seed_quality.worker import native_reference,block_audit,assert_vectors


def diff_vectors(a,b):
    assert len(a)==len(b)
    return [dict(step=i,mean_abs=float((x.float()-y.float()).abs().mean()),
        max_abs=float((x.float()-y.float()).abs().max()),same_top1=int(x.argmax())==int(y.argmax()))
        for i,(x,y) in enumerate(zip(a,b))]


@torch.inference_mode()
def fixed_prefix(adapter,layout,condition,ids,audit=False):
    c=control(adapter,layout,condition,len(ids)+1,audit);cache=None;vectors=[]
    with c.context() if c else nullcontext():
        for t in range(len(ids)):
            v,cache=call(adapter,layout,cache,t,ids[t-1] if t else None)
            vectors.append(v.detach().cpu().clone())
    return vectors


def validate_input(adapter,s,bindings):
    assert sha(s['image_path'])==s['image_sha256'],'Raw image SHA mismatch'
    layout,b=prepare(adapter,s);expected=bindings[s['key']]
    assert b['input_ids']==expected['input_ids'] and b['prompt']==expected['prompt']
    assert b['input_hashes']['pixel_values']==expected['pixel_values_sha256']
    assert b['image_grid_thw']==[expected['image_grid_thw']]
    if s['benchmark']=='SEED':assert candidate_tokens(adapter,b,s['choices'])==expected['candidate_token_ids']
    return layout,b,digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})


def unchanged(adapter,input_hash):
    assert digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})==input_hash


def phase(observer,label,fn,vision='sdpa',capture=False):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    observer.begin(label,feature_hashes=capture,feature_capture=capture,vision=vision)
    value=fn();torch.cuda.synchronize()
    backend,features=observer.finish()
    memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        total_bytes=torch.cuda.get_device_properties(0).total_memory,safety_margin_bytes=512*2**20)
    memory['reserved_safety_margin_bytes']=memory['total_bytes']-memory['peak_reserved_bytes']
    memory['actual_free_bytes_at_request_end']=torch.cuda.mem_get_info()[0]
    assert memory['reserved_safety_margin_bytes']>=512*2**20,'Memory safety margin <512MiB'
    return value,dict(backend=backend,memory=memory),features


def backend_compare(adapter,observer,s,bindings,tag):
    layout,b,ih=validate_input(adapter,s,bindings)
    with vision_backend(adapter,'eager'):
        (e,ev),ea,ef=phase(observer,'Eager-Vision Native',lambda:greedy(adapter,layout,'Native',capture=True),vision='eager',capture=True)
    (n,nv),na,nf=phase(observer,'SDPA-Vision Native',lambda:greedy(adapter,layout,'Native',capture=True),capture=True)
    if e['token_ids']==n['token_ids']:matched=nv;pa=None
    else:
        matched,pa,_=phase(observer,'SDPA-Vision Eager-generated fixed prefix',lambda:fixed_prefix(adapter,layout,'Native',e['token_ids']))
    features=[]
    for i,(x,y) in enumerate(zip(ef,nf)):
        assert x.shape==y.shape
        d=(x.float()-y.float()).abs()
        features.append(dict(feature=i,shape=list(x.shape),mean_abs=float(d.mean()),max_abs=float(d.max())))
    assert len(features)==4
    ld=diff_vectors(ev,matched);g=read(OUT/'protocol.json')['backend_numerical_gates']
    passed=all(x['mean_abs']<=g['features_mean_abs'] and x['max_abs']<=g['features_max_abs'] for x in features)
    passed=passed and all(x['mean_abs']<=g['native_same_prefix_logits_mean_abs'] and x['max_abs']<=g['native_same_prefix_logits_max_abs'] for x in ld)
    unchanged(adapter,ih)
    private=PRIVATE/f'backend_compare_rank{tag["rank"]}.pt'
    torch.save(dict(eager_features=ef,sdpa_features=nf,eager_logits=ev,sdpa_logits=nv,sdpa_same_prefix_logits=matched),private)
    record=dict(**tag,status='PASSED' if passed else 'FAILED_NUMERICAL_GATE',key=s['key'],binding=b,
        features=features,all_same_prefix_logits=ld,eager=e,sdpa=n,tokens_equal=e['token_ids']==n['token_ids'],
        phases=[ea,na]+([pa] if pa else []),input_tensors_unchanged=True,
        raw_arrays_server_only=dict(path=str(private),sha256=sha(private)),at=now())
    atomic(OUT/f'smoke/backend_comparison_rank{tag["rank"]}.json',record)
    assert passed,'Frozen backend numerical thresholds exceeded; see complete comparison'


def smoke(adapter,observer,rank,rows,bindings,tag):
    protocol=read(OUT/'protocol.json');lookup={s['key']:s for s in rows}
    compare=next(x for x in protocol['backend_comparison'] if x['rank']==rank)
    backend_compare(adapter,observer,lookup[compare['key']],bindings,tag)
    completed=[]
    for spec in [x for x in protocol['smoke'] if x['rank']==rank]:
        s=lookup[spec['key']];layout,b,ih=validate_input(adapter,s,bindings);phases=[]
        atomic(OUT/f'smoke/rank{rank}/{s["key"]}_input.json',dict(**tag,key=s['key'],binding=b,
            actual_shapes={k:list(v.shape) for k,v in adapter.inputs.items()}))
        with native_reference(adapter) as refs:
            (native,nv),p,_=phase(observer,'Native full greedy',lambda:greedy(adapter,layout,'Native',capture=True));phases.append(p)
        (noop,noov),p,_=phase(observer,'Native identity No-op',lambda:greedy(adapter,layout,'Native',capture=True,noop=True));phases.append(p)
        assert native['token_ids']==noop['token_ids'];assert_vectors(nv,noov);del noov
        def hf_request():
            with torch.inference_mode():
                return adapter.model.generate(**adapter.inputs,do_sample=False,num_beams=1,max_new_tokens=32,
                    use_cache=True,return_dict_in_generate=True,output_scores=True,logits_to_keep=1)
        hf,p,_=phase(observer,'Native HF generate',hf_request);phases.append(p)
        assert hf.sequences[0,b['input_length']:].tolist()==native['token_ids']
        assert_vectors(nv,[v[0].cpu() for v in hf.scores]);del hf
        with block_audit(adapter,layout,refs) as blocks:
            (physical,pv),p,_=phase(observer,'Physical-Original full greedy',lambda:greedy(adapter,layout,'Physical-C3',capture=True,audit=True));phases.append(p)
        textlen=b['input_length']-b['image_tokens'];textpos=[i for i in range(b['input_length']) if i not in set(b['image_positions'])]
        assert len(blocks)==36*len(physical['token_ids'])
        for j,r in enumerate(blocks):
            step=j//36;layer=r['layer'];want=b['input_length'] if step==0 and layer<22 else textlen if step==0 else 1
            assert r['hidden_rows']==want
            if layer in range(1,7):assert ('q_proj' not in r['qkv']) if step else r['qkv']['q_proj'][1]==b['input_length']-1
        for r in physical['audit']['kv']['layers']:
            assert r['live_tokens']==(0 if r['layer'] in range(1,7) else textlen+len(physical['token_ids'])-1)
        assert physical['audit']['deepstack']==[0,1,2]
        for r in physical['audit']['layers']:
            want=(list(range(b['input_length'])) if r['layer']<22 else textpos) if r['step']==0 else [b['input_length']+r['step']-1]
            assert r['logical_positions']==want
        (dense,dv),p,_=phase(observer,'Dense F full greedy',lambda:greedy(adapter,layout,'Dense-F',capture=True,audit=True));phases.append(p)
        if dense['token_ids']==physical['token_ids']:matched=pv
        else:
            matched,p,_=phase(observer,'Physical along Dense F prefix',lambda:fixed_prefix(adapter,layout,'Physical-C3',dense['token_ids']));phases.append(p)
        diff=diff_vectors(dv,matched)
        strict=all(x['mean_abs']<=0.02 and x['max_abs']<=0.5 and x['same_top1'] for x in diff) and dense['token_ids']==physical['token_ids']
        candidate_checks=[]
        if s['benchmark']=='SEED':
            candidates=candidate_tokens(adapter,b,s['choices'])
            for condition in ('Native','Physical-C3'):
                (fork,fv),p,_=phase(observer,condition+' SEED fork',lambda:rank_choices(adapter,layout,condition,candidates,capture=True));phases.append(p)
                (cold,cv),p,_=phase(observer,condition+' SEED independent cold',lambda:rank_choices(adapter,layout,condition,candidates,capture=True,fork=False));phases.append(p)
                assert fork['mean_nll']==cold['mean_nll'] and fork['token_logps']==cold['token_logps']
                for x,y in zip(fv,cv):assert_vectors(x,y)
                candidate_checks.append(dict(condition=condition,candidate_token_ids=candidates,all_effective_logits_torch_equal=True,
                    mean_nll=fork['mean_nll'],prediction=fork['prediction'],cache_forks_equal_cold=True))
                del fv,cv
        unchanged(adapter,ih)
        raw=PRIVATE/f'smoke_rank{rank}_{s["key"]}.pt';torch.save(dict(native=nv,physical=pv,dense=dv,physical_dense_prefix=matched),raw)
        atomic(OUT/f'smoke/rank{rank}/{s["key"]}.json',dict(**tag,status='PASSED_SEMANTICS_AND_RESOURCE_GATES',
            key=s['key'],reasons=spec['reasons'],binding=b,native=native,physical=physical,dense=dense,
            block_audit=blocks,phases=phases,candidate_checks=candidate_checks,
            native_hf_noop_all_effective_logits_torch_equal=True,inputs_unchanged=True,hook_cleanup=True,
            dense_physical_same_prefix_logits=diff,preserved_strict_numeric_gate_passed=strict,
            physical_numeric_status='Historical Physical-Original limitations retained; not claimed strictly equivalent',
            raw_arrays_server_only=dict(path=str(raw),sha256=sha(raw)),at=now()))
        completed.append(s['key']);print('SMOKE',rank,s['key'],'PASSED',flush=True)
        del nv,pv,dv,matched,refs;gc.collect();torch.cuda.empty_cache()
    atomic(OUT/f'smoke/rank{rank}/completion.json',dict(**tag,status='PASSED',cases=completed,at=now()))


def run(adapter,observer,rank,rows,bindings,tag):
    assert all(read(OUT/f'smoke/rank{i}/completion.json')['status']=='PASSED' for i in (0,1))
    completed=0
    for s in (s for s in rows if s['shard']==rank):
        assert not (OUT/'STOP.json').exists(),'Another worker failed'
        assert shutil.disk_usage('/data').free>=20*2**30
        paths={c:OUT/'checkpoints'/s['key']/(c+'.json') for c in ('Native','Physical-C3')}
        missing=[c for c,p in paths.items() if not checkpoint_complete(p,tag,s)]
        if missing:
            layout,b,ih=validate_input(adapter,s,bindings)
            for condition in missing:
                if s['benchmark']=='MME':
                    (result,_),pa,_=phase(observer,condition+' MME',lambda:greedy(adapter,layout,condition))
                    result['prediction']=mme_parse(result['text'])
                else:
                    candidates=candidate_tokens(adapter,b,s['choices'])
                    (result,_),pa,_=phase(observer,condition+' SEED',lambda:rank_choices(adapter,layout,condition,candidates))
                result['correct']=result['prediction']==s['answer'];unchanged(adapter,ih)
                # Compact per-request evidence, retaining actual per-layer operators/shapes.
                write_checkpoint(paths[condition],dict(**tag,at=now(),key=s['key'],condition=condition,
                    sample_sha256=digest(s),image_sha256=s['image_sha256'],input_binding=b,
                    answer=s['answer'],result=result,execution_audit=pa))
        completed+=1
        if completed%20==0:
            atomic(OUT/f'progress/rank{rank}.json',dict(**tag,status='RUNNING',completed_questions=completed,last_key=s['key'],at=now()))
            print('PROGRESS',rank,completed,s['key'],flush=True)
    atomic(OUT/f'progress/rank{rank}.json',dict(**tag,status='COMPLETE',completed_questions=completed,at=now()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[0,1],required=True)
    p.add_argument('--stage',choices=['smoke','run'],required=True);args=p.parse_args();rank=args.rank
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[rank]
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert read(OUT/'authorization.json')['gpu_authorized_this_round'] is True
    assert read(OUT/'cpu_validation.json')['status']=='PASSED'
    assert read(OUT/'source_manifest.json')['files']==source_hashes(),'Frozen execution source changed'
    for item in read(OUT/'reuse_audit.json')['files']:
        assert sha(OUT/item['file'])==item['sha256'],'Reused frozen artifact changed: '+item['file']
    for path,h in read(OUT/'installed_source_audit.json')['files'].items():
        assert sha(path)==h,'Installed dependency source changed: '+path
    import transformers
    assert transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
    for name,h in read(OUT/'model_manifests.json')['qwen']['files'].items():assert sha(Path(MODELS['qwen'])/name)==h
    tag=dict(kind='qwen',rank=rank,uuid=UUIDS[rank],pid=os.getpid(),stage=args.stage,source_commit=git_head(),
        source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'),
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),model_manifest_sha256=sha(OUT/'model_manifests.json'))
    atomic(OUT/f'environment/rank{rank}_{args.stage}.json',dict(**tag,at=now()))
    try:
        adapter=load()
        with efficient_sdpa_only(),BackendAudit(adapter) as observer:
            (smoke if args.stage=='smoke' else run)(adapter,observer,rank,read(OUT/'sample_manifest.json'),
                read(OUT/'qwen_input_bindings.json.gz'),tag)
        del adapter;gc.collect();torch.cuda.empty_cache()
        atomic(OUT/f'environment/rank{rank}_{args.stage}_exit.json',dict(**tag,status='SUCCESS',at=now()))
    except BaseException:
        err=dict(**tag,status='FAILED_STOP',at=now(),traceback=traceback.format_exc())
        try:
            err.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                memory_summary=torch.cuda.memory_summary(),actual_input_shapes={k:list(v.shape) for k,v in adapter.inputs.items()},
                last_actual_attention_shape=getattr(adapter,'last_attention_shape',None))
        except Exception as e:err['diagnostic_error']=repr(e)
        atomic(OUT/f'environment/rank{rank}_{args.stage}_failure.json',err);atomic(OUT/'STOP.json',err)
        raise


if __name__=='__main__':main()
