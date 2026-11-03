"""Two image-cluster workers; LLaVA audit/release strictly precedes Qwen."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import torch

from .common import *
from .engine import Adapter,PrefillP1SingleLayerBypass,generate,tensor_hash,footprint


def cpu():
    from . import tests
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    proof=dict(passed=True,tests=result.testsRun,sources=source_hashes(),cuda_initialized=False)
    seal(OUT/'cpu_attempts'/(digest(proof)+'.json'),proof)
    atomic(OUT/'cpu_validation.json',proof)


def freeze():
    assert read(OUT/'cpu_validation.json')['sources']==source_hashes()
    assert command('git','status','--porcelain','--','vlm_research/p1_layerwise_accuracy')==''
    for kind,m in MODELS.items():
        folder=OUT/kind; samples=read(folder/'sample_manifest.json')['samples']
        specs=read(folder/'input_bindings.json')
        pool=[s for s in samples if s['accuracy_subset']] if kind=='llava' else samples
        ordered=sorted(pool,key=lambda s:(len(s['input_mapping']['prompt_ids']) if kind=='llava' else specs[key(s)]['layout']['input_length'],key(s)))
        smoke=[key(ordered[i]) for i in (0,len(ordered)//2,len(ordered)-1)]
        if kind=='qwen':assert len({specs[k]['layout']['image_tokens'] for k in smoke})>1
        files=['sample_manifest.json','input_bindings.json','shards.json','baselines.json','legacy_generation.json',
            'first_token_raw_bindings.json','historical_first_token_summary.json','historical_reuse_audit.json']
        weights={str(f):dict(sha256=sha(f),size=f.stat().st_size) for f in sorted(model_path(kind).glob('*.safetensors'))}
        official={str(f):sha(f) for f in sorted(model_path(kind).glob('*.json'))}
        p=dict(kind=kind,model=m,code_commit=command('git','rev-parse','HEAD'),questions=3000,image_clusters=2409,
            conditions=conditions(kind),condition_order='Native Full first, then ascending Drop-P1 layers',
            intervention='Only selected language-layer o_proj original prefill P-1 output row zero; all Decode native',
            native_full_is_not_All_Writer=True,original_input_and_prompt=True,independent_native_KV=True,
            generation=dict(do_sample=False,num_beams=1,max_new_tokens=32,normal_native_eos=True,
                logits_to_keep=m['logits_to_keep']),
            no_attention_mask_changes=True,other_queries_and_MLP_native=True,smoke_keys=smoke,
            logits_shape_reason='LLaVA keep=0 exactly matches frozen original 33-vector/480-generation experiment; Qwen native keep=1',
            historical_reuse_requires_strict_gpu_replay=True,visual_replay='Same-image native features cloned, exact replay checked before reuse',
            bootstrap=dict(repetitions=20000,seed=SEED,image_cluster=True,question_weighted=True,
                ordinary_two_sided_95=True,bonferroni_comparisons=m['layers'],no_noninferiority_claim=True),
            sources=source_hashes(),weights=weights,official_sources=official,
            artifacts={f:sha(folder/f) for f in files})
        seal(folder/'protocol.json',p)
        print('FROZEN',kind,digest(p),smoke,flush=True)


def caches(cache):
    if hasattr(cache,'layers'):
        return [(v.keys,v.values) for v in cache.layers]
    return list(zip(cache.key_cache,cache.value_cache))


def smoke(adapter,samples,specs,baselines,legacy,p,shard):
    by={key(s):s for s in samples};cases=[]
    for identifier in p['smoke_keys']:
        s=by[identifier];adapter.prepare(s,specs[identifier])
        with adapter.visual_replay(enabled=False):
            full,fs,valid,parent=generate(adapter,capture=True,native=True)
        features=adapter.visual_hashes
        assert full['token_ids']==baselines[identifier]['token_ids'],'Historical native tokens mismatch'
        if adapter.kind=='llava':
            meta=read(OUT/'llava/first_token_raw_bindings.json')[identifier]
            assert sha(meta['path'])==meta['sha256']
            with np.load(meta['path']) as data:
                old_full=torch.from_numpy(data['full_logits'].copy())
                old_off=torch.from_numpy(data['ablated_logits'].copy())
            assert torch.equal(fs[0],old_full),'Historical Full logits exact replay failed'
        else:
            assert full['all_step_logits_sha256']==baselines[identifier]['all_step_logits_sha256']
        with adapter.visual_replay():
            noop,ns,nvalid,own=generate(adapter,selected=0,noop=True,capture=True)
        assert full['token_ids']==noop['token_ids'] and len(valid)==len(nvalid)
        assert all(torch.equal(a,b) for a,b in zip(valid,nvalid)),'All valid native/no-op logits must be torch.equal'
        if adapter.kind=='qwen':assert full['native_mrope_positions']==noop['native_mrope_positions']
        assert all(a.data_ptr()!=b.data_ptr() and c.data_ptr()!=d.data_ptr()
            for (a,c),(b,d) in zip(caches(parent),caches(own)))
        assert features==adapter.visual_hashes
        del own,nvalid,ns
        layers=[0,len(adapter.layers)//2,len(adapter.layers)-1]
        for layer in layers:
            with adapter.visual_replay():
                changed,cs,_,own=generate(adapter,selected=layer,capture=True)
            assert sum(e['active'] for e in changed['intervention_events'])==1
            assert all(not e['active'] and e['decode_native'] for e in changed['intervention_events'][1:])
            if adapter.kind=='qwen':
                common=min(len(full['token_ids']),len(changed['token_ids']))
                assert full['native_mrope_positions'][:common]==changed['native_mrope_positions'][:common]
            if adapter.kind=='llava':
                assert torch.equal(cs[0],old_off[layer]),'Legacy single-layer logits mismatch'
                assert changed['token_ids']==legacy[identifier][f'Drop-P1-L{layer}']['token_ids']
            assert all(a.data_ptr()!=b.data_ptr() for (a,_),(b,_) in zip(caches(parent),caches(own)))
            del own,cs
        # Replay features must not change native output, including Qwen DeepStack.
        with adapter.visual_replay():
            repeated,rs,_,own=generate(adapter,capture=True,native=True)
        assert full['all_step_logits_sha256']==repeated['all_step_logits_sha256']
        assert all(torch.equal(a,b) for a,b in zip(fs,rs))
        del parent,own,rs,fs,valid
        counts=footprint(adapter.model)
        try:
            with PrefillP1SingleLayerBypass(adapter.model,adapter.layers,adapter.inputs['input_ids'].shape[1],0):
                raise RuntimeError('deliberate cleanup check')
        except RuntimeError:
            pass
        assert counts==footprint(adapter.model)
        adapter.verify()
        cases.append(dict(key=identifier,input_length=adapter.inputs['input_ids'].shape[1],
            visual_feature_hashes=features,all_valid_noop_torch_equal=True,historical_native_exact=True,
            legacy_layer_replay_exact=adapter.kind=='llava',tested_layers=layers,cache_storage_disjoint=True,
            all_layer_MLP_checked=True,decode_all_native=True,normal_exception_hooks_clean=True))
        adapter.inputs,adapter.visual=None,None
        gc.collect();torch.cuda.empty_cache()
    seal(OUT/p['kind']/f'shard{shard}/smoke.json',dict(passed=True,binding=digest(p),cases=cases))
    print('SMOKE PASS',p['kind'],shard,flush=True)


def full_prefix(adapter,baseline,raw=None):
    """Only a native verification forward, not a second free generation."""
    with torch.inference_mode(),adapter.visual_replay():
        n=adapter.inputs['input_ids'].shape[1]
        out=adapter.model(**adapter.inputs,use_cache=True,past_key_values=None,return_dict=True,
            logits_to_keep=MODELS[adapter.kind]['logits_to_keep'],cache_position=torch.arange(n,device='cuda:0'))
        full=out.logits[0,-1].detach().float().cpu().clone()
    assert torch.isfinite(full).all() and int(full.argmax())==baseline['token_ids'][0]
    if raw:
        assert sha(raw['path'])==raw['sha256']
        with np.load(raw['path']) as old:
            assert torch.equal(full,torch.from_numpy(old['full_logits'])),'Historical Full first-logit mismatch'
    else:
        assert tensor_hash(full)==baseline['all_step_logits_sha256'][0]
    return full


def worker(kind,shard):
    p=verify(kind);folder=OUT/kind/f'shard{shard}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=read(OUT/kind/'gpu_preflight.json')['selected'][shard]
    h=hardware();g=next(g for g in h['gpus'] if g['uuid']==selected['uuid'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==g['uuid'] and idle(g,h)
    samples=read(OUT/kind/'sample_manifest.json')['samples']; by={key(s):s for s in samples}
    specs=read(OUT/kind/'input_bindings.json');base=read(OUT/kind/'baselines.json')
    legacy=read(OUT/kind/'legacy_generation.json');raw=read(OUT/kind/'first_token_raw_bindings.json')
    keys=read(OUT/kind/'shards.json')['keys'][shard]
    for path,expected in p['weights'].items():
        assert Path(path).stat().st_size==expected['size'] and sha(path)==expected['sha256']
    for path,expected in p['official_sources'].items():assert sha(path)==expected
    adapter=Adapter(kind)
    seal(folder/'environment.json',dict(runtime=adapter.environment,binding=digest(p),gpu=g,
        python=sys.executable,model_copies=1,logits_to_keep=MODELS[kind]['logits_to_keep']))
    smoke(adapter,samples,specs,base,legacy,p,shard)
    start_wait=time.time()
    while not all((OUT/kind/f'shard{i}/smoke.json').exists() for i in (0,1)):
        assert time.time()-start_wait < 600,'Peer smoke failed or timed out'
        assert not list((OUT/kind).glob('shard*/failure*.json'))
        time.sleep(1)
    for i in (0,1):assert read(OUT/kind/f'shard{i}/smoke.json')['binding']==digest(p)
    start=time.time();completed=0;new=0;reused=0
    for identifier in keys:
        s=by[identifier]
        paths=[checkpoint(kind,shard,identifier,c) for c in conditions(kind)]
        if all(path.exists() for path in paths):
            for c,path in zip(conditions(kind),paths):validate_checkpoint(read(path),p,s,c)
            completed+=len(paths)
            continue
        assert sha(s['image_path'])==s['image_sha256']
        adapter.prepare(s,specs[identifier])
        full=full_prefix(adapter,base[identifier],raw.get(identifier))
        for c,path in zip(conditions(kind),paths):
            if path.exists():
                validate_checkpoint(read(path),p,s,c);completed+=1;continue
            layer=layer_of(kind,c)
            if c=='Native Full' or identifier in legacy:
                generation=dict(base[identifier] if c=='Native Full' else legacy[identifier][c])
                generation.update(full_first_logits_replayed_exact=True,first_logits_sha256=tensor_hash(full) if layer is None else None)
                origin='HISTORICAL_NATIVE' if c=='Native Full' else 'HISTORICAL_480_DROP'
                reused+=1
            else:
                with adapter.visual_replay():generation,scores,_,_=generate(adapter,selected=layer)
                generation['correct']=int(generation['prediction']==s['answer'].lower())
                generation['full_token_logp_loss']=float(full.log_softmax(-1)[int(full.argmax())]-scores[0].log_softmax(-1)[int(full.argmax())])
                origin='NEW_GENERATION';new+=1
            generation['first_token_flip']=int(generation['token_ids'][0]!=base[identifier]['token_ids'][0])
            record=dict(binding=digest(p),code_commit=p['code_commit'],task=s['task'],question_id=s['question_id'],
                image_id=s['image_id'],image_sha256=s['image_sha256'],sample_sha256=digest(s),gold=s['answer'],
                condition=c,origin=origin,generation=generation,generation_sha256=digest(generation),
                gpu_uuid=g['uuid'],input_tensor_hashes=adapter.input_hashes,visual_feature_hashes=adapter.visual_hashes)
            seal(path,record);completed+=1
        adapter.verify()
        adapter.inputs,adapter.visual=None,None
        atomic(folder/'progress.json',dict(status='RUNNING',completed=completed,total=len(keys)*len(conditions(kind)),
            new_generations=new,reused=reused,elapsed_seconds=time.time()-start,last_key=identifier))
        if completed % (len(conditions(kind))*50)==0:
            print(kind,shard,completed,'/',len(keys)*len(conditions(kind)),f'{new/max(1,time.time()-start):.2f} new/sec',flush=True)
    adapter.verify() if adapter.inputs is not None else None
    del adapter
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    seal(folder/'complete.json',dict(passed=True,binding=digest(p),conditions=completed,questions=len(keys),
        new_generations=new,reused=reused,elapsed_seconds=time.time()-start,gpu_uuid=g['uuid']))
    atomic(folder/'progress.json',dict(status='COMPLETE',completed=completed,total=len(keys)*len(conditions(kind))))


def pipeline():
    assert command('git','branch','--show-current')==BRANCH
    guard=(OUT/'pipeline.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for kind in ('llava','qwen'):
        folder=OUT/kind
        if (folder/'completion.json').exists():
            assert read(folder/'completion.json')['status']=='COMPLETE' and read(folder/'validation.json')['passed']
            assert read(folder/'gpu_release.json')['own_compute_absent']
            continue
        if kind=='qwen':
            assert read(OUT/'llava/completion.json')['status']=='COMPLETE'
            assert read(OUT/'llava/validation.json')['passed'] and read(OUT/'llava/gpu_release.json')['own_compute_absent']
        h,gpus=select_gpus()
        if (folder/'launch.json').exists():
            previous=read(folder/'launch.json')
            for pid in previous['pids']:
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    continue
                raise RuntimeError('Prior worker PID still exists; refuse duplicate/reused PID')
            old_gpus=read(folder/'gpu_preflight.json')['selected']
            gpus=[next(g for g in h['gpus'] if g['uuid']==old['uuid']) for old in old_gpus]
            if not all(idle(g,h) for g in gpus):gpus=[]
        if len(gpus)<2:
            atomic(OUT/'pipeline_status.json',dict(status='STOP_QWEN_NO_TWO_IDLE_GPUS' if kind=='qwen' else 'STOP_LLAVA_NO_TWO_IDLE_GPUS',snapshot=h))
            return
        if (folder/'launch.json').exists():
            attempt=folder/'previous_attempts'/str(time.time_ns())
            for name in ['launch.json','gpu_preflight.json','gpu_release.json']+[
                f'shard{i}/{name}' for i in (0,1) for name in ('smoke.json','environment.json','complete.json','progress.json')]:
                old=folder/name
                if old.exists():
                    target=attempt/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(old),str(target))
            for old in folder.glob('shard*/failure*.json'):
                target=attempt/old.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(old),str(target))
        atomic(folder/'gpu_preflight.json',dict(selected=gpus,snapshot=h,time=time.time()))
        workers=[];streams=[]
        try:
            for shard,g in enumerate(gpus):
                now=hardware();assert idle(g,now)
                log=folder/f'shard{shard}/worker.log';log.parent.mkdir(parents=True,exist_ok=True)
                stream=log.open('a');streams.append(stream)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=g['uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                workers.append(subprocess.Popen([MODELS[kind]['python'],'-u','-m','vlm_research.p1_layerwise_accuracy.run',
                    'worker','--kind',kind,'--shard',str(shard)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT))
            atomic(folder/'launch.json',dict(time=time.time(),pids=[w.pid for w in workers],kind=kind))
            while any(w.poll() is None for w in workers):
                assert all(w.poll() in (None,0) for w in workers),'Worker failed; abort its own peer only'
                time.sleep(3)
            assert all(w.returncode==0 for w in workers)
            snapshot=hardware()
            assert not {str(w.pid) for w in workers} & {r.split(',')[1].strip() for r in snapshot['compute'].splitlines() if r}
            seal(folder/'gpu_release.json',dict(own_compute_absent=True,pids=[w.pid for w in workers],snapshot=snapshot,time=time.time()))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1')
            with (folder/'analysis.log').open('a') as log:
                subprocess.run([MODELS['llava']['python'],'-u','-m','vlm_research.p1_layerwise_accuracy.analyze',kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(folder/'completion.json')['status']=='COMPLETE' and read(folder/'validation.json')['passed']
            print('MODEL COMPLETE AND RELEASED',kind,flush=True)
        finally:
            for w in workers:
                if w.poll() is None:w.terminate()
            for w in workers:
                if w.poll() is None:w.wait(timeout=60)
            for stream in streams:stream.close()
    subprocess.run([MODELS['llava']['python'],'-m','vlm_research.p1_layerwise_accuracy.analyze','cross'],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),check=True)
    subprocess.run(['git','add','vlm_research/p1_layerwise_accuracy',str(OUT.relative_to(ROOT))],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
        'Complete sequential LLaVA/Qwen GQA3000 P-1 single-layer answer accuracy profiles'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True)
    atomic(OUT/'delivery.json',dict(status='PUSHED',commit=command('git','rev-parse','HEAD')))
    print('DELIVERED',command('git','rev-parse','HEAD'),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','pipeline','worker'])
    parser.add_argument('--kind',choices=['llava','qwen']);parser.add_argument('--shard',type=int,choices=[0,1])
    args=parser.parse_args()
    try:
        worker(args.kind,args.shard) if args.stage=='worker' else globals()[args.stage]()
    except BaseException:
        folder=OUT/args.kind/f'shard{args.shard}' if args.stage=='worker' else OUT
        atomic(folder/f'failure_{time.time_ns()}.json',dict(stage=args.stage,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
