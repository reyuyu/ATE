"""Frozen acceptance gates, per-condition checkpoints and paired GPU workers."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest
import numpy as np
import torch
import transformers
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.run import prepared
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import atomic_json,tensor_hashes
from vlm_common import load_model,gpu_memory
from .engine import JointAttention,generate
from .protocol import ROOT,OUT,RAW,GPUS,MODEL_PATH,PROTOCOL,CONDITIONS


def sources():
    names=['paper_rq1_early_joint_attention/'+n+'.py' for n in ['__init__','protocol','data','engine','run','launch']]
    names+=['test_early_joint_attention.py','vlm_common.py','knockout_probe.py','paper_rq1/engine.py',
        'paper_rq1/run.py','paper_rq1/scorers.py','paper_rq1/protocol.py','paper_rq1_single_layer_gqa6/run.py',
        'paper_rq1_single_layer_gqa6/protocol.py','paper_rq1_single_layer_gqa6/data.py',
        'paper_rq1_decode_layerwise_quality/engine.py','paper_rq1_decode_layerwise_quality/statistics.py',
        'benchmarks/protocols.py','benchmarks/dataset_registry.py','benchmarks/smoke.py','benchmarks/scoring.py']
    return {n:sha256(ROOT/n) for n in names}


def cpu():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'execution_manifest.json').exists():raise ValueError('Acceptance already frozen')
    (OUT/'attempts').mkdir(exist_ok=True)
    for name in ['cpu_tests.txt','cpu_validation.json']:
        path=OUT/name
        if path.exists():path.rename(OUT/'attempts'/f'{time.time_ns()}_{name}')
    with (OUT/'cpu_tests.txt').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_early_joint_attention'))
    if not result.wasSuccessful():raise AssertionError('CPU tests failed; inspect preserved log')
    freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source_sha256=sources(),log_sha256=sha256(OUT/'cpu_tests.txt')))
    print('CPU passed',result.testsRun,flush=True)


def freeze():
    acceptance=json.loads((OUT/'cpu_validation.json').read_text())
    if not acceptance['passed'] or acceptance['source_sha256']!=sources():raise ValueError('Exact-source CPU tests required')
    names=['vlm_research/'+s for s in sources()]
    tracked=subprocess.check_output(['git','ls-files','--',*names],cwd=ROOT.parent,text=True).splitlines()
    if set(tracked)!=set(names) or subprocess.check_output(['git','diff','HEAD','--name-only','--',*names],cwd=ROOT.parent,text=True).strip():
        raise ValueError('Commit exact worker sources before freeze')
    parent=json.loads((ROOT/'results/paper_rq1_decode_layerwise_quality/execution_manifest.json').read_text())
    for path,b in parent['model_files'].items():
        if sha256(path)!=b['sha256']:raise ValueError('Pinned model changed')
    freeze_json(OUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=sources(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        model_files=parent['model_files'],bindings={n:sha256(OUT/n) for n in ['experiment_config.json','sample_manifest.json','selection_audit.json','cpu_validation.json']},
        transformers_source={inspect.getfile(llama):sha256(inspect.getfile(llama))}))


def verify():
    config=json.loads((OUT/'execution_manifest.json').read_text())
    if config['source_sha256']!=sources() or config['protocol']!=PROTOCOL:raise ValueError('Frozen code/protocol changed')
    for p,h in config['bindings'].items():
        if sha256(OUT/p)!=h:raise ValueError('Frozen acceptance/data changed')
    for p,h in config['transformers_source'].items():
        if sha256(p)!=h:raise ValueError('Third-party code changed')
    for p,b in config['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=b['bytes'] or st.st_mtime_ns!=b['mtime_ns']:raise ValueError('Frozen model artifact changed')
    for p,h in json.loads((OUT/'sample_manifest.json').read_text())['scorer_bindings'].items():
        if sha256(p)!=h:raise ValueError('Official scorer changed')
    return config


def checkpoint_valid(path,binding,sample,condition):
    if not path.exists():return False
    row=json.loads(path.read_text())
    if row['binding']!=binding or row['sample_sha256']!=digest(sample) or row['condition']!=condition:
        raise ValueError('Checkpoint binding mismatch')
    if row['result_sha256']!=digest(row['generation']):raise ValueError('Generation checkpoint corrupted')
    return True


def preflight(rank):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPUS[rank]:raise ValueError('Wrong frozen GPU UUID')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if any(line.startswith(GPUS[rank]) for line in processes.splitlines()):raise ValueError('GPU occupied; do not interrupt task')
    free=subprocess.check_output(['nvidia-smi',f'--id={GPUS[rank]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.strip())<18000:raise ValueError('Insufficient free memory')
    return dict(uuid=GPUS[rank],memory_free_mib=int(free.strip()),all_compute_processes=processes)


def smoke(model,processor,samples,rank):
    checks=[];raw={};footprint=hook_footprint(model)
    for sample in samples:
        prompt=prepared(model,processor,sample);n=prompt['input_ids'].shape[1];inputs=tensor_hashes(prompt)
        baseline,full_logits=generate(model,prompt,CONDITIONS['S0'],collect=True,audit=True)
        with torch.inference_mode():
            reference=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,n:].tolist()
        if baseline['token_ids']!=reference:raise AssertionError('Full cached generation differs from native generate')
        raw[sample['question_id']]=torch.stack(full_logits).numpy()
        records=[]
        for key,condition in CONDITIONS.items():
            noop,noop_logits=generate(model,prompt,condition,noop=True,collect=True)
            if noop['token_ids']!=baseline['token_ids'] or len(noop_logits)!=len(full_logits) or any(not torch.equal(a,b) for a,b in zip(noop_logits,full_logits)):
                raise AssertionError('Noop effective logits are not exact')
            counts=dict(mlp=[0]*32,attention=[0]*32);handles=[]
            try:
                for k,block in enumerate(model_layers(model)):
                    def observe_mlp(module,args,out,k=k):
                        counts['mlp'][k]+=1
                        if not torch.isfinite(out).all():raise AssertionError('Nonfinite MLP')
                    def observe_attention(module,args,out,k=k):counts['attention'][k]+=1
                    handles.append(block.mlp.register_forward_hook(observe_mlp))
                    handles.append(block.self_attn.o_proj.register_forward_hook(observe_attention))
                changed,changed_logits=generate(model,prompt,condition,collect=True,audit=True)
            finally:
                for h in handles:h.remove()
            if any(c!=changed['length'] for kind in counts.values() for c in kind):raise AssertionError('Attention/MLP skipped unexpectedly')
            # Independent HF driver checks prefill plus evolving own-prefix cache.
            with JointAttention(model,condition['layers'],condition['mode'],n):
                with torch.inference_mode():
                    direct=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,n:].tolist()
            if direct!=changed['token_ids']:raise AssertionError('Intervened own-prefix driver differs from HF generation')
            if condition['mode']=='Decode-only':
                if not torch.equal(changed_logits[0],full_logits[0]) or changed['initial_cache_sha256']!=baseline['initial_cache_sha256']:
                    raise AssertionError('Decode-only prefill changed')
            events=changed['events'];layers=set(condition['layers'])
            if len(events)!=len(layers)*changed['length'] or {e['layer'] for e in events}!=layers:raise AssertionError('Wrong hook layers/calls')
            if any(e['decode']!=(j>=len(layers)) for j,e in enumerate(events)) and layers:
                raise AssertionError('Wrong Prefill/Decode event sequence')
            for e in events:
                expected=(condition['mode']=='Combined' or condition['mode']=='Decode-only' and e['decode']
                    or condition['mode'] in ('P-1-only','Prefill-all-only') and not e['decode'])
                if e['active']!=expected:raise AssertionError('Wrong temporal activation')
            if hook_footprint(model)!=footprint or tensor_hashes(prompt)!=inputs:raise AssertionError('Input change/hook leak')
            records.append(dict(condition=key,noop_all_effective_logits_exact=True,all32_mlp_and_attention_calls=counts,
                generation=changed,hf_token_ids_exact=True))
        checks.append(dict(question_id=sample['question_id'],full=baseline,conditions=records,native_token_ids_exact=True))
    try:
        with JointAttention(model,[0,1],'Combined',n):raise RuntimeError('deliberate cleanup probe')
    except RuntimeError:pass
    if hook_footprint(model)!=footprint:raise AssertionError('Exception hook leak')
    path=RAW/f'smoke/shard{rank}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**raw)
    return dict(passed=True,checks=checks,exception_cleanup=True,raw_path=str(path),raw_sha256=sha256(path))


def gate():
    checks=[];arrays=[];environments=[]
    for rank in range(3):
        value=json.loads((OUT/f'smoke/shard{rank}/validation.json').read_text())
        completion=json.loads((OUT/f'smoke/shard{rank}/completion.json').read_text())
        if not value['passed'] or completion['status']!='complete' or completion['binding']!=digest(verify()):raise ValueError('GPU Smoke acceptance incomplete')
        if sha256(value['raw_path'])!=value['raw_sha256']:raise ValueError('Smoke raw changed')
        checks.append(value)
        with np.load(value['raw_path']) as f:arrays.append({k:f[k] for k in f.files})
        environments.append(json.loads(sorted((OUT/f'smoke/shard{rank}').glob('environment_*.json'))[-1].read_text()))
    env_keys=['python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes','binding','code_commit']
    if len({digest({k:e[k] for k in env_keys}) for e in environments})!=1:raise ValueError('Worker environments differ')
    differences=[]
    for rank in [1,2]:
        for qid in arrays[0]:
            a,b=arrays[0][qid],arrays[rank][qid]
            if a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('Cross-GPU Full logits not exact; do not relax tolerance')
            for first,other in zip(checks[0]['checks'],checks[rank]['checks']):
                if [c['generation']['token_ids'] for c in first['conditions']]!=[c['generation']['token_ids'] for c in other['conditions']]:
                    raise ValueError('Cross-GPU changed tokens differ')
            differences.append(dict(rank=rank,question_id=qid,max_abs_logit_diff=0.))
    value=dict(passed=True,workers=3,cross_gpu=differences,checks=checks)
    if not (OUT/'gpu_smoke_validation.json').exists():freeze_json(OUT/'gpu_smoke_validation.json',value)
    return value


def worker(stage,rank):
    config=verify();binding=digest(config)
    if stage=='generation':gate()
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        old=json.loads((folder/'completion.json').read_text())
        if old['binding']!=binding or old['status']!='complete':raise ValueError('Completed stage binding changed')
        if stage=='generation':
            samples=json.loads((OUT/'sample_manifest.json').read_text())['samples']
            for s in (s for s in samples if s['rank']==rank):
                for key in CONDITIONS:
                    if not checkpoint_valid(folder/'checkpoints'/f"{s['task']}__{s['question_id']}"/f'{key}.json',binding,s,key):
                        raise ValueError('Completed worker missing result')
        print('Existing completed worker validated',stage,rank,flush=True);return
    pre=preflight(rank);model,processor,environment=load_model(MODEL_PATH)
    if transformers.__version__!='4.49.0':raise ValueError('Pinned Transformers required')
    environment.update(preflight=pre,binding=binding,code_commit=config['code_commit'],physical_gpu=rank+1)
    freeze_json(folder/f'environment_{time.time_ns()}.json',environment)
    version={k:p._version for k,p in model.named_parameters()};hooks=hook_footprint(model)
    manifest=json.loads((OUT/'sample_manifest.json').read_text());started=time.perf_counter();completed=0
    if stage=='smoke':
        atomic_json(folder/'validation.json',smoke(model,processor,manifest['smoke'],rank));completed=2
    else:
        selected=[s for s in manifest['samples'] if s['rank']==rank]
        for i,sample in enumerate(selected):
            path=folder/'checkpoints'/f"{sample['task']}__{sample['question_id']}"
            prompt=None;input_hash=None
            for key,condition in CONDITIONS.items():
                file=path/f'{key}.json'
                if checkpoint_valid(file,binding,sample,key):completed+=1;continue
                if prompt is None:prompt=prepared(model,processor,sample);input_hash=tensor_hashes(prompt)
                start=time.perf_counter();generation,_=generate(model,prompt,condition)
                text=processor.tokenizer.decode(generation['token_ids'],skip_special_tokens=True).strip()
                generation.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==sample['answer']))
                record=dict(binding=binding,sample_sha256=digest(sample),condition=key,task=sample['task'],question_id=sample['question_id'],
                    image_id=sample['image_id'],generation=generation,result_sha256=digest(generation),seconds=time.perf_counter()-start)
                atomic_json(file,record);completed+=1
            if hook_footprint(model)!=hooks or prompt is not None and tensor_hashes(prompt)!=input_hash:raise AssertionError('Input mutation or hook leak')
            if version!={k:p._version for k,p in model.named_parameters()}:raise AssertionError('Parameter mutation')
            if (i+1)%10==0 or i+1==len(selected):
                progress=dict(status='running',questions_done=i+1,questions_total=len(selected),conditions_done=completed,
                    seconds=time.perf_counter()-started,binding=binding)
                freeze_json(folder/'progress'/f'{i+1}_{time.time_ns()}.json',progress);print(json.dumps(progress),flush=True)
    if version!={k:p._version for k,p in model.named_parameters()} or hook_footprint(model)!=hooks:raise AssertionError('Model changed')
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,code_commit=config['code_commit'],
        completed=completed,seconds=time.perf_counter()-started,memory=gpu_memory(),model_parameter_versions_unchanged=True,hooks_clean=True))
    print(stage,'complete',completed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','smoke','generation','gate']);parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    try:
        if args.stage=='cpu':cpu()
        elif args.stage=='freeze':freeze()
        elif args.stage=='gate':gate()
        else:worker(args.stage,args.rank)
    except BaseException as exc:
        if args.stage in ('smoke','generation'):
            freeze_json(OUT/f'failures/{args.stage}_shard{args.rank}_{time.time_ns()}.json',
                dict(error=repr(exc),traceback=traceback.format_exc(),stage=args.stage,rank=args.rank))
        raise
