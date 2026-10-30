import argparse
import fcntl
import inspect
import json
import os
import random
import subprocess
import time
import traceback

import numpy as np
import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6 import run as prior
from paper_rq1.run import preflight
from vlm_common import load_model, gpu_memory
from .engine import collect
from .protocol import MODEL_PATH, OLD, OUTPUT, PROTOCOL, RESOURCES, ROOT, SEED


def source_hashes():
    result=prior.source_hashes()
    for p in [*(ROOT/'gqa6_update_decomposition').glob('*.py'),ROOT/'test_gqa6_update_decomposition.py']:
        result[str(p.relative_to(ROOT))]=sha256(p)
    return result


def freeze():
    data=json.loads((OUTPUT/'data_manifest.json').read_text())
    cpu=json.loads((OUTPUT/'cpu_validation.json').read_text())
    if data['questions']!=600 or not cpu['passed'] or cpu['source_sha256']!=source_hashes():
        raise ValueError('Exact-source passing CPU tests and600 frozen samples required')
    paths=['vlm_research/'+p for p in source_hashes()]
    tracked=subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines()
    if set(tracked)!=set(paths) or subprocess.check_output(['git','diff','HEAD','--name-only','--',*paths],cwd=ROOT.parent,text=True).strip():
        raise ValueError('Commit final sources before GPU freeze')
    old=prior.verify()
    for name,record in old['model_files'].items():
        if sha256(name)!=record['sha256']:raise ValueError('Pinned model SHA256 differs')
    third_party={inspect.getfile(c):sha256(inspect.getfile(c)) for c in [LlamaDecoderLayer,LlavaForConditionalGeneration]}
    freeze_json(OUTPUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=source_hashes(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        data_sha256=sha256(OUTPUT/'data_manifest.json'),cpu_sha256=sha256(OUTPUT/'cpu_validation.json'),
        model_files=old['model_files'],third_party_sha256=third_party))


def verify():
    config=json.loads((OUTPUT/'execution_manifest.json').read_text())
    if config['source_sha256']!=source_hashes() or digest(config['protocol'])!=digest(PROTOCOL):
        raise ValueError('Frozen source/protocol changed')
    if sha256(OUTPUT/'data_manifest.json')!=config['data_sha256'] or sha256(OUTPUT/'cpu_validation.json')!=config['cpu_sha256']:
        raise ValueError('Frozen data or CPU acceptance changed')
    data=json.loads((OUTPUT/'data_manifest.json').read_text())
    for name,key in [('sample_manifest.json','sample_manifest_sha256'),('historical_bindings.json','historical_bindings_sha256')]:
        if sha256(OUTPUT/name)!=data[key]:raise ValueError('Frozen samples/historical binding changed')
    for name,h in {**data['source_sha256'],**config['third_party_sha256']}.items():
        if sha256(name)!=h:raise ValueError('Historical source or Transformers changed')
    for name,record in config['model_files'].items():
        stat=os.stat(name)
        if stat.st_size!=record['bytes'] or stat.st_mtime_ns!=record['mtime_ns']:
            raise ValueError('Pinned model artifact changed')
    return config


def samples(rank):
    path=OUTPUT/f'manifests/shard{rank}.json'
    expected=json.loads((OUTPUT/'data_manifest.json').read_text())['shard_sha256'][str(rank)]
    if sha256(path)!=expected:raise ValueError('Changed shard')
    return json.loads(path.read_text())['samples']


def save_raw(path,vectors,logits):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('xb') as f:
            np.savez_compressed(f,vectors=vectors,full_logits=logits)
            f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
    return dict(raw_path=str(path),raw_sha256=sha256(path))


def historical_check(logits,reference):
    if sha256(reference['logits_path'])!=reference['logits_sha256']:raise ValueError('Original raw logits changed')
    with np.load(reference['logits_path']) as f:old=f['full_logits'].copy()
    np.testing.assert_allclose(logits,old,atol=PROTOCOL['historical_logits_atol'],rtol=PROTOCOL['historical_logits_rtol'])
    if int(logits.argmax())!=int(old.argmax()):raise ValueError('Historical Full Top-1 differs')
    return dict(max_abs_logit_difference=float(np.abs(logits-old).max()),
                rms_logit_difference=float(np.sqrt(np.mean((logits-old)**2))),top1_exact=True)


def plain(model,inputs,query):
    with torch.inference_mode():
        out=model(**inputs,use_cache=False,output_attentions=False,return_dict=True)
    return out.logits[0,query].float().cpu().numpy()


def gate(stage):
    c=verify();binding=digest(c)
    completed=[json.loads((OUTPUT/f'{stage}/shard{i}/completion.json').read_text()) for i in range(4)]
    if any(r['status']!='complete' or r['binding']!=binding for r in completed):raise ValueError('Incomplete matching workers')
    envs=[json.loads((OUTPUT/f'{stage}/shard{i}/environment.json').read_text()) for i in range(4)]
    keys=('torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes','actual_seed','binding')
    if len({digest({k:e[k] for k in keys}) for e in envs})!=1:raise ValueError('Worker environment mismatch')
    if stage=='smoke':
        checks=[]
        for i in range(4):
            files=list((OUTPUT/f'smoke/shard{i}/checkpoints').glob('*.json'))
            if len(files)!=1:raise ValueError('Exactly one common Smoke question per GPU required')
            checks.append(json.loads(files[0].read_text()))
        if len({v['sample_sha256'] for v in checks})!=1:raise ValueError('Different cross-GPU Smoke question')
        with np.load(checks[0]['result']['raw_path']) as f:
            reference=f['full_logits'].copy();vectors=f['vectors'].astype(np.float32)
        logit_errors=[];vector_errors=[]
        for value in checks[1:]:
            with np.load(value['result']['raw_path']) as f:
                other=f['full_logits'];raw=f['vectors'].astype(np.float32)
                np.testing.assert_allclose(other,reference,atol=PROTOCOL['historical_logits_atol'],rtol=PROTOCOL['historical_logits_rtol'])
                logit_errors.append(float(np.abs(other-reference).max()));vector_errors.append(float(np.abs(raw-vectors).max()))
        freeze_json(OUTPUT/'cross_gpu_smoke.json',dict(passed=True,sample_sha256=checks[0]['sample_sha256'],
            same_run_observer_logits_exact=all(v['result']['hooks_native_logits_exact'] for v in checks),
            cross_gpu_max_logit_difference=max(logit_errors),cross_gpu_max_raw_vector_difference=max(vector_errors),
            logits_atol=PROTOCOL['historical_logits_atol'],logits_rtol=PROTOCOL['historical_logits_rtol']))
    return completed


def worker(stage,rank):
    c=verify();binding=digest(c)
    if stage=='collect':gate('smoke')
    folder=OUTPUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=preflight(rank)
    if transformers.__version__!='4.49.0':raise ValueError('Wrong Transformers')
    model,processor,env=load_model(MODEL_PATH)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    env.update(binding=binding,actual_seed=SEED,preflight=pre,code_commit=c['code_commit'])
    path=folder/'environment.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=env[k] for k in ['binding','actual_seed','model_revision','parameter_dtypes','transformers']):
            raise ValueError('Changed resume environment')
        freeze_json(folder/'attempts'/f'environment_{time.time_ns()}.json',env)
    else:freeze_json(path,env)
    versions={n:p._version for n,p in model.named_parameters()};footprint=hook_footprint(model)
    refs=json.loads((OUTPUT/'historical_bindings.json').read_text())
    rows=samples(rank) if stage=='collect' else json.loads((OUTPUT/'sample_manifest.json').read_text())['samples'][:1]
    started=time.perf_counter();completed=0;observed_forwards=0;native_checks=0
    for sample in rows:
        name=sample['task']+'__'+sample['question_id'];path=folder/'checkpoints'/f'{name}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['binding']!=binding or old['sample_sha256']!=digest(sample) or sha256(old['result']['raw_path'])!=old['result']['raw_sha256']:
                raise ValueError('Invalid existing checkpoint')
            completed+=1;continue
        try:
            inputs=prior.prepared(model,processor,sample);before=prior.tensor_hashes(inputs)
            query=sample['input_mapping']['p_minus_one']
            if stage=='smoke':native=plain(model,inputs,query);native_checks+=1
            calls=[]
            h=model.register_forward_pre_hook(lambda m,a:calls.append(1))
            try:measured,vectors,logits=collect(model,inputs,query)
            finally:h.remove()
            if calls!=[1]:raise ValueError('Collection must use exactly one Full forward')
            observed_forwards+=1
            comparison=historical_check(logits,refs[name])
            if stage=='smoke':
                after=plain(model,inputs,query);native_checks+=1
                if not np.array_equal(logits,native) or not np.array_equal(logits,after):
                    raise ValueError('Observational hooks changed native Full logits')
            if before!=prior.tensor_hashes(inputs) or hook_footprint(model)!=footprint:
                raise ValueError('Input mutation or residual hook')
            result=dict(measurements=measured,historical_comparison=comparison,
                query=query,nodes_captured=160,observed_full_forwards=1,
                hooks_native_logits_exact=True if stage=='smoke' else None,inputs_unchanged=True,
                **save_raw(RESOURCES/f'raw/{stage}/shard{rank}/{name}.npz',vectors,logits))
            prior.atomic_json(path,dict(binding=binding,sample_sha256=digest(sample),task=sample['task'],
                question_id=sample['question_id'],image_id=sample['image_id'],result=result))
            completed+=1;print(stage,rank,completed,'/',len(rows),name,flush=True)
        except BaseException as exc:
            prior.atomic_json(folder/'failures'/f'{name}_{time.time_ns()}.json',dict(error=repr(exc),
                traceback=traceback.format_exc(),question_id=sample['question_id'],skipped=False,binding=binding))
            raise
    if versions!={n:p._version for n,p in model.named_parameters()} or hook_footprint(model)!=footprint:
        raise ValueError('Changed parameter version or residual hook')
    verify()
    prior.atomic_json(folder/'completion.json',dict(status='complete',binding=binding,rank=rank,
        expected=len(rows),completed=completed,observed_forwards_this_attempt=observed_forwards,
        native_smoke_checks_this_attempt=native_checks,wall_seconds=time.perf_counter()-started,
        model_versions_unchanged=True,hooks_clean=True,memory=gpu_memory()))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','smoke','collect']);p.add_argument('--rank',type=int,choices=range(4));a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.rank is None:p.error('rank required')
    else:worker(a.stage,a.rank)


if __name__=='__main__':main()
