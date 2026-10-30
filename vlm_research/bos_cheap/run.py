import argparse
import fcntl
import inspect
import json
import os
import subprocess
import time
import traceback
import numpy as np
import torch
import transformers
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6 import run as old
from paper_rq1_single_layer_gqa6.engine import forward,metric_rows
from paper_rq1.run import preflight
from vlm_common import load_model,gpu_memory
from .protocol import ROOT,OUT,RAW,OLD,UPDATES,PROTOCOL,MODEL_PATH,SEED,HEAD_FEATURES,INPUT_FEATURES
from .engine import grams,collect,benchmark,projection_benchmark,fitted_predictor_benchmark


def sources():
    return dict(old.source_hashes(),**{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'bos_cheap').glob('*.py')},**{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'test_bos_cheap.py']})


def freeze():
    cpu=json.loads((OUT/'cpu_validation.json').read_text());data=json.loads((OUT/'data_manifest.json').read_text())
    if not cpu['passed'] or cpu['source_sha256']!=sources() or data['historical_questions']!=3000:raise ValueError('Need exact-source CPU acceptance and frozen data')
    paths=['vlm_research/'+p for p in sources()]
    tracked=subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines()
    if set(tracked)!=set(paths) or subprocess.check_output(['git','diff','HEAD','--name-only','--',*paths],cwd=ROOT.parent,text=True).strip():raise ValueError('Commit tested sources before freeze')
    prior=old.verify()
    for p,v in prior['model_files'].items():
        if sha256(p)!=v['sha256']:raise ValueError('Model checksum differs')
    freeze_json(OUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=sources(),data_sha256=sha256(OUT/'data_manifest.json'),cpu_sha256=sha256(OUT/'cpu_validation.json'),model_files=prior['model_files'],transformers_sha256={inspect.getfile(llama):sha256(inspect.getfile(llama))},code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()))


def verify():
    c=json.loads((OUT/'execution_manifest.json').read_text())
    if c['source_sha256']!=sources() or digest(c['protocol'])!=digest(PROTOCOL):raise ValueError('Frozen sources/protocol changed')
    if sha256(OUT/'data_manifest.json')!=c['data_sha256'] or sha256(OUT/'cpu_validation.json')!=c['cpu_sha256']:raise ValueError('Data/CPU changed')
    data=json.loads((OUT/'data_manifest.json').read_text())
    if sha256(OUT/'sample_manifest.json')!=data['sample_sha256']:raise ValueError('Changed sample manifest')
    for p,h in {**data['source_sha256'],**c['transformers_sha256']}.items():
        if sha256(p)!=h:raise ValueError('Changed bound source/result: '+p)
    for p,r in c['model_files'].items():
        s=os.stat(p)
        if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:raise ValueError('Changed model')
    return c


def samples(rank,part):
    p=OUT/f'manifests/shard{rank}.json';d=json.loads((OUT/'data_manifest.json').read_text())
    if sha256(p)!=d['shard_sha256'][str(rank)]:raise ValueError('Changed shard')
    return json.loads(p.read_text())[part]


def gate(stage):
    config=verify();binding=digest(config);values=[]
    for i in range(4):
        s=json.loads((OUT/f'{stage}/shard{i}/completion.json').read_text())
        if s['status']!='complete' or s['binding']!=binding:raise ValueError('Require all4 complete identical frozen workers')
        values.append(s)
    environments=[json.loads((OUT/f'{stage}/shard{i}/environment.json').read_text()) for i in range(4)]
    keys=['python','torch','transformers','cuda_runtime','gpu','model_revision','parameter_dtypes','binding','code_commit','seed']
    if len({digest({k:e[k] for k in keys}) for e in environments})!=1:raise ValueError('Different GPU environments')
    if stage=='smoke':
        checks=[json.loads(next((OUT/f'smoke/shard{i}/checkpoints').glob('*.json')).read_text()) for i in range(4)]
        errors=[]
        with np.load(checks[0]['raw_path']) as f:reference={k:f[k] for k in ['full_logits','qkv','head_features']}
        for ck in checks:
            if not ck['checks']['native_logits_exact'] or len(ck['checks']['cheap_checks'])!=32:raise ValueError('Missing real cheap-path Smoke checks')
            with np.load(ck['raw_path']) as f:
                for k,r in reference.items():np.testing.assert_allclose(f[k],r,**(PROTOCOL['historical_tolerances'] if k=='full_logits' else PROTOCOL['cheap_tolerances']))
                errors.append(float(np.abs(f['full_logits']-reference['full_logits']).max()))
        freeze_json(OUT/'cross_gpu_smoke.json',dict(passed=True,same_run_hook_logits_exact=True,max_cross_gpu_full_logit_difference=max(errors),all32_cheap_rope_qkv_checks=True,all4_hardware_environments_match=True))
    return values


def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()
    return dict(raw_path=str(path),raw_sha256=sha256(path))


def aggregate(hf,base,length):
    result=[]
    for layer in range(32):
        row=dict(layer=layer,**dict(zip(INPUT_FEATURES,[*map(float,base[layer]),length,float(np.log(length))])))
        for j,name in enumerate(HEAD_FEATURES):
            values=hf[layer,:,j]
            for stat,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:row[name+'_'+stat]=float(fn(values))
        result.append(row)
    return result


def worker(stage,rank):
    config=verify();binding=digest(config)
    if stage=='old':gate('smoke')
    if stage=='new':
        gate('old');f=json.loads((OUT/'predictor_freeze.json').read_text())
        if f['binding']!=binding or sha256(OUT/'frozen_predictors.json')!=f['models_sha256']:raise ValueError('Predictors not frozen before heldout access')
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=preflight(rank)
    if transformers.__version__!='4.49.0':raise ValueError('Wrong Transformers')
    model,processor,env=load_model(MODEL_PATH);torch.manual_seed(SEED);np.random.seed(SEED)
    env.update(binding=binding,code_commit=config['code_commit'],preflight=pre,seed=SEED)
    path=folder/'environment.json'
    if not path.exists():freeze_json(path,env)
    else:freeze_json(folder/f'attempt_{time.time_ns()}.json',env)
    footprint=hook_footprint(model);original=llama.eager_attention_forward;versions={n:p._version for n,p in model.named_parameters()}
    gram_start=time.perf_counter();gram=grams(model)
    freeze_json(folder/f'gram_setup_{time.time_ns()}.json',dict(seconds=time.perf_counter()-gram_start,bytes=sum(x.numel()*x.element_size() for x in gram),purpose='Offline projected-Value norm Gram matrices, no fixed BOS K/V anchor assumption.'))
    all_samples=json.loads((OUT/'sample_manifest.json').read_text())
    selected=all_samples['old_samples'][:1] if stage=='smoke' else samples(rank,stage)
    start=time.perf_counter();n=0
    for s in selected:
        name=s['task']+'__'+s['question_id'];path=folder/'checkpoints'/f'{name}.json'
        if path.exists():
            ck=json.loads(path.read_text())
            if ck['binding']!=binding or ck['sample_sha256']!=digest(s) or sha256(ck['raw_path'])!=ck['raw_sha256']:raise ValueError('Invalid resume checkpoint')
            n+=1;continue
        try:
            inputs=old.prepared(model,processor,s);inputs_hash=old.tensor_hashes(inputs);query=s['input_mapping']['p_minus_one']
            if stage=='smoke':native,_,_=forward(model,inputs,query)
            arrays,hf,base,trace=collect(model,inputs,s['input_mapping'],gram,check=stage=='smoke')
            logits=arrays['full_logits'];historical=None
            if stage in ['old','smoke']:
                ref=json.loads((OLD/f"first_token/shard{s['rank']}/checkpoints/{name}.json").read_text())
                if ref['sample_sha256']!=digest(s):raise ValueError('Historical input/sample mismatch')
                reference=ref['result']
                if sha256(reference['logits_path'])!=reference['logits_sha256']:raise ValueError('Changed historical raw logits')
                with np.load(reference['logits_path']) as f:prior=f['full_logits']
                np.testing.assert_allclose(logits,prior,**PROTOCOL['historical_tolerances'])
                historical=float(np.abs(logits-prior).max())
                prior_bos=np.array([r['bos_mass_per_head'] for r in ref['result']['bos']],dtype=np.float32)
                np.testing.assert_allclose(arrays['bos_mass'],prior_bos,atol=.0005,rtol=.005)
            checks=dict(inputs_unchanged=inputs_hash==old.tensor_hashes(inputs),hooks_clean=hook_footprint(model)==footprint and llama.eager_attention_forward is original)
            if not all(checks.values()):raise ValueError('Input mutation or leaked observer')
            if stage=='smoke':
                after,_,_=forward(model,inputs,query)
                if not np.array_equal(logits,native.numpy()) or not np.array_equal(logits,after.numpy()):raise ValueError('Observer changes logits')
                checks.update(native_logits_exact=True,cheap_checks=trace.cheap_checks,cost=benchmark(*trace.bench)+projection_benchmark(trace.projection_bench))
            metrics=None
            if stage=='new':
                changed=[]
                for layer in range(32):
                    value,_,_=forward(model,inputs,query,layer=layer);changed.append(value.numpy())
                arrays['ablated_logits']=np.stack(changed)
                metrics=[dict(layer=i,**r) for i,r in enumerate(metric_rows(logits,arrays['ablated_logits']))]
                if inputs_hash!=old.tensor_hashes(inputs) or hook_footprint(model)!=footprint:raise ValueError('Input mutation/hook leakage after new single-layer labels')
            arrays['head_features']=hf;arrays['input_features']=base
            result=dict(binding=binding,sample_sha256=digest(s),task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],features=aggregate(hf,base,len(s['input_mapping']['prompt_ids'])),checks=checks,historical_full_max_difference=historical,metrics=metrics,**save_npz(RAW/f'raw/{stage}/shard{rank}/{name}.npz',arrays))
            old.atomic_json(path,result);n+=1;print(stage,rank,n,'/',len(selected),name,flush=True)
        except BaseException as exc:
            old.atomic_json(folder/f'failures/{name}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),binding=binding,skipped=False));raise
    if stage=='new':
        models=json.loads((OUT/'frozen_predictors.json').read_text())['eps']['0.03']['selected']
        rule=next(r for r in models if r['name']=='BOS-QK+Residual')
        first=json.loads((folder/'checkpoints'/f"{selected[0]['task']}__{selected[0]['question_id']}.json").read_text())['features'][0]
        freeze_json(folder/f'actual_predictor_cost_{time.time_ns()}.json',fitted_predictor_benchmark(rule,first))
    if footprint!=hook_footprint(model) or llama.eager_attention_forward is not original or versions!={n:p._version for n,p in model.named_parameters()}:raise ValueError('Model/source/hook mutation')
    verify();old.atomic_json(folder/'completion.json',dict(status='complete',rank=rank,binding=binding,completed=n,expected=len(selected),wall_seconds=time.perf_counter()-start,weights_versions_unchanged=True,hooks_clean=True,memory=gpu_memory()))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','smoke','old','new']);p.add_argument('--rank',type=int);a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:worker(a.stage,a.rank)


if __name__=='__main__':main()
