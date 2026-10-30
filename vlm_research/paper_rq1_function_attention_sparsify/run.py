import argparse
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
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from paper_rq1.run import prepared,preflight
from paper_rq1.engine import step_inputs,cache_fingerprint
from paper_rq1_token_role_decode.engine import generate
from paper_rq1_token_role_decode.run import raw_save
from paper_rq1_single_layer_gqa6.run import atomic_json,tensor_hashes
from vlm_common import load_model,gpu_memory
from .protocol import ROOT,OUT,RAW,PROTOCOL,MODEL_PATH,LAYERS,CONDITIONS,OLD,GPUS
from .engine import step,partition,retained,metrics


def sources():
    names=['protocol.py','engine.py','data.py','run.py','__init__.py']
    result={str(ROOT/'paper_rq1_function_attention_sparsify'/n):sha256(ROOT/'paper_rq1_function_attention_sparsify'/n) for n in names}
    for n in ['test_function_attention_sparsify.py','paper_rq1_token_role_decode/engine.py',
              'paper_rq1_token_role_decode/labels.py','paper_rq1_late_attention_sources/engine.py',
              'paper_rq1/engine.py','paper_rq1/run.py','vlm_common.py']:
        result[str(ROOT/n)]=sha256(ROOT/n)
    result[inspect.getfile(llama)]=sha256(inspect.getfile(llama))
    return result


def freeze():
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_function_attention_sparsify'])
    with (OUT/'cpu_tests.txt').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('CPU tests failed')
    model=json.loads((OLD[1]/'execution_manifest.json').read_text())['model_files']
    for path,b in model.items():
        if sha256(path)!=b['sha256']:raise ValueError('Model artifact changed')
    if subprocess.check_output(['git','diff','HEAD','--name-only','--','vlm_research/paper_rq1_function_attention_sparsify','vlm_research/test_function_attention_sparsify.py'],cwd=ROOT.parent,text=True).strip():
        raise ValueError('Commit collector before GPU freeze')
    freeze_json(OUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=sources(),model_files=model,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        data_sha256=sha256(OUT/'sample_manifest.json'),cpu_tests=result.testsRun,cpu_passed=True,cpu_log_sha256=sha256(OUT/'cpu_tests.txt')))


def verify():
    r=json.loads((OUT/'execution_manifest.json').read_text())
    if r['source_sha256']!=sources() or r['protocol']!=json.loads(json.dumps(PROTOCOL)):raise ValueError('Frozen collector changed')
    if sha256(OUT/'sample_manifest.json')!=r['data_sha256']:raise ValueError('Frozen input changed')
    for p,b in r['model_files'].items():
        s=Path(p).stat()
        if s.st_mtime_ns!=b['mtime_ns'] or s.st_size!=b['bytes']:raise ValueError('Model artifact changed')
    return r


def smoke(model,processor,sample):
    prompt=prepared(model,processor,sample);before=tensor_hashes(prompt)
    generation,raw=generate(model,prompt)
    with torch.inference_mode():native=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True,eos_token_id=2)
    if native[0,prompt['input_ids'].shape[1]:].tolist()!=generation['token_ids']:raise AssertionError('Cached generation mismatch')
    history=None;prefix=[];checks=[]
    for t,token in enumerate(generation['token_ids']):
        inputs=step_inputs(prompt,prefix);logits,future,_=step(model,inputs,history)
        np.testing.assert_array_equal(logits.numpy(),raw[t])
        if t in [1,4]:
            groups=partition(prompt,t);fingerprint=cache_fingerprint(history)
            observed,_,trace=step(model,inputs,history,hidden=True)
            np.testing.assert_array_equal(logits.numpy(),observed.numpy())
            if trace.hidden.shape!=(4096,):raise AssertionError('Wrong pre-attention hidden')
            for layer in LAYERS:
                for algorithm,mode in CONDITIONS:
                    noop,_,_=step(model,inputs,history,groups=groups,algorithm=algorithm,mode=mode,layer=layer,noop=True)
                    np.testing.assert_array_equal(logits.numpy(),noop.numpy())
                    changed,_,tr=step(model,inputs,history,groups=groups,algorithm=algorithm,mode=mode,layer=layer,audit=True)
                    if cache_fingerprint(history)!=fingerprint:raise AssertionError('Smoke parent cache changed')
                    checks.append(dict(t=t,algorithm=algorithm,mode=mode,layer=layer,noop_exact=True,cache_exact=True,audits=tr.checks,**metrics(logits,changed)))
        prefix.append(token);history=future
    if {r['t'] for r in checks}!={1,4}:raise AssertionError('Smoke missing decode states')
    if tensor_hashes(prompt)!=before:raise AssertionError('Input mutation')
    return dict(passed=True,native_generate_exact=True,checks=checks),{}


def collect(model,processor,sample):
    if sha256(sample['old_checkpoint'])!=sample['old_sha256']:raise ValueError('Historical checkpoint changed')
    old=json.loads(Path(sample['old_checkpoint']).read_text())
    if sha256(old['raw_path'])!=old['raw_sha256']:raise ValueError('Historical raw changed')
    with np.load(old['raw_path']) as f:expected=f['full_logits']
    prompt=prepared(model,processor,sample);input_hash=tensor_hashes(prompt)
    targets={r['generation_step']:r for r in sample['selected']}
    history=None;prefix=[];rows=[];logits_raw=[];hidden=[];full=[];audits=[];historical_max=0.
    for t,token in enumerate(old['token_ids']):
        inputs=step_inputs(prompt,prefix);base,future,tr=step(model,inputs,history,hidden=True)
        historical_max=max(historical_max,float((base-torch.from_numpy(expected[t])).abs().max()))
        if int(base.argmax())!=token:raise AssertionError('Historical native trajectory differs; cannot silently relabel frozen targets')
        full.append(base.numpy());hidden.append(tr.hidden.numpy())
        if t in targets:
            target=targets[t];groups=partition(prompt,t);fingerprint=cache_fingerprint(history)
            if int(inputs['input_ids'][0,0])!=prefix[-1] or history.get_seq_length()!=prompt['input_ids'].shape[1]+t-1:raise AssertionError('Wrong Decode state')
            for layer in LAYERS:
                for algorithm,mode in CONDITIONS:
                    changed,_,record=step(model,inputs,history,groups=groups,algorithm=algorithm,mode=mode,layer=layer)
                    keep=0 if algorithm=='zero' else len(retained(groups,mode))
                    rows.append(dict(image_id=sample['image_identity'],image_sha256=sample['image_sha256'],cohort=sample['cohort'],split=sample['split'],
                        t=t,word=target['word'],role=target['word_type'],layer=layer,algorithm=algorithm,mode=mode,
                        full_keys=sum(len(groups[n]) for n in ['BOS','IMAGE','PROMPT_TEXT','GENERATED_HISTORY']),
                        retained_keys=keep,**metrics(base,changed)))
                    logits_raw.append(changed.numpy())
            if cache_fingerprint(history)!=fingerprint:raise AssertionError('Formal native cache contaminated')
            audits.append(dict(t=t,groups=groups,parent_cache_sha256=fingerprint,cache_unchanged=True))
        prefix.append(token);history=future
    if tensor_hashes(prompt)!=input_hash or len(rows)!=len(targets)*21:raise AssertionError('Incomplete/mutated collection')
    return dict(rows=rows,audits=audits,token_ids=old['token_ids'],caption=old['caption'],historical_max_logit_difference=historical_max,
        historical_token_sequence_exact=True),dict(full_logits=np.stack(full),h31_in=np.stack(hidden),changed_logits=np.stack(logits_raw))


def worker(stage,rank):
    config=verify();binding=digest(config);physical=rank+1
    if stage=='stage_a':
        for i in range(3):
            r=json.loads((OUT/f'smoke/shard{i}/completion.json').read_text())
            if r['binding']!=binding or r['status']!='complete':raise ValueError('All three smokes required')
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'resource_check.json',preflight(physical))
    model,processor,environment=load_model(MODEL_PATH);atomic_json(folder/'environment.json',environment)
    hooks=hook_footprint(model);original=llama.eager_attention_forward;versions={n:p._version for n,p in model.named_parameters()}
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    samples=[manifest['smoke']] if stage=='smoke' else [s for s in manifest['development'] if s['rank']==rank]
    started=time.perf_counter()
    for i,sample in enumerate(samples):
        name=sample['image_identity'].replace(':','_');path=folder/f'checkpoints/{name}.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['binding']!=binding or r['sample_sha256']!=digest(sample) or sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Invalid checkpoint')
            continue
        try:
            result,raw=smoke(model,processor,sample) if stage=='smoke' else collect(model,processor,sample)
            result.update(raw_save(RAW/f'{stage}/shard{rank}/{name}.npz',**raw))
            if hook_footprint(model)!=hooks or llama.eager_attention_forward is not original:raise AssertionError('Hook/global wrapper leak')
            result.update(binding=binding,sample_sha256=digest(sample),image_id=sample['image_identity'])
            atomic_json(path,result);print(stage,physical,i+1,'/',len(samples),name,flush=True)
        except BaseException as e:
            atomic_json(folder/f'failure_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),binding=binding));raise
    if versions!={n:p._version for n,p in model.named_parameters()}:raise AssertionError('Weights changed')
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,images=len(samples),rank=rank,physical_gpu=physical,
        seconds=time.perf_counter()-started,hooks_clean=True,weights_unchanged=True,memory=gpu_memory()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','smoke','stage_a']);parser.add_argument('--rank',type=int,choices=range(3),default=0)
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:worker(args.stage,args.rank)


if __name__=='__main__':main()
