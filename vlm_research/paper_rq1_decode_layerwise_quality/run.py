"""Frozen-code CPU acceptance, per-GPU Smoke and atomic per-condition generation."""
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
from paper_rq1.engine import cache_fingerprint
from paper_rq1.run import prepared
from paper_rq1_single_layer_gqa6.run import atomic_json,tensor_hashes
from vlm_common import load_model,gpu_memory
from .protocol import ROOT,OUT,RAW,GPUS,MODEL_PATH,PROTOCOL,CONDITIONS,condition_layer,limit
from .engine import DecodeLayer,prefill,generate


def sources():
    names=['paper_rq1_decode_layerwise_quality/'+n+'.py' for n in ['__init__','protocol','data','engine','statistics','run','launch']]
    names+=['test_decode_layerwise_quality.py','vlm_common.py','knockout_probe.py','paper_rq1/engine.py','paper_rq1/run.py',
        'paper_rq1_single_layer_gqa6/run.py','benchmarks/protocols.py','benchmarks/dataset_registry.py','benchmarks/smoke.py',
        'paper_rq1/scorers.py','benchmarks/scoring.py','benchmarks/_vendor/m4c_evaluator.py']
    return {p:sha256(ROOT/p) for p in names}


def cpu():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'cpu_tests.txt').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_decode_layerwise_quality'))
    if not result.wasSuccessful():raise ValueError('CPU validation failed')
    freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source_sha256=sources(),log_sha256=sha256(OUT/'cpu_tests.txt')))
    print('CPU tests passed',result.testsRun,flush=True)


def freeze():
    cpu=json.loads((OUT/'cpu_validation.json').read_text())
    if not cpu['passed'] or cpu['source_sha256']!=sources():raise ValueError('Exact-source CPU validation required')
    paths=['vlm_research/'+s for s in sources()]
    tracked=subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines()
    dirty=subprocess.check_output(['git','diff','HEAD','--name-only','--',*paths],cwd=ROOT.parent,text=True).strip()
    if set(tracked)!=set(paths) or dirty:raise ValueError('Commit exact worker sources first')
    parent=json.loads((ROOT/'results/paper_rq1_function_attention_sparsify/execution_manifest.json').read_text())
    for path,binding in parent['model_files'].items():
        if sha256(path)!=binding['sha256']:raise ValueError('Model artifact changed')
    freeze_json(OUT/'execution_manifest.json',dict(protocol=PROTOCOL,source_sha256=sources(),model_files=parent['model_files'],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        sample_sha256=sha256(OUT/'sample_manifest.json'),cpu_sha256=sha256(OUT/'cpu_validation.json'),
        transformers_source={inspect.getfile(llama):sha256(inspect.getfile(llama))}))


def verify():
    config=json.loads((OUT/'execution_manifest.json').read_text())
    if config['source_sha256']!=sources() or config['protocol']!=PROTOCOL:raise ValueError('Frozen code/config changed')
    for field,name in [('sample_sha256','sample_manifest.json'),('cpu_sha256','cpu_validation.json')]:
        if sha256(OUT/name)!=config[field]:raise ValueError('Frozen input changed')
    for p,h in config['transformers_source'].items():
        if sha256(p)!=h:raise ValueError('Third-party code changed')
    for p,b in config['model_files'].items():
        st=Path(p).stat()
        if st.st_size!=b['bytes'] or st.st_mtime_ns!=b['mtime_ns']:raise ValueError('Model changed')
    return config


def preflight(rank):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPUS[rank]:raise ValueError('Expose only frozen GPU UUID')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    if any(line.startswith(GPUS[rank]) for line in processes.splitlines()):raise ValueError('GPU occupied; will not interrupt existing task')
    return dict(gpu_uuid=GPUS[rank],other_processes=processes)


def checkpoint_valid(path,binding,sample,condition):
    if not path.exists():return False
    row=json.loads(path.read_text())
    if row['binding']!=binding or row['sample_sha256']!=digest(sample) or row['condition']!=condition:raise ValueError('Checkpoint binding mismatch')
    if row['result_sha256']!=digest(row['generation']):raise ValueError('Generation checkpoint corrupted')
    return True


def cache_versions(cache):
    # Inference tensors intentionally have no PyTorch version counter; full parent
    # bytes are SHA-checked before/after all conditions, independently of metadata.
    return [(t.data_ptr(),tuple(t.shape)) for pair in cache.to_legacy_cache() for t in pair]


def smoke(model,processor,samples,rank):
    checks=[];raw={};footprint=hook_footprint(model)
    for sample in samples:
        prompt=prepared(model,processor,sample);n=prompt['input_ids'].shape[1];inputs=tensor_hashes(prompt)
        first,parent=prefill(model,prompt);parent_sha=cache_fingerprint(parent)
        with torch.inference_mode():reference=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=limit(sample),use_cache=True,eos_token_id=2)[0,n:].tolist()
        base,base_logits=generate(model,prompt,limit(sample),parent=parent,first=first,collect=True)
        if base['token_ids']!=reference:raise AssertionError('Full differs from native generate')
        comparisons=[]
        for layer in range(32):
            with DecodeLayer(model,layer,n) as pretrace:new_first,new_cache=prefill(model,prompt)
            if pretrace.edits or not torch.equal(first,new_first) or cache_fingerprint(new_cache)!=parent_sha:raise AssertionError('Decode-only affected Prefill')
            del new_cache
            noop,no_logits=generate(model,prompt,limit(sample),layer,parent,first,noop=True,collect=True)
            if noop['token_ids']!=reference or len(no_logits)!=len(base_logits) or any(not torch.equal(a,b) for a,b in zip(base_logits,no_logits)):raise AssertionError('No-op logits not exact')
            counts={'attention':[0]*32,'mlp':[0]*32,'zero':0};handles=[]
            # Installed after the intervention hook via a wrapper of the real o_proj output.
            with DecodeLayer(model,layer,n) as trace:
                try:
                    for k,block in enumerate(model_layers(model)):
                        def observed(module,args,out,k=k):
                            counts['attention'][k]+=1
                            if k==layer:
                                if not torch.isfinite(out).all() or torch.count_nonzero(out).item()!=0:raise AssertionError('Target o_proj not zero')
                                counts['zero']+=1
                        def mlp(module,args,out,k=k):
                            counts['mlp'][k]+=1
                            if not torch.isfinite(out).all():raise AssertionError('Nonfinite MLP')
                        handles.append(block.self_attn.o_proj.register_forward_hook(observed));handles.append(block.mlp.register_forward_hook(mlp))
                    history=__import__('paper_rq1.engine',fromlist=['clone_cache']).clone_cache(parent);tokens=[int(first.argmax())]
                    from paper_rq1.engine import step_inputs
                    with torch.inference_mode():
                        while len(tokens)<min(limit(sample),8) and tokens[-1]!=2:
                            out=model(**step_inputs(prompt,tokens),past_key_values=history,use_cache=True,return_dict=True)
                            history=out.past_key_values
                            if not torch.isfinite(out.logits).all():raise AssertionError('Nonfinite Smoke logits')
                            tokens.append(int(out.logits[0,-1].argmax()))
                finally:
                    for h in handles:h.remove()
            if any(v!=len(tokens)-1 for key in ['attention','mlp'] for v in counts[key]) or counts['zero']!=len(tokens)-1:raise AssertionError('Wrong target/MLP/layer execution')
            # Same condition with its own newly computed Prefill must reproduce shared-Prefill tokens/logits.
            if layer in [0,14,31]:
                shared,ls=generate(model,prompt,limit(sample),layer,parent,first,collect=True)
                own,lo=generate(model,prompt,limit(sample),layer,collect=True)
                if shared['token_ids']!=own['token_ids'] or any(not torch.equal(a,b) for a,b in zip(ls,lo)):raise AssertionError('Shared native Prefill not equivalent')
            if cache_fingerprint(parent)!=parent_sha or hook_footprint(model)!=footprint:raise AssertionError('Cache contamination or hook leak')
            comparisons.append(dict(layer=layer,noop_all_logits_exact=True,prefill_cache_logits_exact=True,all_blocks_and_mlp_executed=True,zero_outputs=counts['zero']))
        if tensor_hashes(prompt)!=inputs:raise AssertionError('Inputs changed')
        raw[sample['dataset']]=torch.stack(base_logits).numpy()
        checks.append(dict(dataset=sample['dataset'],question_id=sample['question_id'],native_tokens=reference,checks=comparisons))
    path=RAW/f'smoke/shard{rank}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**raw)
    return dict(passed=True,checks=checks,raw_path=str(path),raw_sha256=sha256(path))


def gate():
    checks=[];arrays=[]
    for rank in range(3):
        value=json.loads((OUT/f'smoke/shard{rank}/validation.json').read_text())
        if not value['passed']:raise ValueError('GPU Smoke incomplete')
        checks.append(value)
        with np.load(value['raw_path']) as f:arrays.append({k:f[k] for k in f.files})
    # Same architecture GPUs, deterministic kernels, exact frozen-source comparison.
    differences=[]
    for rank in [1,2]:
        for unit in arrays[0]:
            a,b=arrays[0][unit],arrays[rank][unit]
            if a.shape!=b.shape:raise ValueError('Cross-GPU generation lengths differ')
            maximum=float(np.max(np.abs(a-b)));differences.append(dict(rank=rank,dataset=unit,max_abs_logit_diff=maximum))
            if maximum!=0:raise ValueError('Cross-GPU exact same-hardware logits failed; do not relax tolerance')
    freeze_json(OUT/'gpu_smoke_validation.json',dict(passed=True,workers=3,cross_gpu=differences,checks=checks))


def worker(stage,rank):
    config=verify();binding=digest(config)
    if stage=='generation':gate()
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=preflight(rank);model,processor,env=load_model(MODEL_PATH)
    if transformers.__version__!='4.49.0':raise ValueError('Pinned Transformers required')
    env.update(preflight=pre,binding=binding,code_commit=config['code_commit'],physical_gpu=rank+1)
    freeze_json(folder/f'environment_{time.time_ns()}.json',env)
    weights={k:p._version for k,p in model.named_parameters()};hooks=hook_footprint(model)
    data=json.loads((OUT/'sample_manifest.json').read_text());started=time.perf_counter()
    if stage=='smoke':freeze_json(folder/'validation.json',smoke(model,processor,data['smoke'],rank))
    else:
        rows=[s for s in data['samples'] if s['rank']==rank]
        for i,sample in enumerate(rows):
            name=sample['dataset']+'__'+sample['question_id'];dest=folder/'checkpoints'/name
            if all(checkpoint_valid(dest/(c+'.json'),binding,sample,c) for c in CONDITIONS):continue
            prompt=prepared(model,processor,sample);input_hash=tensor_hashes(prompt)
            first,parent=prefill(model,prompt);parent_before=cache_fingerprint(parent);versions=cache_versions(parent)
            for condition in CONDITIONS:
                path=dest/(condition+'.json')
                if checkpoint_valid(path,binding,sample,condition):continue
                begin=time.perf_counter()
                result,_=generate(model,prompt,limit(sample),condition_layer(condition),parent,first)
                if cache_versions(parent)!=versions:raise AssertionError('Native parent cache modified')
                result['text']=processor.tokenizer.decode(result['token_ids'],skip_special_tokens=True).strip()
                if result['token_ids'][0]!=int(first.argmax()) or hook_footprint(model)!=hooks:raise AssertionError('First-token/hook invariant')
                atomic_json(path,dict(binding=binding,sample_sha256=digest(sample),condition=condition,
                    dataset=sample['dataset'],question_id=sample['question_id'],image_identity=sample['image_identity'],image_sha256=sample['image_sha256'],
                    generation=result,result_sha256=digest(result),seconds=time.perf_counter()-begin,physical_gpu=rank+1))
            if cache_fingerprint(parent)!=parent_before or tensor_hashes(prompt)!=input_hash:raise AssertionError('Native cache/input contaminated')
            freeze_json(dest/'audit.json',dict(parent_cache_sha256=parent_before,parent_cache_unchanged=True,input_unchanged=True,conditions=33,first_token=int(first.argmax())))
            del parent,first,prompt
            print(json.dumps(dict(completed_images=i+1,total_images=len(rows),dataset=sample['dataset'],question_id=sample['question_id'],elapsed_seconds=time.perf_counter()-started)),flush=True)
    if {k:p._version for k,p in model.named_parameters()}!=weights or hook_footprint(model)!=hooks:raise AssertionError('Model weight/hook mutation')
    freeze_json(folder/'completion.json',dict(status='complete',binding=binding,code_commit=config['code_commit'],seconds=time.perf_counter()-started,memory=gpu_memory()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gate','smoke','generation']);parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    try:
        if args.stage in ['cpu','freeze','gate']:globals()[args.stage]()
        else:worker(args.stage,args.rank)
    except BaseException as exc:
        folder=OUT/'failures';folder.mkdir(parents=True,exist_ok=True)
        freeze_json(folder/f'{args.stage}_rank{args.rank}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),stage=args.stage,rank=args.rank))
        raise


if __name__=='__main__':main()
