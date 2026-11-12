"""Reuse the frozen 24-case runner with the independently verified QK executor."""
import argparse
import gc
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..qwen3vl_svh_fasttrack import run as original_run
from ..qwen3vl_svh_fasttrack import freeze as old_freeze
from ..qwen3vl_svh_qk_repair import run as repair
from ..qwen3vl_svh_qk_repair.engine import generate
from ..qwen3vl_svh_preflight48.run import resource
from ..qwen3vl_svh_preflight48.freeze import sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'vlm_research/results/qwen3vl_svh_fasttrack_qk'
GPUS = ['GPU-b08489af-0a8a-eb48-55ba-8346733af417','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']


def sources():
    return {**old_freeze.sources(), **repair.sources(), **{
        str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}}


def freeze():
    assert not torch.cuda.is_initialized()
    assert not (OUT/'protocol.json').exists()
    verified = json.loads((repair.OUT/'verification_completion.json').read_text())
    assert verified['status']=='PASS' and verified['completed']==verified['expected']==9
    p = json.loads((repair.OLD/'protocol.json').read_text())
    for name, digest in p['frozen_files'].items():
        assert sha(repair.OLD/name)==digest
        write(OUT/name,json.loads((repair.OLD/name).read_text()))
        assert sha(OUT/name)==digest
    assert p['gate']==repair.GATE
    assert sha(p['inventory_path'])==p['inventory_sha256']
    for path,digest in p['scorer_sources'].items():
        assert sha(path)==digest
    p.update(base_commit='30535244f2', gpu_uuids=GPUS, source_files=sources(),
        scoped_regression_sha256=sha(repair.OUT/'verification_completion.json'),
        inherited_fasttrack_protocol_sha256=sha(repair.OLD/'protocol.json'),
        original_failures_preserved=str(repair.OLD),
        label='ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL',
        full_59_numerical_validated=False, new_quality_records=0,
        prior_numerical_results_not_counted=True)
    write(OUT/'protocol.json',p)
    write(OUT/'cpu_freeze.json',dict(status='PASS', same_original_cases=True,
        same_original_bindings=True, same_original_shards=True, cuda_initialized=False,
        scoped_regressions=9, numerical_cases=24, pretest_cases=48))


def check():
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['source_files']==sources()
    assert p['gate']==repair.GATE and p['full_59_numerical_validated'] is False
    for name,digest in p['frozen_files'].items():
        assert sha(OUT/name)==digest
    assert sha(p['inventory_path'])==p['inventory_sha256']
    return p


def worker(shard):
    check()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPUS[shard]
    before=resource()
    assert GPUS[shard] not in before['processes']
    state=subprocess.check_output(['nvidia-smi','-i',GPUS[shard],
        '--query-gpu=name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert '3090' in state[0] and int(state[1])>23500 and int(state[2])==0
    write(OUT/f'worker{shard}_gpu_before.json',before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    model=None
    original_run.OUT=OUT
    original_run.GPUS=GPUS
    original_run.fixed_generate=generate
    try:
        processor=AutoProcessor.from_pretrained(repair.MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(repair.MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        original_run.numerical(model,processor,shard)
    except BaseException as exc:
        current=OUT/f'worker{shard}_current.json'
        failure=dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),
            shard=shard,pid=os.getpid(),gpu_uuid=GPUS[shard],
            current=json.loads(current.read_text()) if current.exists() else None,
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        write(OUT/f'worker{shard}_failure.json',failure)
        if not (OUT/'STOP_NUMERICAL.json').exists():
            write(OUT/'STOP_NUMERICAL.json',failure)
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def summarize():
    check()
    cases=json.loads((OUT/'numerical_cases.json').read_text())
    validations=[]
    for case in cases:
        path=OUT/'numerical'/case['sample']['key'].replace(':','_')/'validation.json'
        if path.exists():
            validations.append(json.loads(path.read_text()))
    passed=sum(r['status']=='PASS' for r in validations)
    failed=sum(r['status']!='PASS' for r in validations)
    complete=passed==24 and failed==0 and not (OUT/'STOP_NUMERICAL.json').exists()
    write(OUT/'numerical_completion.json',dict(status='PASS' if complete else 'FAIL',
        completed=len(validations),passed=passed,failed=failed,expected=24,
        pretest8_allowed=complete,full_evaluation_allowed=False,
        full_59_numerical_validated=False,acceptance_label='FAST_TRACK_NUMERICAL_VALIDATED' if complete else 'BLOCKED',
        rows=validations))
    print('FAST_TRACK24',passed,'PASS',failed,'FAIL','completed',len(validations),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','worker','summarize'])
    parser.add_argument('--shard',type=int,choices=[0,1])
    args=parser.parse_args()
    if args.mode=='freeze':
        freeze()
    elif args.mode=='summarize':
        summarize()
    else:
        assert args.shard is not None
        worker(args.shard)
