"""Reuse the frozen 24-case gate with both independently verified repairs."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..qwen3vl_svh_fasttrack import run as previous
from ..qwen3vl_svh_fasttrack.freeze import GPUS, scorers
from ..qwen3vl_svh_sqa7908_repair.run import BASE, MODEL, sources as repair_sources
from ..qwen3vl_svh_sqa7908_repair.engine import generate as repaired
from ..qwen3vl_svh_vision_memory.runtime import RUNTIME, vision_runtime
from ..qwen3vl_svh_preflight48.freeze import prepare, sha, table, write, TASKS
from ..qwen3vl_svh_preflight48.run import resource
from ..qwen3vl_svh_numerical_repair.run import GATE

ROOT = Path(__file__).resolve().parents[2]
OLD = BASE / 'vlm_research/results/qwen3vl_svh_fasttrack_qk'
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_fasttrack_runtime24'
FROZEN = ['numerical_cases.json', 'pretest8_manifest.json', 'input_bindings.json', 'shards.json']


def sources():
    paths = [ROOT / 'vlm_research/qwen3vl_svh_fasttrack/run.py',
             ROOT / 'vlm_research/qwen3vl_svh_fasttrack/freeze.py']
    for name in ['qwen3vl_svh_fasttrack_runtime', 'qwen3vl_svh_vision_memory']:
        paths += list((ROOT / 'vlm_research' / name).glob('*.py'))
    return {**repair_sources(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def freeze():
    assert not torch.cuda.is_initialized() and not (OUT / 'protocol.json').exists()
    p = json.loads((OLD / 'protocol.json').read_text())
    assert p['gate'] == GATE and p['total_questions'] == 35346
    assert json.loads((ROOT / 'vlm_research/results/qwen3vl_svh_sqa7908_repair/verify_completion.json').read_text())['status'] == 'PASS'
    assert json.loads((ROOT / 'vlm_research/results/qwen3vl_svh_vision_memory/memory_accept_completion.json').read_text())['status'] == 'PASS'
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FROZEN:
        assert sha(OLD / name) == p['frozen_files'][name]
        shutil.copyfile(OLD / name, OUT / name)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    bindings = json.loads((OUT / 'input_bindings.json').read_text())
    cases = json.loads((OUT / 'numerical_cases.json').read_text())
    assert len(cases) == len({c['sample']['key'] for c in cases}) == 24
    for case in cases:
        inputs, layout = prepare(processor, case['sample'])
        assert layout == bindings[case['sample']['key']]
        del inputs
    scorer = scorers()
    scorer.adapters()
    assert all(sha(path) == expected for path, expected in p['scorer_sources'].items())
    p.update(runtime=RUNTIME, original_eager_status='BLOCKED_MEMORY',
        current_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_files=sources(), inventory_path=str(BASE / 'vlm_research/results/qwen3vl_svh_expanded_preflight48/inventory.jsonl.gz'),
        inherited_qk_protocol_sha256=sha(OLD / 'protocol.json'),
        full_59_numerical_validated=False, numerical_inputs=24,
        gate=GATE, gpu_uuids=GPUS, numerical_status='NOT_RUN',
        plan_A='BLOCKED', no_speed_claim=True)
    assert sha(p['inventory_path']) == p['inventory_sha256']
    write(OUT / 'protocol.json', p)
    write(OUT / 'cpu_freeze.json', dict(status='PASS', cases=24, cuda_initialized=False,
        runtime=RUNTIME, original_samples_unchanged=True))
    print('FROZEN', RUNTIME, '24 original inputs; full generation still locked', flush=True)


def check():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['source_files'] == sources() and p['gate'] == GATE and p['runtime'] == RUNTIME
    assert sha(p['inventory_path']) == p['inventory_sha256']
    for name in FROZEN:
        assert sha(OUT / name) == p['frozen_files'][name]
    for path, expected in p['scorer_sources'].items():
        assert sha(path) == expected
    return p


def configure():
    previous.OUT, previous.GPUS, previous.fixed_generate = OUT, GPUS, repaired
    previous.check_freeze = check
    original_call = previous.call
    def call(*args, **kwargs):
        with vision_runtime():
            return original_call(*args, **kwargs)
    previous.call = call


def worker(shard):
    check()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPUS[shard]
    before = resource()
    assert GPUS[shard] not in before['processes']
    state = subprocess.check_output(['nvidia-smi', '-i', GPUS[shard], '--query-gpu=name,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    assert '3090' in state[0] and int(state[1]) > 23500 and int(state[2]) == 0
    write(OUT / f'worker{shard}_gpu_before.json', before)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    configure()
    model = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        previous.numerical(model, processor, shard)
    except BaseException as exc:
        current = OUT / f'worker{shard}_current.json'
        failure = dict(status='FAIL', error=repr(exc), traceback=traceback.format_exc(),
            shard=shard, gpu_uuid=GPUS[shard], current=json.loads(current.read_text()) if current.exists() else None,
            pid=os.getpid(), runtime=RUNTIME, allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        write(OUT / f'worker{shard}_failure.json', failure)
        if not (OUT / 'STOP_NUMERICAL.json').exists():
            write(OUT / 'STOP_NUMERICAL.json', failure)
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        write(OUT / f'worker{shard}_gpu_after_inprocess.json', resource())


def summary():
    check()
    cases = json.loads((OUT / 'numerical_cases.json').read_text())
    rows = []
    for case in cases:
        key = case['sample']['key']
        path = OUT / 'numerical' / key.replace(':', '_').replace('/', '_') / 'validation.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    done = len(rows) == 24 and all(r['status'] == 'PASS' for r in rows) and not (OUT / 'STOP_NUMERICAL.json').exists()
    table(OUT / 'numerical_summary.csv', rows)
    write(OUT / 'numerical_completion.json', dict(status='PASS' if done else 'FAIL_OR_PARTIAL',
        completed=len(rows), passed=sum(r['status'] == 'PASS' for r in rows), expected=24, rows=rows,
        pretest_allowed=done, full_evaluation_allowed=False, full_59_numerical_validated=False,
        runtime=RUNTIME, label='FAST_TRACK_NUMERICAL24_PASS' if done else 'NO_EVALUATION_UNLOCK', time=time.time()))
    print('NUMERICAL SUMMARY', len(rows), 'PASS' if done else 'FAIL_OR_PARTIAL', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'numerical', 'summary'])
    parser.add_argument('--shard', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'summary':
        summary()
    else:
        assert args.shard is not None
        worker(args.shard)
