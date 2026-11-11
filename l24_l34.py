"""One user-requested visual reaccess set; historical runtime is read-only."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l11-l24-local-budget-search')
sys.path.insert(0, str(SOURCE))
from common import MODEL, OUT as PREVIOUS, SEED, TASKS, UUIDS, Store, check, digest, prepare, read_store, scored, sha, table, write
sys.path.insert(0, str(SOURCE))

OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_l24_l34_discovery'
LAYERS = list(range(24, 35))
NAME = 'L24_L34'


def freeze():
    check()
    assert not (OUT / 'protocol.json').exists(), 'Do not replace a frozen protocol'
    rows = [r for r in json.loads((PREVIOUS / 'sample_manifest.json').read_text()) if r['split'] == 'Discovery']
    assert len(rows) == 585 and sum(r['analysis_task'] == 'TextVQA' for r in rows) == 250
    assert not any(r.get('historical_blocked_status') for r in rows)
    clusters = {}
    image_hashes = {}
    for row in rows:
        assert clusters.setdefault(row['image_sha256'], row['shard']) == row['shard']
        if row['image_path'] not in image_hashes:
            image_hashes[row['image_path']] = sha(row['image_path'])
        assert image_hashes[row['image_path']] == row['image_sha256']
    write(OUT / 'sample_manifest.json', rows)
    baseline_keys = {r['key'] for r in rows}
    baseline = Store(OUT / 'paired_baselines.sqlite')
    try:
        for key, row in read_store(PREVIOUS / 'historical.sqlite'):
            if row['key'] in baseline_keys:
                baseline.put(key, row)
        assert baseline.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 1170
    finally:
        baseline.conn.close()
    source_files = {path: checksum for path, checksum in check()['sources'].items()}
    source_files.update({str(ROOT / name): sha(ROOT / name) for name in ['l24_l34.py', 'l24_l34_tests.py']})
    write(OUT / 'protocol.json', dict(
        experiment='QWEN3VL_SVHR_L24_L34_DISCOVERY', condition=NAME, layers=LAYERS, k=11,
        model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', dtype='bfloat16', attention='eager',
        scope='EXACT_ORIGINAL_DISCOVERY_ONLY_NO_VALIDATION_OR_HELD_OUT', tasks=dict(Counter(r['analysis_task'] for r in rows)),
        planned_records=585, prefill_and_decode_same_set=True, r_writing_layers=list(range(22)),
        visual_hidden_exit_before=22, answer_skip_layers=list(range(1, 7)),
        storage='FULL_FUNCTIONAL_REFERENCE_NOT_A_PHYSICAL_SPEED_TEST', generation='FROZEN_PER_ROW_GREEDY_CAP_AND_EOS',
        previous_commit='066c037f88931aeb73937e34b663b61de3f24ba5',
        previous_protocol_sha256=sha(PREVIOUS / 'protocol.json'), manifest_sha256=sha(OUT / 'sample_manifest.json'),
        baseline_sha256=sha(OUT / 'paired_baselines.sqlite'), source_sha256=source_files,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        bootstrap=dict(repeats=20000, seed=SEED, cluster='Image SHA256, complete Circular groups'),
        gpu_uuids=UUIDS, historical_strict_status='FAIL_RETAINED',
        stop_on='ANY_NUMERICAL_INPUT_NONFINITE_OR_OOM_FAILURE; preserve and release owned contexts',
        time=time.time()))
    print('FROZEN', NAME, LAYERS, dict(Counter(r['shard'] for r in rows)), flush=True)


def frozen():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['layers'] == LAYERS and len(LAYERS) == 11
    assert sha(OUT / 'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT / 'paired_baselines.sqlite') == p['baseline_sha256']
    for path, checksum in p['source_sha256'].items():
        assert sha(path) == checksum, ('FROZEN_SOURCE_CHANGED', path)
    return p


def resources():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(uuid in processes for uuid in UUIDS), processes
    sample = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv'], text=True)
    assert shutil.disk_usage('/data').free >= 2 * 2**30
    write(OUT / 'gpu_preflight.json', dict(time=time.time(), processes=processes, sample=sample, authorized=UUIDS))


def worker(shard):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from engine import ACTIVE, Request
    from vlm_research.qwen3vl_overnight.engine import run as frozen_run
    p = frozen()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUIDS[shard]
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = None
    baseline = store = None
    write(OUT / f'worker{shard}_start.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=time.time()))
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert UUIDS[shard] not in processes
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        rows = json.loads((OUT / 'sample_manifest.json').read_text())
        pool = [r for r in rows if r['shard'] == shard]
        baseline = Store(OUT / 'paired_baselines.sqlite')
        store = Store(OUT / f'worker{shard}.sqlite')

        def bound(row):
            inputs, layout = prepare(processor, row)
            assert hashlib.sha256(layout['prompt'].encode()).hexdigest() == row['prompt_sha256']
            assert sha(row['image_path']) == row['image_sha256']
            for label in ['Native', 'SVH']:
                old = baseline.get(row['key'] + '|' + label)
                assert old['input_binding_sha256'] == digest(layout), (row['key'], label, 'INPUT_BINDING_MISMATCH')
            return inputs, layout

        smoke = []
        for task in ['TextVQA', 'ScienceQA-IMG', 'MMBench']:
            candidates = sorted([r for r in pool if r['analysis_task'] == task], key=lambda r: r['input_length'])
            row = candidates[len(candidates) // 2]
            inputs, layout = bound(row)
            native, a, _ = frozen_run(model, processor, inputs, layout, 'Native', capture=True, cap=2)
            noop, b, _ = frozen_run(model, processor, inputs, layout, 'Native', capture=True, noop=True, cap=2)
            assert native['token_ids'] == noop['token_ids'] and len(a) == len(b)
            assert all(torch.equal(x, y) for x, y in zip(a, b)), 'NATIVE_NOOP_NOT_TORCH_EQUAL'
            with Request(model, processor, inputs, layout, collect=ACTIVE, audit=True, cap=3) as request:
                zero, _ = request.generate([], capture=True)
            with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=LAYERS, audit=True, cap=3) as request:
                functional, a = request.generate(LAYERS, capture=True)
            with Request(model, processor, inputs, layout, collect=LAYERS, prefill_layers=LAYERS, audit=True, cap=3) as request:
                physical, b = request.generate(LAYERS, physical=True, capture=True)
            error = max(float((x-y).abs().max()) for x, y in zip(a, b))
            assert len(a) == len(b) and error <= .02 and functional['token_ids'] == physical['token_ids']
            assert zero['prefill_non_target_hashes'] == functional['prefill_non_target_hashes'] == physical['prefill_non_target_hashes']
            assert functional['prefill_layers'] == functional['decode_layers'] == LAYERS
            assert all(r['allowed'] == (r['layer'] in LAYERS) for r in functional['permissions'])
            assert all(r['image_mass'] == 0 for r in functional['permissions'] if not r['allowed'])
            smoke.append(dict(key=row['key'], native_noop='TORCH_EQUAL', layers=LAYERS, same_prefill_decode=True,
                              functional_physical_max_logit_error=error, same_tokens=True,
                              non_target_prefill_unchanged=True, permissions=functional['permissions'],
                              source_input_binding_sha256=digest(layout), hooks_weights_inputs_unchanged=True,
                              historical_strict_status='FAIL_RETAINED'))
            del inputs
        write(OUT / f'smoke{shard}.json', dict(status='PASS_NEW_SET_STORAGE_SCOPE_ONLY', cases=smoke, time=time.time(), gpu_uuid=UUIDS[shard]))
        while not (OUT / 'START_QUALITY.json').exists():
            assert not (OUT / 'STOP.json').exists(), 'Stopped before dual-GPU smoke passed'
            time.sleep(.2)
        for row in sorted(pool, key=lambda r: (r['image_sha256'], r['key'])):
            assert not (OUT / 'STOP.json').exists(), 'Other shard failed; stop this shard'
            key = row['key'] + '|' + NAME
            if store.get(key) is not None:
                continue
            assert shutil.disk_usage('/data').free >= 2 * 2**30
            write(OUT / f'current{shard}.json', dict(key=row['key'], condition=NAME, time=time.time()))
            inputs, layout = bound(row)
            with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=LAYERS, cap=row['max_new_tokens']) as request:
                result, _ = request.generate(LAYERS)
            result.update(scored(row, result['text'], result['truncated']))
            result.update(key=row['key'], benchmark=row['benchmark'], analysis_task=row['analysis_task'],
                          image_sha256=row['image_sha256'], circular_group=row.get('circular_group'), gold=row['gold'],
                          question=row['question'], condition=NAME, stage='Discovery', view='Unified', layers=LAYERS,
                          prompt_sha256=row['prompt_sha256'], input_binding_sha256=digest(layout),
                          protocol_sha256=sha(OUT / 'protocol.json'), gpu_uuid=UUIDS[shard],
                          first_token_flip_vs_svh=result['token_ids'][0] != baseline.get(row['key'] + '|SVH')['token_ids'][0])
            assert result['prefill_layers'] == result['decode_layers'] == LAYERS
            store.put(key, result)
            del inputs
            n = store.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            if n % 20 == 0:
                print('PROGRESS', shard, n, len(pool), row['key'], flush=True)
        write(OUT / f'done{shard}.json', dict(status='COMPLETE_SHARD', rows=len(pool), gpu_uuid=UUIDS[shard], time=time.time()))
    except BaseException as exc:
        write(OUT / f'failure{shard}.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        for db in [store, baseline]:
            if db is not None:
                db.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT / f'release{shard}.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], process_exit_releases_context=True, time=time.time()))


def launch():
    frozen()
    resources()
    children = []
    streams = []
    started = time.time()
    assert not (OUT / 'launch.json').exists(), 'Use checkpoints deliberately, never launch a duplicate coordinator'
    try:
        for shard, uuid in enumerate(UUIDS):
            stream = (OUT / f'worker{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                                             cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM='false'),
                                             stdout=stream, stderr=subprocess.STDOUT))
        write(OUT / 'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started))
        while not all((OUT / f'smoke{i}.json').exists() for i in [0, 1]):
            assert all(c.poll() is None for c in children), 'Worker exited before smoke'
            assert time.time() - started < 900, 'GPU smoke timeout'
            time.sleep(1)
        write(OUT / 'START_QUALITY.json', dict(time=time.time(), scope='DISCOVERY_ONLY', both_gpu_smoke='PASS'))
        while not all((OUT / f'done{i}.json').exists() for i in [0, 1]):
            assert not any((OUT / f'failure{i}.json').exists() for i in [0, 1]), 'WORKER_FAILURE'
            assert all(c.poll() in [None, 0] for c in children), 'Worker exit failure'
            assert time.time() - started < 3600, 'Stop rather than leave an unbounded job'
            time.sleep(2)
        for child in children:
            assert child.wait(timeout=40) == 0
        write(OUT / 'generation_complete.json', dict(status='COMPLETE_585_GENERATIONS', time=time.time(), wall_seconds=time.time()-started))
    except BaseException as exc:
        write(OUT / 'coordinator_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        write(OUT / 'STOP.json', dict(time=time.time(), scope='ONLY_THIS_EXPERIMENT_OWN_WORKERS'))
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait()
        for stream in streams:
            stream.close()
        write(OUT / 'gpu_release_audit.json', dict(time=time.time(), exit_codes=[c.returncode for c in children],
              processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv'], text=True)))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--freeze', action='store_true')
    args.add_argument('--launch', action='store_true')
    args.add_argument('--worker', type=int)
    opt = args.parse_args()
    if opt.freeze:
        freeze()
    elif opt.launch:
        launch()
    elif opt.worker is not None:
        worker(opt.worker)
    else:
        args.error('Choose --freeze, --launch or --worker')
