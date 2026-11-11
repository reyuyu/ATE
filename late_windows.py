"""Two fixed Discovery windows, reusing the frozen runtime and scorers."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l24-l34-discovery')
sys.path.insert(0, str(OLD_ROOT))
import l24_l34 as prior
from l24_l34 import MODEL, PREVIOUS, SEED, TASKS, UUIDS, Store, check, digest, prepare, read_store, scored, sha, table, write

OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_late_windows_discovery'
CONDITIONS = {'L24_L35': list(range(24, 36)), 'L20_L34': list(range(20, 35))}


def freeze():
    prior.frozen()
    assert not (OUT / 'protocol.json').exists(), 'Never replace a frozen protocol'
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['sample_manifest.json', 'paired_baselines.sqlite']:
        shutil.copyfile(prior.OUT / name, OUT / name)
    rows = json.loads((OUT / 'sample_manifest.json').read_text())
    assert len(rows) == 585 and all(r['split'] == 'Discovery' for r in rows)
    sources = dict(prior.frozen()['source_sha256'])
    for name in ['late_windows.py', 'late_windows_tests.py', 'cpu_checks.py']:
        sources[str(ROOT / name)] = sha(ROOT / name)
    for name in ['cpu_cache_checks.py']:
        sources[str(OLD_ROOT / name)] = sha(OLD_ROOT / name)
    write(OUT / 'protocol.json', dict(
        experiment='QWEN3VL_SVHR_TWO_FIXED_LATE_WINDOWS_DISCOVERY', conditions=CONDITIONS,
        expected_per_condition=585, expected_total=1170, scope='EXACT_ORIGINAL_DISCOVERY_ONLY',
        manifest_sha256=sha(OUT / 'sample_manifest.json'), baseline_sha256=sha(OUT / 'paired_baselines.sqlite'),
        tasks=dict(Counter(r['analysis_task'] for r in rows)), source_sha256=sources,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        previous_result_commit='566f6725d4', previous_protocol_sha256=sha(prior.OUT / 'protocol.json'),
        model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', dtype='bfloat16', attention='eager',
        prefill_and_decode_same_set=True, r_writing_layers=list(range(22)), visual_hidden_exit_before=22,
        answer_skip_layers=list(range(1, 7)), storage='FULL_FUNCTIONAL_REFERENCE_NOT_A_PHYSICAL_SPEED_TEST',
        generation='FROZEN_PER_ROW_GREEDY_CAP_AND_EOS', performance='NOT_PLANNED',
        bootstrap=dict(repeats=20000, seed=SEED, cluster='Image SHA256; complete Circular groups'),
        gpu_uuids=UUIDS, historical_strict_status='FAIL_RETAINED', time=time.time()))
    print('FROZEN', CONDITIONS, dict(Counter(r['shard'] for r in rows)), flush=True)


def frozen():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['conditions'] == CONDITIONS
    assert sha(OUT / 'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT / 'paired_baselines.sqlite') == p['baseline_sha256']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    return p


def worker(shard):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from engine import ACTIVE, Request
    from vlm_research.qwen3vl_overnight.engine import run as frozen_run
    frozen()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUIDS[shard]
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = baseline = store = None
    write(OUT / f'worker{shard}_start.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=time.time()))
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert UUIDS[shard] not in processes, 'Authorized GPU became occupied'
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
            for name, layers in CONDITIONS.items():
                with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=layers, audit=True, cap=3) as request:
                    functional, a = request.generate(layers, capture=True)
                with Request(model, processor, inputs, layout, collect=layers, prefill_layers=layers, audit=True, cap=3) as request:
                    physical, b = request.generate(layers, physical=True, capture=True)
                assert len(a) == len(b)
                error = max(float((x-y).abs().max()) for x, y in zip(a, b))
                assert error <= .02 and functional['token_ids'] == physical['token_ids']
                assert zero['prefill_non_target_hashes'] == functional['prefill_non_target_hashes'] == physical['prefill_non_target_hashes']
                assert functional['prefill_layers'] == functional['decode_layers'] == layers
                assert all(r['allowed'] == (r['layer'] in layers) for r in functional['permissions'])
                assert all(r['image_mass'] == 0 for r in functional['permissions'] if not r['allowed'])
                smoke.append(dict(key=row['key'], condition=name, layers=layers, native_noop='TORCH_EQUAL',
                                  functional_physical_max_logit_error=error, same_tokens=True,
                                  non_target_prefill_unchanged=True, permissions=functional['permissions'],
                                  input_binding_sha256=digest(layout), hooks_weights_inputs_unchanged=True,
                                  historical_strict_status='FAIL_RETAINED'))
            del inputs
        write(OUT / f'smoke{shard}.json', dict(status='PASS_NEW_SETS_STORAGE_SCOPE_ONLY', cases=smoke, time=time.time(), gpu_uuid=UUIDS[shard]))
        while not (OUT / 'START_QUALITY.json').exists():
            assert not (OUT / 'STOP.json').exists(), 'Stopped before both Smoke passed'
            time.sleep(.2)
        for row in sorted(pool, key=lambda r: (r['image_sha256'], r['key'])):
            assert not (OUT / 'STOP.json').exists(), 'Other shard failed'
            assert shutil.disk_usage('/data').free >= 2 * 2**30
            inputs, layout = bound(row)
            for name, layers in CONDITIONS.items():
                key = row['key'] + '|' + name
                if store.get(key) is not None:
                    continue
                write(OUT / f'current{shard}.json', dict(key=row['key'], condition=name, time=time.time()))
                with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=layers, cap=row['max_new_tokens']) as request:
                    result, _ = request.generate(layers)
                result.update(scored(row, result['text'], result['truncated']))
                result.update(key=row['key'], benchmark=row['benchmark'], analysis_task=row['analysis_task'],
                              image_sha256=row['image_sha256'], circular_group=row.get('circular_group'), gold=row['gold'],
                              question=row['question'], condition=name, stage='Discovery', view='Unified', layers=layers,
                              prompt_sha256=row['prompt_sha256'], input_binding_sha256=digest(layout),
                              protocol_sha256=sha(OUT / 'protocol.json'), gpu_uuid=UUIDS[shard],
                              first_token_flip_vs_svh=result['token_ids'][0] != baseline.get(row['key'] + '|SVH')['token_ids'][0])
                assert result['prefill_layers'] == result['decode_layers'] == layers
                store.put(key, result)
            del inputs
            n = store.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            if n % 40 == 0:
                print('PROGRESS', shard, n, len(pool)*len(CONDITIONS), row['key'], flush=True)
        write(OUT / f'done{shard}.json', dict(status='COMPLETE_SHARD', rows=len(pool)*len(CONDITIONS), time=time.time(), gpu_uuid=UUIDS[shard]))
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
        write(OUT / f'release{shard}.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=time.time()))


def launch():
    frozen()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(uuid in processes for uuid in UUIDS), processes
    assert shutil.disk_usage('/data').free >= 2 * 2**30
    write(OUT / 'gpu_preflight.json', dict(time=time.time(), processes=processes, authorized=UUIDS,
          sample=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)))
    assert not (OUT / 'launch.json').exists(), 'Reject a duplicate launch'
    children, streams = [], []
    started = time.time()
    try:
        for shard, uuid in enumerate(UUIDS):
            stream = (OUT / f'worker{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM='false'),
                            stdout=stream, stderr=subprocess.STDOUT))
        write(OUT / 'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started))
        while not all((OUT / f'smoke{i}.json').exists() for i in [0, 1]):
            assert all(c.poll() is None for c in children), 'Worker exited before Smoke'
            assert time.time()-started < 900, 'Smoke timeout'
            time.sleep(1)
        write(OUT / 'START_QUALITY.json', dict(time=time.time(), both_gpu_smoke='PASS', conditions=CONDITIONS))
        while not all((OUT / f'done{i}.json').exists() for i in [0, 1]):
            assert not any((OUT / f'failure{i}.json').exists() for i in [0, 1]), 'WORKER_FAILURE'
            assert all(c.poll() in [None, 0] for c in children), 'Worker exited with failure'
            assert time.time()-started < 3600, 'Bounded execution timeout'
            time.sleep(2)
        for child in children:
            assert child.wait(timeout=40) == 0
        write(OUT / 'generation_complete.json', dict(status='COMPLETE_1170_GENERATIONS', expected=1170,
                                                    time=time.time(), wall_seconds=time.time()-started))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', type=int)
    opt = parser.parse_args()
    if opt.freeze:
        freeze()
    elif opt.launch:
        launch()
    elif opt.worker is not None:
        worker(opt.worker)
    else:
        parser.error('Choose --freeze, --launch or --worker')
