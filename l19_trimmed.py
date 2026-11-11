"""Two fixed windows on the frozen engine, without repeating GPU Smoke."""
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
PRIOR_ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l19-l25-discovery')
sys.path.insert(0, str(PRIOR_ROOT))
from l19_l25 import OUT as PRIOR_OUT, frozen as previous_frozen
from l24_l34 import MODEL, SEED, TASKS, UUIDS, Store, digest, prepare, read_store, scored, sha, table, write

OUT = ROOT/'vlm_research/results/qwen3vl_svhr_l19_trimmed_discovery'
CONDITIONS = {'L19_L32': list(range(19, 33)), 'L19_L31': list(range(19, 32))}
EXPECTED = 585*len(CONDITIONS)


def freeze():
    old = previous_frozen()
    assert not (OUT/'protocol.json').exists(), 'Do not replace frozen input/protocol'
    OUT.mkdir(parents=True, exist_ok=True)
    for name, binding in [('sample_manifest.json', 'manifest_sha256'), ('paired_baselines.sqlite', 'baseline_sha256')]:
        assert sha(PRIOR_OUT/name) == old[binding]
        shutil.copyfile(PRIOR_OUT/name, OUT/name)
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    assert len(rows) == 585 and len({r['key'] for r in rows}) == 585
    assert all(r['split'] == 'Discovery' for r in rows)
    assert sum(r['analysis_task'] == 'TextVQA' for r in rows) == 250
    clusters = {}
    for r in rows:
        assert clusters.setdefault(r['image_sha256'], r['shard']) == r['shard']
    assert CONDITIONS == {'L19_L32': list(range(19, 33)), 'L19_L31': list(range(19, 32))}
    p = dict(old)
    for field in ['condition', 'layers', 'k', 'user_request_label']:
        p.pop(field, None)
    p.update(experiment='QWEN3VL_SVHR_L19_TRIMMED_DISCOVERY_NO_REPEAT_SMOKE', conditions=CONDITIONS,
        expected_per_condition=585, expected_total=EXPECTED, planned_records=EXPECTED,
        source_sha256=dict(old['source_sha256']), manifest_sha256=sha(OUT/'sample_manifest.json'),
        baseline_sha256=sha(OUT/'paired_baselines.sqlite'), previous_protocol_sha256=sha(PRIOR_OUT/'protocol.json'),
        previous_result_commit='771047b8e586e3ff24ccafbd812f51527aa46f7b',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        smoke='NOT_RUN_USER_REQUEST', new_gpu_numerical_validation='NOT_RUN',
        validation_basis='Reuse unchanged engine and earlier Smoke evidence; not a new numerical PASS',
        time=time.time())
    p['source_sha256'][str(ROOT/'l19_trimmed.py')] = sha(ROOT/'l19_trimmed.py')
    write(OUT/'protocol.json', p)
    print('FROZEN', CONDITIONS, 'EXPECTED', EXPECTED, flush=True)


def frozen():
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['conditions'] == CONDITIONS and p['expected_total'] == EXPECTED
    assert p['smoke'] == 'NOT_RUN_USER_REQUEST'
    assert sha(OUT/'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT/'paired_baselines.sqlite') == p['baseline_sha256']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    return p


def worker(shard):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from engine import ACTIVE, Request
    frozen()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUIDS[shard]
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = baseline = store = None
    write(OUT/f'worker{shard}_start.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=time.time()))
    try:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert UUIDS[shard] not in processes, 'Authorized GPU became occupied'
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        rows = json.loads((OUT/'sample_manifest.json').read_text())
        pool = [r for r in rows if r['shard'] == shard]
        baseline = Store(OUT/'paired_baselines.sqlite')
        store = Store(OUT/f'worker{shard}.sqlite')
        for row in sorted(pool, key=lambda r: (r['image_sha256'], r['key'])):
            assert not (OUT/'STOP.json').exists(), 'Stop requested or other shard failed'
            assert shutil.disk_usage('/data').free >= 2*2**30
            inputs, layout = prepare(processor, row)
            assert hashlib.sha256(layout['prompt'].encode()).hexdigest() == row['prompt_sha256']
            assert sha(row['image_path']) == row['image_sha256']
            for label in ['Native', 'SVH']:
                old = baseline.get(row['key']+'|'+label)
                assert old['input_binding_sha256'] == digest(layout), (row['key'], label, 'INPUT_BINDING_MISMATCH')
            for name, layers in CONDITIONS.items():
                assert not (OUT/'STOP.json').exists(), 'Other shard failed'
                key = row['key']+'|'+name
                existing = store.get(key)
                if existing is not None:
                    assert existing['protocol_sha256'] == sha(OUT/'protocol.json')
                    assert existing['input_binding_sha256'] == digest(layout)
                    assert existing['layers'] == existing['prefill_layers'] == existing['decode_layers'] == layers
                    continue
                write(OUT/f'current{shard}.json', dict(key=row['key'], condition=name, time=time.time()))
                with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=layers,
                             cap=row['max_new_tokens']) as request:
                    result, _ = request.generate(layers)
                result.update(scored(row, result['text'], result['truncated']))
                result.update(key=row['key'], benchmark=row['benchmark'], analysis_task=row['analysis_task'],
                    image_sha256=row['image_sha256'], circular_group=row.get('circular_group'), gold=row['gold'],
                    question=row['question'], condition=name, stage='Discovery', view='Unified', layers=layers,
                    prompt_sha256=row['prompt_sha256'], input_binding_sha256=digest(layout),
                    protocol_sha256=sha(OUT/'protocol.json'), gpu_uuid=UUIDS[shard],
                    first_token_flip_vs_svh=result['token_ids'][0] != baseline.get(row['key']+'|SVH')['token_ids'][0])
                assert result['prefill_layers'] == result['decode_layers'] == layers
                store.put(key, result)
            del inputs
            n = store.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0]
            if n % 40 == 0:
                print('PROGRESS', shard, n, len(pool)*len(CONDITIONS), row['key'], flush=True)
        n = store.conn.execute('SELECT COUNT(*) FROM records').fetchone()[0]
        assert n == len(pool)*len(CONDITIONS)
        write(OUT/f'done{shard}.json', dict(status='COMPLETE_SHARD', rows=n, time=time.time(), gpu_uuid=UUIDS[shard]))
    except BaseException as exc:
        write(OUT/f'failure{shard}.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        for db in [store, baseline]:
            if db is not None:
                db.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'release{shard}.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=time.time()))


def launch():
    frozen()
    assert not (OUT/'launch.json').exists(), 'Reject duplicate launch'
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(uuid in processes for uuid in UUIDS), processes
    assert shutil.disk_usage('/data').free >= 2*2**30
    write(OUT/'gpu_preflight.json', dict(time=time.time(), processes=processes, authorized=UUIDS,
        sample=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)))
    children, streams = [], []
    started = time.time()
    try:
        for shard, uuid in enumerate(UUIDS):
            stream = (OUT/f'worker{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM='false'),
                stdout=stream, stderr=subprocess.STDOUT))
        write(OUT/'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started,
                                    smoke='NOT_RUN_USER_REQUEST'))
        while not all((OUT/f'done{i}.json').exists() for i in [0, 1]):
            assert not any((OUT/f'failure{i}.json').exists() for i in [0, 1]), 'WORKER_FAILURE'
            assert all(c.poll() in [None, 0] for c in children), 'Worker exited with failure'
            assert time.time()-started < 3600, 'Bounded execution timeout'
            time.sleep(2)
        for child in children:
            assert child.wait(timeout=40) == 0
        counts = [json.loads((OUT/f'done{i}.json').read_text())['rows'] for i in [0, 1]]
        assert sum(counts) == EXPECTED
        write(OUT/'generation_complete.json', dict(status='COMPLETE_1170_GENERATIONS', expected=EXPECTED,
            actual=sum(counts), time=time.time(), wall_seconds=time.time()-started))
    except BaseException as exc:
        write(OUT/'coordinator_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        write(OUT/'STOP.json', dict(time=time.time(), scope='ONLY_THIS_EXPERIMENT_OWN_WORKERS'))
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait()
        for stream in streams:
            stream.close()
        write(OUT/'gpu_release_audit.json', dict(time=time.time(), exit_codes=[c.returncode for c in children],
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
