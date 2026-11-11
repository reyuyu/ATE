"""Fixed L19-L35 quality evaluation; reuse immutable model/runtime/scorers."""
import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l11-l24-local-budget-search')
sys.path.insert(0, str(SOURCE))
from common import BASE, FIRST, MODEL, P2, SEED, UUIDS, Store, digest, historical, inventory, prepare, read_store, scored, sha, table, write
from prompt_common import render

OUT = ROOT/'vlm_research/results/qwen_svh_r_l19_l35_sixbench'
DISCOVERY = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l19-l35-discovery/vlm_research/results/qwen3vl_svhr_l19_l35_discovery')
LAYERS = list(range(19, 36))
NAME = 'qwen-SVH-R'
COUNTS = {'GQA': 12578, 'ScienceQA-IMG': 2017, 'TextVQA': 5000, 'POPE': 9000, 'MME': 2374, 'MMBench': 4377}
TASK_ORDER = {'ScienceQA-IMG': 0, 'MMBench': 1, 'GQA': 2, 'POPE': 3, 'TextVQA': 4, 'MME': 5}


def state(record):
    return record.get('state', record.get('status', 'COMPLETE'))


def compact(record):
    out = dict(record)
    out.pop('input_binding', None)
    out.pop('audit', None)
    out.pop('structural', None)
    out['state'] = state(record)
    if out['state'] == 'COMPLETE':
        out['score'] = record.get('score', record.get('scoring', {}).get('score'))
        out['scoring'] = record.get('scoring', {k: record.get(k) for k in ['score', 'parsed', 'parse_failed']})
        assert out['score'] is not None
    out['original_record_sha256'] = digest(record)
    return out


def manifest():
    with gzip.open(OUT/'sample_manifest.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def check():
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['answer_visual_layers'] == LAYERS and p['questions'] == 35346
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('SOURCE_CHANGED', path)
    for name, expected in p['frozen_files'].items():
        assert sha(OUT/name) == expected, ('FROZEN_FILE_CHANGED', name)
    for path, value in p['model_files'].items():
        st = Path(path).stat()
        assert [st.st_size, st.st_mtime_ns] == value['stat'], ('MODEL_CHANGED', path)
    return p


def freeze():
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    assert not (OUT/'protocol.json').exists(), 'Never replace frozen protocol'
    assert shutil.disk_usage('/data').free >= 2*2**30
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((FIRST/'protocol.json').read_text())
    p2 = json.loads((P2/'p2_full_protocol.json').read_text())
    rows = inventory()
    assert Counter(r['benchmark'] for r in rows) == COUNTS
    assert len(rows) == len({r['key'] for r in rows}) == 35346
    mapping = json.loads((FIRST/'shards.json').read_text())
    assert sha(FIRST/'shards.json') == old['shards_sha256']
    hist = historical({r['key'] for r in rows})
    assert len(hist) == 2*len(rows)
    # P2 was aligned on its own workers. Keep original connected groups but
    # inherit each group's actual historical device, not an older shard label.
    devices = defaultdict(set)
    for row in rows:
        for condition in ['Native', 'SVH']:
            devices[mapping[row['key']]['cluster']].add(hist[row['key'], condition]['gpu_uuid'])
    assert all(len(values) == 1 for values in devices.values()), 'Historical group has mixed devices; explicit re-pairing required'
    for value in mapping.values():
        value['shard'] = UUIDS.index(next(iter(devices[value['cluster']])))
    print('Historical bindings read; checking pixels and freezing inputs', flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    baselines = Store(OUT/'baselines.sqlite')
    groups, images, known = {}, {}, Counter()
    for i, row in enumerate(rows):
        if row['benchmark'] in ['ScienceQA-IMG', 'MMBench']:
            row, _, _ = render(processor, row, 'P2')
            rows[i] = row
        row['shard'] = mapping[row['key']]['shard']
        row['cluster'] = row['image_sha256']
        for tag in [('image', row['image_sha256'])] + ([(row['benchmark'], row['pair_id'])] if row['benchmark'] == 'MME' else []) + ([(row['benchmark'], row['circular_group'])] if row['benchmark'] == 'MMBench' else []):
            assert groups.setdefault(tag, row['shard']) == row['shard'], ('SPLIT_GROUP', tag)
        if row['image_path'] not in images:
            assert sha(row['image_path']) == row['image_sha256']
            images[row['image_path']] = row['image_sha256']
        bindings = []
        for condition in ['Native', 'SVH']:
            r = hist[row['key'], condition]
            for field in ['key', 'image_sha256', 'prompt_sha256', 'gold', 'input_length', 'image_tokens']:
                assert r[field] == row[field], ('BASELINE_BINDING', field, row['key'])
            assert r['gpu_uuid'] == UUIDS[row['shard']], ('DEVICE_PAIR', row['key'], condition)
            assert r.get('max_new_tokens', row['max_new_tokens']) == row['max_new_tokens']
            assert state(r) in ['COMPLETE', 'SKIPPED_OOM']
            if 'input_binding' in r:
                assert digest(r['input_binding']) == r['input_binding_sha256']
            bindings.append(r['input_binding_sha256'])
            baselines.put(row['key']+'|'+condition, compact(r))
            known[row['benchmark'], condition, state(r)] += 1
        assert bindings[0] == bindings[1]
        row['input_binding_sha256'] = bindings[0]
        if (i+1) % 5000 == 0:
            print('CPU_FREEZE', i+1, len(rows), flush=True)
    baselines.conn.close()
    with gzip.open(OUT/'sample_manifest.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True)+'\n')
    # Only exact same-input, same-device, same-generation Discovery records qualify.
    specs = {r['key']: r for r in rows}
    reuses = Store(OUT/'reused.sqlite')
    reused = []
    for shard in [0, 1]:
        for _, r in read_store(DISCOVERY/f'worker{shard}.sqlite'):
            row = specs[r['key']]
            if r['gpu_uuid'] != UUIDS[row['shard']]:
                continue
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == LAYERS
            assert r['image_sha256'] == row['image_sha256'] and r['prompt_sha256'] == row['prompt_sha256']
            assert r['input_binding_sha256'] == row['input_binding_sha256']
            r = compact(r)
            r.update(condition=NAME, reused_from=str(DISCOVERY), full_protocol_pending=True, state='COMPLETE')
            reuses.put(row['key'], r)
            reused.append(row['key'])
    reuses.conn.close()
    prior = json.loads((DISCOVERY/'protocol.json').read_text())
    sources = dict(prior['source_sha256'])
    for path in [ROOT/'qwen_svh_r_full.py', ROOT/'test_qwen_svh_r_full.py', ROOT/'summarize_qwen_svh_r.py', SOURCE/'engine.py']:
        sources[str(path)] = sha(path)
    sources.update(old['scorer_sources'])
    for path in [FIRST/'protocol.json', P2/'p2_full_protocol.json', DISCOVERY/'protocol.json']:
        sources[str(path)] = sha(path)
    protocol = dict(experiment='QWEN_SVH_R_L19_L35_SIXBENCH', formal_configuration=NAME,
        plan='B: ATE unified HF; NOT author-official reproduction', plan_A='BLOCKED',
        revision=MODEL.name, model=str(MODEL), model_files=old['model_files'],
        dtype='bfloat16', text_attention='eager', vision_attention='eager', visual_budget=old['visual_budget'],
        answer_visual_layers=LAYERS, prefill_layers=LAYERS, decode_layers=LAYERS,
        writer_layers=list(range(22)), visual_hidden_exit_before=22, answer_attention_skip_layers=list(range(1, 7)),
        quality_execution='FROZEN_FUNCTIONAL_REFERENCE_WITH_CAUSAL_VISUAL_SIDECAR',
        persistent_kv='Functional reference collects all 30 active layers; no physical KV/speed claim',
        strict_numerical_status='FAIL_RETAINED', new_gpu_numerical_validation='NOT_RUN',
        repeat_smoke='NOT_RUN_USER_REQUEST; frozen engine unchanged', full_59_numerical_validated=False,
        questions=len(rows), counts=COUNTS, generation=old['generation'], caps=old['caps'],
        prompts=dict(GQA='Frozen Scheme B, unchanged', TextVQA='Frozen Paper-OCR, unchanged',
                     ScienceQA_IMG='P2', MMBench='P2; ATE internal legacy Dev Circular, not V1.1',
                     p2_suffix=p2['suffix_exact_from_prior'], POPE='COCO 9000; not old 8910', MME='Classic 14-task Acc+Acc+'),
        source_sha256=sources, frozen_files={n: sha(OUT/n) for n in ['sample_manifest.jsonl.gz', 'baselines.sqlite', 'reused.sqlite']},
        gpu_uuids=UUIDS, planned_new_generations=len(rows)-len(reused), exact_reused_records=len(reused),
        baseline_audit=[dict(task=k[0], condition=k[1], state=k[2], n=v) for k, v in sorted(known.items())],
        baseline_scope='Exact input/gold/image/revision/protocol/cap/device; historical OOM retained',
        oom_policy='Retain BLOCKED_OOM, clear own failed request, continue other unchanged inputs; no resize/backend change',
        non_oom_failure_policy='Stop own workers, preserve checkpoint; never write invalid score',
        bootstrap_replicates=20000, bootstrap_seed=SEED, research_scope='Historical exploratory cohort; not independent noninferiority validation',
        base_commit='fc51d1fff7fec26fdc4f964d7c5e90365db89bf2', frozen_at=time.time())
    write(OUT/'protocol.json', protocol)
    write(OUT/'cpu_validation.json', dict(status='PASS_CPU_BINDINGS_ONLY', questions=len(rows), image_paths=len(images),
        task_counts=COUNTS, no_cuda_initialized=not torch.cuda.is_initialized(), exact_reuse=len(reused),
        historical_numerical_status='FAIL_RETAINED', complete_numerical_gate='NOT_PERFORMED', baseline_audit=protocol['baseline_audit']))
    print('FROZEN', len(rows), 'exact reuse', len(reused), flush=True)


def worker(shard):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from engine import ACTIVE, Request
    p = check()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == UUIDS[shard]
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model = output = reused = None
    started = time.time()
    write(OUT/f'start{shard}.json', dict(pid=os.getpid(), gpu_uuid=UUIDS[shard], time=started))
    try:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert UUIDS[shard] not in apps, 'GPU became occupied'
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}).eval()
        output, reused = Store(OUT/f'worker{shard}.sqlite'), Store(OUT/'reused.sqlite')
        rows = [r for r in manifest() if r['shard'] == shard]
        rows.sort(key=lambda r: (TASK_ORDER[r['benchmark']], r['image_sha256'], r['key']))
        counts = Counter()
        for i, row in enumerate(rows):
            assert not (OUT/'STOP.json').exists(), 'Own experiment stop requested'
            assert shutil.disk_usage('/data').free >= 2*2**30, 'STOP_DISK'
            prior = output.get(row['key']) or reused.get(row['key'])
            if prior is not None:
                assert prior['input_binding_sha256'] == row['input_binding_sha256']
                assert prior['gpu_uuid'] == UUIDS[shard] and prior['layers'] == LAYERS
                if not prior.get('reused_from'):
                    assert prior['protocol_sha256'] == sha(OUT/'protocol.json')
                counts[row['benchmark'], state(prior)] += 1
                continue
            write(OUT/f'current{shard}.json', dict(key=row['key'], task=row['benchmark'], index=i,
                total=len(rows), elapsed_s=time.time()-started, time=time.time()))
            inputs, layout = prepare(processor, row)
            assert digest(layout) == row['input_binding_sha256'], ('INPUT_BINDING', row['key'])
            assert hashlib.sha256(layout['prompt'].encode()).hexdigest() == row['prompt_sha256']
            began = time.time()
            record = dict(key=row['key'], benchmark=row['benchmark'], question_id=row['question_id'], question=row['question'],
                gold=row['gold'], image_sha256=row['image_sha256'], image_tokens=row['image_tokens'], input_length=row['input_length'],
                max_new_tokens=row['max_new_tokens'], condition=NAME, layers=LAYERS, prefill_layers=LAYERS, decode_layers=LAYERS,
                prompt_sha256=row['prompt_sha256'], input_binding_sha256=digest(layout), protocol_sha256=sha(OUT/'protocol.json'),
                gpu_uuid=UUIDS[shard], **{k: row.get(k) for k in ['strategy', 'mme_task', 'pair_id', 'circular_group']})
            try:
                with Request(model, processor, inputs, layout, collect=ACTIVE, prefill_layers=LAYERS,
                             cap=row['max_new_tokens']) as request:
                    result, _ = request.generate(LAYERS)
                assert result['prefill_layers'] == result['decode_layers'] == LAYERS
                result.update(scored(row, result['text'], result['truncated']))
                record.update(result, state='COMPLETE')
                record['scoring'] = {k: record.get(k) for k in ['score', 'parsed', 'parse_failed']}
                del result, request
            except torch.OutOfMemoryError as exc:
                record.update(state='BLOCKED_OOM', error=str(exc), failure_trace=traceback.format_exc(),
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
                # Failed request frames can retain tensors until their traceback is released.
                if 'request' in locals():
                    del request
            record['elapsed_s'] = time.time()-began
            output.put(row['key'], record)
            counts[row['benchmark'], state(record)] += 1
            oom = state(record) == 'BLOCKED_OOM'
            del inputs, layout, record
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
            if (i+1) % 50 == 0 or i+1 == len(rows):
                progress = dict(processed=i+1, planned=len(rows), time=time.time(), elapsed_s=time.time()-started,
                    counts=[dict(task=k[0], state=k[1], n=v) for k,v in sorted(counts.items())])
                write(OUT/f'progress{shard}.json', progress)
                print('PROGRESS', json.dumps(progress), flush=True)
        assert len(rows) == sum(counts.values())
        write(OUT/f'done{shard}.json', dict(status='ALL_INPUTS_ATTEMPTED', planned=len(rows),
            counts=[dict(task=k[0], state=k[1], n=v) for k,v in sorted(counts.items())], time=time.time(), elapsed_s=time.time()-started))
    except BaseException as exc:
        write(OUT/f'failure{shard}.json', dict(error=repr(exc), trace=traceback.format_exc(), time=time.time()))
        raise
    finally:
        for store in [output, reused]:
            if store is not None:
                store.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'release{shard}.json', dict(pid=os.getpid(), time=time.time(), gpu_uuid=UUIDS[shard]))


def launch():
    check()
    assert not (OUT/'launch.json').exists(), 'Reject duplicate launch; resume must be explicit'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert not any(uuid in apps for uuid in UUIDS), apps
    write(OUT/'gpu_preflight.json', dict(time=time.time(), authorized=UUIDS, processes=apps,
        sample=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv'], text=True)))
    children, streams = [], []
    started = time.time()
    try:
        for shard, uuid in enumerate(UUIDS):
            stream = (OUT/f'worker{shard}.log').open('a')
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1'),
                stdout=stream, stderr=subprocess.STDOUT))
        write(OUT/'launch.json', dict(pid=os.getpid(), worker_pids=[p.pid for p in children], time=started,
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
        while any(child.poll() is None for child in children):
            assert all(child.poll() in [None, 0] for child in children), 'WORKER_FAILURE'
            assert time.time()-started < 12*3600, 'Bounded twelve-hour execution timeout'
            time.sleep(5)
        assert all(child.returncode == 0 for child in children)
        assert all((OUT/f'done{i}.json').exists() for i in [0, 1])
        write(OUT/'generation_complete.json', dict(status='ALL_INPUTS_ATTEMPTED_NOT_A_NUMERICAL_PASS',
            planned=35346, elapsed_s=time.time()-started, time=time.time()))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=60)
        for stream in streams:
            stream.close()
        write(OUT/'gpu_release_audit.json', dict(time=time.time(), exit_codes=[c.returncode for c in children],
            processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv'], text=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.launch:
        launch()
    elif args.worker is not None:
        worker(args.worker)
    else:
        parser.error('Choose --freeze, --launch or --worker')
