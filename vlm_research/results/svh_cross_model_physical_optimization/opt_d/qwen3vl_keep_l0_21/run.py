"""Audit the exact historical baseline, freeze, and execute only 3,000 new answers."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import unittest

import torch

from .common import *
from .engine import generate, controller
from ..qwen3vl_early_windows.experiment import tensor_digest, hook_counts, gqa_prediction
from ..qwen3vl_query_role.engine import generate as native_generate, feature_hashes
from ..qwen3vl_query_role.run import environment, check_models
from ..qwen3vl_visual_writing_necessity.engine import generate as historical_generate
from ..qwen3vl_visual_writing_necessity.run import checked_inputs, cache_hash


def cpu():
    from . import tests
    from ..qwen3vl_semantic_visual_writers import tests as inherited
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
        unittest.defaultTestLoader.loadTestsFromModule(inherited)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    seal(OUT/'cpu_tests.json', dict(passed=True, tests=result.testsRun, sources=sources(), cuda_initialized=False))


def freeze():
    from tokenizers import Tokenizer
    assert not (OUT/'protocol.json').exists()
    assert read(OUT/'cpu_tests.json')['sources'] == sources()
    assert command('git', 'status', '--porcelain', '--', 'vlm_research/qwen3vl_keep_l0_21') == ''
    names = ['sample_manifest.json', 'input_bindings.json', 'shards.json', 'model_manifest.json',
        'environment.json', 'official_gqa_prediction.json', 'historical_overlap.json']
    provenance = {}
    for name in names + ['protocol.json', 'checkpoint_hashes.json', 'completion.json', 'validation.json',
                         'raw_checkpoints_shard0.tar.gz', 'raw_checkpoints_shard1.tar.gz']:
        provenance[name] = committed(name)
    old = read(HISTORY/'protocol.json')
    assert read(HISTORY/'completion.json')['status'] == 'COMPLETE'
    assert old['model_revision'] == REVISION and old['dtype'] == 'bfloat16' and old['transformers'] == '4.57.6'
    for name in names:
        shutil.copyfile(HISTORY/name, OUT/name)
    samples = read(OUT/'sample_manifest.json')['samples']
    specs = read(OUT/'input_bindings.json')
    shards = read(OUT/'shards.json')['keys']
    assert len(samples) == 3000 and len({s['image_sha256'] for s in samples}) == 2409
    assert sorted(Counter(s['task'] for s in samples).values()) == [500]*6
    assert [len(s) for s in shards] == [1500, 1500] and set(shards[0]).isdisjoint(shards[1])
    index = {r['path']: r['sha256'] for r in read(HISTORY/'checkpoint_hashes.json')}
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    baseline, record_sources = {}, []
    for sample in samples:
        identifier = key(sample)
        shard = 0 if identifier in shards[0] else 1
        rows = {}
        for c in ['All-Writer', 'No-Writer', 'Drop-L35']:
            relative = f'shard{shard}/checkpoints/{identifier}/{c}.json'
            path = HISTORY/relative
            assert sha256(path) == index[relative]
            r = read(path)
            assert r['binding'] == digest(old) and r['sample_sha256'] == digest(sample)
            assert r['input_binding'] == specs[identifier]['binding']
            assert r['image_sha256'] == sample['image_sha256'] and r['gold'] == sample['answer']
            g = r['generation']
            assert digest(g) == r['generation_sha256']
            assert g['P1_Decode_image_off'] and g['logits_to_keep'] == 1
            assert g['writer_layers'] == ([] if c == 'No-Writer' else list(range(35)) if c == 'Drop-L35' else list(range(36)))
            text = tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
            assert text == g['text'] and gqa_prediction(text) == g['prediction']
            assert g['correct'] == int(g['prediction'] == sample['answer'].lower())
            rows[c] = r
            record_sources.append(dict(key=identifier, condition=c, path=str(path), sha256=index[relative]))
        assert rows['Drop-L35']['structural_negative_control_exact']
        assert rows['Drop-L35']['effective_logits_sha256'] == rows['All-Writer']['effective_logits_sha256']
        assert rows['Drop-L35']['generation']['token_ids'] == rows['All-Writer']['generation']['token_ids']
        baseline[identifier] = {c: rows[c] for c in ('All-Writer', 'No-Writer')}
    assert sum(r['All-Writer']['generation']['correct'] for r in baseline.values()) == 2637
    assert sum(r['No-Writer']['generation']['correct'] for r in baseline.values()) == 1634
    seal(OUT/'historical_baselines.json', baseline)
    seal(OUT/'historical_reuse_audit.json', dict(passed=True, source_commit=BASE, artifacts=provenance,
        records_verified=9000, paired_All_Writer=3000, paired_No_Writer=3000, structural_Drop_L35_controls=3000,
        independently_decoded_and_rescored=True, source_records=record_sources))
    images = set(read(OUT/'historical_overlap.json')['historical_images'])
    disjoint = [key(s) for s in samples if s['image_sha256'] not in images]
    assert len(disjoint) == 2640
    seal(OUT/'image_disjoint_manifest.json', dict(keys=disjoint, count=2640, historical_overlap_sha256=sha256(OUT/'historical_overlap.json')))
    check_models(read(OUT/'model_manifest.json'))
    assert environment() == read(OUT/'environment.json'), 'Frozen Qwen runtime changed'
    snapshot, candidates = hardware(), []
    for i in (0, 3, 2, 1):
        gpu = next(g for g in snapshot['gpus'] if int(g['index']) == i)
        try:
            assert_idle(gpu['uuid'], snapshot)
            candidates.append(gpu)
        except AssertionError:
            pass
    if len(candidates) < 2:
        seal(OUT/'gpu_preflight.json', dict(status='STOPPED_INSUFFICIENT_IDLE_GPUS', snapshot=snapshot))
        raise RuntimeError('Two idle GPUs required; no automatic waiting or polling')
    seal(OUT/'gpu_preflight.json', dict(status='READY', timestamp=time.time(), snapshot=snapshot, selected=candidates[:2]))
    artifacts = names+['historical_baselines.json', 'historical_reuse_audit.json', 'image_disjoint_manifest.json', 'cpu_tests.json']
    p = dict(title='Qwen3-VL joint R-to-Image writing permission: Keep-L0-L21', source_commit=BASE,
        code_commit=command('git', 'rev-parse', 'HEAD'), model_revision=REVISION,
        dtype='bfloat16', transformers='4.57.6', attention='eager', open_layers=OPEN_LAYERS,
        closed_layers=list(range(22, 36)), P1_Decode_image_off_all_36_layers=True,
        unchanged_R_definition=old['R_definition'], unchanged_DeepStack_mRoPE=True,
        generation=dict(greedy=True, max_new_tokens=32, logits_to_keep=1, independent_native_KV=True),
        questions=3000, image_clusters=2409, new_conditions=[CONDITION], new_generations=3000,
        reused_conditions=['All-Writer', 'No-Writer'], primary_comparison='Keep-L0-L21 minus All-Writer',
        primary_noninferiority_margin_pp=-1., sensitivity_margin_pp=-.5,
        bootstrap=dict(repetitions=20000, seed=SEED, whole_image_SHA256_cluster_pairs=True,
            question_weighted=True, two_sided_quantiles=[.025, .975], one_sided_95_lower_quantile=.05),
        decision_rule='Support NI only if one-sided 95% lower bound strictly exceeds -1 pp; no multiple correction for one primary comparison',
        No_Writer_reference_only=True, gpus=candidates[:2], smoke_keys=old['smoke_keys'],
        sources=sources(), official_sources=old['official_sources'],
        artifacts={name: sha256(OUT/name) for name in artifacts})
    seal(OUT/'protocol.json', p)
    print('FROZEN', digest(p), 'historical paired baselines independently audited', 'GPUs', [g['index'] for g in p['gpus']], flush=True)


def smoke(model, processor, samples, specs, baselines, p, shard):
    by = {key(s): s for s in samples}
    cases = []
    for identifier in p['smoke_keys']:
        sample, spec = by[identifier], specs[identifier]
        inputs = checked_inputs(processor, sample, spec)
        inputs_before = {k: tensor_digest(v) for k, v in inputs.items()}
        weights = [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        native, nscore, nt, parent = native_generate(model, processor, inputs, spec['roles'], native=True, audit=True, trace=True)
        feature_ref, parent_hash = feature_hashes(nt), cache_hash(parent)
        noop, score, _, own = generate(model, processor, inputs, spec, noop=True, audit=True)
        assert noop['token_ids'] == native['token_ids'] and len(score) == len(nscore)
        assert all(torch.equal(a, b) for a, b in zip(score, nscore))
        assert all(a.keys.data_ptr() != b.keys.data_ptr() and a.values.data_ptr() != b.values.data_ptr() for a, b in zip(parent.layers, own.layers))
        del own, score
        full, fscore, ft, fc = historical_generate(model, processor, inputs, spec, 'All-Writer', audit=True, trace=True)
        old = baselines[identifier]['All-Writer']
        assert full['token_ids'] == old['generation']['token_ids']
        assert [tensor_digest(v) for v in fscore] == old['effective_logits_sha256'], 'Historical All-Writer logits differ'
        drop, score, _, own = historical_generate(model, processor, inputs, spec, 'Drop-L35', audit=True)
        assert drop['token_ids'] == full['token_ids'] and len(score) == len(fscore)
        assert all(torch.equal(a, b) for a, b in zip(score, fscore))
        del own, score
        changed, score, trace, own = generate(model, processor, inputs, spec, audit=True, trace=True)
        assert trace.validate() and trace.mlp_calls == [len(changed['token_ids'])]*36
        assert feature_hashes(trace) == feature_ref == feature_hashes(ft)
        assert cache_hash(parent) == parent_hash
        assert all(a.keys.data_ptr() != b.keys.data_ptr() and a.values.data_ptr() != b.values.data_ptr() for a, b in zip(fc.layers, own.layers))
        for layer in range(22):
            assert torch.equal(ft.records[layer]['attention'], trace.records[layer]['attention'])
        assert torch.equal(ft.records[22]['input'], trace.records[22]['input'])
        keep = [q for q in range(spec['layout']['input_length']) if q not in spec['roles']['R']]
        assert torch.equal(ft.records[22]['attention'][:, keep], trace.records[22]['attention'][:, keep])
        assert changed['source_audit']['blocked_probability_max'] == 0
        counts = hook_counts(model)
        try:
            with controller(model, spec):
                raise RuntimeError('intentional cleanup')
        except RuntimeError as error:
            assert str(error) == 'intentional cleanup'
        assert hook_counts(model) == counts
        assert inputs_before == {k: tensor_digest(v) for k, v in inputs.items()}
        assert weights == [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        cases.append(dict(key=identifier, passed=True, native_noop_all_valid_logits_torch_equal=True,
            historical_All_Writer_Drop_L35_all_logits_exact=True, all_22_open_layers_unchanged=True,
            non_target_Query_at_L22_exact=True, actual_probabilities_zero=True, native_image_queries=True,
            independent_cache_storage=True, weights_inputs_visual_features_unchanged=True,
            native_DeepStack_mRoPE_36_MLP=True, normal_exception_hook_cleanup=True))
        del inputs, nscore, nt, parent, fscore, ft, fc, trace, score, own
        gc.collect()
        torch.cuda.empty_cache()
        print('SMOKE PASS', identifier, flush=True)
    seal(OUT/f'shard{shard}/smoke.json', dict(passed=True, binding=digest(p), cases=cases, gpu_uuid=p['gpus'][shard]['uuid']))


def worker(shard):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    p = verify()
    folder = OUT/f'shard{shard}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert environment() == read(OUT/'environment.json')
    uuid = p['gpus'][shard]['uuid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    seal(folder/'preflight.json', dict(gpu=assert_idle(uuid), binding=digest(p)))
    check_models(read(OUT/'model_manifest.json'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'},
        use_safetensors=True, low_cpu_mem_usage=True).eval()
    assert len(model.model.language_model.layers) == 36
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == 'eager'
    samples = read(OUT/'sample_manifest.json')['samples']
    specs, baselines = read(OUT/'input_bindings.json'), read(OUT/'historical_baselines.json')
    with torch.inference_mode():
        smoke(model, processor, samples, specs, baselines, p, shard)
        wait_started = time.monotonic()
        while not all((OUT/f'shard{i}/smoke.json').exists() for i in range(2)):
            assert not list(OUT.glob('shard*/failure*.json')), 'Peer Smoke failed'
            assert time.monotonic()-wait_started < 600, 'Peer Smoke timeout'
            time.sleep(2)
        assert all(read(OUT/f'shard{i}/smoke.json')['binding'] == digest(p) for i in range(2))
        selected = set(read(OUT/'shards.json')['keys'][shard])
        work = [s for s in samples if key(s) in selected]
        weights = [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
        started = time.time()
        seal(folder/'execution.json', dict(started_unix=started, pid=os.getpid(), binding=digest(p), gpu_uuid=uuid))
        for i, s in enumerate(work):
            identifier, spec = key(s), specs[key(s)]
            path = checkpoint(shard, identifier)
            if path.exists():
                r = read(path)
                assert r['binding'] == digest(p) and r['sample_sha256'] == digest(s)
                assert r['input_binding'] == spec['binding'] and r['generation_sha256'] == digest(r['generation'])
            else:
                inputs = checked_inputs(processor, s, spec)
                before = {k: tensor_digest(v) for k, v in inputs.items()}
                g, scores, _, cache = generate(model, processor, inputs, spec)
                assert before == {k: tensor_digest(v) for k, v in inputs.items()}
                assert weights == [(n, v._version, v.data_ptr()) for n, v in model.named_parameters()]
                full = baselines[identifier]['All-Writer']['generation']
                g.update(correct=int(g['prediction'] == s['answer'].lower()),
                    first_token_flip=int(g['token_ids'][0] != full['token_ids'][0]),
                    first_logits_sha256=tensor_digest(scores[0]))
                seal(path, dict(condition=CONDITION, binding=digest(p), code_commit=p['code_commit'],
                    task=s['task'], question_id=s['question_id'], image_id=s['image_id'], image_sha256=s['image_sha256'],
                    gold=s['answer'], sample_sha256=digest(s), input_binding=spec['binding'], gpu_uuid=uuid,
                    generation=g, generation_sha256=digest(g)))
                del inputs, scores, cache
            elapsed = time.time()-started
            atomic(folder/'progress.json', dict(status='RUNNING', completed=i+1, total=len(work), elapsed_seconds=elapsed,
                per_second=(i+1)/elapsed, last_key=identifier, timestamp=time.time(), binding=digest(p)))
            if (i+1) % 100 == 0:
                print('PROGRESS', i+1, '/', len(work), 'seconds', round(elapsed, 2), flush=True)
        seal(folder/'complete.json', dict(passed=True, generations=len(work), binding=digest(p),
            elapsed_seconds=time.time()-started, finished_unix=time.time(), gpu_uuid=uuid))
        atomic(folder/'progress.json', dict(status='COMPLETE', completed=len(work), total=len(work),
            elapsed_seconds=time.time()-started, timestamp=time.time(), binding=digest(p)))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    seal(folder/'release.json', dict(allocated_bytes=torch.cuda.memory_allocated(), gpu_uuid=uuid, timestamp=time.time()))


def launch():
    p = verify()
    assert command('git', 'branch', '--show-current') == 'codex/qwen3vl-keep-l0-21-joint-drop'
    children, streams = [], []
    try:
        for i, gpu in enumerate(p['gpus']):
            assert_idle(gpu['uuid'])
            folder = OUT/f'shard{i}'
            folder.mkdir(parents=True, exist_ok=True)
            stream = (folder/'worker.log').open('a')
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            children.append(subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.qwen3vl_keep_l0_21.run',
                'worker', '--shard', str(i)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT))
        seal(OUT/'launch.json', dict(timestamp=time.time(), pid=os.getpid(), worker_pids=[c.pid for c in children]))
        while any(c.poll() is None for c in children):
            assert all(c.poll() in (None, 0) for c in children), 'Worker failed; stop only our own peer worker'
            time.sleep(2)
        assert all(c.returncode == 0 for c in children)
        compute = command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
        assert not {str(c.pid) for c in children} & {v.split(',')[0].strip() for v in compute.splitlines()}
        seal(OUT/'gpu_release_audit.json', dict(own_workers_exited=True, own_compute_processes_absent=True,
            worker_pids=[c.pid for c in children], snapshot=compute, timestamp=time.time()))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (OUT/'analysis.log').open('a') as log:
            subprocess.run(['/data/lk/深度学习/.envs/ate-vlm-research/bin/python', '-m',
                'vlm_research.qwen3vl_keep_l0_21.analyze'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(OUT/'completion.json')['status'] == 'COMPLETE'
        subprocess.run(['git', 'add', 'vlm_research/qwen3vl_keep_l0_21', str(OUT.relative_to(ROOT))], cwd=ROOT, check=True)
        subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@localhost', 'commit', '-m',
            'Complete Qwen joint Drop L22-L35: 3000 paired answers and prespecified NI analysis'], cwd=ROOT, check=True)
        commit = command('git', 'rev-parse', 'HEAD')
        subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/codex/qwen3vl-keep-l0-21-joint-drop'],
            cwd=ROOT, env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), check=True)
        atomic(OUT/'delivery_status.json', dict(status='PUSHED', commit=commit))
        print('DELIVERED', commit, flush=True)
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic(OUT/f'launcher_failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'worker', 'launch'])
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    try:
        if args.stage == 'worker':
            worker(args.shard)
        else:
            globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            seal(OUT/f'shard{args.shard}/failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
