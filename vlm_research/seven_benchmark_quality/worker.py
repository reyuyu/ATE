"""One resident LLaVA per authorized GPU; independent paired caches."""
import argparse
from contextlib import nullcontext
import gc
import os
import signal
import time
import traceback

from .common import *


def stop_signal(signum, frame):
    raise RuntimeError(f'Stopping this worker on signal {signum}')


def forced(adapter, layout, condition, tokens):
    from ..mme_seed_quality.engine import control, call
    controller = control(adapter, layout, condition, 32, False)
    cache = None
    vectors = []
    with controller.context() if controller else nullcontext():
        for t in range(min(3, len(tokens))):
            value, cache = call(adapter, layout, cache, t, tokens[t-1] if t else None)
            vectors.append(value.detach().cpu().clone())
    del cache, controller
    return vectors


def smoke(adapter, sample, rank):
    import torch
    from ..mme_seed_quality.engine import prepare, greedy, control
    from ..mme_seed_quality.worker import native_reference, block_audit, assert_vectors
    from ..mme_seed_quality.common import hook_counts, tensor_sha
    from ..physical_visual_handoff.common import TOLERANCES
    layout, binding = prepare(adapter, sample)
    before = hook_counts(adapter.model)
    baseline_inputs = {k: tensor_sha(v) for k, v in adapter.inputs.items()}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with native_reference(adapter) as refs:
        native, nv = greedy(adapter, layout, 'Native', capture=True)
    noop, ov = greedy(adapter, layout, 'Native', capture=True, noop=True)
    assert_vectors(nv, ov)
    assert native['token_ids'] == noop['token_ids']
    hf = adapter.model.generate(**adapter.inputs, do_sample=False, num_beams=1,
        max_new_tokens=32, use_cache=True, return_dict_in_generate=True,
        output_scores=True, logits_to_keep=1)
    assert hf.sequences[0, layout.length:].tolist() == native['token_ids']
    assert_vectors(nv, [v[0].detach().cpu() for v in hf.scores])
    del hf, nv, ov
    with block_audit(adapter, layout, refs) as blocks:
        physical, pv = greedy(adapter, layout, 'Physical-C3', capture=True, audit=True)
    assert len(blocks) == 32 * len(physical['token_ids'])
    audit = physical['audit']
    assert audit['events'] and all(v['blocked_probability_max'] == 0 for v in audit['events'])
    assert all(v['mlp_calls'] == 1 and v['norm_calls'] == 2 for v in blocks)
    for t in range(len(physical['token_ids'])):
        for i in range(32):
            r = blocks[t*32+i]
            expected = layout.length if t == 0 and i < 20 else len(layout.text) if t == 0 else 1
            assert r['hidden_rows'] == expected
            if i in (2, 3, 4, 5):
                if t:
                    assert not r['qkv'], 'Decode Attention projections must be physically absent'
                else:
                    assert r['qkv']['q_proj'][1] == layout.length - 1
    for r in audit['kv']['layers']:
        if r['layer'] in (2, 3, 4, 5):
            assert r.get('used_tokens', r.get('live_tokens')) == 0
        else:
            assert r['original_positions'][:len(layout.text)] == list(layout.text)
    dense = forced(adapter, layout, 'Dense-F', physical['token_ids'])
    tol = TOLERANCES['llava']
    diffs = []
    for x, y in zip(pv, dense):
        delta = (x.float() - y.float()).abs()
        diffs.append(dict(mean_abs=float(delta.mean()), max_abs=float(delta.max()),
                          top1_equal=int(x.argmax()) == int(y.argmax())))
        assert torch.allclose(x, y, atol=tol['logit_atol'], rtol=tol['logit_rtol']), diffs
    # Repeat a cold request: no controller/cache storage is shared across requests.
    repeat, rv = greedy(adapter, layout, 'Physical-C3', capture=True)
    assert physical['token_ids'] == repeat['token_ids']
    assert_vectors(pv, rv)
    funcs = [l.self_attn.forward.__func__ for l in adapter.layers]
    try:
        with control(adapter, layout, 'Physical-C3', 32, False).context():
            raise RuntimeError('intentional cleanup test')
    except RuntimeError as error:
        assert str(error) == 'intentional cleanup test'
    assert funcs == [l.self_attn.forward.__func__ for l in adapter.layers]
    assert before == hook_counts(adapter.model)
    assert baseline_inputs == {k: tensor_sha(v) for k, v in adapter.inputs.items()}
    memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_bytes=torch.cuda.get_device_properties(0).total_memory)
    assert memory['total_bytes'] - memory['peak_reserved_bytes'] >= 512 * 1024**2
    result = dict(status='PASSED', at=now(), key=sample['key'], rank=rank,
        native_noop_all_effective_logits_torch_equal=True, native_hf_tokens_logits_exact=True,
        mask_rotary_non_target_scope_exact=True, mlp_norm_residual_exact=True,
        no_persistent_image_kv=True, physical_query_shapes_correct=True,
        cold_cache_replay_exact=True, normal_exception_cleanup=True,
        input_tensors_unchanged=True, binding=binding, block_audit=blocks,
        dense_reference_same_prefix_diff=diffs, frozen_tolerances=tol,
        native=native, physical=physical, memory=memory)
    atomic(OUT / f'smoke/worker{rank}/{sample["key"]}.json', result)
    print('SMOKE_PASSED', rank, sample['key'], flush=True)
    del refs, blocks, pv, rv, dense
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    rank = args.rank
    assert os.environ['CUDA_VISIBLE_DEVICES'] == DEVICES[rank]
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    signal.signal(signal.SIGTERM, stop_signal)
    protocol = verify()
    safety()
    import torch
    import transformers
    assert transformers.__version__ == '4.49.0'
    from ..mme_seed_quality.engine import load, prepare, greedy
    from ..mme_seed_quality.common import tensor_sha
    model_binding = read(OUT / 'model_binding.json')
    for name, h in model_binding['files'].items():
        assert sha(Path(MODELS['llava']) / name) == h, name
    tag = dict(rank=rank, uuid=DEVICES[rank], pid=os.getpid(), at=now(),
               protocol_sha256=digest(protocol), source_commit=os.popen('git rev-parse HEAD').read().strip())
    atomic(OUT / f'worker{rank}/environment.json', dict(**tag, transformers=transformers.__version__,
        torch=torch.__version__, python=os.sys.version, cuda=torch.version.cuda))
    db = database(rank, protocol)
    adapter = None
    started = time.monotonic()
    total_new = 0
    try:
        adapter = load('llava')
        assert str(next(adapter.model.parameters()).dtype) == 'torch.float16'
        selection = read(OUT / 'smoke_selection.json')
        for bench in ORDER:
            safety()
            rows = [s for s in samples(bench) if s['shard'] == rank]
            lookup = {s['key']: s for s in rows}
            for key in selection[bench][str(rank)]:
                smoke(adapter, lookup[key], rank)
            # All six per-benchmark smoke inputs must pass before either shard scores.
            # The supervisor starts both workers together; an explicit peer gate is
            # checked below, with failures signalled through STOP by the supervisor.
            atomic(OUT / f'worker{rank}/{bench}_smoke_passed.json', dict(status='PASSED', at=now()))
            peer = OUT / f'worker{1-rank}/{bench}_smoke_passed.json'
            deadline = time.monotonic() + 1800
            while not peer.exists():
                safety()
                assert time.monotonic() < deadline, 'Peer smoke did not finish within 30 minutes'
                time.sleep(2)
            complete = 0
            for sample in rows:
                safety()
                existing = dict(db.execute('SELECT condition,payload FROM results WHERE key=?',
                                           (sample['key'],)).fetchall())
                for condition, payload in existing.items():
                    record = json.loads(payload)
                    assert record['sample_sha256'] == digest(sample)
                    assert record['protocol_sha256'] == digest(protocol)
                    saved = db.execute('SELECT sha256 FROM results WHERE key=? AND condition=?',
                                       (sample['key'], condition)).fetchone()[0]
                    assert saved == digest(record)
                missing = [c for c in CONDITIONS if c not in existing]
                if missing:
                    assert sha(sample['image_path']) == sample['image_sha256']
                    layout, binding = prepare(adapter, sample)
                    baseline = {k: tensor_sha(v) for k, v in adapter.inputs.items()}
                    for condition in missing:
                        generation, _ = greedy(adapter, layout, condition)
                        assert baseline == {k: tensor_sha(v) for k, v in adapter.inputs.items()}
                        record = dict(key=sample['key'], condition=condition, benchmark=bench,
                            question_id=sample['source_question_id'], image_sha256=sample['image_sha256'],
                            sample_sha256=digest(sample), protocol_sha256=digest(protocol),
                            input_binding=binding, generation=generation,
                            scoring=score(sample, generation['text']), at=now())
                        put(db, sample, condition, record)
                        total_new += 1
                complete += 1
                if complete % 20 == 0 or complete == len(rows):
                    status = dict(**tag, status='RUNNING', benchmark=bench, paired_questions=complete,
                        benchmark_questions=len(rows), new_generations=total_new,
                        elapsed_seconds=time.monotonic()-started, last_key=sample['key'], updated_at=now())
                    atomic(OUT / f'worker{rank}/progress.json', status)
                    print('PROGRESS', json.dumps(status), flush=True)
            atomic(OUT / f'worker{rank}/{bench}_generation_complete.json',
                   dict(status='GENERATED', paired_questions=len(rows), at=now()))
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic(OUT / f'worker{rank}/complete.json', dict(**tag, status='GENERATED_ALL_SPLITS', finished_at=now()))
    except BaseException:
        error = dict(**tag, status='STOPPED', error=traceback.format_exc(), failed_at=now())
        atomic(OUT / f'worker{rank}/failure.json', error)
        atomic(OUT / 'STOP_FAILURE.json', error)
        print(error['error'], flush=True)
        raise
    finally:
        db.close()
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        atomic(OUT / f'worker{rank}/release.json', dict(**tag, released_at=now(),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()))


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        main()
