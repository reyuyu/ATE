"""GPU1-only smoke and atomic complete-question checkpoint collection."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import torch
from safetensors.torch import load_file, save_file

from .common import (OUT, RAW, QWEN, INTERN, MODELS, read, write, sha256, digest, verify_sources)
from .engine import Adapter, hash_tree
from .metrics import metrics


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def idle():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and torch.cuda.device_count() == 1
    processes = command('nvidia-smi', '-i', '1', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader')
    assert not processes, f'GPU1 occupied: {processes}; stopping, no other GPU permitted'
    return command('nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader')


class Monitor:
    def __init__(self, name, mode):
        self.name, self.mode, self.rows = name, mode, []
        self.stop = threading.Event()
        self.conflict = None

    def sample(self):
        value = command('nvidia-smi', '-i', '1', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits')
        for row in value.splitlines():
            pid, uuid, memory = [v.strip() for v in row.split(',')]
            if int(pid) != os.getpid():
                self.conflict = row
            else:
                self.rows.append(dict(time=time.time(), pid=int(pid), uuid=uuid, memory_mib=int(memory)))

    def __enter__(self):
        def sample():
            while not self.stop.wait(2):
                self.sample()
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        write(OUT / f'{self.name}_{self.mode}_memory.json', dict(rows=self.rows,
            peak_process_memory_mib=max((r['memory_mib'] for r in self.rows), default=0),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2, conflict=self.conflict))


def save_raw(path, full, off):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    save_file(dict(full=full.contiguous(), off=torch.stack(off).contiguous()), str(temporary))
    os.replace(temporary, path)


def smoke(adapter, inputs, samples, monitor):
    name = adapter.name
    old_dir = QWEN if name == 'qwen' else INTERN
    old = sorted(read(old_dir / 'sample_manifest.json')['samples'], key=lambda s: (s['task'], s['question_id']))[:2]
    cases = []
    for historical in old:
        key = historical['task'], historical['question_id']
        sample, frozen = samples[key], inputs[key]
        adapter.prepare(sample, frozen)
        full, reference = adapter.forward(native=True, trace=True)
        visual_hash = hash_tree(adapter.visual)
        replay, _ = adapter.forward()
        noop, _ = adapter.forward(2, noop=True)
        assert torch.equal(full, replay) and torch.equal(full, noop)
        if name == 'qwen':
            record = read(QWEN / 'native_checkpoints' / ('__'.join(key) + '.json'))
            assert sha256(record['raw_path']) == record['raw_sha256']
            historical_full = load_file(record['raw_path'])['logits']
            old_scan = read(QWEN / 'scan_checkpoints' / ('__'.join(key) + '.json'))
            assert sha256(old_scan['raw_path']) == old_scan['raw_sha256']
            old_off = load_file(old_scan['raw_path'])['logits']
            targets = [0, 2, 17, 24, 35]
        else:
            record = read(INTERN / 'checkpoints' / ('__'.join(key) + '.json'))
            assert sha256(record['native_first_logits_path']) == record['native_first_logits_file_sha256']
            raw = load_file(record['native_first_logits_path'])
            historical_full = raw['native_first_logits']
            old_off = None
            targets = [0, 2, 19, 27]
        assert torch.equal(full, historical_full), ('Historical native logits mismatch', name, key,
                                                  float((full.float() - historical_full.float()).abs().max()))
        regressions = []
        for layer in targets:
            off, records = adapter.forward(layer, trace=True)
            if old_off is not None:
                assert torch.equal(off, old_off[layer]), ('Historical OFF logits mismatch', name, key, layer)
            else:
                prior = record['sensitivity'][layer]
                new = metrics(full, off)
                assert prior['layer'] == layer
                assert prior['signed_logp_loss'] == new['signed_loss']
                assert prior['absolute_logp_change'] == new['absolute_change']
                assert prior['off_first_token'] == new['off_token_id']
            for i, (baseline, changed) in enumerate(zip(reference, records)):
                assert baseline['attention_prefix_sha256'] == changed['attention_prefix_sha256']
                assert baseline['output_prefix_sha256'] == changed['output_prefix_sha256']
                assert (torch.count_nonzero(changed['attention']) == 0) if i == layer else (torch.count_nonzero(changed['attention']) > 0)
            regressions.append(dict(layer=layer, historical_off_logits_available=old_off is not None,
                historical_loss_error=0., **metrics(full, off)))
        assert hash_tree(adapter.visual) == visual_hash
        adapter.verify()
        cases.append(dict(task=key[0], question_id=key[1], noop_max_logit_error=0., native_replay_max_logit_error=0.,
            history_native_max_logit_error=0., non_p1_queries_exact=True, residual_equations_exact=True,
            other_layers_not_zeroed=True, all_mlp_executed=True, visual_replay_hash=visual_hash,
            independent_cache_lengths_valid=True, regressions=regressions))
        print(name, 'SMOKE PASS', key, flush=True)
        adapter.clear()
    assert not monitor.conflict
    write(OUT / f'{name}_gpu_smoke.json', dict(passed=True, protocol_sha256=sha256(OUT / 'protocol.json'),
        cases=cases, no_nan_inf_oom=True, no_hook_residue=True, weights_unchanged=True, no_input_contamination=True))
    if name == 'qwen':
        audit = read(OUT / 'qwen_legacy_validation.json')
        audit.update(gpu_regression='passed', gpu_smoke_sha256=sha256(OUT / 'qwen_gpu_smoke.json'))
        write(OUT / 'qwen_legacy_validation.json', audit)


def collect(adapter, inputs, samples, monitor):
    name, layers = adapter.name, adapter.length
    assert read(OUT / f'{name}_gpu_smoke.json')['passed']
    if name == 'qwen':
        assert read(OUT / 'qwen_legacy_validation.json')['gpu_regression'] == 'passed'
    binding = sha256(OUT / 'protocol.json')
    commit = command('git', 'rev-parse', 'HEAD')
    start = time.perf_counter()
    total, reused = 0, 0
    for key, sample in samples.items():
        assert not monitor.conflict, f'External GPU1 task detected: {monitor.conflict}'
        path = OUT / name / 'checkpoints' / ('__'.join(key) + '.json')
        frozen = inputs[key]
        if path.exists():
            old = read(path)
            assert old['complete'] and old['protocol_sha256'] == binding and old['input_binding'] == frozen['input_binding']
            assert len(old['rows']) == layers and [r['layer'] for r in old['rows']] == list(range(layers))
            assert sha256(old['raw_path']) == old['raw_sha256']
            total += 1
            reused += 1
            continue
        began = time.perf_counter()
        try:
            adapter.prepare(sample, frozen)
            full, _ = adapter.forward(native=True)
            # Every sample validates complete native visual replay, not just the smoke inputs.
            replay, _ = adapter.forward()
            assert torch.equal(full, replay), 'Visual replay is not bit-exact; no invalid results permitted'
            visual_hash = hash_tree(adapter.visual)
            off_vectors, rows = [], []
            for layer in range(layers):
                changed, _ = adapter.forward(layer)
                rows.append(dict(layer=layer, relative_depth=layer / (layers - 1), **metrics(full, changed)))
                off_vectors.append(changed)
            adapter.verify()
            assert hash_tree(adapter.visual) == visual_hash
            raw = RAW / name / ('__'.join(key) + '.safetensors')
            save_raw(raw, full, off_vectors)
            write(path, dict(complete=True, model=name, task=key[0], question_id=key[1],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'],
                input_length=frozen['layout']['input_length'], image_tokens=frozen['layout']['image_tokens'],
                input_binding=frozen['input_binding'], input_tensor_hashes=adapter.input_hashes,
                protocol_sha256=binding, code_commit=commit, raw_path=str(raw), raw_sha256=sha256(raw),
                logits_dtype=str(full.dtype), raw_shape=[layers, full.numel()], visual_replay_exact=True,
                visual_sha256=visual_hash, seconds=time.perf_counter() - began, rows=rows))
            total += 1
            adapter.clear()
            elapsed = time.perf_counter() - start
            write(OUT / f'{name}_progress.json', dict(completed=total, total=3000, reused_checkpoints=reused,
                elapsed_seconds=elapsed, new_question_seconds=elapsed / max(total - reused, 1),
                eta_seconds=(3000 - total) * elapsed / max(total - reused, 1), last_sample=key, pid=os.getpid()))
            if total % 10 == 0:
                print(f'{name} COMPLETE {total}/3000 elapsed={elapsed:.1f}s eta={(3000-total)*elapsed/max(total-reused,1)/3600:.2f}h', flush=True)
        except BaseException:
            write(OUT / name / 'failures' / ('__'.join(key) + f'__{time.time_ns()}.json'),
                  dict(task=key[0], question_id=key[1], traceback=traceback.format_exc(), protocol_sha256=binding))
            raise
    write(OUT / f'{name}_completion.json', dict(complete=True, questions=total, full_prefills=total,
        off_prefills=layers * total, extra_noop_replay_prefills=total, seconds=time.perf_counter() - start,
        reused_checkpoints=reused, code_commit=commit, protocol_sha256=binding))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'internvl'])
    parser.add_argument('mode', choices=['smoke', 'run'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_sources()
    assert read(OUT / f'cpu_tests_{MODELS[args.model]["transformers"]}.json')['passed']
    frozen = read(OUT / f'{args.model}_input_manifest.json')['samples']
    inputs = {(s['task'], s['question_id']): s for s in frozen}
    samples = {(s['task'], s['question_id']): s for s in read(OUT / 'shared_3000_manifest.json')['samples']}
    assert set(inputs) == set(samples) and len(samples) == 3000
    uuid = idle()
    write(OUT / f'{args.model}_{args.mode}_preflight.json', dict(physical_gpu=1, uuid=uuid,
        pid=os.getpid(), process_device='cuda:0', all_gpus=command('nvidia-smi')))
    start = time.perf_counter()
    adapter, state = None, dict(passed=False, model=args.model, mode=args.mode)
    try:
        with Monitor(args.model, args.mode) as monitor:
            adapter = Adapter(args.model)
            mapped = command('nvidia-smi', '-i', '1', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader')
            assert any(int(line.split(',')[0]) == os.getpid() and uuid in line for line in mapped.splitlines())
            if args.mode == 'smoke':
                smoke(adapter, inputs, samples, monitor)
            else:
                collect(adapter, inputs, samples, monitor)
            state.update(passed=True, no_nan_inf_oom=True, weights_unchanged=True, hooks_clean=True)
    except BaseException:
        state['traceback'] = traceback.format_exc()
        raise
    finally:
        state.update(seconds=time.perf_counter() - start, pid=os.getpid(), uuid=uuid)
        write(OUT / f'{args.model}_{args.mode}_execution.json', state)
        adapter = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
