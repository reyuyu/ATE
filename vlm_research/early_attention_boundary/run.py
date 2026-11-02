"""GPU1-only sequential-model runner with atomic per-condition checkpoints."""
import argparse
from collections import Counter
import importlib.metadata
import os
import platform
import subprocess
import threading
import time
import traceback

import torch
from safetensors.torch import load_file, save_file

from .common import *
from .engine import GenerationAdapter
from ..internvl3_stage.engine import first_metrics
from ..internvl3_smoke.engine import hooks_count


class Monitor:
    def __init__(self):
        self.rows, self.stop = [], threading.Event()

    def poll(self):
        while not self.stop.is_set():
            row = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip()
            memory, util = [int(v.strip()) for v in row.split(',')]
            self.rows.append(dict(elapsed_seconds=time.monotonic()-self.start, memory_mib=memory, utilization=util))
            self.stop.wait(2)

    def __enter__(self):
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def smoke(adapter, samples, frozen):
    selected = [samples[0], max(samples, key=lambda s: frozen[key(s)]['layout']['input_length'])]
    selected = list({key(s): s for s in selected}.values())
    records = []
    for sample in selected:
        adapter.prepare(sample, frozen[key(sample)])
        native, logits = adapter.generate(audit=True)
        repeat, repeat_logits = adapter.generate()
        assert native['token_ids'] == repeat['token_ids'] and torch.equal(logits, repeat_logits)
        widest = max(CONDITIONS[adapter.name].values(), key=len)
        noop, noop_logits = adapter.generate(widest, noop=True, audit=True)
        assert native['token_ids'] == noop['token_ids'] and torch.equal(logits, noop_logits)
        historical = Path('/data/lk/ATE-resources/experiments/three_vlm_layer_sensitivity_3000') / adapter.name / (key(sample)+'.safetensors')
        previous = load_file(str(historical))['full']
        assert torch.equal(logits, previous), ('Native historical first logits mismatch', float((logits.float()-previous.float()).abs().max()))
        conditions = []
        for label, targets in CONDITIONS[adapter.name].items():
            if not targets:
                continue
            result, _ = adapter.generate(targets, audit=True)
            assert result['visual_hashes'] == native['visual_hashes'], 'Vision/DeepStack changed'
            assert result['prefill_prefix_hashes'] == native['prefill_prefix_hashes'], 'Other prefill queries changed'
            conditions.append(dict(condition=label, decode_steps=result['decode_steps'],
                                   every_decode_query_audited=True, output_zero_and_residual_checks=True))
        before = hooks_count(adapter.model)
        try:
            if adapter.name == 'qwen':
                from ..qwen3vl_early_windows.experiment import AttentionBypass
                context = AttentionBypass(adapter.model, widest, frozen[key(sample)]['layout']['input_length'])
            else:
                from ..internvl3_stage.engine import WindowHook
                context = WindowHook(adapter.layers, widest, frozen[key(sample)]['layout']['input_length'])
            with context:
                raise RuntimeError('intentional cleanup smoke')
        except RuntimeError as exc:
            assert str(exc) == 'intentional cleanup smoke'
        assert hooks_count(adapter.model) == before
        adapter.verify()
        records.append(dict(sample=key(sample), input_length=frozen[key(sample)]['layout']['input_length'],
            native_noop_logits_max_difference=0, repeated_full_exact=True, historical_full_logits_max_difference=0,
            all_non_target_prefill_prefixes_exact=True, native_vision_and_deepstack_unchanged=True,
            exception_hook_cleanup=True, conditions=conditions))
        adapter.clear()
        print('SMOKE PASS', adapter.name, key(sample), flush=True)
    assert any(c['decode_steps'] > 0 for r in records for c in r['conditions'])
    return dict(passed=True, records=records, nan_inf_oom=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(CONDITIONS))
    name = parser.parse_args().model
    protocol = verify()
    assert read(OUT / 'cpu_tests.json')['passed']
    initial = preflight()
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 1
    write(OUT / f'{name}_preflight.json', dict(snapshot=initial, physical_gpu=1, uuid=UUID,
          visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], logical_device='cuda:0'))
    bindings = read(OUT / 'model_and_environment_bindings.json')[name]
    for entry in bindings['verified_model_files']['files']:
        file = model_path(name) / entry['path']
        assert file.stat().st_size == entry['size'] and sha256(file) == entry['sha256'], str(file)
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    write(OUT / f'{name}_environment.json', dict(python=platform.python_version(), packages=packages,
          frozen_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          protocol_sha256=sha256(OUT / 'protocol.json'), weights_all_rehashed=True))
    samples = read(OUT / 'sample_manifest.json')['samples']
    frozen = {key(s): s for s in read(OUT / f'{name}_input_manifest.json')['samples']}
    preflight()
    start = time.monotonic()
    with Monitor() as monitor:
        adapter = GenerationAdapter(name)
        torch.cuda.reset_peak_memory_stats()
        validation = smoke(adapter, samples, frozen)
        write(OUT / f'{name}_smoke.json', validation)
        formal_start = time.monotonic()
        for index, sample in enumerate(samples):
            assert sha256(sample['image_path']) == sample['image_sha256']
            cp = OUT / 'checkpoints' / name / (key(sample)+'.json')
            value = read(cp) if cp.exists() else dict(sample=sample, protocol_sha256=sha256(OUT / 'protocol.json'), conditions={})
            assert value['sample'] == sample and value['protocol_sha256'] == sha256(OUT / 'protocol.json')
            full_logits = None
            adapter.prepare(sample, frozen[key(sample)])
            for label, targets in CONDITIONS[name].items():
                raw = RAW / name / (key(sample)+'__'+label+'.safetensors')
                if label in value['conditions']:
                    assert sha256(raw) == value['conditions'][label]['raw_sha256']
                    logits = load_file(str(raw))['first']
                else:
                    result, logits = adapter.generate(targets)
                    if label == 'Full':
                        full_logits = logits
                    result.update(first_metrics(full_logits, logits))
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    temp = raw.with_suffix('.tmp')
                    save_file({'first': logits}, str(temp))
                    os.replace(temp, raw)
                    result.update(raw_path=str(raw), raw_sha256=sha256(raw))
                    value['conditions'][label] = result
                    write(cp, value)
                if label == 'Full':
                    full_logits = logits
            adapter.verify()
            adapter.clear()
            assert set(value['conditions']) == set(CONDITIONS[name])
            print(f'{name} {index+1}/120 complete; elapsed={time.monotonic()-formal_start:.1f}s', flush=True)
        complete = dict(complete=True, samples=120, generations=120*len(CONDITIONS[name]),
            formal_seconds=time.monotonic()-formal_start, smoke_and_load_seconds=formal_start-start,
            allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            all_inputs_parameters_and_hooks_unchanged=True)
        del adapter
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        complete['allocated_after_model_cleanup_mib'] = torch.cuda.memory_allocated()/2**20
    complete['nvidia_smi_peak_mib'] = max(r['memory_mib'] for r in monitor.rows)
    csv_write(OUT / f'{name}_gpu_log.csv', monitor.rows)
    write(OUT / f'{name}_completion.json', complete)
    print('COMPLETE', name, complete, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
