"""Read immutable parent data; append Sink definitions, smoke and GPU1 results."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.dataset_registry import digest, freeze_json, sha256
from information_flow_window9 import run as parent

OUTPUT = ROOT / 'results/gpu1_information_flow_hit_mass'
P34 = ROOT / 'results/phase3_4_roi_sink_diagnostic'
PARENT_COMMIT = '4d800f0971f8c97bdfbc8db9728fc87247ea376c'


def write_csv(path, rows):
    import csv
    if not rows:
        raise ValueError('Empty CSV')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / p for p in
        ('roi_sink_common.py', 'knockout_probe.py')]
    return {**parent.source_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}}


def historical_sinks():
    manifest = json.loads((P34 / 'manifest.json').read_text())
    definitions = json.loads((P34 / 'sink_manifest.json').read_text())
    samples = {r['question_id']: r for r in manifest['samples']}
    result = {}
    for entry in definitions['samples']:
        sample = samples[entry['question_id']]
        key = sample['image_sha256']
        if key in result and result[key]['sink_indices'] != entry['sink_indices']:
            raise ValueError('Conflicting frozen Sink definitions for same image')
        result[key] = {**entry, 'image_id': sample['img_id']}
    return result


def prepare():
    from roi_sink_common import PROTOCOL, NEUTRAL
    _, manifest = parent.load_experiment()
    old = historical_sinks()
    images = {r['image_sha256']: r for r in manifest['samples']}
    protected = [parent.OUTPUT / n for n in ('manifest.json', 'config.json', 'curves.csv', 'per_question.csv', 'baselines.csv')]
    protected += [parent.OUTPUT / 'checkpoints' / f'{s["dataset"]}__{s["question_id"]}.json' for s in manifest['samples']]
    protected += [P34 / 'manifest.json', P34 / 'sink_manifest.json', P34 / 'preregistered_protocol.json']
    config = {'parent_commit': PARENT_COMMIT, 'parent_manifest_sha256': sha256(parent.OUTPUT / 'manifest.json'),
        'parent_config': json.loads((parent.OUTPUT / 'config.json').read_text()),
        'sample_keys': [[r['dataset'], r['question_id']] for r in manifest['samples']],
        'questions': len(manifest['samples']), 'unique_images': len(images),
        'historical_sink_images': len(set(images) & set(old)), 'new_neutral_images': len(set(images) - set(old)),
        'neutral_prompts': NEUTRAL, 'sink_rule': PROTOCOL['sink_rule'],
        'sink_definition': 'All neutral-persistent sink_indices, including ROI overlap; no ROI subtraction. Empty sets stay empty.',
        'neutral_storage': 'Same two-stage FP16 rounding as Phase3.4: per-head Q-mean/A visual rows, then FP64 head mean cast FP16',
        'native_mass': 'Raw softmax mass summed over keys, averaged equally over heads and true Question queries; P-1 is separate',
        'sink_share': 'Ratio of aggregated raw Sink/Image masses, not mean of per-head ratios; zero denominator -> null',
        'window_mass': 'Arithmetic mean of raw masses over the SAME clipped 9-layer window; share is ratio of these means',
        'first_token_hit': 'Reuse parent strip/lower decoded-first-Gold-token match; NOT complete-answer/VQA accuracy',
        'replay': 'Same batch4, model, inputs, paths and centers. Require exact parent Native and masked fixed-token LogP.',
        'association': 'Descriptive within-question Spearman over 32 overlapping windows; median over questions, no layer-level p-values. '
                       'BOS always P-1; Sink/Image share uses Question for Image->Question and P-1 for other paths. '
                       'Sensitivity = signed negative relative probability change; negative rho means higher occupancy accompanies less drop.',
        'protected_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in protected}}
    freeze_json(OUTPUT / 'config.json', config)
    print('FROZEN', config['questions'], 'questions;', config['historical_sink_images'], 'historical Sink images;',
          config['new_neutral_images'], 'new neutral images', flush=True)


def load_config():
    config = json.loads((OUTPUT / 'config.json').read_text())
    for path, expected in config['protected_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Protected historical file changed: {path}')
    _, manifest = parent.load_experiment()
    if [[r['dataset'], r['question_id']] for r in manifest['samples']] != config['sample_keys']:
        raise ValueError('Parent sample order changed')
    return config, manifest


def binding(config):
    if parent.git('status', '--porcelain', '--untracked-files=no', '--',
                  'information_flow_hit_mass', 'information_flow_window9', 'roi_sink_common.py', 'vlm_common.py'):
        raise ValueError('Commit source before GPU execution')
    return {'code_commit': parent.git('rev-parse', 'HEAD'), 'source_sha256': source_hashes(),
            'config_sha256': digest(config)}


def forward(model, inputs):
    import torch
    with torch.inference_mode():
        result = model(**inputs, use_cache=False, output_attentions=False, logits_to_keep=1, return_dict=True)
    logits = result.logits[:, -1].detach().float()
    if result.past_key_values is not None or not bool(torch.isfinite(logits).all()):
        raise AssertionError('Cache or nonfinite logits')
    return logits


def identify_sinks():
    import numpy as np
    from roi_sink_common import sink_candidates, NEUTRAL, AttentionRows
    from attention_position_probe import locate_question_tokens
    from vlm_common import load_model, prepare_inputs
    config, manifest = load_config()
    code = binding(config)
    freeze_json(OUTPUT / 'sink_execution_binding.json', code)
    old = historical_sinks()
    unique = {r['image_sha256']: r for r in manifest['samples']}
    model = processor = None
    initial_hooks = None
    try:
        for index, (image_sha, sample) in enumerate(sorted(unique.items())):
            path = OUTPUT / 'sinks' / f'{image_sha}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['binding'] != code or saved['image_sha256'] != image_sha:
                    raise ValueError('Sink resume binding mismatch')
                continue
            if image_sha in old:
                entry = old[image_sha]
                neutral_path = P34 / 'attention_rows' / f'{entry["question_id"]}_neutral.npz'
                if sha256(neutral_path) != entry['neutral_npz_sha256']:
                    raise ValueError('Historical neutral cache changed')
                maps = np.load(neutral_path)['attention']
                identified = sink_candidates(maps)
                if identified['indices'] != entry['sink_indices']:
                    raise ValueError('Historical rule does not reproduce frozen Sink indices')
                neutral_source = str(neutral_path.relative_to(ROOT))
                origin = 'phase3_4_frozen'
            else:
                if model is None:
                    pre = parent.preflight()
                    model, processor, environment = load_model(parent.MODEL)
                    initial_hooks = parent.hooks(model)
                    if not (OUTPUT / 'sink_environment.json').exists():
                        freeze_json(OUTPUT / 'sink_environment.json', {**environment, **pre, 'binding': code})
                    elif json.loads((OUTPUT / 'sink_environment.json').read_text())['binding'] != code:
                        raise ValueError('Neutral resume environment binding mismatch')
                views = []
                for text in NEUTRAL:
                    inputs, _ = prepare_inputs(model, processor, sample['image_path'], text)
                    tokens = inputs['input_ids'][0].cpu().numpy()
                    positions, _ = locate_question_tokens(processor, text, tokens, model.config.image_token_index)
                    question = [r['sequence_index'] for r in positions]
                    with AttentionRows(model, inputs, question, len(tokens) - 1) as stream:
                        forward(model, inputs)
                    rows = np.stack(stream.rows)
                    views.append(rows.astype(np.float64).mean(1).astype(np.float16))
                    del inputs, stream, rows
                maps = np.stack(views)
                neutral_path = OUTPUT / 'neutral_maps' / f'{image_sha}.npz'
                neutral_path.parent.mkdir(parents=True, exist_ok=True)
                if neutral_path.exists():
                    raise ValueError('Unexpected neutral artifact without completed checkpoint')
                np.savez_compressed(neutral_path, attention=maps)
                identified = sink_candidates(maps)
                neutral_source = str(neutral_path.relative_to(ROOT))
                origin = 'same_rule_new_neutral'
                if parent.hooks(model) != initial_hooks:
                    raise AssertionError('Neutral hook leak')
            freeze_json(path, {'binding': code, 'image_sha256': image_sha, 'image_id': sample['image_id'],
                        'source': origin, 'neutral_path': neutral_source, 'neutral_sha256': sha256(neutral_path),
                        'sink_indices': identified['indices'], 'identification': identified})
            print(f'SINK {index+1}/{len(unique)} {origin} image={sample["image_id"]} n={len(identified["indices"])}', flush=True)
        entries = [json.loads((OUTPUT / 'sinks' / f'{key}.json').read_text()) for key in sorted(unique)]
        freeze_json(OUTPUT / 'sink_manifest.json', {'binding': code, 'images': entries,
                    'rule': config['sink_rule'], 'frozen_before_masked_forwards': True})
    finally:
        if model is not None and parent.hooks(model) != initial_hooks:
            raise AssertionError('Neutral hook leak on exit')
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def run(smoke=False):
    import numpy as np
    import torch
    from information_flow_window9.masks import PATHS, intervention, first_token_hit, probability_change, window
    from .attention import observe_mass, layer_rows, window_rows
    from vlm_common import load_model, prepare_inputs, gpu_memory
    config, manifest = load_config()
    code = binding(config)
    sink_manifest = json.loads((OUTPUT / 'sink_manifest.json').read_text())
    if sink_manifest['binding']['source_sha256'] != code['source_sha256']:
        raise ValueError('Sink identification source differs')
    sinks = {r['image_sha256']: r for r in sink_manifest['images']}
    for entry in sinks.values():
        if sha256(ROOT / entry['neutral_path']) != entry['neutral_sha256']:
            raise ValueError('Neutral artifact changed')
    code['sink_manifest_sha256'] = sha256(OUTPUT / 'sink_manifest.json')
    if not smoke:
        prior = json.loads((OUTPUT / 'smoke.json').read_text())
        if prior['binding'] != code or not prior['passed']:
            raise ValueError('Passing smoke with the same code required')
        freeze_json(OUTPUT / 'execution_binding.json', code)
    pre = parent.preflight()
    model, processor, environment = load_model(parent.MODEL)
    layers = model.language_model.model.layers
    initial_hooks = parent.hooks(model)
    samples = manifest['samples']
    if smoke:
        samples = [next(r for r in samples if r['dataset'] == task) for task in parent.TASKS]
    elif not (OUTPUT / 'environment.json').exists():
        freeze_json(OUTPUT / 'environment.json', {**environment, **pre, 'binding': code})
    elif json.loads((OUTPUT / 'environment.json').read_text())['binding'] != code:
        raise ValueError('Resume environment binding mismatch')
    started = time.time()
    smoke_records = []
    try:
        for index, sample in enumerate(samples):
            key = f'{sample["dataset"]}__{sample["question_id"]}'
            checkpoint = OUTPUT / 'checkpoints' / f'{key}.json'
            if not smoke and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['binding'] != code or len(saved['rows']) != 96:
                    raise ValueError('Resume checkpoint mismatch')
                print('RESUME', key, flush=True)
                continue
            historical = json.loads((parent.OUTPUT / 'checkpoints' / f'{key}.json').read_text())
            old = historical['baseline']
            old_rows = {(r['path'], r['center']): r for r in historical['rows']}
            mapping = sample['metadata']['input_mapping']
            entry = sinks[sample['image_sha256']]
            inputs, _ = prepare_inputs(model, processor, sample['image_path'], mapping['text_payload'])
            if inputs['input_ids'][0].tolist() != mapping['prompt_ids'] or int(inputs['input_ids'][0, 0]) != processor.tokenizer.bos_token_id:
                raise AssertionError('Original input mapping/BOS changed')
            inputs = {k: v.repeat(parent.BATCH, *([1] * (v.ndim - 1))) for k, v in inputs.items()}
            if smoke:
                plain = forward(model, inputs)
            with observe_mass(layers, mapping, entry['sink_indices']) as observed:
                full = forward(model, inputs)
            heads = np.stack([observed[l] for l in range(32)])
            native_mass = layer_rows(heads)
            window_mass = window_rows(native_mass)
            if smoke and not torch.equal(full, plain):
                raise AssertionError('Native mass observer changed logits')
            if not all(torch.equal(full[0], full[b]) for b in range(parent.BATCH)):
                raise AssertionError('Native replica mismatch')
            token = int(full[0].argmax())
            full_logp = float(full[0].log_softmax(-1)[token])
            if token != old['target_token_id'] or full_logp != old['full_logp']:
                raise AssertionError('Native Full does not exactly reproduce parent')
            full_hit = first_token_hit(processor.tokenizer.decode([token]), old['gold_first_token_texts'])
            if full_hit != old['full_first_token_hit']:
                raise AssertionError('Native hit definition changed')
            rows = []
            for path in PATHS:
                centers = [0, 4, 16, 31] if smoke else list(range(32))
                for start in range(0, len(centers), parent.BATCH):
                    chunk = centers[start:start + parent.BATCH]
                    with intervention(layers, path, mapping, chunk, validate=smoke):
                        masked = forward(model, inputs)
                    logps = masked.log_softmax(-1)[:, token].cpu().tolist()
                    predicted = masked.argmax(-1).cpu().tolist()
                    for b, center in enumerate(chunk):
                        prior = old_rows[path, center]
                        if logps[b] != prior['masked_logp']:
                            raise AssertionError(f'Parent masked LogP drift at {key} {path} {center}: {logps[b]} != {prior["masked_logp"]}')
                        text = processor.tokenizer.decode([predicted[b]])
                        hit = first_token_hit(text, old['gold_first_token_texts'])
                        span = window(center)
                        rows.append({'dataset': sample['dataset'], 'question_id': sample['question_id'],
                            'image_id': sample['image_id'], 'path': path, 'center': center,
                            'window_start': span[0], 'window_end': span[-1], 'window_layers': len(span),
                            'native_token_id': token, 'native_token_text': old['target_token_text'],
                            'masked_argmax_token_id': predicted[b], 'masked_argmax_token_text': text,
                            'argmax_changed': predicted[b] != token, 'full_first_token_hit': full_hit,
                            'masked_first_token_hit': hit, 'hit_to_miss': full_hit and not hit,
                            'miss_to_hit': not full_hit and hit,
                            'full_logp': full_logp, 'masked_logp': logps[b], 'delta_logp': logps[b] - full_logp,
                            'full_probability': math.exp(full_logp), 'masked_fixed_token_probability': math.exp(logps[b]),
                            'probability_change_pct': probability_change(full_logp, logps[b]),
                            'parent_probability_replay_exact': True})
            restored = forward(model, inputs)
            if not torch.equal(full, restored) or parent.hooks(model) != initial_hooks:
                raise AssertionError('Native Full/hooks not exactly restored')
            if smoke:
                smoke_records.append({'sample': key, 'rows': len(rows), 'parent_replay_exact': True,
                    'native_observer_logits_exact': True, 'native_restored_bitwise': True,
                    'sink_count': len(entry['sink_indices']), 'mass_layers': len(native_mass)})
            else:
                head_path = OUTPUT / 'native_head_mass' / f'{key}.npz'
                head_path.parent.mkdir(parents=True, exist_ok=True)
                if head_path.exists():
                    raise ValueError('Unexpected per-head artifact without complete checkpoint')
                np.savez_compressed(head_path, mass=heads)
                freeze_json(checkpoint, {'binding': code, 'sample_sha256': digest(sample), 'baseline': old,
                    'sink_indices': entry['sink_indices'], 'sink_source': entry['source'], 'rows': rows,
                    'native_mass': native_mass, 'window_mass': window_mass,
                    'head_mass_path': str(head_path.relative_to(ROOT)), 'head_mass_sha256': sha256(head_path),
                    'native_restored_bitwise': True})
            print(f'{"SMOKE" if smoke else "RUN"} {index+1}/{len(samples)} {key} sink={len(entry["sink_indices"])} '
                  f'elapsed={time.time()-started:.1f}s peak_mib={gpu_memory()["peak_allocated_mib"]:.0f}', flush=True)
            del inputs, full, masked, restored, heads, observed
        if smoke:
            freeze_json(OUTPUT / 'smoke.json', {'binding': code, 'passed': True, 'records': smoke_records,
                        'environment': environment, 'elapsed_seconds': time.time()-started})
        else:
            freeze_json(OUTPUT / 'completion.json', {'status': 'scan_complete', 'questions': 96, 'masked_rows': 9216,
                        'native_mass_rows': 3072, 'parent_replay_exact_all': True,
                        'elapsed_seconds': time.time()-started, 'binding': code})
    finally:
        if parent.hooks(model) != initial_hooks:
            raise AssertionError('Leaked hooks')
        del model, layers
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'sinks', 'smoke', 'run'))
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
    elif stage == 'sinks':
        identify_sinks()
    else:
        run(stage == 'smoke')


if __name__ == '__main__':
    main()
