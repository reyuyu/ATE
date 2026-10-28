"""Fixed single-case handoff runner; path adaptation, no new research conditions."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from attention_inheritance import VisualAttentionInheritance, TOLERANCES
from attention_inheritance_pilot import sha, native_signature, plot, HISTORY, CACHE, QID
from knockout_probe import resource_preflight, model_layers, write_csv
from roi_sink_common import forward
from roi_sink_probe import inputs_and_history, score
from vlm_common import ROOT, load_model, write_json


PINS = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'transformers': '4.49.0',
        'accelerate': '1.4.0', 'huggingface_hub': '0.29.3', 'safetensors': '0.5.3',
        'pillow': '11.1.0', 'numpy': '1.26.4', 'matplotlib': '3.10.1', 'sentencepiece': '0.2.0'}


def remap_sample(sample, image_root):
    result = copy.deepcopy(sample)
    result['image_path'] = str(image_root.resolve() / f"image_{sample['img_id']}.jpg")
    if sha(result['image_path']) != sample['image_sha256']:
        raise ValueError('Frozen image hash mismatch; do not substitute or re-encode images')
    return result


def checked_command(command, log):
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Prerequisite failed ({result.returncode}); inspect {log}')


def run(args, output):
    started = time.perf_counter()
    versions = {name: importlib.metadata.version(name) for name in PINS}
    if versions != PINS or sys.version_info[:2] != (3, 11):
        raise ValueError(f'Fixed environment required, actual={versions}, Python={platform.python_version()}')
    manifest_path = HISTORY / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    historical_sample = next(s for s in manifest['samples'] if s['question_id'] == QID)
    sample = remap_sample(historical_sample, args.image_root)
    library = native_signature()
    code_files = ['attention_inheritance_remote.py', 'cpu_tests_remote.py', 'attention_inheritance.py',
                  'attention_inheritance_pilot.py', 'test_attention_inheritance.py', 'vlm_common.py',
                  'roi_sink_common.py', 'roi_sink_probe.py', 'physical_exit_pilot.py']
    config = dict(task='single_case_attention_inheritance_remote',
                  started=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                  question_id=QID, sample=sample, historical_sample=historical_sample,
                  image_path_mapping=dict(original=historical_sample['image_path'], actual=sample['image_path'],
                                          rule='image_root/image_<img_id>.jpg; runtime deep copy only'),
                  model_path=str(args.model_path.resolve()), source_layers=[23, 24],
                  target_layers=list(range(24, 32)), lambdas=[0, .5, 1], query='P-1 only',
                  formula='m*((1-lambda)*a/m + lambda*p_full_reference)',
                  tolerance_preregistered=TOLERANCES, historical_manifest_sha256=sha(manifest_path),
                  cache_sha256=sha(CACHE), library=library, packages=versions,
                  python=platform.python_version(), interpreter=sys.executable,
                  code_sha256={f: sha(ROOT / f) for f in code_files},
                  previous_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(output / 'experiment_config.json', config)
    checked_command([sys.executable, 'handoff/verify_recovery.py', '--image-root', str(args.image_root),
                     '--verify-source-images', '--model-dir', str(args.model_path),
                     '--output', str(output / 'recovery_audit.json')], output / 'recovery_audit.txt')
    checked_command([sys.executable, 'handoff/verify_history_remote.py', '--image-root', str(args.image_root),
                     '--output', str(output / 'historical_checksum_audit.json')], output / 'historical_checksum_audit.txt')
    for suite in ['inheritance', 'all']:
        checked_command([sys.executable, 'cpu_tests_remote.py', '--model-path', str(args.model_path),
                         '--suite', suite, '--output-dir', str(output / ('cpu_' + suite))],
                        output / ('cpu_' + suite + '.txt'))
    resource = resource_preflight()
    write_json(output / 'gpu_resource.json', resource)
    if not resource['ready']:
        raise RuntimeError(resource.get('reason', 'GPU not ready'))
    model, processor, environment = load_model(model_path=str(args.model_path.resolve()))
    write_json(output / 'environment.json', environment)
    prompt, candidates, qpos, apos, history, qrecords, counts = inputs_and_history(model, processor, sample)
    visual = torch.where(prompt['input_ids'][0] == model.config.image_token_index)[0]
    if (len(model_layers(model)) != 32 or visual.cpu().tolist() != list(range(5, 581))
            or apos != 602 or apos in qpos or sample['candidates'] != ['left', 'right']
            or sample['answer'] != 'right'):
        raise AssertionError('Unexpected frozen case structure')
    write_json(output / 'input_mapping.json', dict(question_id=QID, counts=counts,
               question_mapping=qrecords, answer_query=apos, visual_indices=visual.cpu().tolist(),
               candidate_mapping=history['candidate_mapping']))
    full = [forward(model, x, qpos, apos, observe=(i == 0)) for i, (x, rec) in enumerate(candidates)]
    logs = [score(o, rec) for o, (x, rec) in zip(full, candidates)]
    historical_logs = [c['logp'] for c in history['full_candidates']]
    with np.load(CACHE) as saved:
        cached = saved['attention'].copy()
    comparison = dict(same_machine_logp=logs, historical_logp=historical_logs,
                      logp_difference=[a - b for a, b in zip(logs, historical_logs)],
                      historical_full_exact=logs == historical_logs,
                      source_cache_bitwise_exact=np.array_equal(full[0]['rows'], cached),
                      cache_changed_elements=int(np.count_nonzero(full[0]['rows'] != cached)),
                      cache_max_abs_error=float(np.max(np.abs(full[0]['rows'].astype(np.float32) - cached.astype(np.float32)))))
    # Preserve actual cross-machine differences BEFORE enforcing original assertions.
    write_json(output / 'historical_comparison.json', comparison)
    np.savez_compressed(output / 'native_full_before_checks.npz',
                        prediction_logits=np.stack([o['prediction_logits'].numpy() for o in full]),
                        question_hidden=full[0]['question_hidden'].numpy(), attention=full[0]['rows'])
    if not comparison['historical_full_exact']:
        raise AssertionError('Historical Full mismatch; numerical audit saved, no tolerance changed')
    if not comparison['source_cache_bitwise_exact']:
        raise AssertionError('Fresh Full rows differ from Phase3.4 cache; audit saved, no bypass')
    if not torch.equal(full[0]['prediction_logits'], full[1]['prediction_logits']):
        raise AssertionError('Future candidate leakage')
    target = sample['candidates'].index(sample['answer'])
    wrong = 1 - target
    full_margin = logs[target] - logs[wrong]
    old = next(r for r in csv.DictReader((HISTORY / 'per_sample_results.csv').open())
               if r['question_id'] == QID and r['cutoff'] == '24')
    physical = float(old['answer_margin'])
    refs = {layer: full[0]['rows'][layer, :, 1, :].astype(np.float32) for layer in [23, 24]}
    np.savez_compressed(output / 'reference_distributions.npz',
                        **{f'layer{layer}_raw': ref for layer, ref in refs.items()},
                        **{f'layer{layer}_conditional': ref / ref.sum(-1, keepdims=True) for layer, ref in refs.items()})
    modules = [layer.self_attn for layer in model_layers(model)]
    rows, audits = [], []
    measurements = [o['measurement'] for o in full]
    all_logits = {'native_full': np.stack([o['prediction_logits'].numpy() for o in full])}
    for source in [23, 24]:
        for strength in [0, .5, 1]:
            outs, conditions = [], []
            for i, (x, rec) in enumerate(candidates):
                intervention = VisualAttentionInheritance(modules, visual.cpu(), apos, refs[source], strength)
                try:
                    with intervention:
                        value = forward(model, x, qpos, apos)
                        intervention.assert_complete()
                finally:
                    write_json(output / f'audit_L{source}_lambda{strength:g}_candidate{i}.json', intervention.records)
                if any('forward' in module.__dict__ for module in modules):
                    raise AssertionError('Override leaked')
                if not torch.equal(value['question_hidden'], full[i]['question_hidden']):
                    raise AssertionError('Earlier Question states changed')
                if strength == 0 and not torch.equal(value['prediction_logits'], full[i]['prediction_logits']):
                    raise AssertionError('lambda0 differs from same-machine Full')
                conditions.append(dict(candidate=sample['candidates'][i], layer_audits=intervention.records,
                                       measurement=value['measurement'], question_hidden_bitwise_full=True,
                                       lambda_zero_logits_bitwise_full=True if strength == 0 else None))
                outs.append(value)
                audits.extend(intervention.records)
                measurements.append(value['measurement'])
            if not torch.equal(outs[0]['prediction_logits'], outs[1]['prediction_logits']):
                raise AssertionError('Candidate leaked into prediction')
            logp = [score(o, rec) for o, (x, rec) in zip(outs, candidates)]
            margin = logp[target] - logp[wrong]
            token_ids = [rec[0]['token_id'] for x, rec in candidates]
            row = dict(question_id=QID, source_layer=source, **{'lambda': strength},
                       target_first_layer=24, target_last_layer=31, answer_query=apos,
                       margin=margin, gain_vs_full=margin-full_margin, full_margin=full_margin,
                       historical_physical_k24_margin=physical, correct_candidate=sample['answer'],
                       predicted_candidate=sample['answer'] if margin > 0 else sample['candidates'][wrong] if margin < 0 else 'tie',
                       correct=margin > 0, correct_logp=logp[target], wrong_logp=logp[wrong],
                       correct_logit=float(outs[0]['prediction_logits'][token_ids[target]]),
                       wrong_logit=float(outs[0]['prediction_logits'][token_ids[wrong]]),
                       correct_candidate_probability=float(np.exp(logp[target])),
                       right_vs_left_probability=float(1/(1+np.exp(-margin))),
                       forward_seconds_two_candidates=sum(o['measurement']['forward_seconds'] for o in outs),
                       peak_allocated_mib=max(o['measurement']['gpu_memory']['peak_allocated_mib'] for o in outs),
                       peak_reserved_mib=max(o['measurement']['gpu_memory']['peak_reserved_mib'] for o in outs),
                       all_local_checks_passed=True, question_hidden_bitwise_full=True,
                       candidate_logits_bitwise_identical=True)
            rows.append(row)
            write_json(output / f'audit_L{source}_lambda{strength:g}.json', conditions)
            write_json(output / 'per_condition_results.json', rows)
            write_csv(output / 'per_condition_results.csv', rows)
            all_logits[f'L{source}_lambda{strength:g}'] = np.stack([o['prediction_logits'].numpy() for o in outs])
            np.savez_compressed(output / 'prediction_logits.npz', **all_logits)
            print(json.dumps(row), flush=True)
    after = forward(model, candidates[0][0], qpos, apos)
    measurements.append(after['measurement'])
    if not torch.equal(after['prediction_logits'], full[0]['prediction_logits']) or not torch.equal(after['question_hidden'], full[0]['question_hidden']):
        raise AssertionError('Post-cleanup native Full changed')
    if native_signature() != library:
        raise AssertionError('Installed library changed')
    if len(rows) != 6 or len(audits) != 96:
        raise AssertionError('Incomplete condition/layer coverage')
    validation = dict(status='completed', passed=True, question_id=QID, condition_count=6,
                      target_layer_audit_count=len(audits), historical_comparison=comparison,
                      lambda_zero_both_sources_exact=True, question_hidden_all_exact=True,
                      post_cleanup_native_exact=True, third_party_source_unchanged=True,
                      all_local_audits_passed=all(a['passed'] for a in audits),
                      max_row_sum_abs_error=max(a['row_sum_max_abs_error'] for a in audits),
                      max_visual_mass_abs_error=max(a['visual_mass_max_abs_error'] for a in audits),
                      max_visual_mass_relative_error=max(a['visual_mass_max_relative_error'] for a in audits),
                      max_conditional_l1_error=max(a['conditional_l1_max_error'] for a in audits),
                      nominal_conditional_threshold_exceeded_audits=sum(a['conditional_nominal_exceeded'] for a in audits),
                      per_cell_formula_rounding_exact=all(a['formula_rounded_target_bitwise_exact'] for a in audits),
                      dtype_derived_conditional_bound_passed=all(a['conditional_rounding_bound_passed'] for a in audits),
                      peak_allocated_mib=max(m['gpu_memory']['peak_allocated_mib'] for m in measurements),
                      peak_reserved_mib=max(m['gpu_memory']['peak_reserved_mib'] for m in measurements),
                      wall_seconds_including_prerequisites=time.perf_counter()-started,
                      diagnostic_forward_seconds_sum=sum(m['forward_seconds'] for m in measurements),
                      measured_forwards=len(measurements), measurements=measurements)
    plot(output, rows, full_margin, physical)
    table = '\n'.join(f"| L{r['source_layer']} | {r['lambda']:g} | {r['margin']:+.8f} | {r['gain_vs_full']:+.8f} | {r['predicted_candidate']} |" for r in rows)
    (output / 'report.md').write_text(
        '# Remote single-case Attention inheritance: 15133706\n\n'
        f'Completed on {socket.gethostname()}; native Full margin {full_margin:+.8f}. '
        f'Physical K24 {physical:+.8f} is historical only, not rerun.\n\n'
        '| Full reference | Lambda | Margin | Gain vs same-machine Full | Prediction |\n'
        '| --- | ---: | ---: | ---: | --- |\n' + table + '\n\n'
        '![Measured margins](margin_comparison.png)\n\n'
        'Only P-1=602 to 576 image keys in L24..L31 is changed after softmax and before current V. '
        'Original per-head Full L23/L24 references, current visual mass, fixed candidates, no EOS, FP16/eager are unchanged. '
        'Seven inheritance and 101 original CPU tests passed; see cpu_inheritance/ and cpu_all/. '
        'Both lambda-zero controls and post-cleanup Full are exact on this machine; original historical checks remain enforced. '
        'Per-cell rounding, dtype-derived bounds, original nominal warnings and non-target invariance are in audit_*.json and validation.json.\n\n'
        'This is one post-hoc selected weak-margin case, not an accuracy estimate. '
        'Observed margin changes do not establish semantic improvement, harmful sinks, or an explanation of physical exit. '
        'No training, extra conditions or 240-image intervention was run.\n')
    write_json(output / 'validation.json', validation)
    print('COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--image-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    args.model_path = args.model_path.resolve()
    args.image_root = args.image_root.resolve()
    base = ROOT / 'results/attention_inheritance_15133706'
    output = (args.output_dir or base / f'remote_{socket.gethostname()}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if not output.is_relative_to(base.resolve()) or output == base.resolve():
        parser.error('Use a NEW subdirectory of results/attention_inheritance_15133706')
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args, output)
    except BaseException as exc:
        write_json(output / 'failure.json', dict(status='incomplete', experiment_completed=False,
                   error=repr(exc), traceback=traceback.format_exc(),
                   timestamp=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
