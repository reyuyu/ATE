"""Read-only CPU verification, or one explicit publication-only finalization.

No model forward, scoring, selection, plotting or scientific aggregation is run.
The executed source remains frozen at the manifest's original code commit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from phase3_7.protocol import CONDITIONS, RULES, digest, read, sha, sources
from phase3_7.run import checkpoint, installed_sources
from phase3_7.analyze import verify_body


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def files(exclude):
    return {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name not in exclude and '__pycache__' not in p.parts}


def compare_csv(path, rows):
    with path.open(newline='') as stream:
        actual = list(csv.DictReader(stream))
    expected = [{k: str(v) for k, v in row.items()} for row in rows]
    require(actual == expected, 'CSV does not exactly match completed checkpoints: ' + path.name)


def comparison(rows, config):
    grouped = {(r['question_id'], r['condition']): r for r in rows}
    wide = []
    for sample in config['samples']:
        qid = sample['question_id']
        row = dict(question_id=qid, image_id=sample['img_id'], gold=sample['answer'])
        for condition in CONDITIONS:
            value = grouped[qid, condition]
            for key in ['margin', 'gain', 'correct_logp', 'wrong_logp', 'absolute_correct_logp_gain',
                        'correct', 'tie', 'full_correct_to_wrong', 'full_wrong_to_correct']:
                row[condition + '_' + key] = value[key]
        wide.append(row)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(wide[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(wide)
    lines = ['# Paired Comparison with Native Full', '',
             'Same 16 frozen images and original complete-candidate protocol; B0 is current-machine Native Full, not historical Full. All masks affect L0/L1 only. Positive Margin is correct; ties are wrong.', '',
             '| Condition | Correct / N | Mean Margin | Mean Gain vs Full | Mean absolute per-image Gain | Mean gold absolute LogP |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    stats = {}
    for condition in CONDITIONS:
        rr = [r for r in rows if r['condition'] == condition]
        n = len(rr)
        values = dict(n=n, correct=sum(r['margin'] > 0 for r in rr),
                      mean_margin=math.fsum(r['margin'] for r in rr) / n,
                      mean_gain=math.fsum(r['gain'] for r in rr) / n,
                      mean_abs_gain=math.fsum(abs(r['gain']) for r in rr) / n,
                      mean_gold_logp=math.fsum(r['correct_logp'] for r in rr) / n)
        stats[condition] = values
        lines.append(f"| {condition} | {values['correct']}/{n} | {values['mean_margin']:+.6f} | {values['mean_gain']:+.6f} | {values['mean_abs_gain']:.6f} | {values['mean_gold_logp']:.6f} |")
    lines += ['', 'No condition changes correctness on any image. Gold absolute LogP and the two-candidate Margin are different measurements; increasing both candidates can leave ranking unchanged. Positive average Gain does not establish a beneficial effect in the population.', '',
              '## Every Image: Final Margin', '',
              '| Question ID | Gold | Full | B1 | B2 | B3 | B4 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in wide:
        lines.append('| ' + ' | '.join([row['question_id'], row['gold']] +
                     [f"{row[c + '_margin']:+.6f}" for c in CONDITIONS]) + ' |')
    lines += ['', 'The exact wide CSV contains each condition\'s Margin/Gain, full-vocabulary gold/wrong LogP, signed gold LogP change and correctness/flip/tie flags. See full_comparison.csv and the existing margin_vs_full.png, whose diagonal is equality with Full.', '',
              'This table only reformats completed checkpoint scores; it is not a new run, a new condition or a second scientific aggregation.']
    return stream.getvalue(), '\n'.join(lines) + '\n', stats


def validate_science():
    envelope = read(OUT / 'manifest.json')
    c, binding = envelope['config'], envelope['binding']
    require(digest(c) == binding and c['rules'] == RULES and c['sources'] == sources(),
            'Frozen manifest/config/source mismatch')
    subprocess.run(['git', 'merge-base', '--is-ancestor', c['code_commit'], 'HEAD'], cwd=ROOT, check=True)
    for path, expected in c['sources'].items():
        data = subprocess.check_output(['git', 'show', c['code_commit'] + ':vlm_research/' + path], cwd=ROOT)
        require(hashlib.sha256(data).hexdigest() == expected, 'Executed source commit mismatch')
    for path, expected in c['original_inputs_sha256'].items():
        require(sha(ROOT / path) == expected, 'Original source/history changed: ' + path)
    cpu = read(OUT / 'acceptance/validation.json')
    require(cpu['passed'] and cpu['binding'] == binding and cpu['tests']['all']['tests'] == 129,
            'CPU acceptance gate failed')
    require(cpu['tests']['focused']['tests'] == 21, 'Focused test coverage changed')
    expected_sources = installed_sources()
    bodies = {}
    audit_counts = {}
    raw_arrays = 0
    for stage, ids in [('smoke', c['smoke_ids']), ('pilot', [s['question_id'] for s in c['samples']])]:
        root = OUT / stage
        gate = read(root / 'validation.json')
        require(gate['passed'] and gate['status'] == 'completed' and gate['binding'] == binding,
                'Incomplete stage: ' + stage)
        require(gate['question_ids'] == ids and gate['count'] == len(ids), 'Wrong stage coverage')
        require(gate['weights_sha256_before'] == gate['weights_sha256_after'] and gate['third_party_source_unchanged'],
                'Weight/library mutation gate failed')
        env_paths = list(root.glob('environment_*.json'))
        require(len(env_paths) == 1, 'Expected one completed stage environment record')
        env = read(env_paths[0])
        require(env['installed_sources'] == expected_sources, 'Installed Transformers source changed')
        require({p.stem for p in (root / 'checkpoints').glob('*.json')} == set(ids), 'Unexpected/missing checkpoint')
        layers = local = 0
        stage_bodies = []
        for qid in ids:
            sample = next(s for s in c['samples'] if s['question_id'] == qid)
            require(sha(sample['image_path']) == sample['image_sha256'], 'Image changed')
            path = root / 'checkpoints' / (qid + '.json')
            require(sha(path) == gate['checkpoints_sha256'][qid], 'Checkpoint SHA changed')
            body = checkpoint(path, binding, sample, root)
            count, local_count = verify_body(body, sample, root)
            layers += count
            local += local_count
            for name in body['artifacts']:
                if name.endswith('.npz'):
                    with np.load(root / name, allow_pickle=False) as arrays:
                        for key in arrays.files:
                            require(np.isfinite(arrays[key]).all(), 'Nonfinite saved observation')
                        raw_arrays += 1
            stage_bodies.append(body)
        bodies[stage] = stage_bodies
        audit_counts[stage] = dict(layer_audits=layers, local_mask_audits=local)
    rows = [r for b in bodies['pilot'] for r in b['b']['rows']]
    arows = [r for b in bodies['pilot'] if b['a'] for r in b['a']['rows']]
    require(len(rows) == 80 and len(arows) == 48, 'Incomplete formal A/B results')
    compare_csv(OUT / 'per_sample_b.csv', rows)
    compare_csv(OUT / 'per_pair_a.csv', arows)
    for row in rows:
        require(math.isclose(row['margin'], row['correct_logp'] - row['wrong_logp'], abs_tol=2e-12), 'Margin arithmetic failed')
        require(math.isclose(row['gain'], row['margin'] - row['full_margin'], abs_tol=2e-12), 'Gain arithmetic failed')
    original = read(OUT / 'aggregation_validation_generated.json')
    differences = {name: dict(original_sha256=value, final_sha256=sha(OUT / name))
                   for name, value in original['artifacts'].items() if sha(OUT / name) != value}
    require(set(differences) == {'aggregation_console.txt', 'report.md'}, 'Unexpected post-aggregation mutation')
    require(sha(OUT / 'machine_report.md') == original['artifacts']['report.md'], 'Machine report archive changed')
    log = (OUT / 'aggregation_console.txt').read_bytes()
    suffix = b'SUMMARY_COMPLETE 16 48\n'
    require(log.endswith(suffix) and hashlib.sha256(log[:-len(suffix)]).hexdigest() == original['artifacts']['aggregation_console.txt'],
            'Log difference is not exactly the final completion line')
    figures = {}
    for path in sorted(OUT.rglob('*.png')):
        with Image.open(path) as image:
            data = np.asarray(image.convert('RGB'))
            require(min(image.size) >= 500 and data.std() > 1, 'Blank/undersized figure')
            figures[str(path.relative_to(OUT))] = dict(width=image.width, height=image.height,
                                                     pixel_std=float(data.std()))
    require(len(figures) == 17, 'Wrong figure coverage')
    require(not list(OUT.rglob('failure*.json')) and not list(OUT.rglob('.lock')), 'Failure or live execution lock remains')
    return c, binding, rows, dict(audit_counts=audit_counts, checked_npz_files=raw_arrays,
                                  figures=figures, disclosed_original_gate_differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', help='One-time publication-only hash finalization')
    args = parser.parse_args()
    c, binding, rows, checks = validate_science()
    csv_text, markdown, stats = comparison(rows, c)
    if args.finalize:
        require(not (OUT / 'publication_validation.json').exists(), 'Publication was already finalized')
        require(read(OUT / 'summary_validation.json') == read(OUT / 'aggregation_validation_generated.json'),
                'Pre-finalization gate is not the original generated gate')
        for name, text in [('full_comparison.csv', csv_text), ('FULL_COMPARISON.md', markdown)]:
            with (OUT / name).open('x') as stream:
                stream.write(text)
        summary_gate = read(OUT / 'aggregation_validation_generated.json')
        summary_gate.update(publication_finalized=True, publication_only_changes=checks['disclosed_original_gate_differences'],
                            original_gate_archive='aggregation_validation_generated.json')
        summary_gate['artifacts'] = files({'summary_validation.json', 'publication_validation.json', '.lock'})
        with (OUT / 'summary_validation.json').open('w') as stream:
            json.dump(summary_gate, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        write_json(OUT / 'publication_validation.json', dict(
            passed=True, status='completed', finalized_at=datetime.now(timezone.utc).isoformat(),
            binding=binding, code_commit=c['code_commit'], baseline=c['baseline'],
            images_b=16, rows_b=80, pairs_a=12, rows_a=48, scientific_aggregation_count=1,
            cpu_tests=129, figures_visually_reviewed=17, comparison_with_full=stats, checks=checks,
            artifacts=files({'publication_validation.json', '.lock'})))
    require((OUT / 'full_comparison.csv').read_text() == csv_text and (OUT / 'FULL_COMPARISON.md').read_text() == markdown,
            'Full comparison does not match checkpoint scores')
    for name in ['summary_validation.json', 'publication_validation.json']:
        gate = read(OUT / name)
        require(gate['passed'] and gate['binding'] == binding, 'Publication binding failed')
        expected = files({name, '.lock'} if name == 'publication_validation.json' else
                         {'summary_validation.json', 'publication_validation.json', '.lock'})
        require(gate['artifacts'] == expected, 'Publication artifact hashes failed: ' + name)
    print('PUBLICATION_VERIFIED', 'B=16x5', 'A=12x4', 'CPU=129', 'figures=17', 'aggregation=1', flush=True)


if __name__ == '__main__':
    main()
