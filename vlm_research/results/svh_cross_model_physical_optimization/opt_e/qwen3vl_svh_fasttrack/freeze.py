"""Bind the existing Plan-B inventory without changing its inputs or budgets."""
import gzip
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor

from ..qwen3vl_svh_preflight48.freeze import MODEL, TASKS, COUNTS, digest, grouped_exact, prepare, sha, table, write
from ..qwen3vl_svh_numerical_repair.run import GATE, source_files
from ..qwen3vl_svh_preflight48 import score as scoring

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'vlm_research/results/qwen3vl_svh_expanded_preflight48'
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_fasttrack_sixbench'
AUDIT = Path('/data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090')
GPUS = ['GPU-b08489af-0a8a-eb48-55ba-8346733af417', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']


def scorers():
    scoring.SOURCES = AUDIT / 'p0_provenance_v2/sources'
    return scoring


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'vlm_research/qwen3vl_svh_l28_repair').glob('*.py'))
    paths += list((ROOT / 'vlm_research/qwen3vl_svh_preflight48').glob('*.py'))
    return {**source_files(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def inventory():
    with gzip.open(OLD / 'inventory.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def choose_cases(rows):
    selected = []
    for task in TASKS:
        pool = sorted((r for r in rows if r['benchmark'] == task), key=lambda r: (r['input_length'], r['key']))
        for label, q in [('min', 0), ('median', .5), ('p99', .99), ('max', 1)]:
            selected.append(dict(sample=pool[math.ceil(q * (len(pool) - 1))], quantile=label))
    assert len(selected) == len({r['sample']['key'] for r in selected}) == 24
    assert any(r['sample']['key'] == 'GQA:20705853' for r in selected)
    # All shortest cases precede the more expensive lengths; no output selection.
    return sorted(selected, key=lambda c: (['min', 'median', 'p99', 'max'].index(c['quantile']), TASKS.index(c['sample']['benchmark'])))


def choose_pretest(rows):
    chosen = []
    for task in TASKS:
        pool = [r for r in rows if r['benchmark'] == task]
        if task in ('MME', 'MMBench'):
            chosen += grouped_exact(pool, 'pair_id' if task == 'MME' else 'circular_group', 8, 'FAST_TRACK8_' + task)
        elif task == 'POPE':
            for strategy, n in [('random', 2), ('popular', 3), ('adversarial', 3)]:
                chosen += sorted((r for r in pool if r['strategy'] == strategy), key=lambda r: digest(['FAST_TRACK8', r['key']]))[:n]
        else:
            chosen += sorted(pool, key=lambda r: digest(['FAST_TRACK8', r['key']]))[:8]
    assert Counter(r['benchmark'] for r in chosen) == {t: 8 for t in TASKS}
    return chosen


def shards(rows):
    parent = {r['key']: r['key'] for r in rows}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    seen = {}
    for r in rows:
        tags = [('image', r['image_sha256'])]
        if r['benchmark'] == 'MME':
            tags.append(('mme', r['pair_id']))
        if r['benchmark'] == 'MMBench':
            tags.append(('mmbench', r['circular_group']))
        for tag in tags:
            if tag in seen:
                parent[root(r['key'])] = root(seen[tag])
            else:
                seen[tag] = r['key']
    groups = defaultdict(list)
    for r in rows:
        groups[root(r['key'])].append(r)
    work = lambda group: sum(r['input_length'] ** 2 + 20 * r['input_length'] for r in group)
    loads, mapping = [0, 0], {}
    for key, group in sorted(groups.items(), key=lambda p: (-work(p[1]), digest(p[0]))):
        shard = min(range(2), key=lambda s: (loads[s], s))
        loads[shard] += work(group)
        for r in group:
            mapping[r['key']] = dict(shard=shard, cluster=key)
    return mapping, loads


def freeze():
    assert not torch.cuda.is_initialized()
    assert not (OUT / 'protocol.json').exists(), 'Do not replace a frozen run'
    torch.set_num_threads(1)
    old = json.loads((OLD / 'protocol.json').read_text())
    assert sha(OLD / 'inventory.jsonl.gz') == old['frozen_files']['inventory.jsonl.gz']
    assert old['gate'] == GATE and old['tasks'] == COUNTS
    rows = inventory()
    assert len(rows) == 35346 and Counter(r['benchmark'] for r in rows) == COUNTS
    assert len({r['key'] for r in rows}) == len(rows)
    scorer = scorers()
    scorer.adapters()
    scorer_paths = scorer.scorer_sources()
    for path in scorer_paths.values():
        assert sha(path) in old['source_annotations'].values(), ('Scorer changed', path)
    cases, pretest = choose_cases(rows), choose_pretest(rows)
    mapping, loads = shards(rows)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    selected = {c['sample']['key']: c['sample'] for c in cases}
    selected.update({r['key']: r for r in pretest})
    bindings = {}
    for r in selected.values():
        inputs, layout = prepare(processor, r)
        assert __import__('hashlib').sha256(layout['prompt'].encode()).hexdigest() == r['prompt_sha256']
        bindings[r['key']] = layout
        del inputs
    write(OUT / 'numerical_cases.json', cases)
    write(OUT / 'pretest8_manifest.json', pretest)
    write(OUT / 'input_bindings.json', bindings)
    write(OUT / 'shards.json', mapping)
    table(OUT / 'numerical_cases.csv', [dict(key=c['sample']['key'], task=c['sample']['benchmark'],
          quantile=c['quantile'], image_tokens=c['sample']['image_tokens'], input_length=c['sample']['input_length'],
          shard=mapping[c['sample']['key']]['shard']) for c in cases])
    frozen = {n: sha(OUT / n) for n in ['numerical_cases.json', 'pretest8_manifest.json', 'input_bindings.json', 'shards.json']}
    write(OUT / 'protocol.json', dict(base_commit='2e885efa8b237312d3a744cbdfa7814a7757e3e0',
        plan=old['plan'], plan_A='BLOCKED', gate=GATE, full_59_numerical_validated=False,
        acceptance_label='FAST_TRACK_VALIDATED only if all gates pass', numerical_inputs=24,
        pretest_per_task=8, counts=COUNTS, total_questions=35346, total_full_generations=70692,
        pope='9000 original; report verified common 8910 separately', mmbench=old['mmbench'],
        generation=old['generation'], caps=old['caps'], min_pixels=old['min_pixels'], max_pixels=old['max_pixels'],
        model=old['model'], revision=old['revision'], dtype=old['dtype'], attention=old['attention'],
        logits_to_keep=1, gpu_uuids=GPUS, shard_estimated_costs=loads,
        source_files=sources(), scorer_sources={p:sha(p) for p in scorer_paths.values()}, frozen_files=frozen,
        inventory_path=str(OLD / 'inventory.jsonl.gz'), inventory_sha256=sha(OLD / 'inventory.jsonl.gz'),
        inherited_protocol_sha256=sha(OLD / 'protocol.json'), frozen_model_files=old.get('model_files'),
        number_of_canary_pairs_between_checks=128, no_speed_claim=True))
    write(OUT / 'cpu_freeze.json', dict(status='PASS', rows=len(rows), bound_inputs=len(bindings),
        cuda_initialized=torch.cuda.is_initialized(), cases=24, pretest=48))
    print('FREEZE PASS', len(rows), 'cases', len(cases), 'bindings', len(bindings), flush=True)


if __name__ == '__main__':
    freeze()
