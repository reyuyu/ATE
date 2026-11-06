"""CPU-only freeze. No model or large dataset copies."""
from collections import defaultdict
import gzip
import os
import subprocess

from .common import *


def main():
    safety()
    assert not (OUT / 'protocol.json').exists(), 'Never overwrite an experiment freeze'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifests').mkdir(exist_ok=True)
    safety_audit = read(AUDIT / 'dataset_safety_final.json')
    prior_models = read(Path('/data/lk/ATE-worktrees/mme-seed-physical-c3-quality') /
        'vlm_research/results/mme_seed_physical_c3_quality/model_manifests.json')
    for name, expected in prior_models['llava']['files'].items():
        assert sha(Path(MODELS['llava']) / name) == expected, name
    atomic(OUT / 'model_binding.json', prior_models['llava'])
    original_commit = '213d4e3be766444e1ecd9a90ac479836f7756e9b'
    engine_paths = ('mme_seed_quality/engine.py', 'accuracy_autoresearch/engine.py',
                    'ate_fulljoint_v2/engine.py', 'physical_handoff_fastpath/engine.py')
    original_bindings = {}
    import hashlib
    for rel in engine_paths:
        path = 'vlm_research/' + rel
        content = subprocess.check_output(['git', 'show', original_commit + ':' + path], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == sha(ROOT / path), path
        original_bindings[path] = sha(ROOT / path)
    science = read('/data/lk/ATE-resources/benchmarks/phase4_0/sources/scienceqa/data/scienceqa/problems.json')
    source_files = {}
    rows = {}
    image_manifest = {}
    clusters = defaultdict(int)
    for bench in ORDER:
        inp, ans = DATA / bench / 'inputs.jsonl', DATA / bench / 'answers.json'
        audit = read(AUDIT / f'{bench}_validated.json')
        assert audit['status'] == 'COMPLETE' and audit['questions'] == COUNTS[bench]
        source_files[str(inp)] = sha(inp)
        source_files[str(ans)] = sha(ans)
        assert sha(inp) == safety_audit['checks'][bench]['inputs_sha256']
        answers = read(ans)
        infos = {v['path']: v for v in read(AUDIT / f'{bench}_image_validation.json')}
        entries = []
        with inp.open() as stream:
            for index, line in enumerate(stream):
                raw = json.loads(line)
                info = infos[raw['image_path']]
                assert info['readable']
                path = raw['image_path']
                if path not in image_manifest:
                    assert sha(path) == info['sha256'], path
                    image_manifest[path] = dict(sha256=info['sha256'], bytes=Path(path).stat().st_size)
                qid = str(raw['question_id'])
                assert qid in answers
                item = dict(key=f'{bench}__{index:07d}', benchmark=bench, record_index=index,
                    source_question_id=qid, image_path=path, image_sha256=info['sha256'],
                    image_id=raw['image_id'], question=prompt(raw, bench), raw=raw, answer=answers[qid])
                if bench == 'ScienceQA-IMG':
                    original = science[qid]
                    assert original['answer'] == answers[qid] and original['question'] == raw['question']
                    assert original['choices'] == raw['choices'] and original['hint'] == raw['hint']
                    item['groups'] = {k: original.get(k) for k in
                        ('subject', 'topic', 'category', 'skill', 'grade', 'task')}
                clusters[info['sha256']] += 1
                entries.append(item)
        assert len(entries) == COUNTS[bench]
        rows[bench] = entries
        print(f'Validated {bench}: {len(entries)} records', flush=True)
    loads = [0, 0]
    assignment = {}
    for h in sorted(clusters, key=lambda h: (-clusters[h], digest(['seven-benchmark', 20261106, h]))):
        rank = min(range(2), key=lambda r: (loads[r], r))
        assignment[h] = rank
        loads[rank] += clusters[h]
    smoke = {}
    for bench, entries in rows.items():
        for s in entries:
            s['shard'] = assignment[s['image_sha256']]
        path = OUT / f'manifests/{bench}.jsonl.gz'
        with gzip.open(path, 'wt') as stream:
            for s in entries:
                stream.write(json.dumps(s, ensure_ascii=False, separators=(',', ':')) + '\n')
        # Input-only choice: shortest, median, longest prompt per shard.
        smoke[bench] = {}
        for rank in range(2):
            ordered = sorted((s for s in entries if s['shard'] == rank),
                             key=lambda s: (len(s['question']), s['key']))
            smoke[bench][str(rank)] = [ordered[i]['key'] for i in (0, len(ordered)//2, -1)]
    atomic(OUT / 'image_manifest.json', image_manifest)
    atomic(OUT / 'smoke_selection.json', smoke)
    artifacts = {str(p.relative_to(OUT)): sha(p) for p in (OUT / 'manifests').glob('*.gz')}
    artifacts.update({name: sha(OUT / name) for name in
                      ('image_manifest.json', 'smoke_selection.json', 'model_binding.json')})
    protocol = dict(created_at=now(), model='llava', revision=Path(MODELS['llava']).name,
        dtype='float16', attention='eager', transformers='4.49.0', max_new_tokens=32,
        greedy=True, logits_to_keep=1, order=ORDER, counts=COUNTS, conditions=CONDITIONS,
        devices=DEVICES, sources=sources(), artifacts=artifacts, source_files=source_files,
        original_engine_commit=original_commit, original_engine_sha256=original_bindings,
        shard_loads=loads, image_clusters=len(clusters), model_count_per_gpu=1,
        qwen_status='STOPPED_PREEXISTING_SDPA_NUMERICAL_GATE_FAILED',
        qwen_evidence_commit='53e9cc1f52481de338ab1a3149a41b63eaa5d162',
        mmvet_status='Generate only; paid official judge not authorized',
        mmbench_status='Generate all cyclic records; official matching/circular score pending',
        bootstrap_replicates=20000, bootstrap_seed=20261106,
        ci_scope='Exploratory, image SHA256 whole-cluster paired, question-weighted',
        disk_free_floor_GiB=20, overwrite=False, historical_generation_reused=False,
        timings=False, stop_on_oom_or_nonfinite=True)
    atomic(OUT / 'protocol.json', protocol)
    print('FROZEN', digest(protocol), 'loads', loads, flush=True)


if __name__ == '__main__':
    main()
