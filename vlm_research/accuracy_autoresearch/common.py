import collections
import datetime
import json
from pathlib import Path
import subprocess
import time

from ..physical_visual_handoff.common import atomic, digest, read, sha, write_csv, MODELS, TOLERANCES
from ..ate_fulljoint_vqav2.common import score as vqa_score, conditional_writers
from ..fastp2_bypass_writer.common import hardware

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/accuracy_autoresearch_6h'
BASE = '3b46d36dc80e46fde778c6fc93191c8e33f24122'
SEED = 20261105
VQA_ROOT = Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation/vlm_research/results/receiver_cross_dataset_validation/vqa_v2')
V2_ROOT = Path('/data/lk/ATE-worktrees/ate-fulljoint-v2-answer-only/vlm_research/results/ate_fulljoint_v2_answer_only')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def deadline(field='deliver_by'):
    return datetime.datetime.fromisoformat(read(ROOT/'task_deadline.json')[field]).timestamp()


def ensure_time(gpu=False):
    assert time.time() < deadline('stop_new_gpu_work_by' if gpu else 'deliver_by'), 'Fixed deadline reached'


def git_json(commit, path):
    return json.loads(subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT))


def identifier(sample):
    return sample['dataset']+'__'+sample['task']+'__'+sample['question_id']


def score(sample, text):
    if sample['dataset'] == 'GQA':
        return float(text.rstrip('.').lower() == sample['answer'])
    return vqa_score(text, sample['answers'])


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    # Analysis/dispatch may be extended between rounds; GPU execution is immutable.
    paths = [p for p in paths if p.name in ('engine.py', 'common.py', 'worker.py', 'tests.py', '__init__.py')]
    paths += [ROOT/'vlm_research'/p for p in (
        'ate_fulljoint_v2/engine.py', 'physical_handoff_fastpath/engine.py',
        'physical_handoff_fastpath/ledger.py', 'p1_layerwise_accuracy/engine.py',
        'receiver_cross_dataset/official_vqa_eval.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def config(writer='all', answer='none', cutoff=20, prompt='original', head=0):
    return dict(writer=writer, answer_attention=answer, cutoff=cutoff,
                prompt_variant=prompt, logits_to_keep=head, max_new_tokens=32)


def config_id(c):
    return f"{c['writer']}-{c['answer_attention']}-L{c['cutoff']}-{c['prompt_variant']}-H{c['logits_to_keep']}"


def prompt_text(sample, variant):
    original = sample['prompt_text']
    if variant == 'original':
        return original
    if variant == 'repeat-after-instruction':
        return original+'\nQuestion: '+sample['question']
    if variant == 'repeat-before-instruction':
        return sample['question']+'\nQuestion: '+sample['question']+'\nAnswer the question using a single word or phrase.'
    raise ValueError(variant)


def freeze_data():
    from transformers import AutoProcessor
    ensure_time()
    assert not (OUT/'data_freeze.json').exists()
    pool = git_json(BASE, 'vlm_research/results/gqa3000_p1_layerwise_accuracy/llava/sample_manifest.json')['samples']
    gbindings = git_json(BASE, 'vlm_research/results/gqa3000_p1_layerwise_accuracy/llava/input_bindings.json')
    quick = read(V2_ROOT/'sample_manifest.json')
    vmanifest = read(VQA_ROOT/'sample_manifest.json')
    assert len(pool) == 3000 and len(vmanifest['rows']) == 500
    historical_smoke = git_json('b483cdc524ce568fdcff5a871b1608b4aeecb322',
        'vlm_research/results/receiver_cross_dataset_validation/vqa_v2/smoke_selection.json')['question_ids']
    samples, buckets = {}, collections.defaultdict(list)
    for old in pool:
        s = dict(old, dataset='GQA')
        k = identifier(s)
        assert s['input_mapping']['prompt_ids'] == gbindings[s['task']+'__'+s['question_id']]['prompt_ids']
        samples[k] = s
        buckets[s['image_sha256']].append(k)
    for old in vmanifest['rows']:
        s = dict(old, dataset='VQAv2', task='VQAv2', prompt_text=old['question']+'\nAnswer the question using a single word or phrase.')
        spec = read(VQA_ROOT/'input_specs'/(s['question_id']+'.json'))
        s['input_mapping'] = dict(prompt_ids=spec['prompt_ids'])
        s['historical_input_hashes'] = spec['input_hashes']
        samples[identifier(s)] = s
        buckets[s['image_sha256']].append(identifier(s))
    assert len(samples) == 3500
    for s in samples.values():
        assert str(Path(s['image_path']).resolve()).startswith('/data/')
        assert sha(s['image_path']) == s['image_sha256']
    # No predictions, correctness, log-probabilities, or candidate outputs enter selection.
    forced = {s['image_sha256'] for s in samples.values() if s['dataset']=='VQAv2' and
              s['question_id'] in set(quick)|set(historical_smoke)}
    assignment = {h:'DEV' for h in forced}
    def allocate(dataset, split, targets):
        count = collections.Counter(s['task'] if dataset=='GQA' else s['answer_type']
            for s in samples.values() if s['dataset']==dataset and assignment.get(s['image_sha256'])==split)
        ranked = sorted(buckets, key=lambda h:digest([SEED, dataset, split, h]))
        for category in sorted(targets):
            for h in ranked:
                if count[category] >= targets[category]:
                    break
                if h in assignment:
                    continue
                rows = [samples[k] for k in buckets[h] if samples[k]['dataset']==dataset]
                if not any((s['task'] if dataset=='GQA' else s['answer_type'])==category for s in rows):
                    continue
                assignment[h] = split
                count.update(s['task'] if dataset=='GQA' else s['answer_type'] for s in rows)
        return dict(count)
    tasks = sorted({s['task'] for s in samples.values() if s['dataset']=='GQA'})
    allocate('GQA','DEV',dict.fromkeys(tasks,40))
    allocate('GQA','CHECK',dict.fromkeys(tasks,100))
    vcounts = collections.Counter(s['answer_type'] for s in samples.values() if s['dataset']=='VQAv2')
    # Exact 120/180/200 for this image-unique VQA cohort, with answer-type strata.
    for split, size in (('DEV',120),('CHECK',180)):
        for _ in range(size):
            current = [s for s in samples.values() if s['dataset']=='VQAv2' and assignment.get(s['image_sha256'])==split]
            if len(current)>=size:
                break
            have = collections.Counter(s['answer_type'] for s in current)
            category = max(vcounts, key=lambda t:size*vcounts[t]/500-have[t])
            options = [s for s in samples.values() if s['dataset']=='VQAv2' and s['image_sha256'] not in assignment
                       and s['answer_type']==category]
            selected = min(options, key=lambda s:digest([SEED,split,s['image_sha256']]))
            assignment[selected['image_sha256']] = split
    for h in buckets:
        assignment.setdefault(h,'FINAL')
    load, shard = [0,0], {}
    for h in sorted(buckets, key=lambda h:(-len(buckets[h]),digest([SEED,'shard',h]))):
        rank = min(range(2), key=lambda r:(load[r],r))
        shard[h] = rank
        load[rank] += len(buckets[h])
    coarse = []
    for t in tasks:
        options = [k for k,s in samples.items() if s['task']==t and assignment[s['image_sha256']]=='DEV']
        coarse += sorted(options, key=lambda k:digest([SEED,'coarse',k]))[:10]
    coarse += [k for k,s in samples.items() if s['dataset']=='VQAv2' and s['question_id'] in quick]
    counts = collections.Counter((s['dataset'],assignment[s['image_sha256']]) for s in samples.values())
    atomic(OUT/'sample_manifest.json',samples)
    atomic(OUT/'split_manifest.json',dict(image_split=assignment,image_shard=shard,coarse=coarse,
        forced_dev_vqa_ids=sorted(set(quick)|set(historical_smoke)), counts={str(k):v for k,v in counts.items()},
        rule='Input hash + task/answer type; whole image SHA clusters; historical quick60 and smoke cases forced DEV'))
    p = read(V2_ROOT/'protocol.json')
    atomic(OUT/'data_freeze.json',dict(at=now(), seed=SEED, base=BASE, model=MODELS['llava'],
        weights=p['weights'], tolerances=TOLERANCES['llava'],
        samples_sha256=sha(OUT/'sample_manifest.json'), splits_sha256=sha(OUT/'split_manifest.json'),
        quality_selection='CHECK minimizes worst dataset loss vs same-prompt Native; <=0.25pp ties use paired E2E then KV',
        final_selection='Before FINAL: quality-primary and at most one speed secondary; never reselect on FINAL',
        noninferiority='-1pp; image-cluster 20000 paired bootstrap; one-sided Bonferroni across frozen candidates x 2 datasets',
        timing='Paired same-device cold-KV, recompute Vision, 3 warmup+5 interleaved measured; Processor/I/O excluded for all',
        historical_limitation='All images belong to repeatedly researched historical cohorts; FINAL is only held out within this task'))
    print('DATA FROZEN',dict(counts),'coarse',len(coarse),'shards',load,flush=True)


if __name__ == '__main__':
    freeze_data()
