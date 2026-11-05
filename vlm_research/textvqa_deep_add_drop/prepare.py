"""Freeze image-cluster selection before reading any candidate outcomes."""
import subprocess
import sys

from PIL import Image

from .common import *
from ..fastp2_bypass_writer.common import hardware


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    test = subprocess.run([sys.executable, '-m', 'vlm_research.textvqa_deep_add_drop.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(test.stdout)
    assert test.returncode == 0, test.stdout
    previous_freeze = read(PREVIOUS / 'execution_freeze.json')
    assert previous_freeze['sources'] == previous.sources()
    assert sha(PREVIOUS / 'sample_manifest.json') == previous_freeze['sample_manifest_sha256']
    original = read(PREVIOUS / 'sample_manifest.json')
    assert len(original) == 500
    samples, order, end = select_samples(original)
    atomic(OUT / 'sample_manifest.json', samples)
    atomic(OUT / 'selection.json', dict(seed=SEED, target_questions=200,
        rule='Sort Image SHA clusters by sha256(seed|image_sha); choose closest-to-200 whole-cluster prefix, shorter prefix on ties.',
        selected_questions=len(samples), selected_image_clusters=end, all500_image_sha_order=order,
        selected_image_sha=order[:end], original500_manifest_sha256=sha(PREVIOUS / 'sample_manifest.json'),
        no_score_based_selection=True))
    groups = {h: [k for k, s in samples.items() if s['image_sha256'] == h] for h in order[:end]}
    assignment, loads = {}, [0, 0]
    for image in sorted(groups, key=lambda h: (-len(groups[h]), stable(h))):
        rank = min(range(2), key=lambda r: (loads[r], r))
        assignment[image] = rank
        loads[rank] += len(groups[image])
    smoke, seen = [], set()
    for k in sorted(samples, key=lambda k: stable('smoke|'+k)):
        if samples[k]['image_sha256'] not in seen:
            smoke.append(k)
            seen.add(samples[k]['image_sha256'])
        if len(smoke) == 3:
            break
    atomic(OUT / 'partition.json', dict(image_shard=assignment, question_counts=loads,
        smoke=smoke, same_image_all_conditions_same_device=True))
    for path, expected in {s['image_path']: s['image_sha256'] for s in samples.values()}.items():
        assert sha(path) == expected, path
        with Image.open(path) as im:
            im.verify()
    for path, expected in previous_freeze['weights'].items():
        assert sha(path) == expected, path
    roots = {'Native': previous.HISTORY, 'E': previous.PREVIOUS, 'Late': PREVIOUS}
    freezes = {k: read(root / 'execution_freeze.json') for k, root in roots.items()}
    totals, records = dict.fromkeys(BASELINES, 0.), []
    for key, sample in samples.items():
        assert sample['dataset'] == 'textvqa' and len(sample['answer']) == 10
        bindings, logits = [], []
        for label in BASELINES:
            path, value = historical_path(key, label), historical(key, label)
            assert value['key'] == key and value['condition'] == label
            assert value['sample_sha256'] == digest(sample) and value['sources'] == freezes[label]['sources']
            assert value['score'] == score(sample, value['generation']['text'])
            totals[label] += value['score']*100/len(samples)
            bindings.append(value['input_binding'])
            logits.append(value['generation']['first_logits_sha256'])
            records.append(dict(key=key, condition=label, path=str(path), sha256=sha(path),
                input_binding_sha256=digest(value['input_binding'])))
        assert bindings[0] == bindings[1] == bindings[2] and logits[0] == logits[1] == logits[2]
    atomic(OUT / 'source_manifest.json', dict(base_commit=BASE_COMMIT, reused_records=records,
        weights=previous_freeze['weights'], baseline_subset_accuracy=totals,
        previous_execution_freeze_sha256=sha(PREVIOUS / 'execution_freeze.json')))
    atomic(OUT / 'protocol.json', dict(created_at=now(), base_commit=BASE_COMMIT,
        model='LLaVA-1.5-7B', revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        dtype='FP16', attention='eager', transformers='4.49.0', logits_to_keep=1,
        greedy=True, max_new_tokens=32, questions=len(samples), image_clusters=end,
        entire_prefill_and_p1='Native in every layer, all Image/Prompt Attention and MLP unchanged',
        decode_text_attention='Native in all32 layers; no o_proj bypass',
        decode_image_layers={'Native': list(range(32)), 'E': list(range(11, 21)),
                             'Late': list(range(11, 32)), **KEEP},
        intervention='Only actual Decode Query -> original Image Key pre-softmax -inf hard mask',
        conditions=list(LABELS), new_generations=len(samples)*22, reused_generations=len(samples)*3,
        original_prompt_and_official_ten_answer_scorer=True,
        no_performance_tests=True, no_physical_optimization=True, no_extra_layer_combinations=True,
        exploratory_ordering_only=True, no_noninferiority_or_confirmatory_claim=True,
        comparisons='Add-Li minus E; Drop-Li minus Late; each minus Native as descriptive reference',
        bootstrap=dict(cluster='image_sha256', replicates=20000, seed=SEED, question_equal=True),
        source_hashes=sources(), tolerances=TOLERANCES['llava']))
    atomic(OUT / 'preflight_resources.json', dict(at=now(), **hardware()))
    print(dict(status='CPU_READY', CPU_tests=16, questions=len(samples), image_clusters=end,
               shard_questions=loads, baseline_subset_accuracy=totals), flush=True)


if __name__ == '__main__':
    main()
