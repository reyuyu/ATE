"""CPU-only preparation; never authorize or load a GPU implicitly."""
import subprocess
import sys

from PIL import Image

from .common import *
from ..fastp2_bypass_writer.common import hardware
from ..middle10_attribution.common import sources as previous_sources


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'execution_freeze.json').exists(), 'Already frozen'
    result = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_decode_regions.tests'],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    samples = read(PREVIOUS / 'sample_manifest.json')
    previous = read(PREVIOUS / 'execution_freeze.json')
    assert previous['sources'] == previous_sources()
    assert len(samples) == 500 and len({s['image_sha256'] for s in samples.values()}) == 309
    for path, expected in {s['image_path']: s['image_sha256'] for s in samples.values()}.items():
        assert sha(path) == expected, path
        with Image.open(path) as im:
            im.verify()
    for path, expected in previous['weights'].items():
        assert sha(path) == expected, path
    records, totals = [], dict.fromkeys(BASELINES, 0.)
    freezes = {'Native': read(HISTORY / 'execution_freeze.json'), 'E': previous}
    for key, s in samples.items():
        assert s['dataset'] == 'textvqa' and len(s['answer']) == 10
        bindings = []
        for label in BASELINES:
            path = historical_path(key, label)
            v = historical(key, label)
            assert v['sources'] == freezes[label]['sources']
            assert v['key'] == key and v['condition'] == label and v['sample_sha256'] == digest(s)
            assert v['score'] == score(s, v['generation']['text'])
            totals[label] += v['score']/5
            bindings.append(v['input_binding'])
            records.append(dict(key=key, condition=label, path=str(path), sha256=sha(path),
                input_binding_sha256=digest(v['input_binding']),
                first_logits_sha256=v['generation']['first_logits_sha256']))
        assert bindings[0] == bindings[1]
        assert historical(key, 'E')['generation']['first_logits_sha256'] == historical(key, 'Native')['generation']['first_logits_sha256']
    assert abs(totals['Native']-49.68) < 1e-8 and abs(totals['E']-45.14) < 1e-8
    atomic(OUT / 'sample_manifest.json', samples)
    atomic(OUT / 'partition.json', read(PREVIOUS / 'partition.json'))
    atomic(OUT / 'source_manifest.json', dict(base_commit=BASE_COMMIT, weights=previous['weights'],
        previous_freeze_sha256=sha(PREVIOUS / 'execution_freeze.json'), reused_records=records,
        historical_scores=totals, images_sha_verified=309))
    atomic(OUT / 'protocol.json', dict(created_at=now(), base_commit=BASE_COMMIT,
        model='LLaVA-1.5-7B', revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        dtype='FP16', attention='eager', transformers='4.49.0', logits_to_keep=1,
        greedy=True, max_new_tokens=32, questions=500, image_clusters=309,
        same_prompt=True, same_scoring='Frozen ten-answer TextVQA consensus soft accuracy',
        entire_prefill_including_p1='Native attention, visual access and all MLP in every layer',
        decode_image_layers={'Native': list(range(32)), **KEEP},
        intervention='Only true Decode Query -> original Image Key logits are -inf outside allowed layers',
        dense_mask_only=True, physical_exit=False, attention_output_bypass=False,
        new_generations=1000, reused_generations=1000,
        performance_requests=0, performance_forbidden_even_if_threshold_met=True,
        threshold=THRESHOLD,
        stop_rule='If neither Early nor Late reaches 47.68%, stop and submit; otherwise only propose a physical follow-up. Never benchmark or extend.',
        comparisons=['Early-E', 'Late-E', 'Early-Native', 'Late-Native', 'Late-Early', 'E-Native'],
        bootstrap=dict(cluster='image_sha256', replicates=20000, seed=SEED, question_equal=True),
        exploratory=True, tolerances=TOLERANCES['llava'], inference_sources=sources(),
        GPU_authorization='PENDING_FRESH_USER_CONFIRMATION'))
    atomic(OUT / 'preflight_resources.json', dict(at=now(), **hardware()))
    atomic(OUT / 'validation.json', dict(status='CPU_READY_GPU_AUTHORIZATION_PENDING',
        CPU_passed=True, CPU_tests=18, images_verified=309, historical_records_audited=1000,
        historical_scores=totals, GPU_model_loads=0, GPU_smoke=False,
        new_quality_generations=0, performance_requests=0))
    print('CPU_READY: same500/309, 1000 Native/E records audited; no GPU model loaded.')


if __name__ == '__main__':
    main()
