"""CPU-only input/baseline audit and explicit quality-first protocol freeze."""
import subprocess
import sys

from PIL import Image

from .common import *
from ..fastp2_bypass_writer.common import hardware
from ..middle10_recovery.common import sources as previous_sources


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'execution_freeze.json').exists(), 'Do not alter frozen GPU inputs'
    cpu = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_attribution.tests'],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / 'cpu_tests.log').write_text(cpu.stdout)
    assert cpu.returncode == 0, cpu.stdout
    samples = read(PREVIOUS / 'sample_manifest.json')
    previous = read(PREVIOUS / 'execution_freeze.json')
    assert previous['sources'] == previous_sources(), 'Prior inference code changed'
    assert len(samples) == 500 and len({s['image_sha256'] for s in samples.values()}) == 309
    images = {s['image_path']: s['image_sha256'] for s in samples.values()}
    for path, expected in images.items():
        assert sha(path) == expected, path
        with Image.open(path) as im:
            im.verify()
    for path, expected in previous['weights'].items():
        assert sha(path) == expected, path
    reused, totals = [], dict.fromkeys(BASELINES, 0.)
    for key, s in samples.items():
        assert s['dataset'] == 'textvqa' and len(s['answer']) == 10
        bindings = []
        for label in BASELINES:
            root = PREVIOUS if label == 'E3' else HISTORY
            path = root / 'checkpoints' / key / (label+'.json')
            v = historical(key, label)
            expected_freeze = previous if label == 'E3' else read(HISTORY / 'execution_freeze.json')
            assert v['sources'] == expected_freeze['sources']
            assert v['key'] == key and v['sample_sha256'] == digest(s)
            assert v['score'] == score(s, v['generation']['text'])
            totals[label] += v['score']/5
            bindings.append(v['input_binding'])
            reused.append(dict(key=key, condition=label, path=str(path), sha256=sha(path),
                input_binding_sha256=digest(v['input_binding']),
                first_logits_sha256=v['generation']['first_logits_sha256']))
        assert all(v == bindings[0] for v in bindings)
    assert all(abs(totals[k]-v)<1e-8 for k, v in
               [('Native', 49.68), ('C3+Middle10', 39.46), ('E3', 41.38)])
    atomic(OUT / 'sample_manifest.json', samples)
    atomic(OUT / 'partition.json', read(PREVIOUS / 'partition.json'))
    atomic(OUT / 'source_manifest.json', dict(base_commit=BASE_COMMIT,
        previous_freeze_sha256=sha(PREVIOUS / 'execution_freeze.json'),
        weights=previous['weights'], baseline_scores=totals, reused_records=reused,
        images_sha_verified=309, GPU_reproduction='PENDING'))
    atomic(OUT / 'protocol.json', dict(created_at=now(), base_commit=BASE_COMMIT,
        model='LLaVA-1.5-7B', revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        dtype='FP16', attention='eager', transformers='4.49.0', logits_to_keep=1,
        greedy=True, max_new_tokens=32, questions=500, image_clusters=309,
        no_extra_OCR=True, same_prompt=True, conditions={k: policy(k) for k in LABELS},
        C_D_E_prefill_p1='All32 native Attention including L2-L5; otherwise all32 visual access is not implemented.',
        R_definition='Original positions strictly after final Image Token and before P-1, as in prior experiment.',
        dense_attribution_only=True, physical_speed_claim=False,
        stage1=dict(allowed='Only correctness Smoke and full500 quality A-E',
            timing_forbidden=True, CUDA_timing_events_forbidden=True, FLOPs_forbidden=True,
            new_generations=2500, quality_gate=THRESHOLD,
            if_best_below_gate='STOP_AND_SUBMIT_QUALITY_ATTRIBUTION_ONLY'),
        stage2=dict(allowed_only_if_dense_accuracy_at_least=THRESHOLD,
            action='Identify recovery source, then attempt physical translation preserving true visual exit.',
            physical_correctness_required=True, physical_full500_accuracy_gate=THRESHOLD),
        stage3=dict(performance48_allowed_only_if='Both dense and physical full500 gates pass, plus numerical checks',
            auto_start=False, otherwise='FORBIDDEN'),
        comparisons=['A-E3', 'B-A', 'C-A', 'D-B', 'D-C', 'E-D',
                     'A-Native', 'B-Native', 'C-Native', 'D-Native', 'E-Native'],
        interaction='D-B-C+A; descriptive, no additive-causality claim',
        bootstrap=dict(cluster='image_sha256', replicates=20000, seed=SEED, question_equal=True),
        inference_sources=sources(), tolerances=TOLERANCES['llava'],
        exploratory=True, independent_confirmation=False,
        GPU_authorization='PENDING_FRESH_USER_CONFIRMATION'))
    atomic(OUT / 'preflight_resources.json', dict(at=now(), **hardware()))
    atomic(OUT / 'validation.json', dict(status='CPU_READY_GPU_AUTHORIZATION_PENDING',
        CPU_tests=19, CPU_passed=True, image_sha_verified=309,
        historical_records_audited=1500, baseline_scores=totals,
        GPU_model_loads=0, GPU_smoke=False, quality_generations=0, performance_requests=0))
    print('CPU_READY: 19 tests, same500/309 images, 1500 baseline bindings. No GPU authorized or started.')


if __name__ == '__main__':
    main()
