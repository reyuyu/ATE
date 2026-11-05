"""CPU-only audit; GPU execution additionally requires fresh user authorization."""
import collections
import subprocess
import sys

from PIL import Image

from .common import *
from ..c3v2_middle_reaccess.common import sources as historical_sources
from ..fastp2_bypass_writer.common import hardware


def main():
    assert not (OUT / 'execution_freeze.json').exists(), 'Do not alter a frozen experiment'
    OUT.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run([sys.executable, '-m', 'vlm_research.middle10_recovery.tests'],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUT / 'cpu_tests.log').write_text(tests.stdout)
    assert tests.returncode == 0, tests.stdout
    samples = read(HISTORY / 'sample_manifest.json')
    old_freeze = read(HISTORY / 'execution_freeze.json')
    old_protocol = read(HISTORY / 'protocol.json')
    assert old_freeze['sources'] == historical_sources(), 'Historical inference sources changed'
    assert len(samples) == 500 and len({s['image_sha256'] for s in samples.values()}) == 309
    assert all(s['dataset'] == 'textvqa' and len(s['answer']) == 10 for s in samples.values())
    assert old_protocol['revision'] == 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    assert old_protocol['logits_to_keep'] == 1 and old_protocol['max_new_tokens'] == 32
    assert old_protocol['tolerances'] == TOLERANCES['llava']
    images = {s['image_path']: s['image_sha256'] for s in samples.values()}
    for path, expected in images.items():
        assert sha(path) == expected, path
        with Image.open(path) as image:
            image.verify()
    for path, expected in old_freeze['weights'].items():
        assert sha(path) == expected, 'Frozen weights changed: ' + path
    reused = []
    for key, sample in samples.items():
        bindings = []
        for label in BASELINES:
            path = HISTORY / 'checkpoints' / key / (label + '.json')
            old = load(path)
            assert old['sources'] == old_freeze['sources']
            assert old['sample_sha256'] == digest(sample)
            assert old['key'] == key and old['condition'] == label
            assert old['score'] == score(sample, old['generation']['text'])
            bindings.append(old['input_binding'])
            reused.append(dict(key=key, condition=label, path=str(path), sha256=sha(path),
                sample_sha256=digest(sample), input_binding_sha256=digest(old['input_binding']),
                token_ids_sha256=digest(old['generation']['token_ids']),
                first_logits_sha256=old['generation']['first_logits_sha256']))
        assert all(binding == bindings[0] for binding in bindings)
    partition = read(HISTORY / 'partition.json')
    assert len(partition['performance48']) == 48
    assert len({samples[k]['image_sha256'] for k in partition['performance48']}) == 48
    atomic(OUT / 'sample_manifest.json', samples)
    atomic(OUT / 'partition.json', partition)
    atomic(OUT / 'source_manifest.json', dict(at=now(), base_commit=BASE_COMMIT,
        historical_result_root=str(HISTORY), historical_freeze_sha256=sha(HISTORY / 'execution_freeze.json'),
        historical_protocol_sha256=sha(HISTORY / 'protocol.json'),
        historical_sources_verified=True, weights=old_freeze['weights'],
        images_verified=len(images), records_verified=len(reused), reused_records=reused,
        note='CPU binding audit only; same-device Native/C3/M9/M10 reproduction remains a GPU gate.'))
    atomic(OUT / 'protocol.json', dict(created_at=now(), model='LLaVA-1.5-7B',
        revision=old_protocol['revision'], dtype='FP16', attention='eager', transformers='4.49.0',
        logits_to_keep=1, greedy=True, max_new_tokens=32, samples=500, image_clusters=309,
        labels=LABELS, new_conditions=NEW, prefill_p1_image_layers=PREFILL_KEEP,
        persistent_image_kv_and_decode_image_layers=KEEP, hidden_exit=CUTOFF,
        receiver_last_image_layer=WRITER_LAST, answer_attention_skip_layers=[2, 3, 4, 5],
        native_exception='Native is completely unmodified: all queries read images, all Attention runs, all32 Image KV layers persist.',
        E4='Restore only Prefill P-1 text Attention in L2-L5, no persistent KV in these layers; Decode still physically skipped.',
        E5='Middle9 image access with Hidden exit21; no R/P-1/Decode image access or persistent Image KV at L20.',
        comparisons=COMPARISONS, factorial='E3 - E2 - E1 + C3+Middle10',
        bootstrap=dict(replicates=20000, seed=SEED, cluster='image_sha256', question_equal=True),
        exploratory=True, noninferiority_claim=False, tolerances=TOLERANCES['llava'],
        physical_reference_gate='Unchanged FP16 numerical gates plus exact greedy tokens; failure stops candidate.',
        performance=dict(performance48=partition['performance48'], warmups=3, repetitions=5,
            modes=['natural_eos', 'fixed16'], primary_cost='fixed16 E2E',
            condition_selection='Native,C3-V2,C3+Middle10 plus all E candidates with positive full500 mean score change versus Middle10; none positive => skip performance.',
            timing='Exact frozen c3v2_middle_reaccess.generate boundaries, fresh vision, independent cold KV, interleaved labels.',
            tensor_shape_audit=True), gpu_authorization='PENDING; no historical GPU authorization reused'))
    atomic(OUT / 'validation.json', dict(status='CPU_READY_GPU_PENDING', at=now(),
        cpu_passed=True, cpu_tests=19, historical_quality_records_audited=2000,
        image_sha_verified=True, frozen_weights_verified=True,
        gpu_smoke_passed=False, formal_generations=0, performance_measurements=0,
        gpu_authorization='PENDING'))
    atomic(OUT / 'preflight_resources.json', dict(at=now(), **hardware()))
    print('CPU_READY: 19 tests; 500 questions / 309 image clusters; 2000 historical records audited. GPU authorization pending.', flush=True)


if __name__ == '__main__':
    main()
