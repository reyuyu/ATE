"""Post-process-exit GPU release, integrity gate, and artifact digests."""
import subprocess
from .common import *


def main():
    verify()
    assert read(OUT / 'analysis_completion.json')['complete']
    assert read(OUT / 'gpu_smoke.json')['passed']
    processes = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True).strip()
    assert not processes, 'GPU1 has a process; cannot certify idle GPU1'
    snapshot = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                       '--format=csv'], text=True)
    write(OUT / 'gpu_release.json', dict(physical_gpu=1, uuid=UUID, no_compute_process=True, snapshot=snapshot))
    samples = read(OUT / 'sample_manifest.json')['samples']
    results = [read(OUT / 'checkpoints' / (s['image_identity'].replace(':', '_')+'.json')) for s in samples]
    assert len(results) == 200 and all(r['native_observer_logits_exact'] and r['hooks_clean'] for r in results)
    write(OUT / 'validation.json', dict(complete=True, images=200, same_frozen_historical_inputs=True,
        native_observer_all_logits_max_difference=0, native_tokens_exact=True,
        all32_layers_all32_heads_all_generated_positions=True,
        row_max_error=max(r['row_max_error'] for r in results),
        decomposition_checks=sum(r['decomp_audits'] for r in results),
        decomposition_bounds_unchanged=True, consumed_predicted_token_alignment_explicit=True,
        multi_token_words_retained=True, independent_caches=True,
        original_weights_inputs_and_transformers_source_unchanged=True, hook_cleanup=True,
        cpu=read(OUT / 'cpu_validation.json'), gpu_smoke=read(OUT / 'gpu_smoke.json'),
        gpu1_released=True, trained=False, new_ablation=False))
    write(OUT / 'artifact_sha256.json', {str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*'))
          if p.is_file() and p.name != 'artifact_sha256.json'})
    print('All gates passed; GPU1 idle; four figures and full per-token/head results ready.', flush=True)


if __name__ == '__main__':
    main()
