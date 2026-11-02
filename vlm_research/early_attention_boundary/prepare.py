"""Freeze an output-blind, image-diverse subset and all execution bindings."""
from .common import *


def main():
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite a frozen protocol'
    source = read(HISTORY / 'shared_3000_manifest.json')
    samples = select(source['samples'])
    assert len(samples) == len({key(s) for s in samples}) == 120
    assert all(sum(s['task'] == t for s in samples) == 20 for t in TASKS)
    for sample in samples:
        assert sha256(sample['image_path']) == sample['image_sha256']
    write(OUT / 'sample_manifest.json', dict(seed=SEED, selection='Round-robin task, input SHA256 order, prefer unused image SHA256',
          source=str(HISTORY / 'shared_3000_manifest.json'), source_sha256=sha256(HISTORY / 'shared_3000_manifest.json'),
          samples=samples, image_clusters=len({s['image_sha256'] for s in samples})))
    bindings = read(HISTORY / 'model_and_environment_bindings.json')
    for model in CONDITIONS:
        original = read(HISTORY / f'{model}_input_manifest.json')
        lookup = {key(s): s for s in original['samples']}
        write(OUT / f'{model}_input_manifest.json', dict(model=MODELS[model],
              source_sha256=sha256(HISTORY / f'{model}_input_manifest.json'),
              samples=[lookup[key(s)] for s in samples]))
    write(OUT / 'model_and_environment_bindings.json', bindings)
    files = ['sample_manifest.json', 'qwen_input_manifest.json', 'internvl_input_manifest.json',
             'model_and_environment_bindings.json']
    write(OUT / 'protocol.json', dict(status='frozen_before_GPU', sample_count=120, seed=SEED,
          conditions=CONDITIONS, model=MODELS, physical_gpu=1, gpu_uuid=UUID,
          intervention='Only o_proj output at original P-1 and every real incremental decode query; native KV updates',
          generation=dict(do_sample=False, num_beams=1, max_new_tokens=32, native_eos=True, cache=True),
          scoring="Pinned GQA conversion: generated text.strip().rstrip('.').lower() == original gold",
          sampling_uses_model_outputs=False, full_reuse=False,
          full_reuse_reason='Previous 3000-question single-layer study saved first-token forwards, not complete answers',
          statistics=dict(unit='Image SHA256 cluster, question-weighted paired bootstrap', repetitions=20000,
              ordinary_confidence=.95, comparison_family='All nine window-vs-Full comparisons across both models',
              simultaneous_method='Bonferroni percentile intervals, tail probability 0.025/9',
              reference_delta_pp=-1, exploratory=True, noninferiority_claim=False,
              zero_discordance_warning='Empirical [0,0] bootstrap is not a population safety guarantee',
              descriptive_low_harm='Point estimate delta >= -1 pp; never equivalent to safety or confirmation'),
          code_sha256=source_hashes(), file_sha256={name: sha256(OUT / name) for name in files}))
    print('Frozen:', len(samples), 'questions;', len({s['image_sha256'] for s in samples}), 'images', flush=True)


if __name__ == '__main__':
    main()
