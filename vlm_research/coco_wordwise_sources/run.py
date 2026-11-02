"""Single UUID-gated GPU; eight smoke images gate the rest of the fixed 200."""
import argparse
import gzip
import inspect
import os
import platform
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
import transformers
import transformers.models.llama.modeling_llama as llama

from benchmarks.smoke import hook_footprint
from knockout_probe import ROW_ATOL
from paper_rq1.run import prepared
from paper_rq1_single_layer_gqa6.run import tensor_hashes
from vlm_common import load_model, gpu_memory
from .common import *
from .engine import Observation, labels


def freeze():
    assert not (OUT / 'protocol.json').exists(), 'Never replace a frozen protocol'
    OUT.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_late_attention_sources', 'coco_wordwise_sources.tests'])
    with (OUT / 'cpu_tests.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert result.wasSuccessful(), 'CPU tests failed'
    write(OUT / 'cpu_validation.json', dict(passed=True, tests=result.testsRun, log_sha256=sha256(OUT / 'cpu_tests.txt')))
    manifest = read(OLD / 'sample_manifest.json')
    rows = manifest['samples']
    assert len(rows) == len({s['image_sha256'] for s in rows}) == 200
    assert all(s['prompt_text'] == 'Describe the image in one sentence.' for s in rows)
    for sample in rows:
        assert sha256(sample['image_path']) == sample['image_sha256']
    write(OUT / 'sample_manifest.json', dict(samples=rows, source_sha256=sha256(OLD / 'sample_manifest.json'),
          source_commit='6964407e6b', smoke_images=[s['image_identity'] for s in rows[:8]],
          representative_image=rows[0]['image_identity'], selection='All original 200, unchanged order; no output-dependent filtering'))
    old = read(OLD / 'execution_manifest.json')
    for path, entry in old['model_files'].items():
        assert sha256(path) == entry['sha256'], path
    write(OUT / 'protocol.json', dict(status='frozen_before_GPU', images=200, physical_gpu=1, gpu_uuid=UUID,
        model_revision=MODEL_REVISION, model_path=MODEL_PATH, dtype='float16', attention='eager', transformers='4.49.0',
        generation=dict(do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True, eos_token_id=2),
        observation='Two independent native model.generate calls per image, first bare then read-only; all effective logits and tokens bit exact',
        sources=SOURCES, composite_history=['GENERATED_OLDER', 'SELF'], layers=list(range(32)), contribution_layers=SELECTED,
        prefill='t=0 is P-1 predicting y[0], reported separately; SELF/generated history empty',
        decode='t>=1 consumes y[t-1] at original N+t-1, predicts y[t]; all layer heads reduced immediately',
        labels='Inherited align/classify and lexicons unchanged; Ambiguous/Special -> Unknown with raw label preserved; all subwords kept',
        word_primary='First predicted subword at t>=1; all remaining subwords and EOS retained separately',
        numerical=dict(row_atol=ROW_ATOL, native_logits='bit exact, no tolerance',
            decomposition='Original contributions/decompose audit unchanged; inherited History split bound also applied at L14'),
        statistics=dict(bootstrap=10000, seed=SEED, unit='image SHA256', confidence=.95,
            role='Within-image paired role means; image fixed effects adjusted for t, relative t, LogP, entropy and word token length',
            history_count='History key count equals t; t is the adjustment, not a collinear extra predictor. Also report per-key mass.',
            role_weight='Each image equal in paired means; fixed-effect regressions weight captions equally',
            inference='Descriptive/exploratory intervals, no layer-skipping or causal-necessity claims'),
        descriptive_similarity=dict(raw_source_js_max=1e-4, nonbos_source_js_max=1e-3,
            contribution_changed='Adjacent norm relative change >=0.20 OR adjacent projected-source cosine <=0.90',
            scope='Predeclared descriptive thresholds, not tests of equivalence or causal necessity'),
        sample_sha256=sha256(OUT / 'sample_manifest.json'), cpu_sha256=sha256(OUT / 'cpu_validation.json'),
        source_sha256=sources(), model_files=old['model_files'],
        transformers_source_sha256={inspect.getfile(llama):sha256(inspect.getfile(llama))},
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip()))
    print('Frozen 200 historical images; eight smoke images and representative fixed before GPU.', flush=True)


@torch.inference_mode()
def generate(model, prompt):
    output = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
        eos_token_id=2, return_dict_in_generate=True, output_logits=True, output_attentions=False,
        output_hidden_states=False)
    ids = output.sequences[0, prompt['input_ids'].shape[1]:].tolist()
    logits = torch.stack([v[0].detach().float().cpu() for v in output.logits])
    assert logits.shape[0] == len(ids) and torch.isfinite(logits).all()
    assert ids == logits.argmax(-1).tolist()
    n = prompt['input_ids'].shape[1]
    assert all(output.past_key_values.get_seq_length(i) == n+len(ids)-1 for i in range(32))
    return ids, logits


def head_csv(path, mass):
    import csv
    with gzip.open(path, 'wt', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['t', 'layer', 'head', *SOURCES])
        for t in range(len(mass)):
            for layer in range(32):
                for head in range(mass.shape[2]):
                    writer.writerow([t, layer, head, *[format(float(v), '.9g') for v in mass[t, layer, head]]])


def collect(model, processor, sample):
    start = time.monotonic()
    prompt = prepared(model, processor, sample)
    before = tensor_hashes(prompt)
    footprint = hook_footprint(model)
    original = llama.eager_attention_forward
    ids, native = generate(model, prompt)
    with Observation(model, prompt, model.config.image_token_index) as observer:
        repeated_ids, observed = generate(model, prompt)
    assert ids == repeated_ids and torch.equal(native, observed), 'Read-only observer changed native logits/tokens'
    assert tensor_hashes(prompt) == before and hook_footprint(model) == footprint
    assert llama.eager_attention_forward is original
    mass, counts = np.stack(observer.mass), np.array(observer.key_counts)
    assert mass.shape[:2] == (len(ids), 32) and mass.shape[2] == 32 and mass.shape[3] == 5
    annotation, token_labels = labels(processor.tokenizer, ids)
    logp = native.log_softmax(-1)
    token_logp = logp.gather(-1, torch.tensor(ids)[:, None]).squeeze(-1).tolist()
    entropy = (-(logp.exp()*logp).sum(-1)).tolist()
    name = sample['image_identity'].replace(':', '_')
    raw = RAW / (name+'.npz')
    raw.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, full_logits=native.numpy(), mass=mass, key_counts=counts)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, raw)
    heads = OUT / 'per_head_mass' / (name+'.csv.gz')
    heads.parent.mkdir(parents=True, exist_ok=True)
    head_csv(heads.with_suffix('.tmp.gz'), mass)
    os.replace(heads.with_suffix('.tmp.gz'), heads)
    historical = read(OLD / f"full/shard{sample['rank']}/checkpoints/{name}.json")
    historical_difference = None
    if historical['token_ids'] == ids:
        with np.load(historical['raw_path']) as prior:
            historical_difference = float(np.abs(native.numpy()-prior['full_logits']).max())
    worst = {}
    for audit in observer.audits:
        for field, value in audit.items():
            if isinstance(value, (float, int)) and ('error' in field):
                worst[field] = max(worst.get(field, 0), value)
    return dict(image_id=sample['image_identity'], image_sha256=sample['image_sha256'], input_binding=digest(sample['input_mapping']),
        input_tensor_sha256=before, token_ids=ids, caption=annotation['text'], words=annotation['words'], labels=token_labels,
        token_logp=token_logp, token_entropy=entropy, contributions=observer.contribution_rows,
        eos=ids[-1] == 2, truncated=len(ids) == 64 and ids[-1] != 2, empty=not bool(annotation['text'].strip()),
        native_observer_logits_exact=True, max_logit_difference=0, cache_lengths=observer.cache_lengths,
        parent_cache_isolation='Separate native generate calls; each begins with empty cache',
        all_layers_heads_steps_complete=True, hooks_clean=True, input_unchanged=True,
        row_max_error=float(np.abs(mass.sum(-1)-1).max()), decomp_audits=len(observer.audits), decomp_worst=worst,
        historical_tokens_exact=historical['token_ids'] == ids, historical_logits_max_abs_difference=historical_difference,
        raw_path=str(raw), raw_sha256=sha256(raw), head_csv=str(heads.relative_to(OUT)), head_sha256=sha256(heads),
        seconds=time.monotonic()-start, memory=gpu_memory())


def run():
    config = verify()
    assert transformers.__version__ == '4.49.0'
    initial = preflight()
    for path, binding in config['transformers_source_sha256'].items():
        assert sha256(path) == binding
    for path, record in config['model_files'].items():
        assert sha256(path) == record['sha256']
    preflight()
    torch.set_num_threads(1)
    model, processor, environment = load_model(MODEL_PATH)
    environment.update(preflight=initial, pid=os.getpid(), physical_gpu=1, uuid=UUID,
        source_commit=config['code_commit'], protocol_sha256=sha256(OUT / 'protocol.json'))
    write(OUT / 'environment.json', environment)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    own = [line for line in apps.splitlines() if int(line.split(',')[1]) == os.getpid()]
    assert len(own) == 1 and own[0].split(',')[0].strip() == UUID
    write(OUT / 'runtime_device.json', dict(pid=os.getpid(), actual_process_mapping=own, physical_gpu=1))
    versions = {n: (p._version, p.data_ptr()) for n, p in model.named_parameters()}
    rows = read(OUT / 'sample_manifest.json')['samples']
    start = time.monotonic()
    for index, sample in enumerate(rows):
        if index == 8:
            write(OUT / 'gpu_smoke.json', dict(passed=True, images=8, all_steps_all_logits_exact=True,
                image_ids=[s['image_identity'] for s in rows[:8]], all32_layers_and_heads=True,
                independent_native_caches=True, decomposition_bounds_unchanged=True, hooks_clean=True))
            print('EIGHT-IMAGE GPU SMOKE PASS; remaining formal collection permitted.', flush=True)
        checkpoint = OUT / 'checkpoints' / (sample['image_identity'].replace(':', '_')+'.json')
        if checkpoint.exists():
            result = read(checkpoint)
            assert result['binding'] == digest(config) and result['sample_sha256'] == digest(sample)
            assert sha256(result['raw_path']) == result['raw_sha256']
            assert sha256(OUT / result['head_csv']) == result['head_sha256']
        else:
            result = collect(model, processor, sample)
            assert versions == {n: (p._version, p.data_ptr()) for n, p in model.named_parameters()}
            result.update(binding=digest(config), sample_sha256=digest(sample))
            write(checkpoint, result)
        print(index+1, '/200', sample['image_identity'], 'tokens', len(result['token_ids']),
              'elapsed', round(time.monotonic()-start, 1), flush=True)
    verify()
    write(OUT / 'collection_completion.json', dict(complete=True, images=200, gpu_seconds=time.monotonic()-start,
          model_weights_versions_unchanged=True, native_observer_exact_all_images=True, memory=gpu_memory()))
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print('COMPLETE: GPU process exiting; release will be audited after exit.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    try:
        freeze() if args.mode == 'freeze' else run()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / f'failure_{time.time_ns()}.json', dict(status='failed_stopped_no_tolerance_change', traceback=traceback.format_exc()))
        raise
