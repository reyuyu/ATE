"""Frozen C0/C2 confirmation on all original 240 images; no Sink/ROI dependency."""
import argparse
from contextlib import AbstractContextManager
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import torch
from PIL import Image
from fixed_k_exit import atomic_json
from knockout_probe import model_layers, resource_preflight
from phase38_sink_hooks import zero_query, thash
from phase38_sink_stage import (ROOT, MODEL, IMAGES, PINS, read, git, sources,
                                library_hash, weights_hash, hooks_state, evaluate)
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from roi_sink_probe import inputs_and_history, score
from vlm_common import load_model

OUTPUT = ROOT / 'results/phase3_8b_c2_confirm240'
HISTORY = ROOT / 'results/phase3_2_fixed_k_exit'
PILOT = ROOT / 'results/phase3_8_sink_stage'
BRANCH = 'exp/phase3_8b-c2-confirm240'
GPU_UUID = 'b08489af-0a8a-eb48-55ba-8346733af417'
BASE = '2415b69264d20c5da1c47c18bdb0b5adcafa69a7'
PLAN = dict(conditions=['C0', 'C2'], layers=[2, 3, 4], query='actual prompt P-1',
            offline_right_bias=1.0, bootstrap_seed=382402026, bootstrap_draws=10000,
            ties='incorrect', score='original full-vocabulary conditional candidate sequence LogP',
            smoke_ids=['02474907', '00784767'], primary='Confirmation224')


def cohorts(original, discovery):
    samples = sorted(copy.deepcopy(original), key=lambda s: s['question_id'])
    ids = [s['question_id'] for s in samples]
    images = [s['img_id'] for s in samples]
    discovered = {s['question_id'] for s in discovery}
    if len(samples) != 240 or len(set(ids)) != 240 or len(set(images)) != 240:
        raise ValueError('Expected 240 unique frozen questions and images')
    if len(discovery) != 16 or len(discovered) != 16 or not discovered <= set(ids):
        raise ValueError('Invalid discovery subset')
    for s in samples:
        if sorted(s['candidates']) != ['left', 'right'] or s['answer'] not in s['candidates']:
            raise ValueError('Frozen binary candidate protocol changed')
        s['cohort'] = 'Discovery16' if s['question_id'] in discovered else 'Confirmation224'
    for name, n in [('Discovery16', 8), ('Confirmation224', 112)]:
        for answer in ['left', 'right']:
            if sum(s['cohort'] == name and s['answer'] == answer for s in samples) != n:
                raise ValueError('Unbalanced or overlapping frozen cohorts')
    return samples


class C2Trace(AbstractContextManager):
    """Same zero_query MLP edit as Phase3.8, without any Attention modification."""
    def __init__(self, layers, query, condition):
        if condition not in ['C0', 'C2'] or len(layers) != 32 or query < 0:
            raise ValueError('Only frozen 32-layer C0/C2 protocol is allowed')
        self.layers, self.query, self.condition = layers, query, condition
        self.handles, self.edits = [], []
        self.prefix, self.hidden, self.mid, self.mlp, self.attention = {}, {}, {}, {}, {}
        self.mlp_calls = []

    def __enter__(self):
        try:
            for i, layer in enumerate(self.layers):
                self.handles.append(layer.self_attn.register_forward_hook(self.attention_hook(i)))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.mid_hook(i)))
                self.handles.append(layer.mlp.register_forward_hook(self.mlp_hook(i)))
                self.handles.append(layer.register_forward_hook(self.layer_hook(i)))
            return self
        except BaseException:
            self.__exit__(); raise

    def attention_hook(self, i):
        def hook(module, args, out):
            self.attention[i] = thash(out[0][0, self.query])
        return hook

    def mid_hook(self, i):
        def hook(module, args):
            self.mid[i] = args[0][0, self.query].detach().clone()
        return hook

    def mlp_hook(self, i):
        def hook(module, args, out):
            self.mlp_calls.append(i)
            result = out
            if self.condition == 'C2' and i in PLAN['layers']:
                result, audit = zero_query(out, self.query)
                self.edits.append(dict(layer=i, kind='mlp_before_residual', **audit))
            self.mlp[i] = result[0, self.query].detach().clone()
            return result
        return hook

    def layer_hook(self, i):
        def hook(module, args, out):
            value = out[0]
            if not torch.isfinite(value).all() or not torch.equal(value[0, self.query], self.mid[i] + self.mlp[i]):
                raise AssertionError('Nonfinite hidden or edit not before actual residual addition')
            self.hidden[i] = thash(value[0, self.query])
            self.prefix[i] = thash(value[0, :self.query])
        return hook

    def assert_complete(self):
        if list(self.hidden) != list(range(32)) or self.mlp_calls != list(range(32)) or len(self.attention) != 32:
            raise AssertionError('All 32 Attention/MLP/layer modules must execute exactly once')
        expected = PLAN['layers'] if self.condition == 'C2' else []
        if [e['layer'] for e in self.edits] != expected:
            raise AssertionError('Wrong intervention scope')

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def prepare():
    if OUTPUT.exists() or git('branch', '--show-current') != BRANCH or git('status', '--porcelain'):
        raise ValueError('Use new output and clean committed isolated branch')
    pilot = read(PILOT / 'manifest.json')['config']
    samples = cohorts(read(HISTORY / 'manifest.json')['samples'], pilot['samples'])
    inputs = [HISTORY/'manifest.json', HISTORY/'per_sample_results.csv', PILOT/'manifest.json',
              PILOT/'per_sample_scores.csv', PILOT/'publication_validation.json', ROOT/'configs/model.json', ROOT/'requirements.txt']
    for s in samples:
        path = IMAGES / f"image_{s['img_id']}.jpg"
        if sha256(path) != s['image_sha256']:
            raise ValueError('Original image bytes changed')
        with Image.open(path) as image:
            image.load()
            s['validated_image_size'] = list(image.size)
            if list(image.size) != s['image_size']:
                raise ValueError('Frozen image dimensions changed')
        inputs.append(HISTORY/'checkpoints/run'/f"{s['question_id']}.json")
        if s['cohort'] == 'Discovery16':
            inputs.extend([PILOT/'run'/f"sample_{s['question_id']}"/'sample.json',
                           PILOT/'run'/f"sample_{s['question_id']}"/'raw_vectors_logits_attention.npz'])
    model_files = {item['name']: sha256(MODEL/item['name']) for item in read(ROOT/'configs/model.json')['files']}
    if model_files != pilot['model_files_sha256']:
        raise ValueError('Model differs from actual Phase3.8 fixed snapshot')
    for name, h in pilot['sources'].items():
        if sha256(ROOT/name) != h:
            raise ValueError('Original Phase3.8/shared source changed')
    config = dict(phase='3.8b', base_commit=BASE, code_commit=git('rev-parse', 'HEAD'),
                  sources=sources(), inputs={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                  model_path=str(MODEL), image_root=str(IMAGES), model_files_sha256=model_files,
                  model_revision=MODEL.name, packages=PINS, python=platform.python_version(),
                  library=library_hash(), plan=PLAN, physical_gpu=1, gpu_uuid=GPU_UUID,
                  gpu_name='NVIDIA GeForce RTX 3090', interpreter=sys.executable,
                  limitations='Frozen upstream correct_question cohort; confirmation is independent of C2 discovery, not unbiased GQA.')
    binding = digest(dict(config=config, samples=samples))
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT/'experiment_config.json', config)
    atomic_json(OUTPUT/'manifest.json', dict(binding=binding, samples=samples,
        discovery_ids=[s['question_id'] for s in samples if s['cohort']=='Discovery16'],
        confirmation_ids=[s['question_id'] for s in samples if s['cohort']=='Confirmation224'],
        source_manifest_sha256=sha256(HISTORY/'manifest.json'), images_unique=True))
    print('FROZEN', config['code_commit'], binding, flush=True)


def protocol():
    c, m = read(OUTPUT/'experiment_config.json'), read(OUTPUT/'manifest.json')
    if digest(dict(config=c, samples=m['samples'])) != m['binding'] or c['plan'] != PLAN or sources() != c['sources']:
        raise ValueError('Frozen code/config/data mismatch')
    if git('rev-parse', 'HEAD') != c['code_commit'] or library_hash() != c['library']:
        raise ValueError('Code commit or installed implementation changed')
    if {k: importlib.metadata.version(k) for k in PINS} != PINS or platform.python_version() != c['python']:
        raise ValueError('Fixed environment changed')
    for p, h in c['inputs'].items():
        if sha256(ROOT/p) != h:
            raise ValueError('Historical input changed: '+p)
    return c, m


def cpu():
    c, m = protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Hide GPUs for CPU acceptance')
    import vlm_common
    vlm_common.default_model = lambda: str(MODEL)
    folder = OUTPUT/'tests'; folder.mkdir()
    suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test*.py')
    with (folder/'cpu_tests.txt').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (folder/'pip_check.txt').write_text(pip.stdout+pip.stderr)
    (folder/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    passed = result.wasSuccessful() and not result.skipped and pip.returncode == 0
    atomic_json(folder/'validation.json', dict(passed=passed, binding=m['binding'], tests=result.testsRun,
                failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped)))
    if not passed:
        raise RuntimeError('CPU gate failed; do not relax')
    print('CPU PASS', result.testsRun, flush=True)


def sample_run(model, processor, frozen, folder):
    sample = copy.deepcopy(frozen)
    sample['image_path'] = str(IMAGES/f"image_{sample['img_id']}.jpg")
    if sha256(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Image changed')
    prompt, candidates, qpos, apos, history, qrecords, counts = inputs_and_history(model, processor, sample)
    ids = prompt['input_ids'][0]
    visual = torch.where(ids == model.config.image_token_index)[0].tolist()
    if len(visual) != 576 or apos != len(ids)-1 or max(visual+qpos) >= apos:
        raise AssertionError('Dynamic query/image mapping failed')
    clean = [evaluate(model, x, qpos, apos) for x, rec in candidates]
    values, traces, rows, audits, arrays = {}, {}, [], [], {}
    gold = sample['candidates'].index(sample['answer'])
    for condition in ['C0', 'C2']:
        values[condition], traces[condition] = [], []
        for j, (x, rec) in enumerate(candidates):
            trace = C2Trace(model_layers(model), apos, condition)
            value = evaluate(model, x, qpos, apos, trace)
            if condition == 'C0':
                if value['effective_text_sha256'] != clean[j]['effective_text_sha256'] or not torch.equal(value['question_hidden'], clean[j]['question_hidden']):
                    raise AssertionError('Native observer changed actual text logits')
            else:
                full_trace = traces['C0'][j]
                if trace.prefix != full_trace.prefix or trace.hidden[1] != full_trace.hidden[1]:
                    raise AssertionError('Earlier causal positions or pre-intervention L1 changed')
                if trace.attention[2] != full_trace.attention[2]:
                    raise AssertionError('First intervention must retain native L2 Attention')
                if not torch.equal(value['question_hidden'], values['C0'][j]['question_hidden']):
                    raise AssertionError('Question hidden causal invariant violated')
            values[condition].append(value); traces[condition].append(trace)
            audits.append(dict(condition=condition, candidate=sample['candidates'][j], edits=trace.edits,
                prefix_all_tokens_layer_sha256=trace.prefix, prediction_hidden_layer_sha256=trace.hidden,
                attention_update_sha256=trace.attention, question_sha256=thash(value['question_hidden']),
                input_sha256=value['input_sha256'], text_logits_sha256=value['effective_text_sha256'],
                measurement=value['measurement'],
                all_32_layers_attention_mlp_executed=True, residual_equations_exact=True, hooks_cleaned=True))
        if not torch.equal(values[condition][0]['prediction_logits'], values[condition][1]['prediction_logits']):
            raise AssertionError('Future candidate leaked into P-1 logits')
        if traces[condition][0].hidden != traces[condition][1].hidden:
            raise AssertionError('Future candidate leaked into P-1 hidden')
        logs = [score(v, rec) for v, (x, rec) in zip(values[condition], candidates)]
        full = [score(v, rec) for v, (x, rec) in zip(values['C0'], candidates)]
        if not np.isfinite(logs).all():
            raise AssertionError('Nonfinite scores')
        margin, fm = logs[gold]-logs[1-gold], full[gold]-full[1-gold]
        rows.append(dict(question_id=sample['question_id'], img_id=sample['img_id'], question=sample['question'],
            gold=sample['answer'], cohort=sample['cohort'], condition=condition,
            left_logp=logs[sample['candidates'].index('left')], right_logp=logs[sample['candidates'].index('right')],
            gold_logp=logs[gold], wrong_logp=logs[1-gold], margin=margin, correct=margin>0,
            full_margin=fm, gain=margin-fm, gold_logp_gain=logs[gold]-full[gold],
            wrong_logp_gain=logs[1-gold]-full[1-gold], correct_to_wrong=fm>0 and margin<=0,
            wrong_to_correct=fm<=0 and margin>0))
        arrays[condition+'_prediction_logits'] = np.stack([v['prediction_logits'].numpy() for v in values[condition]])
    for j, (x, rec) in enumerate(candidates):
        restored = evaluate(model, x, qpos, apos)
        if restored['effective_text_sha256'] != clean[j]['effective_text_sha256'] or not torch.equal(restored['question_hidden'], clean[j]['question_hidden']):
            raise AssertionError('Post-C2 native restoration failed')
    regression = None
    if sample['cohort'] == 'Discovery16':
        old = read(PILOT/'run'/f"sample_{sample['question_id']}"/'sample.json')
        regression = {}
        with np.load(PILOT/'run'/f"sample_{sample['question_id']}"/'raw_vectors_logits_attention.npz') as raw:
            for condition in ['C0', 'C2']:
                key = condition+'_prediction_logits'
                delta = float(np.max(np.abs(arrays[key].astype(float)-raw[key].astype(float))))
                regression[condition] = dict(logits_max_abs_delta=delta, bitwise_exact=np.array_equal(arrays[key], raw[key]),
                    margin_delta=next(r for r in rows if r['condition']==condition)['margin']-next(r for r in old['scores'] if r['condition']==condition)['margin'])
                if not regression[condition]['bitwise_exact'] or regression[condition]['margin_delta'] != 0:
                    raise AssertionError('Same-environment Pilot regression not exact; investigate, do not relax')
    np.savez_compressed(folder/'logits.npz', **arrays)
    result = dict(passed=True, question_id=sample['question_id'], scores=rows, audit=audits,
        mapping=dict(prompt_ids=ids.tolist(), answer_query=apos, visual_positions=visual, question=qrecords,
                     candidate_mapping=history['candidate_mapping'], token_counts=counts),
        pilot_regression=regression, historical_logp_deltas=[score(v, rec)-h['logp'] for v, (x, rec), h in zip(values['C0'], candidates, history['full_candidates'])],
        native_restoration_exact=True, raw_sha256=sha256(folder/'logits.npz'))
    atomic_json(folder/'sample.json', result)
    return result


def checkpoint(folder, sample, binding, execution):
    if not (folder/'checkpoint.json').exists():
        return None
    record = read(folder/'checkpoint.json')
    if record != dict(binding=binding, execution=execution, sample_sha256=digest(sample), sample_json_sha256=sha256(folder/'sample.json')):
        raise ValueError('Incompatible or modified checkpoint')
    data = read(folder/'sample.json')
    if not data['passed'] or data['question_id'] != sample['question_id'] or data['raw_sha256'] != sha256(folder/'logits.npz'):
        raise ValueError('Invalid checkpoint payload')
    return data


def execute(stage):
    c, m = protocol(); binding = m['binding']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or torch.cuda.device_count() != 1 or str(torch.cuda.get_device_properties(0).uuid) != GPU_UUID:
        raise ValueError('Must expose physical RTX3090 GPU1 only')
    for gate in ['tests'] + (['smoke'] if stage=='run' else []):
        previous = read(OUTPUT/gate/'validation.json')
        if not previous['passed'] or previous['binding'] != binding:
            raise ValueError('CPU/smoke gate incomplete')
    selected = m['samples'] if stage=='run' else [next(s for s in m['samples'] if s['question_id']==q) for q in PLAN['smoke_ids']]
    out = OUTPUT/stage; out.mkdir(exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource = resource_preflight()
        if not resource['ready']:
            raise RuntimeError('GPU occupied; no other jobs stopped')
        for name, h in c['model_files_sha256'].items():
            if sha256(MODEL/name) != h:
                raise ValueError('Model disk snapshot changed')
        model, processor, environment = load_model(model_path=str(MODEL))
        before = weights_hash(model)
        execution = digest(dict(binding=binding, weights=before['sha256'], gpu=GPU_UUID, library=c['library'], packages=PINS, python=c['python']))
        env_path = out/'environment.json'
        payload = dict(execution=execution, binding=binding, environment=environment, resource=resource,
                       python=platform.python_version(), packages=PINS, weights_before=before, gpu_uuid=GPU_UUID)
        if env_path.exists():
            if read(env_path)['execution'] != execution:
                raise ValueError('Environment/weight incompatible resume')
        else:
            atomic_json(env_path, payload)
        versions = {k: p._version for k, p in model.named_parameters()}
        if set(p.name for p in out.glob('sample_*')) - {f"sample_{s['question_id']}" for s in selected}:
            raise ValueError('Foreign sample checkpoint')
        completed = []
        start = time.perf_counter()
        try:
            for s in selected:
                folder = out/f"sample_{s['question_id']}"
                data = checkpoint(folder, s, binding, execution) if folder.exists() else None
                if data is None:
                    if folder.exists() and list(folder.iterdir()):
                        raise RuntimeError('Partial attempt preserved; inspect explicitly before resuming')
                    folder.mkdir(exist_ok=True)
                    if sources() != c['sources'] or library_hash() != c['library']:
                        raise ValueError('Source changed while executing')
                    data = sample_run(model, processor, s, folder)
                    if versions != {k: p._version for k, p in model.named_parameters()}:
                        raise AssertionError('Parameter mutation')
                    atomic_json(folder/'checkpoint.json', dict(binding=binding, execution=execution,
                        sample_sha256=digest(s), sample_json_sha256=sha256(folder/'sample.json')))
                completed.append(s['question_id'])
                print(stage, len(completed), '/', len(selected), s['question_id'], flush=True)
            after = weights_hash(model)
            if after != before or library_hash() != c['library'] or hooks_state(model) != [(name, 0, 0) for name, _ in model.named_modules()]:
                raise AssertionError('Model/library/Hook contamination')
            atomic_json(out/'validation.json', dict(passed=True, binding=binding, execution=execution,
                code_commit=c['code_commit'], count=len(completed), question_ids=completed,
                full_state_dict_exact=True, weights_before_sha256=before['sha256'], weights_after_sha256=after['sha256'],
                wall_seconds=time.perf_counter()-start, gpu_uuid=GPU_UUID,
                samples={q: sha256(out/f'sample_{q}'/'sample.json') for q in completed}))
            print('COMPLETE', stage, len(completed), flush=True)
        except BaseException as error:
            path = out/f'failure_{time.time_ns()}.json'
            atomic_json(path, dict(status='partial', error=repr(error), traceback=traceback.format_exc(), completed=completed))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'cpu', 'smoke', 'run'])
    stage = parser.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage == 'cpu': cpu()
    else: execute(stage)
