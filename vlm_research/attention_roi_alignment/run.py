"""Only Native Full forwards; never rerun sliding-window interventions."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from attention_position_probe import locate_question_tokens
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.protocols import MODEL_PATH
from benchmarks.smoke import hook_footprint, preflight
from p1_sliding_window.run import plans
from residual_only_pilot.run import actual_inputs
from roi_sink_common import AttentionRows, NEUTRAL, PROTOCOL, sink_candidates
from vlm_common import load_model, prepare_inputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/attention_roi_alignment_gpu0'
PARENT = ROOT/'results/p1_sliding_window_gpu0'
PILOT = ROOT/'results/l2_l4_residual_only_pilot/manifest.json'
ROI = ROOT/'results/phase3_4_roi_sink_diagnostic'
OLD_RAW = Path('/data/lk/ATE-worktrees/p1-sliding-window-gpu0/vlm_research/results/p1_sliding_window_gpu0')
UNITS = ('gqa_chooserel', 'scienceqa_img', 'textvqa')


def sources():
    paths = list((ROOT/'attention_roi_alignment').glob('*.py'))
    paths += [ROOT/p for p in ('roi_sink_common.py', 'prepare_roi_sink.py', 'vlm_common.py',
              'attention_position_probe.py', 'p1_sliding_window/run.py', 'residual_only_pilot/run.py',
              'physical_exit_pilot.py', 'exit_boundary_analysis.py', 'test_attention_roi_alignment.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def freeze():
    samples = json.loads(PILOT.read_text())['samples']
    roi = {s['question_id']: s for s in json.loads((ROI/'manifest.json').read_text())['samples']}
    sinks = {s['question_id']: s for s in json.loads((ROI/'sink_manifest.json').read_text())['samples']}
    selected = [s['question_id'] for s in samples['gqa_chooserel']]
    if any(roi[s['question_id']]['image_sha256'] != s['image_sha256'] for s in samples['gqa_chooserel']):
        raise ValueError('ROI geometry belongs to a different image')
    primary = [q for q in selected if not roi[q]['geometry_exclusion_reasons']]
    if len(primary) != 16 or sum(map(len, samples.values())) != 64:
        raise ValueError('Unexpected frozen cohort; do not reselect')
    files = [PILOT, PARENT/'manifest.json', PARENT/'window_summary.csv', PARENT/'per_question.csv',
             ROI/'manifest.json', ROI/'sink_manifest.json', ROI/'preregistered_protocol.json']
    config = dict(parent_commit='2360834fae22cdb28dd66758893948066e84e69b',
                  input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in files}, source_sha256=sources(),
                  samples=samples, primary_ids=primary,
                  roi={q: {k: roi[q][k] for k in ('image_sha256', 'geometry', 'objects', 'roi_indices',
                       'geometry_exclusion_reasons')} for q in selected},
                  frozen_sinks={q: sinks[q] for q in selected if q in sinks},
                  missing_sink_ids=[q for q in selected if q not in sinks],
                  sink_rule=PROTOCOL['sink_rule'], neutral_prompts=NEUTRAL,
                  statistics='Match Phase3.4: average real Question token rows per head first; compute '
                  'ROI/visual ratios per head, then equal head and image means. Enrichment divides '
                  'conditional ROI share by positive-intersection ROI patch count /576. BOS mass is '
                  'head mean at the true BOS token. Sink mass includes ROI overlap, non-ROI also saved.',
                  input_layout='Same first candidate N+1 layout as frozen sliding-window Full for closed '
                  'tasks; original N-token prompt for TextVQA. Original P-1 and actual Question positions.',
                  scope='64 Native Full task forwards; neutral Full only for26 uncovered ChooseRel. '
                  'No intervention forwards. ScienceQA/TextVQA BOS/Image only. No decoding or training.',
                  window_alignment='Each start s=0..29 is mean of Native Full layers s,s+1,s+2; '
                  'not a single-layer observation or attention under A-Skip.')
    freeze_json(OUTPUT/'manifest.json', config)
    return config


def verify():
    config = json.loads((OUTPUT/'manifest.json').read_text())
    if config['source_sha256'] != sources():
        raise ValueError('Frozen source changed')
    if any(sha256(ROOT/p) != h for p, h in config['input_sha256'].items()):
        raise ValueError('Historical input changed')
    return config


def regional(rows, roi, sinks):
    """rows: layers x heads x (Question mean, P-1) x576, historical FP16 storage."""
    a = np.asarray(rows, dtype=np.float64)
    total = a.sum(-1)
    if not np.isfinite(a).all() or (a < 0).any() or (total <= 0).any():
        raise ValueError('Invalid visual attention')
    result = {}
    for group, g in [('q', 0), ('p', 1)]:
        result[group+'_image'] = total[:, :, g].mean(1)
        for name, indices in [('roi', roi), ('sink', sinks), ('sink_nonroi', sorted(set(sinks)-set(roi)))]:
            mass = a[:, :, g, indices].sum(-1)
            share = mass/total[:, :, g]
            result[group+'_'+name+'_mass'] = mass.mean(1)
            result[group+'_'+name+'_share'] = share.mean(1)
            if name == 'roi':
                result[group+'_roi_enrichment'] = share.mean(1)/(len(roi)/576) if roi else np.full(len(a), np.nan)
    return result


class NativeRows(AttentionRows):
    """Extend validated streaming observer only to retain two BOS scalars per layer."""
    def __init__(self, model, inputs, qpos, apos, bos_id):
        super().__init__(model, inputs, qpos, apos)
        locations = torch.where(inputs['input_ids'][0] == bos_id)[0].tolist()
        if locations != [0]:
            raise ValueError('Unexpected real BOS layout')
        self.bos = []

    def capture(self, index):
        parent = super().capture(index)

        def hook(module, args, output):
            a = output[1][0]
            self.bos.append([float(a[:, self.qpos, 0].float().mean()),
                             float(a[:, self.apos, 0].float().mean())])
            return parent(module, args, output)
        return hook


def observed(model, processor, inputs, qpos, apos):
    before = hook_footprint(model)
    with torch.inference_mode(), NativeRows(model, inputs, qpos, apos, processor.tokenizer.bos_token_id) as stream:
        output = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if hook_footprint(model) != before or output.past_key_values is not None:
        raise AssertionError('Hook leak or cache')
    logits = output.logits[0, apos].cpu().numpy()
    if len(stream.rows) != 32 or not np.isfinite(logits).all():
        raise AssertionError('Missing layers/nonfinite logits')
    return np.stack(stream.rows), np.array(stream.bos), logits, stream.audit


def safe_number(x):
    return float(x) if np.isfinite(x) else None


def run(config):
    OUTPUT.mkdir(exist_ok=True)
    lock = (OUTPUT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight()
    started = time.perf_counter()
    model, processor, environment = load_model(MODEL_PATH)
    freeze_json(OUTPUT/'environment.json', dict(environment, preflight=pre,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=config['source_sha256']))
    versions = {n: p._version for n, p in model.named_parameters()}
    binding = digest(config)
    completed, seconds, smoke_checks = [], [], []
    for unit in UNITS:
        for sample in config['samples'][unit]:
            qid = sample['question_id']
            path = OUTPUT/'checkpoints'/unit/f'{qid}.json'
            if path.exists():
                old = json.loads(path.read_text())
                if old['binding'] != binding or old['image_sha256'] != sample['image_sha256']:
                    raise ValueError('Checkpoint binding mismatch')
                completed.append((unit, qid))
                continue
            begin = time.perf_counter()
            if sha256(sample['image_path']) != sample['image_sha256']:
                raise ValueError('Image changed')
            prompt = actual_inputs(model, processor, sample)
            inputs, _, _ = plans(processor, prompt, sample)
            mapping = sample['metadata']['input_mapping']
            qpos, apos = mapping['question_positions'], mapping['p_minus_1']
            rows, bos, logits, audit = observed(model, processor, inputs, qpos, apos)
            # Same GPU and exact original input layout, not a fresh accuracy baseline.
            with np.load(OLD_RAW/unit/qid/'Full.npz') as old:
                exact = np.array_equal(logits, old['logits'])
                max_error = float(np.max(np.abs(logits.astype(float)-old['logits'].astype(float))))
            if not exact:
                raise AssertionError(f'Frozen same-GPU Full logits differ: {unit}/{qid}, max={max_error}')
            if not completed:
                with torch.inference_mode():
                    native = model(**inputs, use_cache=False).logits[0, apos].cpu().numpy()
                if not np.array_equal(native, logits):
                    raise AssertionError('Attention observer changed logits')
                smoke_checks.append(dict(question_id=qid, observer_native_logits_exact=True,
                                         historical_full_logits_exact=exact))
                freeze_json(OUTPUT/'smoke.json', dict(passed=True, checks=smoke_checks))
            roi, sinks, sink_record = [], [], None
            if unit == 'gqa_chooserel':
                roi = config['roi'][qid]['roi_indices']
                if qid in config['frozen_sinks']:
                    sinks = config['frozen_sinks'][qid]['sink_indices']
                    sink_record = dict(source='Phase3.4 frozen', indices=sinks)
                else:
                    maps = []
                    for text in NEUTRAL:
                        neutral, _ = prepare_inputs(model, processor, sample['image_path'], text)
                        records, _ = locate_question_tokens(processor, text, neutral['input_ids'][0].cpu().numpy(),
                                                            model.config.image_token_index)
                        nr, _, _, _ = observed(model, processor, neutral, [r['sequence_index'] for r in records],
                                               neutral['input_ids'].shape[1]-1)
                        maps.append(nr.astype(np.float64).mean(1).astype(np.float16))
                    maps = np.stack(maps)
                    identified = sink_candidates(maps)
                    sinks = identified['indices']
                    sink_record = dict(source='Phase3.4 neutral rule, newly identified', **identified)
                    neutral_path = OUTPUT/'neutral_maps'/f'{qid}.npz'
                    neutral_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(neutral_path, attention=maps)
                    sink_record['neutral_sha256'] = sha256(neutral_path)
            metrics = regional(rows, roi, sinks)
            records = []
            for layer in range(32):
                record = dict(dataset=unit, question_id=qid, layer=layer, primary=qid in config['primary_ids'],
                              p_bos=float(bos[layer, 1]), q_bos=float(bos[layer, 0]),
                              p_image=float(metrics['p_image'][layer]), q_image=float(metrics['q_image'][layer]))
                if unit == 'gqa_chooserel':
                    record.update({k: safe_number(v[layer]) for k, v in metrics.items()})
                    record.update(roi_patches=len(roi), sink_patches=len(sinks),
                                  roi_area_fraction=len(roi)/576, roi_eligible=qid in config['primary_ids'])
                records.append(record)
            elapsed = time.perf_counter()-begin
            freeze_json(path, dict(binding=binding, image_sha256=sample['image_sha256'],
                records=records, sink=sink_record, historical_full_logits_exact=exact,
                historical_max_logit_error=max_error, row_sum_max_error=max(a['row_sum_error'] for a in audit),
                input_mapping=mapping, seconds=elapsed))
            completed.append((unit, qid))
            seconds.append(elapsed)
            print(f'Full {len(completed)}/64 {unit}/{qid}, {elapsed:.2f}s; sink={len(sinks)}', flush=True)
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise AssertionError('Parameter version changed')
    freeze_json(OUTPUT/'completion.json', dict(status='complete', questions=len(completed),
        native_task_forwards=64, native_neutral_forwards=3*len(config['missing_sink_ids']),
        interventions_rerun=0, all_historical_full_logits_exact=True, weights_versions_unchanged=True,
        wall_seconds=time.perf_counter()-started, current_attempt_sample_seconds=sum(seconds)))
    print('COMPLETE', len(completed), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    else:
        run(verify())


if __name__ == '__main__':
    main()
