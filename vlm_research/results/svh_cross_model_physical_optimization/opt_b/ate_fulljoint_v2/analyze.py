"""Official rescoring and paired image-cluster statistics; no new GPU evaluation."""
import collections
import datetime
import sys
import subprocess

import numpy as np

from .common import OUT, ROOT, atomic, digest, read, sha, write_csv, verify, score, hardware
from .engine import CONDITIONS
from .run import load

LABELS = ['Native Full', 'Fast-P2', '+ Conditional-15', '+ Answer-only', 'FullJoint V2']


def bootstrap(values, images, seed):
    _, inverse = np.unique(images, return_inverse=True)
    n = inverse.max() + 1
    totals = np.zeros((n, values.shape[1]))
    np.add.at(totals, inverse, values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    out = np.empty((20000, values.shape[1]))
    for start in range(0, 20000, 500):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=500)
        out[start:start + 500] = (weights @ totals) / (weights @ counts)[:, None]
    return out


def main():
    p = verify()
    samples, specs = read(OUT / 'sample_manifest.json'), read(OUT / 'input_specs.json')
    assert len(samples) == 60 and len({s['image_sha256'] for s in samples.values()}) == 60
    records = {q: load(OUT / 'checkpoints' / (q + '.json'), p, samples[q], specs[q]) for q in sorted(samples)}
    for record in records.values():
        cs = record['conditions']
        prefill = {c: sum(r['estimated_flops'] for r in cs[c]['flops'] if r['phase'] == 'prefill') for c in CONDITIONS}
        assert prefill['C1'] == prefill['C2'] and prefill['C3-V2'] == prefill['C4-V2']
        assert cs['C3-V2']['quality']['kv']['storage_bytes'] == cs['C4-V2']['quality']['kv']['storage_bytes']
        mlp = {c: sum(r['estimated_flops'] for r in cs[c]['flops']
                      if r['phase'] == 'prefill' and r['component'] == 'mlp') for c in ('C1', 'C3-V2', 'C4-V2')}
        assert mlp['C1'] == mlp['C3-V2'] == mlp['C4-V2']
        n = record['layout']['length']
        assert prefill['C1'] - prefill['C3-V2'] == 4 * (4 * 4096**2 + 4 * n * 4096)
    worker = read(OUT / 'worker_completion.json')
    smoke = read(OUT / 'gpu_smoke.json')
    assert worker['status'] == 'COMPLETE' and smoke['status'] == 'PASS'
    ids = sorted(records)
    values = np.array([[records[q]['conditions'][c]['quality']['score'] for c in CONDITIONS] for q in ids])
    draws = bootstrap(values * 100, [samples[q]['image_sha256'] for q in ids], p['bootstrap_seed'])
    per, accuracy, paired, latency, flops, raw_perf, census, paired_cases = [], [], [], [], [], [], [], []
    for j, c in enumerate(CONDITIONS):
        gens = [records[q]['conditions'][c]['quality'] for q in ids]
        delta = draws[:, j] - draws[:, 0]
        accuracy.append(dict(condition=c, n=60, score_pct=100 * values[:, j].mean(),
            delta_native_pp=100 * (values[:, j] - values[:, 0]).mean(),
            paired95_low=np.quantile(delta, .025), paired95_high=np.quantile(delta, .975),
            eos=sum(g['eos'] for g in gens), truncated=sum(g['truncated'] for g in gens),
            empty=sum(g['empty'] for g in gens), repeated=sum(g['repeated'] for g in gens),
            mean_content_tokens=np.mean([g['content_tokens'] for g in gens])))
        for q, g in zip(ids, gens):
            s, spec = samples[q], specs[q]
            assert score(g['text'], s['answers']) == g['score']
            b = records[q]['conditions']['C0']['quality']
            per.append(dict(question_id=q, image_id=s['image_id'], image_sha256=s['image_sha256'],
                question=s['question'], official_answers=s['answers'], answer_type=s['answer_type'],
                condition=c, answer=g['text'], token_ids=g['token_ids'], score=g['score'],
                native_answer=b['text'], delta_native=g['score'] - b['score'],
                first_token_flip=g['token_ids'][0] != b['token_ids'][0],
                token_sequence_changed=g['token_ids'] != b['token_ids'],
                native_first_token_logp_loss=b['native_first_token_logp'] - g['native_first_token_logp'],
                eos=g['eos'], truncated=g['truncated'], empty=g['empty'], repeated=g['repeated'],
                question_tokens=len(spec['Q']), writer_tokens=len(spec['writers']), R_tokens=len(spec['R']),
                triggered=spec['triggered'], input_hashes=spec['input_hashes']))
            for row in records[q]['conditions'][c]['flops']:
                census.append(dict(question_id=q, condition=c, **row))
    for c, b in (('C4-V2', 'C1'), ('C4-V2', 'C3-V2'), ('C4-V2', 'C0'), ('C2', 'C1'),
                 ('C3-V2', 'C1'), ('C3-V2', 'C0')):
        j, k = CONDITIONS.index(c), CONDITIONS.index(b)
        d = draws[:, j] - draws[:, k]
        gs = [records[q]['conditions'][c]['quality'] for q in ids]
        bs = [records[q]['conditions'][b]['quality'] for q in ids]
        for q, g, baseline in zip(ids, gs, bs):
            paired_cases.append(dict(question_id=q, image_sha256=samples[q]['image_sha256'],
                candidate=c, baseline=b, candidate_answer=g['text'], baseline_answer=baseline['text'],
                candidate_score=g['score'], baseline_score=baseline['score'],
                score_delta=g['score']-baseline['score'], score_down=g['score']<baseline['score'],
                score_up=g['score']>baseline['score'],
                first_token_flip=g['token_ids'][0]!=baseline['token_ids'][0],
                token_sequence_changed=g['token_ids']!=baseline['token_ids']))
        paired.append(dict(candidate=c, baseline=b, n=60, delta_pp=100 * (values[:, j] - values[:, k]).mean(),
            paired95_low=np.quantile(d, .025), paired95_high=np.quantile(d, .975),
            score_down=int((values[:, j] < values[:, k]).sum()), score_up=int((values[:, j] > values[:, k]).sum()),
            fullcredit_lost=int(((values[:, k] == 1) & (values[:, j] < 1)).sum()),
            fullcredit_gained=int(((values[:, k] < 1) & (values[:, j] == 1)).sum()),
            one_to_zero=int(((values[:, k] == 1) & (values[:, j] == 0)).sum()),
            zero_to_one=int(((values[:, k] == 0) & (values[:, j] == 1)).sum()),
            token_sequence_changes=sum(g['token_ids'] != a['token_ids'] for g, a in zip(gs, bs)),
            first_token_flips=sum(g['token_ids'][0] != a['token_ids'][0] for g, a in zip(gs, bs))))
    history = ROOT / 'vlm_research/results/ate_fulljoint_vqav2_quick'
    assert sha(history / 'sample_manifest.json') == sha(OUT / 'sample_manifest.json')
    assert sha(history / 'input_specs.json') == sha(OUT / 'input_specs.json')
    hp = read(history / 'protocol.json')
    assert hp['model'] == p['model'] and hp['weights'] == p['weights']
    assert hp['device'] == p['device'] and hp['performance_ids'] == p['performance_ids']
    old_records = {q: load(history / 'checkpoints' / (q+'.json'), hp, samples[q], specs[q]) for q in ids}
    history_rows = []
    for q in ids:
        for c in ('C0', 'C1', 'C2'):
            old, new = old_records[q]['conditions'][c]['quality'], records[q]['conditions'][c]['quality']
            history_rows.append(dict(question_id=q, condition=c,
                tokens_equal=old['token_ids'] == new['token_ids'],
                first_logits_sha256_equal=old['first_logits_sha256'] == new['first_logits_sha256'],
                official_score_equal=old['score'] == new['score']))
    write_csv(OUT / 'history_binding_audit.csv', history_rows)
    legacy = [old_records[q]['conditions']['C3']['quality'] for q in ids]
    actual = [records[q]['conditions']['C3-V2']['quality'] for q in ids]
    legacy_delta = np.array([a['score']-b['score'] for a,b in zip(actual, legacy)])*100
    for q, g, b in zip(ids, actual, legacy):
        paired_cases.append(dict(question_id=q, image_sha256=samples[q]['image_sha256'],
            candidate='C3-V2', baseline='Legacy-C3', candidate_answer=g['text'], baseline_answer=b['text'],
            candidate_score=g['score'], baseline_score=b['score'], score_delta=g['score']-b['score'],
            score_down=g['score']<b['score'], score_up=g['score']>b['score'],
            first_token_flip=g['token_ids'][0]!=b['token_ids'][0],
            token_sequence_changed=g['token_ids']!=b['token_ids']))
    d = bootstrap(legacy_delta[:, None], [samples[q]['image_sha256'] for q in ids], p['bootstrap_seed'])[:, 0]
    paired.append(dict(candidate='C3-V2', baseline='Legacy-C3', n=60,
        delta_pp=legacy_delta.mean(), paired95_low=np.quantile(d,.025), paired95_high=np.quantile(d,.975),
        score_down=int((legacy_delta<0).sum()), score_up=int((legacy_delta>0).sum()),
        fullcredit_lost=sum(b['score']==1 and a['score']<1 for a,b in zip(actual,legacy)),
        fullcredit_gained=sum(b['score']<1 and a['score']==1 for a,b in zip(actual,legacy)),
        one_to_zero=sum(b['score']==1 and a['score']==0 for a,b in zip(actual,legacy)),
        zero_to_one=sum(b['score']==0 and a['score']==1 for a,b in zip(actual,legacy)),
        token_sequence_changes=sum(a['token_ids']!=b['token_ids'] for a,b in zip(actual,legacy)),
        first_token_flips=sum(a['token_ids'][0]!=b['token_ids'][0] for a,b in zip(actual,legacy)),
        comparison_scope='Historical quality only; old A2 not rerun or timed in this matrix'))
    by_input = {}
    for q in p['performance_ids']:
        for c in CONDITIONS:
            entry = records[q]['conditions'][c]
            ms = entry['measurements']
            assert len(ms) == 5 and all(m['token_ids'] == entry['quality']['token_ids'] for m in ms)
            r = dict(question_id=q, condition=c)
            for field in ('latency_ms', 'vision_ms', 'language_prefill_ms', 'ttft_ms', 'decode_ms',
                          'peak_allocated', 'peak_reserved'):
                r[field] = float(np.median([m[field] for m in ms]))
            steps = entry['quality']['decode_steps']
            r['decode_ms_per_token'] = r['decode_ms'] / steps if steps else None
            phase = collections.Counter()
            for f in entry['flops']:
                phase[f['phase']] += f['estimated_flops']
            for name in ('vision', 'prefill', 'decode'):
                r[name + '_matrix_flops'] = phase[name]
            r['estimated_prefill_tflops'] = phase['prefill'] / r['language_prefill_ms'] / 1e9
            r['estimated_decode_tflops'] = phase['decode'] / r['decode_ms'] / 1e9 if r['decode_ms'] else 0.
            r['decode_matrix_flops_per_token'] = phase['decode'] / steps if steps else 0.
            r['live_kv_MiB'] = entry['quality']['kv']['live_bytes'] / 2**20
            r['capacity_kv_MiB'] = entry['quality']['kv']['storage_bytes'] / 2**20
            r['peak_allocated_MiB'], r['peak_reserved_MiB'] = r['peak_allocated'] / 2**20, r['peak_reserved'] / 2**20
            by_input[q, c] = r
            raw_perf.append(r)
    for c in CONDITIONS:
        rs = [by_input[q, c] for q in p['performance_ids']]
        row = dict(condition=c, n=20, warmups=3, repetitions=5)
        for field in ('latency_ms', 'vision_ms', 'language_prefill_ms', 'ttft_ms', 'decode_ms_per_token'):
            data = [r[field] for r in rs if r[field] is not None]
            row[field] = float(np.median(data)) if data else None
            row[field + '_p10'], row[field + '_p90'] = [float(v) for v in np.quantile(data, [.1, .9])] if data else [None, None]
        for b in ('C0', 'C1', 'C3-V2'):
            ratios = [by_input[q, b]['latency_ms'] / by_input[q, c]['latency_ms'] for q in p['performance_ids']]
            row['E2E_speedup_vs_' + b] = float(np.median(ratios))
            row['E2E_reduction_pct_vs_' + b] = float(np.median([100 * (1 - 1 / v) for v in ratios]))
            same = [q for q in p['performance_ids'] if records[q]['conditions'][c]['quality']['token_ids'] ==
                    records[q]['conditions'][b]['quality']['token_ids']]
            row['same_sequence_n_vs_' + b] = len(same)
            row['same_sequence_speedup_vs_' + b] = float(np.median([
                by_input[q, b]['latency_ms'] / by_input[q, c]['latency_ms'] for q in same])) if same else None
            for field in ('language_prefill_ms', 'decode_ms_per_token'):
                ratios = [by_input[q,b][field]/by_input[q,c][field] for q in p['performance_ids']
                          if by_input[q,b][field] and by_input[q,c][field]]
                row[field+'_speedup_vs_'+b] = float(np.median(ratios)) if ratios else None
        latency.append(row)
        fr = dict(condition=c, n=20)
        for field in ('vision_matrix_flops', 'prefill_matrix_flops', 'decode_matrix_flops',
                      'estimated_prefill_tflops', 'estimated_decode_tflops', 'decode_matrix_flops_per_token',
                      'live_kv_MiB', 'capacity_kv_MiB', 'peak_allocated_MiB', 'peak_reserved_MiB'):
            fr[field] = float(np.median([r[field] for r in rs]))
        fr['prefill_flops_reduction_vs_C1_pct'] = float(np.mean([
            100 * (1 - by_input[q, c]['prefill_matrix_flops'] / by_input[q, 'C1']['prefill_matrix_flops'])
            for q in p['performance_ids']]))
        flops.append(fr)
    for name, rows in (('accuracy_summary.csv', accuracy), ('paired_comparisons.csv', paired),
                       ('per_question.csv', per), ('latency_summary.csv', latency),
                       ('flops_kv_summary.csv', flops), ('performance_per_question.csv', raw_perf),
                       ('shape_census.csv', census), ('dense_reference_audit.csv', smoke['numerical']),
                       ('visual_prefill_preservation_audit.csv', smoke['preservation'])):
        write_csv(OUT / name, rows)
    write_csv(OUT / 'flops_summary.csv', flops)
    write_csv(OUT / 'kv_memory.csv', [{k:v for k,v in r.items()
        if k in ('condition', 'n') or 'kv_' in k or 'peak_' in k} for r in flops])
    write_csv(OUT / 'negative_cases.csv', [r for r in per if r['delta_native'] < 0 or r['empty'] or r['truncated']])
    write_csv(OUT / 'paired_case_changes.csv', paired_cases)
    plot(accuracy, latency, draws)
    from .report import report
    report(p, accuracy, paired, latency, flops, worker, history_rows, smoke)
    assert not (OUT / 'STOP.json').exists()
    actual_weights = {path: sha(path) for path in p['weights']}
    assert actual_weights == p['weights']
    state = hardware()
    assert p['device']['uuid'] not in state['processes'], 'Worker GPU context still alive'
    atomic(OUT / 'resource_audit.json', dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected_device=p['device'], selected_device_has_no_compute_process=True, hardware=state,
        no_external_process_terminated=True))
    atomic(OUT / 'validation.json', dict(status='COMPLETE', quality_records=300, unique_images=60,
        timing_inputs=20, timed_requests=500, warmup_requests=300, bootstrap=20000,
        cpu_tests_pass=9, gpu_smoke_pass=3, protocol_sha256=digest(p),
        sources_match_freeze=True, model_weights_postrun_unchanged=True, scorer_recomputed=True,
        device_released=True, no_fixed16_or64=True, no_new_data_download=True,
        mask_gate='Actual forbidden Softmax probabilities zero; conditional mask used by answer-only path',
        visual_prefill_attention_preserved=smoke['visual_prefill_attention_preserved'],
        preformal_failures=[read(OUT / 'failed_smoke_1/STOP.json')],
        runtime_failures=[], source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        conditional_writer_dense_FLOPs_unchanged=True, C4_C3_KV_capacity_equal=True,
        all_prefill_MLP_flops_preserved=True, historical_tolerances=p['numerical_tolerances']))
    hash_artifacts()
    print('COMPLETE', accuracy, paired, flush=True)


def plot(accuracy, latency, draws):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    colors = ['#666666', '#277da8', '#9e6f20', '#70a37f', '#218c53']
    x = np.arange(5)
    values = np.array([r['score_pct'] for r in accuracy])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    axes[0].bar(x, values, color=colors)
    axes[0].errorbar(x, values, yerr=[values-low, high-values], fmt='none', capsize=3, color='black', lw=.8)
    axes[0].set(ylabel='Official VQAv2 soft accuracy (%)', title='60-image exploratory pilot')
    axes[1].bar(x, [r['E2E_speedup_vs_C0'] for r in latency], color=colors)
    axes[1].axhline(1, color='#555555', lw=.8, ls='--')
    axes[1].set(ylabel='Natural EOS E2E speedup vs Native', title='20 paired inputs | 3 warmup + 5 timed')
    for ax in axes:
        ax.set_xticks(x, [f'{c}\n{label}' for c, label in zip(CONDITIONS, LABELS)], rotation=25, ha='right')
        ax.grid(axis='y', alpha=.15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('accuracy_speedup.' + extension), dpi=240)
    plt.close(fig)


def hash_artifacts():
    hashes = {str(path.relative_to(OUT)): sha(path) for path in sorted(OUT.rglob('*'))
              if path.is_file() and path.name != 'artifact_hashes.json'}
    atomic(OUT / 'artifact_hashes.json', hashes)


if __name__ == '__main__':
    if '--hash-only' in sys.argv:
        hash_artifacts()
    else:
        main()
