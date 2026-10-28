"""A single complete-cohort summary; --verify-existing is read-only."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from knockout_probe import tensor_hash, ROW_ATOL
from roi_sink_probe import score
from phase3_7.run import installed_sources, checkpoint as old_checkpoint
from .protocol import (OUTPUT, OLD, ROOT, GROUPS, CONDITIONS, RULES, atomic,
                       protocol, read, sha)
from .run import checkpoint
from .statistics import summarize_cohorts


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def validate_body(body, sample, root):
    name = next(p for p in body['artifacts'] if p.endswith('audit.json.gz'))
    with gzip.open(root / name, 'rt') as stream:
        audit = json.load(stream)
    for flag in ['noop_effective_text_bitwise_exact', 'post_cleanup_native_bitwise_exact',
                 'question_all32_layers_and_final_bitwise_exact']:
        require(audit[flag], 'Per-sample correctness gate failed')
    mapping = sample['input_mapping']
    qpos = [r['sequence_index'] for r in mapping['question_mapping']]
    apos = mapping['answer_query']
    n = len(mapping['candidate_mapping'][0]['input_ids'])
    row = body['row']
    layer_count = local_count = 0
    for condition in CONDITIONS:
        records = audit['conditions'][condition]
        require(len(records) == 2, 'Candidate audit missing')
        require(records[0]['prediction_logits_sha256'] == records[1]['prediction_logits_sha256'], 'Candidate contamination')
        for index, record in enumerate(records):
            require(all(record[k] for k in ['input_unchanged', 'weights_version_unchanged', 'hooks_removed']), 'Mutation/hook leak')
            layers = record['layer_attention']
            require([x['layer'] for x in layers] == list(range(32)), 'Not all32 layers audited')
            require(len(record['question_layer_sha256']) == 32, 'Question hidden audit incomplete')
            for item in layers:
                require(item['sequence_length'] == n and item['visual_tokens'] == 576 and item['row_sum_error'] <= ROW_ATOL,
                        'Sequence/token retention/normalization failed')
                require(item['future_attention_max'] == 0 and item['padding_attention_max'] == 0, 'Causality/padding violated')
                layer_count += 1
            if condition == 'C1':
                require(record['question_layer_sha256'] == audit['conditions']['C0'][index]['question_layer_sha256'], 'Question states changed')
                require(record['final_question_sha256'] == audit['conditions']['C0'][index]['final_question_sha256'], 'Final Question changed')
                masks = record['mask_records']
                require([x['applied'] for x in masks] == [i < 2 for i in range(32)], 'Wrong active layers')
                for item in masks:
                    require(item['native_mask_unmodified'] and item['visual_tokens_retained'] == 576, 'Native mask/token mutation')
                    if item['layer'] >= 2:
                        require(item['effective_mask_sha256'] == item['native_mask_sha256'], 'Later-layer mask differs')
                for item in record['local_redistribution']:
                    require(item['query_indices'] == [apos] and item['key_indices'] == qpos, 'Wrong dynamic Query->Key mask')
                    require(item['masked_mass_after'] == 0 and item['non_target_queries_bitwise_exact'], 'Blocked/non-target weights incorrect')
                    require(item['max_target_row_mass_error'] <= ROW_ATOL, 'Masked row normalization failed')
                    local_count += 1
        relative = next(p for p in body['artifacts'] if p.endswith('/' + condition + '.npz'))
        with np.load(root / relative, allow_pickle=False) as stored:
            require(all(np.isfinite(stored[key]).all() for key in stored.files), 'Nonfinite raw observation')
            logits = torch.from_numpy(stored['prediction_logits'])
            require(tensor_hash(logits) == records[0]['prediction_logits_sha256'], 'Raw logits hash mismatch')
            logps = [score(dict(prediction_logits=logits), entry['answer_records']) for entry in mapping['candidate_mapping']]
            prefix = 'full' if condition == 'C0' else 'b4'
            require(logps == [row[prefix + '_logp_left'], row[prefix + '_logp_right']], 'Raw full-vocabulary LogP disagrees with checkpoint')
            attention = stored['answer_attention']
            mass = stored['mass']
            require(attention.shape == (2, 32, n) and mass.shape == (2, 32, 4), 'Raw diagnostic axes incorrect')
            require((attention[:, :, apos+1:] == 0).all(), 'Future answer attended by P-1')
            require(np.max(np.abs(attention.astype(float).sum(-1)-1)) <= ROW_ATOL, 'Raw Attention rows invalid')
            require(np.max(np.abs(mass.sum(-1)-1)) <= ROW_ATOL, 'Mass partition invalid')
            if condition == 'C1':
                require((attention[:, :, qpos] == 0).all() and (mass[:, :, 1] == 0).all(), 'Question not actually blocked')
                for layer in (0, 1):
                    before = np.asarray(records[0]['local_redistribution'][layer]['masked_mass_before']).reshape(-1)
                    require(np.array_equal(stored['local_native_mass'][layer, :, 1], before), 'Independent local replay disagrees')
            for layer in (0, 1):
                for j, group in enumerate(GROUPS):
                    require(float(mass[layer, :, j].mean()) == row[f'l{layer}_{prefix}_{group}_mass'], 'CSV mass disagrees with raw array')
    require(len(audit['noop_audits']) == 2, 'No-op incomplete')
    for actual, noop in zip(audit['conditions']['C0'], audit['noop_audits']):
        require(actual['effective_text_logits_sha256'] == noop['effective_text_logits_sha256'], 'No-op logits not exact')
    require(audit['restored_audit']['effective_text_logits_sha256'] == audit['conditions']['C0'][0]['effective_text_logits_sha256'], 'Restored Full not exact')
    gold = sample['answer']
    wrong = 'right' if gold == 'left' else 'left'
    for prefix in ('full', 'b4'):
        require(row[prefix + '_margin'] == row[prefix + '_logp_' + gold] - row[prefix + '_logp_' + wrong], 'Margin arithmetic invalid')
    require(row['gain'] == row['b4_margin']-row['full_margin'], 'Gain arithmetic invalid')
    sign = 1 if gold == 'right' else -1
    require(abs(row['gain']-sign*row['right_minus_left_change']) < 2e-12, 'Candidate bias/margin orientation invalid')
    return layer_count, local_count


def collect(config, binding):
    cpu = read(OUTPUT / 'acceptance/validation.json')
    require(cpu['passed'] and cpu['binding'] == binding, 'CPU acceptance missing')
    audits = {}
    bodies = {}
    for stage, ids in [('smoke', [s['question_id'] for s in config['samples'] if s['question_id'] in config['smoke_ids']]),
                       ('run', [s['question_id'] for s in config['samples']])]:
        root = OUTPUT / stage
        gate = read(root / 'validation.json')
        require(gate['passed'] and gate['status'] == 'completed' and gate['binding'] == binding, 'Partial stage; no complete-cohort claims permitted')
        require(gate['count'] == len(ids) and gate['question_ids'] == ids, 'Stage coverage incorrect')
        require(gate['weights_sha256_before'] == gate['weights_sha256_after'] and gate['third_party_source_unchanged'], 'Model/source mutation')
        require({p.stem for p in (root / 'checkpoints').glob('*.json')} == set(ids), 'Duplicate/foreign/missing checkpoint')
        envs = list(root.glob('environment_*.json'))
        require(bool(envs) and all(read(p)['installed_sources'] == installed_sources() for p in envs), 'Installed Transformers changed')
        layers = local = 0
        stage_bodies = []
        for qid in ids:
            sample = next(s for s in config['samples'] if s['question_id'] == qid)
            path = root / 'checkpoints' / (qid + '.json')
            require(sha(path) == gate['checkpoints_sha256'][qid], 'Checkpoint changed after completion')
            body = checkpoint(path, binding, sample, root)
            require(body['weights_reference_sha256'] == gate['weights_sha256_before'], 'Checkpoint model hash differs across attempts')
            a, b = validate_body(body, sample, root)
            layers += a
            local += b
            stage_bodies.append(body)
        audits[stage] = dict(layer_audits=layers, local_mask_audits=local, count=len(ids))
        bodies[stage] = stage_bodies
    rows = [body['row'] for body in bodies['run']]
    require(len(rows) == 240 and sum(r['cohort'] == 'confirmation224' for r in rows) == 224, 'Main cohort incomplete')
    return rows, audits, bodies['run']


def replication(rows, bodies):
    old_manifest = read(OLD / 'manifest.json')
    original = {r['question_id']: r for r in csv.DictReader((OLD / 'per_sample_b.csv').open()) if r['condition'] == 'B4'}
    output = []
    for row, body in zip(rows, bodies):
        if row['cohort'] != 'discovery16':
            continue
        old_sample = next(s for s in old_manifest['config']['samples'] if s['question_id'] == row['question_id'])
        previous = old_checkpoint(OLD / 'pilot/checkpoints' / (row['question_id'] + '.json'),
                                  old_manifest['binding'], old_sample, OLD / 'pilot')
        record = dict(question_id=row['question_id'], old_gain=float(original[row['question_id']]['gain']), new_gain=row['gain'],
                      gain_change=row['gain']-float(original[row['question_id']]['gain']),
                      old_full_margin=float(original[row['question_id']]['full_margin']), new_full_margin=row['full_margin'],
                      old_b4_margin=float(original[row['question_id']]['margin']), new_b4_margin=row['b4_margin'])
        for condition, old_condition in [('C0', 'B0'), ('C1', 'B4')]:
            prior_path = next(p for p in previous['artifacts'] if p.endswith('/' + old_condition + '.npz'))
            current_path = next(p for p in body['artifacts'] if p.endswith('/' + condition + '.npz'))
            with np.load(OLD / 'pilot' / prior_path, allow_pickle=False) as prior, np.load(OUTPUT / 'run' / current_path, allow_pickle=False) as current:
                a, b = prior['prediction_logits'].astype(float), current['prediction_logits'].astype(float)
                record[condition + '_archived_logits_max_abs_difference'] = float(np.abs(a-b).max())
                record[condition + '_archived_logits_different_count'] = int(np.count_nonzero(a != b))
        output.append(record)
    require(len(output) == 16, 'Discovery replication coverage incorrect')
    return output


def plots(rows, summaries):
    import matplotlib.pyplot as plt
    colors = dict(full='#356da8', b4='#168565', left='#3576a8', right='#b94c72')
    cohorts = ['confirmation224', 'discovery16', 'all240']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for i, cohort in enumerate(cohorts):
        r = summaries[cohort]['all']
        x = np.array([i-.17, i+.17])
        for j, key in enumerate(['full_accuracy', 'b4_accuracy']):
            lo, hi = r['bootstrap_ci95'][key]
            axes[0].bar(x[j], r[key], width=.28, color=colors['full' if j == 0 else 'b4'], label=('Full' if j == 0 else 'B4') if i == 0 else None)
            axes[0].vlines(x[j], lo, hi, color='black')
        lo, hi = r['bootstrap_ci95']['accuracy_difference']
        axes[1].scatter(i, 100*r['accuracy_difference'], color=colors['b4'])
        axes[1].vlines(i, 100*lo, 100*hi, color=colors['b4'])
        axes[2].bar(i-.17, r['harmed'], .28, color='#bd5461', label='Full correct -> wrong' if i == 0 else None)
        axes[2].bar(i+.17, r['recovered'], .28, color=colors['b4'], label='Full wrong -> correct' if i == 0 else None)
    for ax in axes:
        ax.set_xticks(range(3), ['224 primary', '16 discovery', '240 all'])
    axes[0].set(ylabel='Accuracy', ylim=(0, 1), title='Paired Full and B4; image bootstrap CI95')
    axes[0].legend()
    axes[1].axhline(0, color='gray', lw=.8)
    axes[1].set(ylabel='B4 - Full (percentage points)', title='Paired accuracy difference')
    axes[2].set(ylabel='Images', title='Discordant predictions')
    axes[2].legend(fontsize=8)
    fig.savefig(OUTPUT / 'accuracy_and_flips.png', dpi=140)
    plt.close(fig)
    rr = [r for r in rows if r['cohort'] == 'confirmation224']
    gain = np.array([r['gain'] for r in rr])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].hist(gain, bins=30, color=colors['b4'], edgecolor='white')
    axes[0].axvline(0, color='black', lw=1)
    axes[0].set(xlabel='B4 - Full Margin', ylabel='Images', title='All 224 confirmation gains; tails retained')
    for label in ['left', 'right']:
        group = [r for r in rr if r['answer'] == label]
        axes[1].scatter([r['full_margin'] for r in group], [r['b4_margin'] for r in group], s=16, alpha=.7, color=colors[label], label=label)
    limits = [min(min(r['full_margin'], r['b4_margin']) for r in rr), max(max(r['full_margin'], r['b4_margin']) for r in rr)]
    axes[1].plot(limits, limits, '--', color='gray')
    axes[1].axhline(0, color='gray', lw=.6)
    axes[1].axvline(0, color='gray', lw=.6)
    axes[1].set(xlabel='Full Margin', ylabel='B4 Margin', title='Same-image paired final margins')
    axes[1].legend()
    fig.savefig(OUTPUT / 'margin_gain_confirmation224.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    rng = np.random.default_rng(20261029)
    for j, label in enumerate(['left', 'right']):
        values = np.array([r['gain'] for r in rr if r['answer'] == label])
        ax.boxplot([values], positions=[j], widths=.35, showfliers=False)
        ax.scatter(j+rng.uniform(-.16, .16, len(values)), values, s=14, alpha=.6, color=colors[label])
    ax.axhline(0, color='black', lw=.8)
    ax.set(xticks=[0, 1], xticklabels=['Gold left', 'Gold right'], ylabel='B4 - Full Margin', title='Confirmation224: every image, no tail exclusion')
    fig.savefig(OUTPUT / 'left_right_gain.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for label in ['left', 'right']:
        group = [r for r in rr if r['answer'] == label]
        axes[0].hist([r['right_minus_left_change'] for r in group], bins=25, alpha=.55, label=label, color=colors[label])
        axes[1].scatter([r['full_right_minus_left'] for r in group], [r['right_minus_left_change'] for r in group], s=16, alpha=.65, color=colors[label], label=label)
    axes[0].axvline(0, color='black', lw=.8)
    axes[0].set(xlabel='Change in logP(right)-logP(left)', ylabel='Images', title='Confirmation224: direction preference shift')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set(xlabel='Full logP(right)-logP(left)', ylabel='B4 - Full right preference', title='Same-image direction changes')
    axes[0].legend()
    axes[1].legend()
    fig.savefig(OUTPUT / 'right_left_preference.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    group_colors = ['#168565', '#ba536d', '#d6ab39', '#8b9095']
    r = summaries['confirmation224']['all']
    for layer in (0, 1):
        bottom = np.zeros(2)
        for j, group in enumerate(GROUPS):
            values = [r[f'l{layer}_{condition}_{group}_mass'] for condition in ['full', 'b4']]
            axes[layer, 0].bar([0, 1], values, bottom=bottom, color=group_colors[j], label=group)
            bottom += values
        axes[layer, 0].set(xticks=[0, 1], xticklabels=['Native Full', 'B4'], ylim=(0, 1.02), ylabel='P-1 Attention Mass', title=f'L{layer}: actual Full vs B4')
        for col, prefix in [(1, f'l{layer}_'), (2, f'l{layer}_local_')]:
            values = [r[prefix+group+('_mass_change' if col == 1 else '_redistribution')] for group in GROUPS]
            axes[layer, col].bar(range(4), values, color=group_colors)
            axes[layer, col].axhline(0, color='black', lw=.8)
            axes[layer, col].set(xticks=range(4), xticklabels=GROUPS, ylabel='Mass difference',
                                  title=f'L{layer}: ' + ('global B4 - Full' if col == 1 else 'local redistribution; identical layer input'))
        axes[layer, 0].legend(fontsize=8)
    fig.savefig(OUTPUT / 'attention_mass_redistribution.png', dpi=140)
    plt.close(fig)


def report(config, summaries, replicated, audits):
    c = summaries['confirmation224']['all']
    d = summaries['discovery16']['all']
    def ci(result, key, percentage=False):
        scale = 100 if percentage else 1
        lo, hi = result['bootstrap_ci95'][key]
        return f"{scale*result[key]:+.6f} [{scale*lo:+.6f}, {scale*hi:+.6f}]"
    lines = ['# Phase 3.7B: Independent Full vs B4 Validation', '',
             '**Status: completed.** All 240 original images were paired on physical GPU0 in this run; confirmation224 is primary, discovery16 and all240 supplementary.',
             f"Base `{config['baseline']}`; executed frozen code `{config['code_commit']}`. FP16/eager LLaVA-1.5-7B, Transformers4.49.0, original prompt and complete candidate scoring; no EOS. Ties are wrong.", '',
             'Only conditions: C0 Native Full and C1 unmodified Phase3.7 B4 (P-1 Query -> real Question Keys blocked only in L0/L1). No training, new mask, layer/strength scan, decode, KV cache, exit or other dataset.', '',
             '## Frozen Cohorts and Inference', '', RULES['split'],
             'Question IDs remain strings including leading zeroes. Every image SHA256, actual input/Question/P-1/BOS mapping, candidate token and original source is frozen before formal GPU scoring. The upstream correct_question ChooseRel cohort is preselected; these results are not unbiased GQA-population estimates. The 224 images are disjoint from the 16 B4 discovery images, not a newly sampled external dataset.',
             '20,000 image-paired bootstrap resamples, fixed seed 20261029 plus deterministic cohort/subset label hash; percentile CI95. All paired quantities use the same sampled image indices. Exact two-sided McNemar uses only discordant classifications. Confirmation224 accuracy difference/McNemar is primary; subgroup and other metric intervals are descriptive without multiplicity adjustment. No head or layer is an independent sample.', '',
             '## Full vs B4', '',
             '| Cohort | Full correct / N | B4 correct / N | Accuracy difference pp [CI95] | Correct->wrong | Wrong->correct | Exact McNemar p |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for cohort in ['confirmation224', 'discovery16', 'all240']:
        s = summaries[cohort]['all']
        lines.append(f"| {cohort} | {s['full_correct']}/{s['n']} | {s['b4_correct']}/{s['n']} | {ci(s, 'accuracy_difference', True)} | {s['harmed']} | {s['recovered']} | {s['exact_mcnemar_p']:.8g} |")
    lines += ['', '### Primary Confirmation224 Metrics', '',
              '| Metric | Estimate [paired CI95] |', '| --- | --- |']
    for key in ['mean_gain', 'median_gain', 'mean_absolute_gain', 'p5_gain', 'p95_gain',
                'mean_gold_logp_change', 'mean_wrong_logp_change', 'mean_right_minus_left_change']:
        lines.append(f'| {key} | {ci(c, key)} |')
    lines += ['', 'Gold/wrong LogP changes are signed changes of absolute full-vocabulary conditional log probabilities, not absolute values and not a two-choice normalized probability. Positive mean Margin Gain alone is not accuracy improvement.', '',
              '### Left / Right and Full-Correct / Full-Wrong', '',
              '| Primary subgroup | N | Full correct | B4 correct | Harmed | Recovered | Mean Gain [CI95] | Mean right-left shift |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for name in ['left', 'right', 'full_correct', 'full_wrong']:
        s = summaries['confirmation224'][name]
        if s['n']:
            lines.append(f"| {name} | {s['n']} | {s['full_correct']} | {s['b4_correct']} | {s['harmed']} | {s['recovered']} | {ci(s, 'mean_gain')} | {s['mean_right_minus_left_change']:+.6f} |")
    lines += ['', '### Worst Primary Losses (All Retained)', '',
              '| Question ID | Gold | Full Margin | B4 Margin | Gain |', '| --- | --- | ---: | ---: | ---: |']
    for r in c['worst_samples']:
        lines.append(f"| {r['question_id']} | {r['answer']} | {r['full_margin']:+.6f} | {r['b4_margin']:+.6f} | {r['gain']:+.6f} |")
    lines += ['', '## Attention Mass and Redistribution', '',
              'Actual Full/B4 mass is compared within each image. Local redistribution separately holds the actual layer input fixed. In L1, B4 hidden input can already differ because of L0; its local-native replay is not global Native Full. Image, Question, BOS and other valid text keys are disjoint and exhaustive. Every P-1 raw attention row and per-head mass is saved.', '',
              '| Layer | Key group | Full mass | B4 mass | Actual change [CI95] | Local same-input redistribution |',
              '| --- | --- | ---: | ---: | --- | ---: |']
    for layer in (0, 1):
        for group in GROUPS:
            key = f'l{layer}_{group}_mass_change'
            lines.append(f"| L{layer} | {group} | {c[f'l{layer}_full_{group}_mass']:.6f} | {c[f'l{layer}_b4_{group}_mass']:.6f} | {ci(c, key)} | {c[f'l{layer}_local_{group}_redistribution']:+.6f} |")
    old_gain = float(np.mean([r['old_gain'] for r in replicated]))
    raw_diff = max(r[key] for r in replicated for key in ['C0_archived_logits_max_abs_difference', 'C1_archived_logits_max_abs_difference'])
    strongest = [max(['image', 'bos', 'other_text'], key=lambda g: c[f'l{layer}_local_{g}_redistribution']) for layer in (0, 1)]
    if c['accuracy_difference'] > 0 and c['bootstrap_ci95']['accuracy_difference'][0] > 0 and c['exact_mcnemar_p'] < .05:
        empirical = 'The primary accuracy evidence supports an empirical benefit of this fixed early direct block within the frozen cohort, but does not identify delayed Question reading as the unique explanation.'
    elif c['accuracy_difference'] > 0:
        empirical = 'The point estimate is favorable, but the primary accuracy evidence does not establish a reliably positive improvement; the proposed delayed-reading mechanism remains unconfirmed.'
    else:
        empirical = 'The primary result does not support an accuracy-improving delayed-reading claim in these224 images.'
    lines += ['', '## Seven Research Answers', '',
              f"1. **Does B4 improve accuracy on the 224 previously untested images?** Full {c['full_correct']}/224 versus B4 {c['b4_correct']}/224; paired difference {ci(c, 'accuracy_difference', True)} pp, exact McNemar p={c['exact_mcnemar_p']:.8g}. " +
              ('The observed accuracy difference is positive.' if c['accuracy_difference'] > 0 else 'There is no observed positive accuracy difference.') +
              (' The primary CI excludes zero on the positive side.' if c['bootstrap_ci95']['accuracy_difference'][0] > 0 else 'The primary bootstrap interval does not establish a strictly positive effect.') +
              ' Do not replace this endpoint with a favorable mean Margin.',
              f"2. **Are Full errors corrected?** {c['recovered']} previously wrong/tied images become correct. The current-Full-wrong subgroup is reported separately, not removed from the denominator.",
              f"3. **Are Full-correct images harmed?** {c['harmed']} become wrong/tied. Tail losses and the Full-correct subgroup remain visible; correction counts alone are insufficient.",
              f"4. **Does the discovery effect reproduce?** Original16 mean Gain {old_gain:+.6f}; independent rerun {d['mean_gain']:+.6f}, median {d['median_gain']:+.6f}, accuracy {d['full_correct']}/16 -> {d['b4_correct']}/16. " +
              ('The original positive signed mean reproduces descriptively.' if d['mean_gain'] > 0 else 'The original positive signed mean does not reproduce.') +
              f" Maximum absolute archived-versus-current raw prediction-logit difference across all16 Full/B4 is {raw_diff:.8g}. These cross-run FP16 differences are recorded, not suppressed or used to relax exact same-run controls.",
              f"5. **Is this mainly a right-candidate preference shift?** Mean delta(logP(right)-logP(left)) is {ci(c, 'mean_right_minus_left_change')}; gold-left and gold-right mean Gains are {summaries['confirmation224']['left']['mean_gain']:+.6f}/{summaries['confirmation224']['right']['mean_gain']:+.6f}. " +
              'Inspect subgroup flips and the preference plot together. A positive rightward shift favors right-labelled samples and disfavors left-labelled ones, so pooled improvement may reflect directional bias rather than more reliable semantic grounding. These observations do not causally prove that all benefit is explained by bias.',
              f"6. **Does B4 increase actual P-1 Image Attention?** Actual paired Image Mass changes: L0 {ci(c, 'l0_image_mass_change')}; L1 {ci(c, 'l1_image_mass_change')}. " +
              f'The largest positive local redistribution target is {strongest[0]} in L0 and {strongest[1]} in L1. Compare these BOS/other-text increments with Image rather than assuming a visual destination. A mechanical probability redistribution is not proof of better visual information use.',
              '7. **Does this support delayed Question->Answer reading?** ' + empirical + ' This intervention tests direct P-1->Question access only in the first two layers. Later layers can still read Question, and other template-text paths remain available. Compensation, lexical/logit shifts and alternative paths are possible. This experiment does not establish that text is irrelevant, that all early interaction is unnecessary, or that delayed reading is a generally better mechanism.', '',
              '## Correctness and Reproducibility', '',
              f"CPU acceptance and two-image GPU smoke precede formal scoring. Smoke requires new diagnostic runner logits/observations to be exactly equal to the original Phase3.7 B4 computation in the same run. Formal audits: {audits['run']['layer_audits']} layer/candidate records and {audits['run']['local_mask_audits']} local masked records.",
              'Per image: dynamic mappings; target weights zero; all32 Question decoder states and final RMSNorm exact Full; no-op valid-text logits exact Full; restored Full exact; original causal/non-target local connections and L2-L31 masks preserved; 576 visual tokens retained. Input/parameter versions and hook footprint checked every forward, full parameter+buffer byte hashes before/after stage, installed Transformers source hashes unchanged. No accepted NaN/Inf/OOM/partial sample. Original ROW_ATOL=.002 unchanged.',
              'Historical RTX4090 Full is used only for a numerical-difference audit, never as the paired control. Discovery16 archived logits and scores are compared in discovery_replication.csv. Resume accepts only hash-bound complete checkpoints, with no duplicate count or silent override. Failed attempts, if any, remain partial and must be disclosed, not counted as additional images.', '',
              '## Artifacts', '',
              'manifest.json; per_sample_results.csv (240 wide paired rows); per_condition_results.csv (480 rows); summary.json; statistics_ci95.csv; discovery_replication.csv; acceptance/; smoke/; run/; summary_validation.json. Raw NPZ contains prediction vocabulary logits, P-1 attention rows, per-head four-way mass, local native mass and P-1 hidden states; compressed audits preserve both candidates and all-layer Question hashes.',
              'Executed source/configuration, actual package/Python/GPU environment and immutable original-file hashes are recorded. Reproduce on the frozen code commit in a fresh output directory; --verify-existing permits a result-only descendant commit without rerunning the experiment.', '',
              '![Accuracy and flips](accuracy_and_flips.png)', '![Primary margin gains](margin_gain_confirmation224.png)',
              '![Left and right gains](left_right_gain.png)', '![Directional preference](right_left_preference.png)',
              '![Four-way mass](attention_mass_redistribution.png)', '',
              '**Research stop:** Full/B4 only; GPU resources released after execution. No additional intervention or automatic merge is authorized.']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n')


def inventory():
    return {str(p.relative_to(OUTPUT)): sha(p) for p in sorted(OUTPUT.rglob('*'))
            if p.is_file() and p.name not in ['summary_validation.json', '.lock'] and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    config, binding = protocol(execution=not args.verify_existing)
    rows, audits, bodies = collect(config, binding)
    if args.verify_existing:
        gate = read(OUTPUT / 'summary_validation.json')
        require(gate['passed'] and gate['binding'] == binding and gate['artifacts'] == inventory(), 'Publication artifact gate failed')
        with (OUTPUT / 'per_sample_results.csv').open(newline='') as stream:
            actual = list(csv.DictReader(stream))
        require(actual == [{k: str(v) for k, v in row.items()} for row in rows], 'CSV differs from checkpoints')
        print('VERIFIED_EXISTING', 'Full/B4=240', 'confirmation=224', 'discovery=16', flush=True)
        return
    if (OUTPUT / 'summary.json').exists():
        raise FileExistsError('Scientific aggregation is unique; use --verify-existing')
    summaries = summarize_cohorts(rows)
    replicated = replication(rows, bodies)
    write_csv(OUTPUT / 'per_sample_results.csv', rows)
    narrow = []
    for row in rows:
        for condition, prefix in [('C0', 'full'), ('C1', 'b4')]:
            narrow.append(dict(question_id=row['question_id'], img_id=row['img_id'], cohort=row['cohort'],
                               answer=row['answer'], condition=condition, logp_left=row[prefix+'_logp_left'],
                               logp_right=row[prefix+'_logp_right'], gold_logp=row[prefix+'_gold_logp'],
                               wrong_logp=row[prefix+'_wrong_logp'], margin=row[prefix+'_margin'], correct=row[prefix+'_correct']))
    write_csv(OUTPUT / 'per_condition_results.csv', narrow)
    write_csv(OUTPUT / 'discovery_replication.csv', replicated)
    intervals = []
    for cohort, subsets in summaries.items():
        for subset, values in subsets.items():
            if values['n']:
                for metric, bounds in values['bootstrap_ci95'].items():
                    intervals.append(dict(cohort=cohort, subset=subset, n=values['n'], metric=metric,
                                          estimate=values[metric], ci95_low=bounds[0], ci95_high=bounds[1]))
    write_csv(OUTPUT / 'statistics_ci95.csv', intervals)
    atomic(OUTPUT / 'summary.json', dict(binding=binding, code_commit=config['code_commit'], rules=RULES,
           cohorts=summaries, audits=audits, replication=replicated))
    plots(rows, summaries)
    report(config, summaries, replicated, audits)
    atomic(OUTPUT / 'summary_validation.json', dict(passed=True, status='completed', binding=binding,
           code_commit=config['code_commit'], images=240, conditions=CONDITIONS, confirmation=224,
           discovery=16, aggregation_count=1, artifacts=inventory()))
    print('SUMMARY_COMPLETE', '240 paired images', 'primary=224', flush=True)


if __name__ == '__main__':
    main()
