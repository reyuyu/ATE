"""Descriptive false-skip inventory for frozen rules, without retuning."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.dataset_registry import freeze_json, sha256
from paper_rq1.analyze import write_csv
from bos_cheap.protocol import OUT, TASKS
from bos_cheap.predict import load_rows, score_rule, skips, danger
from bos_cheap.analyze import save, risk_values


def main():
    rows = load_rows('new')
    rules = json.loads((OUT / 'frozen_predictors.json').read_text())['eps']['0.03']['selected']
    y = danger(rows, .03).astype(bool)
    failures = []
    region_rows = []
    for name, layers in [('L0-L1', range(2)), ('L2-L31', range(2, 32))]:
        mask = np.array([r['layer'] in layers for r in rows])
        region_rows.append(dict(region=name, **risk_values(rows, mask, .03),
            mean_native_bos_mass=float(np.mean([r['oracle_bos_mass'] for r, s in zip(rows, mask) if s])),
            mean_cheap_cos=float(np.mean([r['bos_cos_mean'] for r, s in zip(rows, mask) if s])),
            mean_value_norm=float(np.mean([r['v_norm_mean'] for r, s in zip(rows, mask) if s]))))
    write_csv(OUT / 'early_region_native_summary.csv', region_rows)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
    for rule in rules:
        selected = skips(rows, rule)
        scores = score_rule(rows, rule)
        for i in np.flatnonzero(selected & y):
            r = rows[i]
            failures.append(dict(predictor=rule['name'], epsilon=.03, score=float(scores[i]),
                threshold=rule.get('per_layer_thresholds', [rule['threshold']] * 32)[r['layer']],
                **{k: r[k] for k in ['task', 'question_id', 'image_id', 'image_sha256', 'layer', 'delta_logp', 'top1_flip', 'js', 'oracle_bos_mass']}))
        if rule['name'] not in ['BOS-QK+Residual', 'Region-threshold']:
            continue
        column = 0 if rule['name'] == 'BOS-QK+Residual' else 1
        for ax, groups, key in [(axes[0, column], list(range(32)), 'layer'), (axes[1, column], list(TASKS), 'task')]:
            n = np.array([sum(s and r[key] == g for s, r in zip(selected, rows)) for g in groups])
            d = np.array([sum(s and bad and r[key] == g for s, bad, r in zip(selected, y, rows)) for g in groups])
            rate = np.divide(d, n, out=np.full(len(n), np.nan), where=n > 0)
            ax.plot(range(len(groups)), rate, 'o-', color='#247d95', ms=4)
            for i in np.flatnonzero(n):
                ax.annotate(f'{d[i]}/{n[i]}', (i, rate[i]), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7, rotation=70 if key == 'layer' else 0)
            ax.axhline(.01, color='#ac3c3c', ls=':', label='1% reference, not certified')
            ax.set_xticks(range(len(groups)), groups, rotation=45 if key == 'task' else 0, ha='right' if key == 'task' else 'center', fontsize=8)
            ax.set_ylabel('Dangerous / skipped; empty groups omitted')
            ax.set_title(rule['name'] + ' by ' + key)
            ax.set_ylim(-.001, .12)
            ax.grid(alpha=.15)
    fig.suptitle('New554 images, epsilon=.03: frozen-policy errors, counts shown explicitly; descriptive only')
    save(fig, 'figure6b_false_skip_layers_tasks')
    write_csv(OUT / 'false_skip_examples.csv', failures)
    freeze_json(OUT / 'error_review.json', dict(script_sha256=sha256(Path(__file__)),
        predictors_sha256=sha256(OUT / 'frozen_predictors.json'), epsilon=.03,
        false_skip_rows=len(failures), model_forwards=0, predictor_fits=0, threshold_changes=0))


if __name__ == '__main__':
    main()
