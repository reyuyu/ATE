"""Publication layout only; frozen experimental/statistical outputs stay intact."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from benchmarks.dataset_registry import freeze_json, sha256
from .run import OUT, ROOT, read, verify


def main():
    verify()
    result = read(OUT / 'result.json')
    assert result['status'] == 'complete'
    assert all(sha256(OUT / name) == expected for name, expected in result['artifact_sha256'].items())
    with (OUT / 'accuracy_summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))[1:]
    assert [r['condition'] for r in rows] == [f'W{s:02d}' for s in range(29)]
    x = np.arange(29)
    y, low, high, ordinary_low, ordinary_high = [np.array([float(r[key]) for r in rows])
        for key in ['delta_pp', 'corrected95_low_pp', 'corrected95_high_pp', 'ci95_low_pp', 'ci95_high_pp']]
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9})
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    shade = ax.fill_between(x, ordinary_low, ordinary_high, color='#b8c7d2', alpha=.5,
        label='Ordinary 95% CI')
    bars = ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), color='#3b5567',
        ecolor='#76858f', fmt='o-', markersize=2.8, linewidth=1.15, elinewidth=.85,
        capsize=1.7, label='29-comparison corrected 95% CI')
    ax.axhline(0, color='#888888', linewidth=.7)
    boundary = ax.axhline(-1, color='#b24e43', linestyle='--', linewidth=1,
        label='Noninferiority boundary: -1 pp')
    star, = ax.plot(2, y[2], marker='*', color='#087e75', markersize=9,
        zorder=5, label='L2-L5 (reused)')
    ax.set_xlim(-.5, 28.5)
    ax.set_ylabel('Accuracy change relative to Full (pp)')
    ax.set_xlabel('Four-layer window start s (Ls-Ls+3)', labelpad=7)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18, linewidth=.5)
    ax.set_axisbelow(True)
    fig.suptitle('Four-Layer Attention Ablation: P-1 + Decode', fontsize=10, y=.985)
    fig.legend([shade, bars, boundary, star], [h.get_label() for h in [shade, bars, boundary, star]],
        loc='upper center', bbox_to_anchor=(.54, .94), frameon=False, fontsize=7, ncol=2)
    fig.text(.54, .035, '3,000 questions / 2,409 image clusters; 1,000,000 paired bootstrap draws\n'
        'Exploratory on the existing cohort; not independent confirmation or evidence of speedup',
        ha='center', va='bottom', fontsize=6.5, linespacing=1.5)
    fig.subplots_adjust(left=.10, right=.99, bottom=.22, top=.81)
    target = OUT / 'publication'
    target.mkdir(exist_ok=True)
    for suffix in ['pdf', 'png']:
        fig.savefig(target / ('four_layer_window_curve.' + suffix), dpi=300)
    plt.close(fig)
    freeze_json(target / 'presentation.json', dict(statistics_unchanged=True,
        original_result_sha256=sha256(OUT / 'result.json'), csv_sha256=sha256(OUT / 'accuracy_summary.csv'),
        renderer_sha256=sha256(ROOT / 'four_layer_attention/render.py'),
        reason='Move legend and footer outside axes; remove label overlap. Same29 points and confidence intervals.',
        artifact_sha256={suffix: sha256(target / ('four_layer_window_curve.' + suffix)) for suffix in ['pdf', 'png']}))


if __name__ == '__main__': main()
