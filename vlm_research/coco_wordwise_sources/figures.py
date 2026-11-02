import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import OUT, MAIN, SELECTED, read
from .statistics import ci

COLORS = ['#888888', '#2687b8', '#d09335', '#318d69']
NAMES = ['BOS', 'Image', 'Prompt', 'History']


def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def figures(checkpoints, raw, role_summary, paired, contributions, consistency):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    representative = read(OUT / 'sample_manifest.json')['representative_image']
    result, mass = checkpoints[representative], raw[representative]
    labels = result['labels']
    words = [r['lexical_word'] if r['word_first'] else ('\u00b7'+str(r['subword_index']+1) if r['subword_index'] is not None else r['decoded_piece'].strip()) for r in labels]
    words = ['EOS' if r['role'] == 'EOS' else w or repr(r['decoded_piece']) for w, r in zip(words, labels)]
    primary = np.stack([mass[:, :, 0], mass[:, :, 1], mass[:, :, 2], mass[:, :, 3]+mass[:, :, 4]], -1)
    width = max(10, .42*len(labels))
    fig, ax = plt.subplots(figsize=(width, 3.6))
    x = np.arange(len(labels))
    ax.stackplot(x, *primary[:, 31].T, labels=NAMES, colors=COLORS, step='mid')
    ax.set_xlim(-.5, len(labels)-.5)
    ax.set_ylim(0, 1.015)
    ax.set_ylabel('L31 attention mass')
    ax.set_xticks(x, words, rotation=50, ha='right')
    roles = {'Function': 'F', 'Visual': 'V', 'Relation': 'R', 'Unknown': 'U', 'Punctuation': 'P', 'EOS': 'E'}
    for i, r in enumerate(labels):
        ax.text(i, -.29, roles[r['role']], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
    ax.axvline(.5, color='#222222', linestyle=':', linewidth=1)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.17), frameon=False)
    ax.set_xlabel('Predicted word / subword     (t=0: Prefill)', labelpad=32)
    fig.subplots_adjust(bottom=.35, top=.85)
    save(fig, 'figure1_caption_timeline')

    fig, axes = plt.subplots(2, 1, figsize=(width, 6), sharex=True)
    maximum = max(primary[:, :, 1].max(), primary[:, :, 3].max())
    for ax, index, name in zip(axes, [1, 3], ['Image', 'Generated history']):
        image = ax.imshow(primary[:, :, index].T, aspect='auto', origin='upper', cmap='viridis', vmin=0, vmax=maximum)
        ax.set_yticks([0, 5, 10, 15, 20, 25, 31])
        ax.set_ylabel('Decoder layer')
        ax.set_title(name, loc='left')
        ax.axhline(30.5, color='white', linewidth=1)
        ax.axvline(.5, color='white', linestyle=':', linewidth=.7)
        fig.colorbar(image, ax=ax, pad=.01, fraction=.025, label='Mass')
    axes[-1].set_xticks(x, words, rotation=50, ha='right')
    axes[-1].set_xlabel('Predicted word / subword')
    fig.tight_layout()
    save(fig, 'figure2_layer_token_heatmap')

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    for j, source in enumerate(MAIN):
        ax = axes[0, j]
        for i, role in enumerate(['Function', 'Visual', 'Relation']):
            entry = next(r for r in role_summary if r['role'] == role+'_common3' and r['outcome'] == source)
            ax.errorbar(i, entry['mean'], yerr=[[entry['mean']-entry['ci95_low']], [entry['ci95_high']-entry['mean']]],
                        fmt='o', capsize=4, color=COLORS[j])
        ax.set_title(NAMES[j])
        ax.set_xticks([0, 1, 2], ['Function', 'Visual', 'Relation'], rotation=25)
        ax.set_ylabel('L31 mass' if j == 0 else '')
        ax.grid(axis='y', alpha=.15)
        ax = axes[1, j]
        for i, contrast in enumerate(['Visual-Function', 'Relation-Function']):
            entry = next(r for r in paired if r['contrast'] == contrast and r['outcome'] == source)
            ax.errorbar(i, 100*entry['mean'], yerr=[[100*(entry['mean']-entry['ci95_low'])],
                [100*(entry['ci95_high']-entry['mean'])]], fmt='o', capsize=4, color=COLORS[j])
        ax.axhline(0, color='#666666', linewidth=.8)
        ax.set_xticks([0, 1], ['Visual\u2212Function', 'Relation\u2212Function'], rotation=25, ha='right')
        ax.set_ylabel('Paired mass difference (pp)' if j == 0 else '')
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    save(fig, 'figure3_word_role_distribution')

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for j, layer in enumerate(SELECTED):
        ax = axes[0, j]
        for k, source in enumerate(MAIN):
            grouped = {}
            for row in contributions:
                if row['layer'] == layer and row['source'] == source:
                    grouped.setdefault(row['image_sha256'], []).append(row)
            xs = [np.mean([r['mass'] for r in group]) for group in grouped.values()]
            ys = [np.mean([r['relative_residual_ratio'] for r in group]) for group in grouped.values()]
            ax.scatter(xs, ys, s=9, alpha=.5, color=COLORS[k], label=NAMES[k], rasterized=False)
        ax.set_title(f'L{layer}')
        ax.set_xlabel('Source mass (caption mean)')
        ax.set_ylabel('Projected norm / residual norm' if j == 0 else '')
        if j == 2:
            ax.legend(fontsize=8, frameon=False)
        ax = axes[1, j]
        for k, source in enumerate(['IMAGE', 'GENERATED_HISTORY']):
            entries = [r for r in consistency if r['layer'] == layer and r['source'] == source and not r['include_eos']]
            if entries:
                entry = entries[0]
                ax.errorbar(k, 100*entry['mean'], yerr=[[100*(entry['mean']-entry['ci95_low'])],
                    [100*(entry['ci95_high']-entry['mean'])]], fmt='o', color=COLORS[1 if k == 0 else 3], capsize=4)
        ax.set_xticks([0, 1], ['Image', 'History'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Contribution change given similar mass (%)' if j == 0 else '')
        ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    save(fig, 'figure4_mass_vs_contribution')
