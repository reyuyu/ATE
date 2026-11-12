"""Standalone CPU plotting entrypoint, independent of the model environment."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS=['GQA','ScienceQA-IMG','TextVQA','POPE','MME','MMBench']


def number(value):
    return None if value in ('',None,'None') else float(value)


def main(root):
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    with (root/'all_stage_accuracy.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    grid=json.loads((root/'candidate_grid.json').read_text())
    discovery=[r for r in rows if r['stage']=='Discovery']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    colors={'K':'#267a66','Middle':'#ba5c30','Late':'#6264aa'}
    for task,ax in zip(TASKS,axes.flat):
        for row in discovery:
            if row['task']!=task or number(row['score']) is None:
                continue
            name=row['condition']
            color=colors['Middle' if name.startswith('Middle') else 'Late' if name.startswith('Late') else 'K']
            ax.scatter(grid[name]['k'],number(row['score']),color=color,s=27)
            if name in ['K0','K1','K8','Middle8','Middle16','Late8']:
                offset=(2,-9) if name=='Middle8' else (2,4)
                ax.annotate(name,(grid[name]['k'],number(row['score'])),fontsize=6,xytext=offset,textcoords='offset points')
        ax.set_title(task,fontsize=10)
        ax.set_xlabel('Allowed re-access layers K')
        ax.set_ylabel('MME score' if task=='MME' else 'Score (%)')
        ax.grid(alpha=.15)
    for suffix in ['png','pdf']:
        fig.savefig(folder/f'discovery_quality_vs_layers.{suffix}',dpi=220)
    plt.close(fig)
    text=[r for r in discovery if r['task']=='TextVQA']
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for i,row in enumerate(text):
        y=number(row['delta_K0'])
        if y is None:
            continue
        low,high=number(row['delta_K0_low']),number(row['delta_K0_high'])
        ax.errorbar(i,y,yerr=None if low is None else [[max(0,y-low)],[max(0,high-y)]],fmt='o',color='#267a66',capsize=3)
    ax.set_xticks(range(len(text)),[r['condition'] for r in text],rotation=45,ha='right')
    ax.axhline(0,color='#555555',lw=.8)
    ax.set_ylabel('TextVQA change vs functional K0 (pp)')
    ax.set_title('Discovery only: frozen Paper-OCR inputs')
    for suffix in ['png','pdf']:
        fig.savefig(folder/f'textvqa_reaccess_comparison.{suffix}',dpi=220)
    plt.close(fig)
    (root/'plot_runtime.json').write_text(json.dumps(dict(python=sys.executable,matplotlib=matplotlib.__version__,
        cpu_only=True,model_runtime_not_modified=True,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.suffix in ['.png','.pdf']}),indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]))
