"""Post-run publication hashes and resource audit; never runs a model."""
from pathlib import Path
import csv
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.dataset_registry import sha256
from paper_rq1_single_layer_gqa6.run import atomic_json
from rq2_sparse_caption import OUT, GPU, read, verify, history


p=verify()
assert read(OUT/'validation.json')['passed']
# Publication-only label reflow from frozen CSV; measurements stay unchanged.
with (OUT/'accuracy_and_quality_summary.csv').open() as f:
    summary=list(csv.DictReader(f))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
labels=['A\nFull\nHistorical','B\nRelay\nHistorical','S\nBoth: 2L\nNew',
        'DS\nDecode: 2L\nNew','DN\nDecode: 0L\nNew']
for ax,metric in zip(axs,['CIDEr','SPICE']):
    values=[float(r[metric]) for r in summary]
    bars=ax.bar(range(5),values,color=['#777777','#aaaaaa','#be6d39','#267e67','#477cac'],width=.65)
    ax.set_xticks(range(5),labels,fontsize=8);ax.set_title(metric)
    ax.set_ylim(0,max(values)*1.18);ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Caption Quality (60 paired COCO images)',fontsize=12)
for ext in ['png','pdf']:fig.savefig(OUT/f'figure_a_caption_quality.{ext}',dpi=240)
plt.close(fig)
for name, binding in p['model_files'].items():
    assert sha256(name)==binding['sha256']
for s in read(OUT/'sample_manifest.json')['samples']:
    assert sha256(s['image_path'])==s['image_sha256']
table,apps=history.snapshot()
assert not any(row.startswith(GPU) for row in apps.splitlines())
atomic_json(OUT/'gpu_release.json',dict(released=True,physical_gpu=0,uuid=GPU,
    snapshot=table,compute_apps=apps,checked_epoch=time.time(),gpu1_used=False))
completion=read(OUT/'completion.json')
atomic_json(OUT/'delivery_audit.json',dict(passed=True,
    code_commit=p['code_commit'],sources_unchanged=True,model_files_rehashed=True,
    images_rehashed=60,reference_captions_sha256=sha256(OUT/'sample_manifest.json'),
    formal_generation_seconds=completion['formal_seconds'],
    elapsed_from_worktree_creation_seconds=time.time()-p['worktree_created_epoch'],
    timing_scope='End-to-end worktree creation through publication audit; excludes initial attachment/code inspection and subsequent Git push.',
    git_main_modified=False,gpu_released=True))
files=[f for f in OUT.rglob('*') if f.is_file() and f.name not in ['worker.lock','artifact_hashes.json']]
hashes={str(f.relative_to(OUT)):sha256(f) for f in sorted(files)}
atomic_json(OUT/'artifact_hashes.json',hashes)
assert all(sha256(OUT/name)==h for name,h in hashes.items())
print('Verified artifacts:',len(hashes),'GPU0 released; GPU1 unused.',flush=True)
