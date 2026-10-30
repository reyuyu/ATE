"""Final finite artifact/immutable-code check and explicit GPU-release record."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from benchmarks.dataset_registry import freeze_json,sha256
from .protocol import ROOT,OUT
from .run import verify


def main():
    config=verify()
    for p,b in config['model_files'].items():
        if sha256(p)!=b['sha256']:raise ValueError('Final pinned model hash mismatch')
    accepted=json.loads((OUT/'analysis_cpu_validation.json').read_text())
    if not accepted['passed']:raise ValueError('Analysis tests not accepted')
    for p,h in accepted['source_sha256'].items():
        if sha256(ROOT/p)!=h:raise ValueError('Postprocessing changed after CPU tests: '+p)
    if not json.loads((OUT/'table_consistency_audit.json').read_text())['passed']:raise ValueError('Table verification missing')
    coverage=json.loads((OUT/'generation_coverage.json').read_text())
    expected=len(json.loads((OUT/'sample_manifest.json').read_text())['samples'])*11
    if coverage['generations']!=expected or not coverage['all_scores_independently_replayed']:raise ValueError('Incomplete core coverage')
    with (OUT/'checkpoint_manifest.csv').open() as f:
        for row in csv.DictReader(f):
            if sha256(OUT/row['path'])!=row['sha256']:raise ValueError('Raw checkpoint hash mismatch')
    checks=[]
    for path in sorted((OUT/'figures').glob('*.png')):
        with Image.open(path) as image:
            pixels=np.array(image.convert('RGB'))
            if min(image.size)<500 or float(pixels.std())<5:raise ValueError('Blank/tiny scientific figure')
            checks.append(dict(path=str(path.relative_to(OUT)),size=list(image.size),pixel_std=float(pixels.std())))
    if len(checks)!=6 or len(list((OUT/'figures').glob('*.pdf')))!=6:raise ValueError('Missing six PNG/PDF figures')
    for rank in range(3):
        smoke=json.loads((OUT/f'smoke/shard{rank}/validation.json').read_text())
        if sha256(smoke['raw_path'])!=smoke['raw_sha256']:raise ValueError('External raw Smoke changed')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
    own=[]
    for line in processes.splitlines()[1:]:
        parts=[x.strip() for x in line.split(',')]
        if len(parts)<2:continue
        path=Path('/proc')/parts[1]/'cmdline'
        if path.exists() and b'paper_rq1_early_joint_attention' in path.read_bytes():own.append(parts[1])
    if own:raise ValueError('Experiment GPU worker still running')
    freeze_json(OUT/'gpu_release.json',dict(own_workers_remaining=[],own_resources_released=True,
        gpus=gpu,compute_processes=processes,other_processes_not_terminated=True))
    # All worker contexts have ended; stale zero-byte lock files are not research outputs.
    for p in (OUT/'generation').glob('shard*/worker.lock'):p.unlink()
    for p in (OUT/'smoke').glob('shard*/worker.lock'):p.unlink()
    excluded={'delivery_inventory.json'}
    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in excluded:
            files.append(dict(path=str(p.relative_to(OUT)),sha256=sha256(p),bytes=p.stat().st_size))
    freeze_json(OUT/'delivery_inventory.json',dict(passed=True,artifacts=len(files),files=files,figure_checks=checks,
        frozen_collector_commit=config['code_commit'],collector_source_unchanged=True,weights_final_sha256_unchanged=True,
        complete_frozen_core=True,requested_balanced_six_task_confirmation='partial_data_shortfall',
        analysis_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'paper_rq1_early_joint_attention').glob('*.py')}))
    print('DELIVERY VERIFIED',len(files),'artifacts;',expected,'generations; own GPUs released',flush=True)


if __name__=='__main__':main()
