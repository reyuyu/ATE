"""Freeze disjoint inputs and fixed candidate grid without inspecting accuracy."""
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from ..qwen3vl_overnight.common import OUT as FIRST, ROOT, inventory, digest, sha, write, table, TASKS, check as check_first
from ..qwen3vl_overnight.report import records
from ..qwen3vl_overnight.metrics import complete_keys

OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_reaccess_layer_search'
CANDIDATES = dict(K0=[], K1=[26], K2=[23,26], K3=[23,24,26], K4=[21,23,24,26],
    K6=[12,21,23,24,25,26], K8=[10,12,21,23,24,25,26,27],
    Middle4=list(range(16,20)), Middle8=list(range(14,22)), Middle12=list(range(12,24)),
    Middle16=list(range(10,26)), Middle18=list(range(9,27)), Middle20=list(range(8,28)),
    Middle24=list(range(6,30)), Early8=list(range(8)), Late8=list(range(28,36)))


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'vlm_research/qwen3vl_overnight'/name for name in ['metrics.py','report.py','archive.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def choose(pool, n, label, field=None):
    if field:
        groups = defaultdict(list)
        for row in pool:
            groups[row[field]].append(row)
        ordered = sorted(groups.values(), key=lambda g:digest([20261111,label,sorted(r['key'] for r in g)]))
        out=[]
        for group in ordered:
            if len(out)>=n:
                break
            out += group
        return out
    return sorted(pool,key=lambda r:digest([20261111,label,r['key']]))[:n]


def freeze():
    check_first()
    first = json.loads((FIRST/'completion.json').read_text())
    assert first['status']=='TERMINAL' and first['stage2_allowed']
    assert not (OUT/'protocol.json').exists()
    planned = inventory()
    completed,_ = records()
    by_key=defaultdict(dict)
    for row in completed:
        if row['status']=='COMPLETE':
            by_key[row['key']][row['condition']]=row
    available={k for k,p in by_key.items() if set(p)=={'Native','SVH'}}
    valid=set()
    for task in TASKS:
        pool=[r for r in planned if r['benchmark']==task]
        valid |= complete_keys(task,pool,{r['key'] for r in pool if r['key'] in available})
    rows=[r for r in planned if r['key'] in valid]
    all_bindings={r['key']:r['input_binding_sha256'] for r in completed if r['condition']=='Native'}
    assert len(all_bindings)==len(planned)
    full_manifest=[dict(r,expected_input_binding_sha256=all_bindings[r['key']]) for r in planned]
    write(OUT/'Full_manifest.json',full_manifest)
    # Extract the exact historical samples from Git, without materializing old trees.
    ref='codex/qwen-internvl-sparse-visual-3000'
    path='vlm_research/results/qwen_internvl_sparse_visual_3000/sample_manifest.json'
    raw=subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT)
    history_commit=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
    historic=json.loads(raw)['samples']
    historic_images={r['image_sha256'] for r in historic}
    mapping=json.loads((FIRST/'shards.json').read_text())
    cluster_historic={mapping[r['key']]['cluster'] for r in rows if r['image_sha256'] in historic_images}
    assignment={}
    for row in rows:
        c=mapping[row['key']]['cluster']
        value=int(digest([20261111,'DISJOINT',c])[:16],16)/2**64
        split='Discovery' if value<.2 else 'Validation' if value<.4 else 'Held-out'
        if c in cluster_historic and split=='Held-out':
            split='Discovery' if value<.7 else 'Validation'
        assignment[c]=split
    selected={}
    for split in ['Discovery','Validation','Held-out']:
        cohort=[r for r in rows if assignment[mapping[r['key']]['cluster']]==split]
        chosen=[]
        for task in TASKS:
            pool=[r for r in cohort if r['benchmark']==task]
            if task=='MME':
                for category in sorted({r['task'] for r in planned if r['benchmark']=='MME'}):
                    chosen += choose([r for r in pool if r['task']==category],16,split+task+category,'pair_id')
            elif task=='POPE':
                for strategy in ['random','popular','adversarial']:
                    chosen += choose([r for r in pool if r['strategy']==strategy],32,split+task+strategy)
            else:
                chosen += choose(pool,200 if task in ['GQA','TextVQA'] else 96,split+task,
                    'circular_group' if task=='MMBench' else None)
        chosen=[dict(r,expected_input_binding_sha256=by_key[r['key']]['Native']['input_binding_sha256']) for r in chosen]
        assert all(by_key[r['key']]['Native']['input_binding_sha256']==by_key[r['key']]['SVH']['input_binding_sha256'] for r in chosen)
        selected[split]=chosen
        write(OUT/(split+'_manifest.json'),chosen)
    sets={s:{mapping[r['key']]['cluster'] for r in pool} for s,pool in selected.items()}
    assert all(not sets[a]&sets[b] for a,b in [('Discovery','Validation'),('Discovery','Held-out'),('Validation','Held-out')])
    assert not any(r['image_sha256'] in historic_images for r in selected['Held-out'])
    write(OUT/'sample_counts.json',{s:dict(Counter(r['benchmark'] for r in pool)) for s,pool in selected.items()})
    ranking_tasks=[task for task in TASKS if all(any(r['benchmark']==task for r in pool) and
        (task!='MME' or len({r['task'] for r in pool if r['benchmark']=='MME'})==14) for pool in selected.values())]
    assert ranking_tasks, 'No usable task; do not invent a selection score'
    history={}
    for ref,path in [
        ('codex/qwen-internvl-sparse-visual-3000','vlm_research/results/qwen_internvl_sparse_visual_3000/report_zh.md'),
        ('codex/qwen-internvl-sparse-visual-3000','vlm_research/results/qwen_internvl_sparse_visual_3000/layer_selection_audit.json'),
        ('codex/qwen3vl-keep-l26-generation-240','vlm_research/results/qwen3vl_keep_l26_generation_240/report_zh.md'),
        ('codex/qwen3vl-roi-sensitivity-alignment-240','vlm_research/results/qwen3vl_roi_sensitivity_alignment_240/report_zh.md'),
        ('codex/qwen3vl-four-stage-pilot-120','vlm_research/results/qwen3vl_four_stage_pilot_120/report_zh.md'),
        ('codex/textvqa500-early-middle-late-budget','vlm_research/results/textvqa500_early_middle_late_budget/report_zh.md')]:
        command=subprocess.run(['git','show',f'{ref}:{path}'],cwd=ROOT,capture_output=True)
        if command.returncode:
            history[ref+':'+path]=dict(status='NOT_FOUND_LOCAL',error=command.stderr.decode()[:1000])
        else:
            history[ref+':'+path]=dict(status='READ_LOCAL_FROZEN',sha256=hashlib.sha256(command.stdout).hexdigest(),
                commit=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip())
    write(OUT/'history_evidence.json',history)
    candidates={name:dict(layers=layers,k=len(layers),status='NOT_APPLICABLE_FROZEN_SKIP' if set(layers)&set(range(1,7)) else 'ELIGIBLE',
        conflicting_skip_layers=sorted(set(layers)&set(range(1,7)))) for name,layers in CANDIDATES.items()}
    write(OUT/'candidate_grid.json',candidates)
    write(OUT/'protocol.json',dict(sources=sources(),first_protocol_sha256=sha(FIRST/'protocol.json'),
        first_completion_sha256=sha(FIRST/'completion.json'),mode='FUNCTIONAL_REACCESS_ONLY',
        frozen_svh='UNCHANGED',candidate_grid_sha256=sha(OUT/'candidate_grid.json'),
        manifests={s:sha(OUT/(s+'_manifest.json')) for s in list(selected)+['Full']},seed=20261111,
        history_manifest_sha256=hashlib.sha256(raw).hexdigest(),heldout_historical_image_overlap=0,
        history_commit=history_commit,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selection='six task normalized equally: percentage/100 and MME/2800; within .003 of best, smallest K; no unmeasured E2E tie-break',
        ranking_tasks=ranking_tasks,unavailable_tasks=[t for t in TASKS if t not in ranking_tasks],
        reduced_task_fallback='Only if common successful inventory lacks a whole metric before candidate outputs. Explicit reduced-task objective, not six-task evidence.',
        candidate_count=len(candidates),not_applicable=[n for n,c in candidates.items() if c['status']!='ELIGIBLE'],
        physical_svhr='INDEPENDENT_PORT_NOT_GPU_VALIDATED; never label dense Mask as physical KV saving',
        optional_physical_port='Independent prewritten port; GPU structural validation after functional selection, selected visual KV only; extra construction charged',
        optional_physical_source_sha256=sha(Path(__file__).parent/'physical.py'),
        full_physical_policy='Only the pre-heldout quality choice; both-device physical validation, no structure failure, free disk >= safety floor + 1GiB estimate; never pick by heldout accuracy',
        original_visual_token_budget_unchanged=True,independent_external_validation=False))


def check():
    check_first()
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['sources']==sources() and sha(OUT/'candidate_grid.json')==p['candidate_grid_sha256']
    assert sha(FIRST/'completion.json')==p['first_completion_sha256']
    for split,expected in p['manifests'].items():
        assert sha(OUT/(split+'_manifest.json'))==expected
    return p
