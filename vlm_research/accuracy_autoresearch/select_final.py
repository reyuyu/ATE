"""Freeze once from CHECK only; never read FINAL outputs here."""
import csv
import datetime
import subprocess

from .common import *
from .dispatch import job, keys_for, with_native


def table(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def main():
    assert time.time()<deadline('candidate_freeze_by')
    assert not (OUT/'final_candidate_freeze.json').exists()
    p=read(OUT/'check_promotion.json')
    choices=p['candidates']+[p['reference']]
    q=table(OUT/'analysis/07_check/accuracy_summary.csv')
    perf=table(OUT/'analysis/08_check_performance/performance_paired.csv')
    timing=table(OUT/'analysis/08_check_performance/latency_flops_kv.csv')
    native=config_id(config('native')); ranking=[]
    def timing_pair(a,b,d):
        return next(r for r in perf if r['dataset']==d and r['condition']==config_id(a) and r['baseline']==config_id(b))
    for c in choices:
        rows=[next(r for r in q if r['dataset']==d and r['config_id']==config_id(c)) for d in ('GQA','VQAv2')]
        speed=[timing_pair(c,config('native'),d) for d in ('GQA','VQAv2')]
        assert all(int(r['same_sequence_n'])>=6 for r in speed),'Too few matched sequences to resolve timing tie'
        kv=[next(r for r in timing if r['dataset']==d and r['config_id']==config_id(c)) for d in ('GQA','VQAv2')]
        ranking.append(dict(config=c,config_id=config_id(c),
            worst_loss_pp=max(-float(r['delta_original_native_pp']) for r in rows),
            matched_sequence_min_speedup=min(float(r['same_sequence_e2e_speedup']) for r in speed),
            all_input_min_speedup=min(float(r['all_e2e_speedup']) for r in speed),
            mean_KV_capacity_MiB=sum(float(r['kv_capacity_mib']) for r in kv)/2,
            quality_by_dataset={r['dataset']:r for r in rows}))
    best=min(r['worst_loss_pp'] for r in ranking)
    ties=[r for r in ranking if r['worst_loss_pp']<=best+.25+1e-10]
    winner=min(ties,key=lambda r:(-r['matched_sequence_min_speedup'],r['mean_KV_capacity_MiB'],r['worst_loss_pp'],r['config_id']))
    primary=winner['config']; faster=[]
    for r in ranking:
        if r['config']==primary: continue
        speed=[timing_pair(r['config'],primary,d) for d in ('GQA','VQAv2')]
        if all(int(v['same_sequence_n'])>=6 and float(v['same_sequence_e2e_speedup'])>=1.02 for v in speed):
            faster.append((min(float(v['same_sequence_e2e_speedup']) for v in speed),r))
    secondary=max(faster,key=lambda v:v[0])[1]['config'] if faster else None
    selected=[primary]+([secondary] if secondary else [])
    controls=[config(),config(answer='both')]
    original=[]
    for c in controls+selected:
        if c not in original: original.append(c)
    evaluations=with_native(original+[dict(c,logits_to_keep=1) for c in original])
    frozen=dict(at=now(),quality_primary=primary,speed_secondary=secondary,selected_candidates=selected,
        evaluation_configs=evaluations,noninferiority_family_size=2*2*len(selected),
        noninferiority_family='Selected architectures x 2 datasets x H0/H1 engineering bindings; -1pp one-sided Bonferroni lower bound',
        ranking=ranking,quality_tie_configs=[r['config_id'] for r in ties],
        selection_rule='Worst ORIGINAL Native dataset quality loss; <=0.25pp ties use minimum matched-sequence E2E speedup then KV. Speed secondary requires >=2% matched-sequence speedup on both datasets with >=6 paired inputs each.',
        check_quality_sha256=sha(OUT/'analysis/07_check/accuracy_summary.csv'),
        check_performance_sha256=sha(OUT/'analysis/08_check_performance/performance_paired.csv'),
        sources=source_hashes(),final_opened_before_freeze=False,
        final_quality_job='09_final',final_perf_job='10_final_performance',
        code_commit_at_selection=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    assert not any((OUT/'records'/config_id(c)/(k+'.json')).exists() for k in keys_for('FINAL') for c in evaluations)
    atomic(OUT/'final_candidate_freeze.json',frozen)
    atomic(OUT/'protocol.json',dict(clock=read(ROOT/'task_deadline.json'),data_freeze=read(OUT/'data_freeze.json'),
        execution_freeze=read(OUT/'execution_freeze.json'),selection=frozen,
        selection_addendum=read(OUT/'selection_addendum_before_prompt_search.json'),
        semantics='All non-Native P-1/Decode directly image-blind, pre-P-1 Attention updates retained, original positions, independent text-only persistent KV, no training'))
    job('09_final','FINAL',evaluations,hypothesis='FROZEN FINAL: Native, Fast-P2, C3-V2 and CHECK-selected architecture(s), with independent H0/H1 bindings. No output may change candidate selection. All FINAL pictures were excluded from this task DEV/CHECK.')
    pool=read(OUT/'sample_manifest.json'); final=keys_for('FINAL'); selected_perf=[]
    for dataset in ('GQA','VQAv2'):
        candidates=sorted((k for k in final if pool[k]['dataset']==dataset),key=lambda k:digest([SEED,'final-performance',k]))
        seen=set()
        for k in candidates:
            h=pool[k]['image_sha256']
            if h not in seen:
                seen.add(h); selected_perf.append(k)
                if len(seen)==20: break
    assert len(selected_perf)==40
    atomic(OUT/'final_timing_selection.json',dict(at=now(),samples=selected_perf,
        rule='Twenty unique image SHA inputs per dataset, stable input hash before FINAL scores; not selected for correct answers or equal output length'))
    job('10_final_performance','FINAL',evaluations,samples=selected_perf,performance=True,
        hypothesis='FROZEN paired performance: 20 input-hash-selected unique images per dataset, all original and uniformly H1 controls, 3 warmups +5 interleaved repeats, natural EOS, cold KV and fresh Vision. Same-sequence subset additionally reported; no performance-driven reselection.')
    print('FROZEN primary',config_id(primary),'secondary',config_id(secondary) if secondary else None,
        'conditions',len(evaluations),'FINAL questions',len(final),flush=True)


if __name__=='__main__': main()
