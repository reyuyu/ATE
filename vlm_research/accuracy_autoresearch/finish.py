"""Final completeness, paired case audits and report tables; no selection logic."""
import argparse
import collections
import csv
import datetime
import math
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import *
from .dispatch import keys_for
from .worker import read_record


def csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def audit():
    frozen=read(OUT/'final_candidate_freeze.json'); pool=read(OUT/'sample_manifest.json')
    parts=read(OUT/'split_manifest.json'); execution=read(OUT/'execution_freeze.json')
    assert source_hashes()==execution['sources']
    assert sha(OUT/'sample_manifest.json')==execution['samples_sha256']
    assert sha(OUT/'split_manifest.json')==execution['splits_sha256']
    assert datetime.datetime.fromisoformat(frozen['at']).timestamp()<deadline('candidate_freeze_by')
    assert all(sha(p)==h for p,h in execution['weights'].items())
    question_bindings=collections.defaultdict(dict); counts=collections.Counter(); first_changed=[]
    for path in sorted((OUT/'records').glob('*/*.json')):
        blob=read(path); r=blob['payload']; s=pool[r['sample']]; c=r['config']
        assert digest(r)==blob['sha256'] and r['sample_sha256']==digest(s) and r['sources']==execution['sources']
        assert path.parent.name==config_id(c) and path.stem==r['sample']
        b=r['input_binding']; ids=b['prompt_ids']; image=[i for i,x in enumerate(ids) if x==32000]
        assert len(image)==576 and len(ids)-1 not in b['writers'] and set(b['writers'])<=set(b['R'])
        if c['prompt_variant']=='original': assert ids==s['input_mapping']['prompt_ids']
        assert set(b['Q'])<=set(b['R'])
        if c['writer']=='qsa': assert set(b['writers'])==set(b['Q'])|set(b['ASSISTANT'])
        if c['writer'] in ('all','native'): assert b['writers']==b['R']
        pixels=b['input_hashes']['pixel_values']
        assert pixels==question_bindings[r['sample']].setdefault('pixels',pixels)
        prompt_key=c['prompt_variant']; hashes=b['input_hashes']
        assert hashes==question_bindings[r['sample']].setdefault(prompt_key,hashes)
        g=r['generation']; assert g['token_ids'] and math.isfinite(g['native_first_token_logp'])
        assert len(g['token_ids'])<=32 and g['score']==score(s,g['text'])
        assert g['eos'] or g['truncated']
        counts[parts['image_split'][s['image_sha256']],config_id(c)]+=1
    final_keys=keys_for('FINAL'); final_expected=len(final_keys)*len(frozen['evaluation_configs'])
    for k in final_keys:
        for c in frozen['evaluation_configs']:
            assert read_record(c,k,pool[k]) is not None,(k,config_id(c))
    for i in range(len(execution['devices'])):
        assert read(OUT/f'worker{i}/smoke.json')['status']=='PASS'
        assert not (OUT/f'worker{i}/STOP.json').exists()
        done=read(OUT/f"worker{i}/jobs/{frozen['final_quality_job']}.json")
        assert done['status']=='COMPLETE'
    for s in pool.values(): assert sha(s['image_path'])==s['image_sha256']
    for c in frozen['evaluation_configs']:
        if c['logits_to_keep']!=1: continue
        other=dict(c,logits_to_keep=0)
        if other not in frozen['evaluation_configs']: continue
        for k in final_keys:
            a=read_record(c,k,pool[k])['generation']; b=read_record(other,k,pool[k])['generation']
            if a['token_ids']!=b['token_ids'] or a['first_logits_sha256']!=b['first_logits_sha256']:
                first_changed.append(dict(sample=k,dataset=pool[k]['dataset'],config_H1=config_id(c),
                    first_logits_changed=a['first_logits_sha256']!=b['first_logits_sha256'],
                    tokens_changed=a['token_ids']!=b['token_ids'],score_H1=a['score'],score_H0=b['score'],
                    answer_H1=a['text'],answer_H0=b['text']))
    write_csv(OUT/'last_row_head_changes.csv',first_changed)
    atomic(OUT/'completion.json',dict(status='COMPLETE',at=now(),final_questions=len(final_keys),
        final_conditions=len(frozen['evaluation_configs']),final_records=final_expected,
        counts={str(k):v for k,v in counts.items()},all_runtime_sources_unchanged=True,
        all_inputs_and_weights_bound=True,all_final_records_complete=True,
        exact_head_change_audit_rows=len(first_changed),data_cohort='Historical exploratory cohorts; not external confirmation'))
    print('COMPLETE: FINAL',final_expected,'records; all retained record/source/input/image/weight checks pass')


def cases():
    frozen=read(OUT/'final_candidate_freeze.json'); pool=read(OUT/'sample_manifest.json'); rows=[]
    for candidate in frozen['selected_candidates']:
        for baseline in (config(),config(answer='both')):
            for k in keys_for('FINAL'):
                s=pool[k]; a=read_record(candidate,k,s)['generation']; b=read_record(baseline,k,s)['generation']
                rows.append(dict(sample=k,dataset=s['dataset'],task=s['task'],answer_type=s.get('answer_type',s['task']),
                    condition=config_id(candidate),baseline=config_id(baseline),question=s['question'],
                    gold=s.get('answer',s.get('answers')),prediction=a['text'],baseline_prediction=b['text'],
                    score=a['score'],baseline_score=b['score'],delta=a['score']-b['score'],
                    harm=a['score']<b['score'],rescue=a['score']>b['score'],tokens=a['token_ids'],baseline_tokens=b['token_ids']))
    write_csv(OUT/'case_changes_vs_C1_C3.csv',rows)
    groups=collections.defaultdict(list)
    for r in rows: groups[r['dataset'],r['task'],r['answer_type'],r['condition'],r['baseline']].append(r)
    write_csv(OUT/'category_changes_vs_C1_C3.csv',[
        dict(dataset=k[0],task=k[1],answer_type=k[2],condition=k[3],baseline=k[4],n=len(v),
            delta_pp=100*np.mean([r['delta'] for r in v]),harm=sum(r['harm'] for r in v),rescue=sum(r['rescue'] for r in v))
        for k,v in sorted(groups.items())])


def tables_and_figures():
    frozen=read(OUT/'final_candidate_freeze.json'); primary=frozen['quality_primary']; secondary=frozen.get('speed_secondary')
    quality=csv_rows(OUT/'analysis'/frozen['final_quality_job']/'accuracy_summary.csv')
    perf=csv_rows(OUT/'analysis'/frozen['final_perf_job']/'latency_flops_kv.csv')
    paired=csv_rows(OUT/'analysis'/frozen['final_perf_job']/'performance_paired.csv')
    configs=[config('native'),config(),config(answer='both')]
    for c in (primary,secondary):
        if c is not None and c not in configs: configs.append(c)
    def label(c):
        tags=[]
        if c['writer']=='native': tags.append('Native')
        elif c==config(): tags.append('C1')
        elif c==config(answer='both'): tags.append('C3-V2')
        else: tags.append('Candidate')
        if c==primary: tags.append('Primary')
        if c==secondary: tags.append('Speed')
        return ' / '.join(tags)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    quality_rows=[]; latency_rows=[]
    for col,dataset in enumerate(('GQA','VQAv2')):
        x=np.arange(len(configs)); values=[]; lows=[]; highs=[]; speeds=[]
        for c in configs:
            q=next(r for r in quality if r['dataset']==dataset and r['config_id']==config_id(c))
            quality_rows.append(dict(label=label(c),**q))
            values.append(float(q['delta_original_native_pp'])); lows.append(float(q['original_delta_ci_low'])); highs.append(float(q['original_delta_ci_high']))
            h=dict(c,logits_to_keep=1); n=config('native',head=1)
            p=next(r for r in perf if r['dataset']==dataset and r['config_id']==config_id(h))
            speed=next(r for r in paired if r['dataset']==dataset and r['condition']==config_id(h) and r['baseline']==config_id(n))
            speeds.append(float(speed['all_e2e_speedup'])); latency_rows.append(dict(label=label(c),**p,**{k:v for k,v in speed.items() if k not in p}))
        ax=axes[0,col]; err=[np.array(values)-lows,np.array(highs)-values]
        ax.errorbar(x,values,yerr=err,fmt='o',capsize=4,color='#187d70'); ax.axhline(0,color='.4',lw=.8)
        ax.axhline(-1,color='#bf453b',ls='--',lw=1); ax.set_ylabel('Accuracy difference (pp, H0)'); ax.set_title(dataset)
        ax=axes[1,col]; ax.bar(x,speeds,color=['#777777' if c['writer']=='native' else '#187d70' if c==primary else '#c8962e' for c in configs],width=.6)
        ax.axhline(1.1,color='#bf453b',ls='--',lw=1); ax.set_ylabel('Measured E2E speedup (uniform H1)')
        for i,speed in enumerate(speeds): ax.text(i,speed+.02,f'{speed:.2f}x',ha='center',fontsize=9)
        ax.set_ylim(0,max(speeds)*1.2)
        for ax in axes[:,col]:
            ax.set_xticks(x,[label(c) for c in configs],rotation=20,ha='right',fontsize=9)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'final_quality_speed.png',dpi=240); fig.savefig(OUT/'final_quality_speed.pdf'); plt.close(fig)
    write_csv(OUT/'final_quality_table.csv',quality_rows); write_csv(OUT/'final_performance_table.csv',latency_rows)
    lines=['# ATE Accuracy Autoresearch: Six-Hour Budget','',
        '## Frozen Recommendation','',
        f"- Primary (selected using CHECK only): `{config_id(primary)}`.",
        f"- Speed secondary: `{config_id(secondary) if secondary else 'none'}`.",
        f"- Candidate frozen at: {frozen['at']}.",
        '- Only LLaVA-1.5-7B, fixed FP16/eager weights. P-1 and every Decode query are image-blind in every non-Native candidate.',
        '- All pre-P-1 Image and text queries retain L2-L5 Attention updates. No legacy I+A bypass.',
        '- H0 is the original all-returned-rows LM Head path; H1 is a separately generated, uniformly last-row-only engineering control. Accuracy is never spliced between them.','',
        '## FINAL Quality (H0)','',
        '| Dataset | Condition | N | Accuracy % | Delta Native pp [95% cluster CI] | Harm / rescue |',
        '|---|---|---:|---:|---|---|']
    for r in quality_rows:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['n']} | {float(r['accuracy']):.3f} | {float(r['delta_original_native_pp']):+.3f} [{float(r['original_delta_ci_low']):+.3f}, {float(r['original_delta_ci_high']):+.3f}] | {r['score_down_original_native']} / {r['score_up_original_native']} |")
    lines+=['','## Uniform H1 Performance','',
        'Paired same-GPU inputs, 3 warmups and 5 interleaved repeats; median per request, then question mean. Every request starts with a cold KV and recomputes Vision. Processor and disk I/O are excluded for all conditions. TTFT is the CUDA model phase (Vision + language Prefill), not a CPU-inclusive wall TTFT. E2E is synchronized request wall time.',
        '', '| Dataset | Condition | N | Prefill ms | E2E ms | Decode ms/token | E2E speedup | Matrix TFLOPs | KV live / capacity MiB | Peak allocated MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in latency_rows:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['n']} | {float(r['language_prefill_ms']):.3f} | {float(r['latency_ms']):.3f} | {float(r['decode_ms_per_token']):.3f} | {float(r['all_e2e_speedup']):.3f}x | {float(r['total_flops'])/1e12:.3f} | {float(r['kv_live_mib']):.2f} / {float(r['kv_capacity_mib']):.2f} | {float(r['peak_allocated'])/2**20:.1f} |")
    lines+=['','## Interpretation and Limits','',
        '- The search hypotheses, coarse/full DEV promotions and CHECK ranking are recorded before each job. FINAL was generated only after the candidate freeze, and did not change the selection.',
        '- GQA and VQAv2 are reported separately. VQAv2 uses the pinned official ten-answer consensus soft score, not exact match. VQA is not counted as a second benchmark.',
        '- The -1 pp target uses one-sided image-cluster bootstrap lower bounds with the frozen comparison-family correction. A confidence interval containing zero does not demonstrate no harm.',
        '- All cohorts are historical and repeatedly researched. This FINAL is held out only within this task, not a new external confirmation dataset.',
        '- Matrix FLOPs are independently counted from executed Linear/Conv/QK/AV shapes (2 FLOPs/MAC); nonlinear, norm, softmax and RoPE are excluded. Hardware TFLOPS and DRAM bandwidth are N/A. Dense writer masks do not create physical FLOP savings.',
        '- KV is persistent text KV, not renamed visual KV. Missing L2-L5 caches are legal only because all subsequent Attention queries in those layers are bypassed. The Decode-only variant still computes P-1 text Attention using transient full Prefill K/V.',
        '- Natural EOS output length and exactly matched complete-sequence subsets are reported in the performance paired CSV. Shorter output is not treated as evidence of a faster kernel.',
        '- Repetition is a repeated three-token n-gram heuristic, not a proof of every possible repetition pattern. Empty, EOS and truncation fields and raw tokens are retained.',
        '', '## Artifacts','',
        '- `results.tsv`: every evaluated candidate/round, including negative results.',
        '- `records/`: atomic, individually hashed question/configuration answers and input bindings.',
        '- `final_candidate_freeze.json`, `data_freeze.json`, `execution_freeze.json`: selection, data and runtime bindings.',
        '- `analysis/`: quality/paired intervals, task scores, harm cases and measured timing/FLOPs/KV.',
        '- `case_changes_vs_C1_C3.csv`: all new damage and corrections, not just successes.',
        '- `last_row_head_changes.csv`: exact logit and sequence changes caused by H1.',
        '- `completion.json`, worker Smoke files and release audit: execution validity and resource cleanup.','']
    (OUT/'final_report_zh.md').write_text('\n'.join(lines))
    print('Final tables and figure written; append Chinese research conclusions after reviewing frozen FINAL.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); a=ap.parse_args()
    if a.action=='audit': audit()
    elif a.action=='cases': cases()
    elif a.action=='tables': tables_and_figures()
