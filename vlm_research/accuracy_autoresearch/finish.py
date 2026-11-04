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
    assert read(ROOT/'task_deadline.json')==execution['clock'], 'The unified task clock changed'
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
        if parts['image_split'][s['image_sha256']]=='FINAL':
            assert datetime.datetime.fromisoformat(r['at'])>datetime.datetime.fromisoformat(frozen['at'])
    final_keys=keys_for('FINAL'); final_expected=len(final_keys)*len(frozen['evaluation_configs'])
    for k in final_keys:
        for c in frozen['evaluation_configs']:
            assert read_record(c,k,pool[k]) is not None,(k,config_id(c))
    for i in range(len(execution['devices'])):
        assert read(OUT/f'worker{i}/smoke.json')['status']=='PASS'
        assert not (OUT/f'worker{i}/STOP.json').exists()
        for job_name in (frozen['final_quality_job'],frozen['final_perf_job']):
            done=read(OUT/f'worker{i}/jobs/{job_name}.json')
            assert done['status']=='COMPLETE' and done['job_sha256']==digest(read(OUT/'jobs'/(job_name+'.json')))
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
    smokes=[a for i in range(len(execution['devices'])) for a in read(OUT/f'worker{i}/smoke.json')['audits']]
    assert all(a['numerical']['passed'] and a['native_noop_exact_both_head_shapes'] for a in smokes)
    assert all(a['numerical']['tolerances']==TOLERANCES['llava'] for a in smokes)
    assert all(e['blocked_probability_max']==0 for a in smokes for e in a['events'])
    assert all(r['passed'] for a in smokes for r in a['preservation'])
    perf_job=read(OUT/'jobs'/(frozen['final_perf_job']+'.json'))
    perf_configs={config_id(c) for c in perf_job['configs']}
    perf_measurements=0
    for k in perf_job['samples']:
        p=read(OUT/'performance'/perf_job['id']/(k+'.json'))
        assert p['sample']==k and p['sample_sha256']==digest(pool[k])
        assert p['sources']==execution['sources'] and p['configs']==perf_job['configs']
        assert len(p['measurements'])==5*len(perf_configs)
        assert {(r['config_id'],r['rep']) for r in p['measurements']}=={
            (cid,rep) for cid in perf_configs for rep in range(5)}
        for r in p['measurements']:
            c=next(c for c in perf_job['configs'] if config_id(c)==r['config_id'])
            assert r['token_ids']==read_record(c,k,pool[k])['generation']['token_ids']
            assert all(math.isfinite(r[f]) and r[f]>0 for f in
                       ('vision_ms','language_prefill_ms','ttft_ms','latency_ms','gpu_model_ms'))
            assert r['kv']['storage_bytes']>=r['kv']['live_bytes']>0
        assert {r['config_id'] for r in p['flops']}==perf_configs
        assert len(p['flops'])==len(perf_configs)
        assert all(r['estimated_flops']>=0 for c in p['flops'] for r in c['rows'])
        perf_measurements+=len(p['measurements'])
    atomic(OUT/'performance_completeness.json',dict(status='PASS',at=now(),
        inputs=len(perf_job['samples']),conditions=len(perf_configs),
        measured_requests=perf_measurements,repeats=5,warmup=3,
        measured_sequences_match_independent_quality=True,all_sources_and_input_bindings_match=True))
    atomic(OUT/'validation.json',dict(status='PASS',at=now(),cpu_unique_tests=12,real_GPU_smoke_inputs=len(smokes),
        native_noop_exact_all_returned_logits_H0_H1=True,
        max_physical_dense_reference_logits_error=max(a['numerical']['max_abs_logits'] for a in smokes),
        forbidden_probability_max=0,pre_Pminus1_Attention_updates_preserved=True,
        full_MLP_preserved=True,original_positions_and_text_KV_checked=True,
        no_runtime_source_change=True,unified_clock_unchanged=True,
        all_image_bytes_and_weights_rechecked=True,all_retained_checkpoint_hashes_checked=True,
        final_only_after_selection=True,no_worker_STOP=True,
        probability_and_operator_checks='Real GPU Smoke plus immutable execution paths, not full Attention matrices for every quality sample',
        noninferiority='Only frozen selected configurations are predeclared; DEV/CHECK and baseline-only CI rows are exploratory'))
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


def full_cohort():
    frozen=read(OUT/'final_candidate_freeze.json'); pool=read(OUT/'sample_manifest.json')
    configs=[config('native'),config(),config(answer='both')]
    for c in frozen['selected_candidates']:
        if c not in configs: configs.append(c)
    rows=[]
    for dataset in ('GQA','VQAv2'):
        keys=[k for k,s in pool.items() if s['dataset']==dataset]
        for c in configs:
            pairs=[(read_record(c,k,pool[k]),read_record(config('native'),k,pool[k])) for k in keys]
            assert all(a is not None and b is not None for a,b in pairs)
            a=np.array([a['generation']['score'] for a,b in pairs]); b=np.array([b['generation']['score'] for a,b in pairs])
            rows.append(dict(dataset=dataset,config_id=config_id(c),n=len(keys),accuracy=100*a.mean(),
                delta_native_pp=100*(a-b).mean(),harm=int(sum(a<b)),rescue=int(sum(a>b)),
                interpretation='Supportive only: includes DEV/CHECK used for selection; not FINAL confirmation'))
    write_csv(OUT/'supportive_full_cohort.csv',rows)
    print('Full historical cohort supportive table written; not used for reselection')


def accounting():
    clock=read(ROOT/'task_deadline.json'); release=read(OUT/'gpu_release_audit.json')
    complete=read(OUT/'completion.json'); frozen=read(OUT/'final_candidate_freeze.json')
    assert complete['status']=='COMPLETE' and read(OUT/'validation.json')['status']=='PASS'
    assert release['own_CUDA_processes_absent'] and (OUT/'results.tsv').exists()
    start=datetime.datetime.fromisoformat(clock['started_at'])
    stamp=datetime.datetime.fromisoformat(now())
    gpu_end=datetime.datetime.fromisoformat(release['at'])
    atomic(OUT/'task_accounting.json',dict(at=stamp.isoformat(),unified_clock=clock,
        minutes_to_GPU_release=(gpu_end-start).total_seconds()/60,
        minutes_to_delivery_preflight=(stamp-start).total_seconds()/60,
        candidate_frozen_minutes=(datetime.datetime.fromisoformat(frozen['at'])-start).total_seconds()/60,
        retained_unique_quality_records=sum(complete['counts'].values()),final_quality_records=complete['final_records'],
        performance=read(OUT/'performance_completeness.json'),
        status='CPU_COMPLETE_GPU_RELEASED_PENDING_GIT_PUSH',
        jobs=[dict(worker=i,job=p.stem,**read(p)) for i in range(2)
              for p in sorted((OUT/f'worker{i}/jobs').glob('*.json'))],
        no_additional_GPU_work=True,no_shared_environment_or_main_changes=True,
        task_stops_early_reason='Completed frozen selection and FINAL; quality target not met. No credible authorized next direction without reusing FINAL for tuning.'))
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*'))
               if p.is_file() and p.name!='artifact_hashes.json'}
    source_inventory={str(p.relative_to(ROOT)):sha(p)
                      for p in sorted((ROOT/'vlm_research/accuracy_autoresearch').glob('*.py'))}
    atomic(OUT/'artifact_hashes.json',dict(created_at=now(),artifacts=inventory,
        final_CPU_and_GPU_source_files=source_inventory,
        immutable_GPU_execution_files=source_hashes()))
    print('Delivery preflight and',len(inventory),'artifact SHA256 bindings written')


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
    quality_rows=[]; engineering_quality=[]; latency_rows=[]
    for col,dataset in enumerate(('GQA','VQAv2')):
        x=np.arange(len(configs)); values=[]; lows=[]; highs=[]; speeds=[]
        for c in configs:
            q=next(r for r in quality if r['dataset']==dataset and r['config_id']==config_id(c))
            quality_rows.append(dict(label=label(c),**q))
            h=dict(c,logits_to_keep=1); n=config('native',head=1)
            hq=next(r for r in quality if r['dataset']==dataset and r['config_id']==config_id(h))
            engineering_quality.append(dict(label=label(c),**hq))
            values.append(float(hq['delta_original_native_pp'])); lows.append(float(hq['original_delta_ci_low'])); highs.append(float(hq['original_delta_ci_high']))
            p=next(r for r in perf if r['dataset']==dataset and r['config_id']==config_id(h))
            speed=next(r for r in paired if r['dataset']==dataset and r['condition']==config_id(h) and r['baseline']==config_id(n))
            speeds.append(float(speed['all_e2e_speedup'])); latency_rows.append(dict(label=label(c),**p,**{k:v for k,v in speed.items() if k not in p}))
        ax=axes[0,col]; err=[np.array(values)-lows,np.array(highs)-values]
        ax.errorbar(x,values,yerr=err,fmt='o',capsize=4,color='#187d70'); ax.axhline(0,color='.4',lw=.8)
        ax.axhline(-1,color='#bf453b',ls='--',lw=1); ax.set_ylabel('Accuracy difference (pp, uniform H1)'); ax.set_title(dataset)
        ax=axes[1,col]; ax.bar(x,speeds,color=['#777777' if c['writer']=='native' else '#187d70' if c==primary else '#c8962e' for c in configs],width=.6)
        ax.axhline(1.1,color='#bf453b',ls='--',lw=1); ax.set_ylabel('Measured E2E speedup (uniform H1)')
        for i,speed in enumerate(speeds): ax.text(i,speed+.02,f'{speed:.2f}x',ha='center',fontsize=9)
        ax.set_ylim(0,max(speeds)*1.2)
        for ax in axes[:,col]:
            ax.set_xticks(x,[label(c) for c in configs],rotation=20,ha='right',fontsize=9)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'final_quality_speed.png',dpi=240); fig.savefig(OUT/'final_quality_speed.pdf'); plt.close(fig)
    write_csv(OUT/'final_quality_table.csv',quality_rows); write_csv(OUT/'final_performance_table.csv',latency_rows)
    write_csv(OUT/'engineering_quality_table.csv',engineering_quality)
    lines=['# ATE 六小时自主研究：最终报告','',
        '## 冻结方案','',
        f"- 主候选只依据CHECK选择：`{config_id(primary)}`。",
        f"- 速度优先备选：`{config_id(secondary) if secondary else '无'}`。",
        f"- 冻结时间：{frozen['at']}，早于FINAL生成及最晚4小时冻结线。",
        '- 仅使用LLaVA-1.5-7B固定权重、FP16/eager；全部非Native候选的原始P−1和所有真实Decode均禁止直接读取Image Key。',
        '- L2–L5所有P−1之前的Image及文本Query保留Attention更新，没有回退到旧I+A跳算。',
        '- H0为历史全返回行LM Head路径；H1为独立重新生成的统一last-row-only工程对照。两者的Accuracy和速度不会拼接。','',
        '## FINAL质量：原始H0','',
        'GQA使用原冻结评分器；VQAv2使用官方十答案共识软评分。VQAv2的损害/纠正指软分下降/上升，不冒充二元错误翻转。区间为20,000次图片SHA256整簇、问题等权的普通配对95%区间。',
        '', '| 数据集 | 条件 | 题数 | Accuracy % | Δ原始Native pp [95%图片簇CI] | 损害 / 纠正 |',
        '|---|---|---:|---:|---|---|']
    for r in quality_rows:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['n']} | {float(r['accuracy']):.3f} | {float(r['delta_original_native_pp']):+.3f} [{float(r['original_delta_ci_low']):+.3f}, {float(r['original_delta_ci_high']):+.3f}] | {r['score_down_original_native']} / {r['score_up_original_native']} |")
    lines+=['','## 统一H1：独立质量绑定','',
        '以下为H1自己的完整生成结果，并非把H0 Accuracy套在H1时延上。主图上、下两行都使用统一H1绑定。H0/H1首步Logits或完整序列的变化全部保存在`last_row_head_changes.csv`。',
        '', '| 数据集 | 条件 | 题数 | H1 Accuracy % | ΔH1 Native pp [95%图片簇CI] |',
        '|---|---|---:|---:|---|']
    for r in engineering_quality:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['n']} | {float(r['accuracy']):.3f} | {float(r['delta_original_native_pp']):+.3f} [{float(r['original_delta_ci_low']):+.3f}, {float(r['original_delta_ci_high']):+.3f}] |")
    lines+=['','## 预声明−1pp非劣效目标','',
        f"比较族大小为{frozen['noninferiority_family_size']}：冻结候选×两数据集×H0/H1；采用单侧Bonferroni校正图片簇Bootstrap下界。只有下界大于−1pp才支持该口径非劣效。其他基线、DEV和CHECK区间均为探索性，不因CI跨0宣布无损。",
        '', '| 数据集 | 冻结候选绑定 | ΔNative pp | 校正单侧下界 pp | 支持−1pp非劣效 |',
        '|---|---|---:|---:|---|']
    for r in quality:
        if r['ni_comparison_predeclared']=='True':
            lines.append(f"| {r['dataset']} | {r['config_id']} | {float(r['delta_original_native_pp']):+.3f} | {float(r['ni_one_sided_lower']):+.3f} | {'是' if r['ni_minus1_supported']=='True' else '否'} |")
    lines+=['','## 统一H1：真实性能','',
        '每数据集20张按输入哈希固定的不同图片，同GPU配对，3次预热、5次条件交错重复；先取单请求中位数，再对题目求均值。每请求冷KV、重新计算Vision。所有条件统一排除Processor、文件I/O及输入位置定位；E2E为同步的推理引擎请求墙钟。TTFT为Vision＋语言Prefill的CUDA模型阶段时间，不冒称含CPU的首Token墙钟。',
        '', '| 数据集 | 条件 | 题数 | Prefill ms | E2E ms | Decode ms/token | E2E加速 | 矩阵TFLOPs | KV Live / Capacity MiB | Peak Allocated MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in latency_rows:
        lines.append(f"| {r['dataset']} | {r['label']} | {r['n']} | {float(r['language_prefill_ms']):.3f} | {float(r['latency_ms']):.3f} | {float(r['decode_ms_per_token']):.3f} | {float(r['all_e2e_speedup']):.3f}x | {float(r['total_flops'])/1e12:.3f} | {float(r['kv_live_mib']):.2f} / {float(r['kv_capacity_mib']):.2f} | {float(r['peak_allocated'])/2**20:.1f} |")
    lines+=['','## 解释边界','',
        '- 每轮假设、配置和晋级理由均在运行前保存；FINAL只检验冻结选择，不回流选层、调规则或重新命名赢家。',
        '- GQA和VQAv2分别报告，不能用前者改善抵消后者下降；不把VQA与VQAv2当成两个独立Benchmark。',
        '- 所有样本来自反复研究过的历史队列。FINAL只是本轮图片隔离留出，不是独立外部确认。',
        '- FLOPs由实际Linear/Conv/QK/AV执行形状独立计数，2 FLOPs/MAC，不含Norm、Softmax、RoPE等非矩阵运算；硬件TFLOPS及DRAM带宽未采集，标记N/A。Dense写入Mask不等于实际省算。',
        '- 持久KV是真实文本位置的K/V，不是视觉KV改名。L2–L5因全部后续Decode都跳过Attention，才可省去这四层持久缓存；Prefill中P−1文本Attention使用真实的临时全量K/V。',
        '- 同完整Token序列子集的时延、自然EOS输出长度均另表保存，不把提前EOS当作Kernel加速。',
        '- 重复检测采用三Token n-gram启发式，不等于覆盖所有重复形式；保留原Token、EOS、截断及空答案。',
        '', '## 产物','',
        '- `results.tsv`：全部候选和轮次，保留负结果。',
        '- `records/`：逐题逐配置原子Checkpoint、答案、Token及输入绑定。',
        '- `final_candidate_freeze.json`、`data_freeze.json`、`execution_freeze.json`：选择、数据与源码冻结。',
        '- `analysis/`：配对区间、任务得分、损害案例，以及真实计时/FLOPs/KV。',
        '- `case_changes_vs_C1_C3.csv`：对C1/C3的全部损害与纠正，不只成功案例。',
        '- `last_row_head_changes.csv`：H1造成的精确Logits与序列变化。',
        '- `completion.json`、`validation.json`、Smoke及释放审计：完整性、正确性及资源状态。','']
    (OUT/'final_report_zh.md').write_text('\n'.join(lines))
    print('Final tables and figure written; append Chinese research conclusions after reviewing frozen FINAL.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); a=ap.parse_args()
    if a.action=='audit': audit()
    elif a.action=='cases': cases()
    elif a.action=='full-cohort': full_cohort()
    elif a.action=='accounting': accounting()
    elif a.action=='tables': tables_and_figures()
