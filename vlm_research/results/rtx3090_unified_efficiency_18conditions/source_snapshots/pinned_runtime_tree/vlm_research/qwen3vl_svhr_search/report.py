"""CPU-only final interpretation. Never select or retune on held-out scores."""
import csv
import json
from collections import defaultdict

import numpy as np

from .freeze import OUT,FIRST,ROOT,TASKS,check
from ..qwen3vl_overnight.common import table,write,scorers
from ..qwen3vl_overnight.report import records as baseline_records
from ..qwen3vl_overnight.metrics import metric,paired,complete_keys


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value in ('',None,'None') else float(value)


def fmt(value):
    value=number(value)
    return 'N/A' if value is None else f'{value:.3f}'


def pareto(rows):
    grouped=defaultdict(dict)
    for row in rows:
        grouped[row['condition']][row['task']]=number(row['score'])
    valid={n:s for n,s in grouped.items() if len(s)==6 and all(x is not None for x in s.values())}
    return [n for n,s in valid.items() if not any(all(t[k]>=s[k] for k in TASKS) and any(t[k]>s[k] for k in TASKS)
        for m,t in valid.items() if m!=n)]


def preview(stage):
    rows=read(OUT/stage/'accuracy_summary.csv')
    grid=json.loads((OUT/'candidate_grid.json').read_text())
    names=list(dict.fromkeys(r['condition'] for r in rows))
    prefix=(FIRST/'report_zh.md').read_text().split('| SVH-R候选 |')[0].rstrip()
    lines=[prefix,'','| SVH-R候选 | 层 / K | GQA / SQA / TextVQA / POPE / MME / MMB | 类型 / KV / 速度 | 阶段 |',
        '|---|---|---|---|---|']
    for name in names:
        own={r['task']:r for r in rows if r['condition']==name}
        scores=' / '.join(fmt(own.get(t,{}).get('score')) for t in TASKS)
        lines.append(f"| {name} | {grid[name]['layers']} / {grid[name]['k']} | {scores} | FUNCTIONAL_REACCESS_ONLY；实际完整视觉KV；速度N/A | {stage} 完成 |")
    lines+=['','当前只是阶段性结果，不是最终候选或全量SVH-R分数。每个阶段样本不同，不能跨阶段直接比较绝对分数。',
        '相对冻结SVH、Native、功能性K0的差值与图片整簇95%CI、损害/纠正数量均在本阶段accuracy_summary.csv。',
        '严格数值门禁FAIL保留，MME双题与MMBench Circular完整组核验通过；缺失题不补错分。',
        '原始SVH不变；回看L1-L6与冻结跳算冲突的Early8、Middle24没有执行。',
        '正在按已授权顺序继续；不根据Held-out修改层集合，不训练、不推送、不修改main。']
    text='\n'.join(lines)+'\n'
    (OUT/stage/'report_zh.md').write_text(text)
    (ROOT/'overnight_report_zh.md').write_text(text)


def physical_summary(scope='Held-out'):
    folder=OUT/('Physical' if scope=='Held-out' else 'Physical-Full')
    if not folder.exists():
        return [],dict(status='NOT_STARTED')
    if (folder/'STOP_STRUCTURE.json').exists():
        return [],dict(status='BLOCKED_PHYSICAL_PORT',evidence=str(folder/'STOP_STRUCTURE.json'))
    from .run import stage_records
    rows=stage_records('Physical' if scope=='Held-out' else 'Physical-Full')
    if not rows:
        return [],dict(status='NO_VALIDATED_PHYSICAL_METHOD',functional_only=True)
    manifest=json.loads((OUT/(scope+'_manifest.json')).read_text())
    ids={r['key']:r for r in manifest}
    mapping=json.loads((FIRST/'shards.json').read_text())
    source,_=baseline_records()
    base=defaultdict(dict)
    for row in source:
        if row['key'] in ids and row['status']=='COMPLETE':
            base[row['key']][row['condition']]=row
    functional=defaultdict(dict)
    for row in stage_records('Held-out') if scope=='Held-out' else []:
        if row['status']=='COMPLETE':
            functional[row['key']][row['condition']]=row
    data=defaultdict(dict)
    scoring=scorers()
    for row in rows:
        if row['status']=='COMPLETE':
            assert row['input_binding_sha256']==ids[row['key']]['expected_input_binding_sha256']
            assert row['gold']==ids[row['key']]['gold'] and row['scoring']==scoring.score(ids[row['key']],row['text'])
            data[row['key']][row['condition']]=row
    output=[]
    for task in TASKS:
        pool=[r for r in manifest if r['benchmark']==task]
        for name in sorted({r['condition'] for r in rows}):
            available={r['key'] for r in pool if name in data[r['key']] and (scope=='Full' or name in functional[r['key']]) and len(base[r['key']])==2}
            valid=sorted(complete_keys(task,pool,available))
            own=[data[k][name] for k in valid]
            for reference in ['SVH','Native']+(['Functional'] if scope=='Held-out' else []):
                before=[functional[k][name] if reference=='Functional' else base[k][reference] for k in valid]
                ci=paired(task,before,own,mapping)
                output.append(dict(task=task,condition=name,reference=reference,n=len(valid),planned=len(pool),full_metric=len(valid)==len(pool),
                    score=metric(task,own,mapping),baseline=metric(task,before,mapping),**ci,
                    harm=sum(b['scoring']['score']<a['scoring']['score'] for a,b in zip(before,own)),
                    rescue=sum(b['scoring']['score']>a['scoring']['score'] for a,b in zip(before,own)),
                    token_changes=sum(a['token_ids']!=b['token_ids'] for a,b in zip(before,own)),
                    actual_image_KV_MiB=np.mean([r['kv']['persistent_image_bytes']/2**20 for r in own]) if own else None,
                    mean_extra_construction_ms=np.mean([r['late_visual_KV_construction']['extra_prefill_ms'] for r in own]) if own else None,
                    mode='PHYSICAL_SVHR_RELAXED_NUMERICAL_QUALITY_ONLY'))
    table(folder/'physical_accuracy.csv',output)
    table(folder/'per_question.csv',[{k:v for k,v in r.items() if k!='kv'} for r in rows])
    return output,dict(status='PHYSICAL_STRUCTURAL_VALIDATED; STRICT_STATUS_PER_SMOKE',records=len(rows),
        full_six_task_run=scope=='Full' and all(r['full_metric'] for r in output),quality_scope=scope,
        raw_status_counts={state:sum(r['status']==state for r in rows) for state in ['COMPLETE','SKIPPED_OOM','BLOCKED']})


def timing_summary():
    if (OUT/'Physical/STOP_STRUCTURE.json').exists():
        return []
    data=[]
    for shard in [0,1]:
        data+=read(OUT/'Physical'/f'timing_raw{shard}.csv')
    if not data:
        return []
    grouped=defaultdict(list)
    for row in data:
        grouped[(row['key'],row['condition'])].append(row)
    names=sorted({r['condition'] for r in data})
    valid={k for k,_ in grouped if all(len(grouped.get((k,n),[]))==5 for n in names)}
    per=defaultdict(dict)
    for (key,name),items in grouped.items():
        if key not in valid:
            continue
        tokens={r['token_ids'] for r in items}
        assert len(tokens)==1, ('NONDETERMINISTIC_TIMING_OUTPUT',key,name)
        per[key][name]=dict(tokens=next(iter(tokens)),**{f:np.mean([float(r[f]) for r in items]) for f in
            ['e2e_ms','ttft_ms','decode_ms_per_token','peak_allocated','live_KV_bytes','capacity_KV_bytes','extra_construction_ms','generated_tokens']})
    output=[]
    for name in names:
        for reference in ['Native','SVH']:
            for same in [False,True]:
                keys=[k for k in valid if not same or per[k][name]['tokens']==per[k][reference]['tokens']]
                if not keys:
                    continue
                result=dict(condition=name,reference=reference,n=len(keys),same_complete_sequence_only=same,
                    cap=32,independent_requests=True,warmups=3,repeats=5,includes_extra_KV_construction=True,
                    no_hardware_TFLOPS_claim=True)
                for field in ['e2e_ms','ttft_ms','decode_ms_per_token','peak_allocated','live_KV_bytes','capacity_KV_bytes','extra_construction_ms','generated_tokens']:
                    result[field]=float(np.mean([per[k][name][field] for k in keys]))
                reference_ms=float(np.mean([per[k][reference]['e2e_ms'] for k in keys]))
                result['e2e_speedup']=reference_ms/result['e2e_ms']
                output.append(result)
    table(OUT/'Physical/timing_summary.csv',output)
    return output


def figures(all_rows,grid):
    import os,subprocess
    from pathlib import Path
    # Reuse a pre-existing CPU plotting environment; do not install into Qwen's.
    python='/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
    subprocess.run([python,str(Path(__file__).with_name('plot.py')),str(OUT)],check=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(OUT/'figure_cache')))


def generate():
    check()
    grid=json.loads((OUT/'candidate_grid.json').read_text())
    selection=json.loads((OUT/'final_selection.json').read_text())
    all_rows=[]
    frontier=[]
    for stage in ['Discovery','Validation','Held-out']:
        rows=read(OUT/stage/'accuracy_summary.csv')
        all_rows += [dict(stage=stage,**r) for r in rows]
        frontier += [dict(stage=stage,condition=n) for n in pareto(rows)]
    table(OUT/'all_stage_accuracy.csv',all_rows)
    table(OUT/'pareto_frontier.csv',frontier)
    physical,physical_status=physical_summary()
    full_physical,full_status=physical_summary('Full')
    timing=timing_summary()
    write(OUT/'physical_status.json',dict(heldout=physical_status,full=full_status))
    figures(all_rows,grid)
    latest={}
    for row in all_rows:
        latest.setdefault(row['condition'],{})[row['task']]=row
    first=(FIRST/'report_zh.md').read_text()
    prefix=first.split('| SVH-R候选 |')[0]
    lines=[prefix.rstrip(),'','| SVH-R候选 | 层集合 / K | GQA / SQA / TextVQA / POPE / MME / MMB | ΔSVH / ΔNative（逐任务） | 类型 / 实际KV | 阶段 |',
        '|---|---|---|---|---|---|']
    for name,candidate in grid.items():
        if candidate['status']!='ELIGIBLE':
            lines.append(f"| {name} | {candidate['layers']} / {candidate['k']} | N/A | N/A | NOT_APPLICABLE_FROZEN_SKIP | 未执行 |")
            continue
        rows=latest.get(name,{})
        scores=' / '.join(fmt(rows.get(t,{}).get('score')) for t in TASKS)
        differences='; '.join(t+': '+fmt(rows.get(t,{}).get('delta_svh'))+'/'+fmt(rows.get(t,{}).get('delta_native')) for t in TASKS)
        kv=[number(r['mean_actual_KV_bytes']) for r in rows.values() if number(r['mean_actual_KV_bytes']) is not None]
        mean_kv=np.mean(kv)/2**20 if kv else None
        stages='/'.join(sorted({r['stage'] for r in rows.values()}))
        lines.append(f"| {name} | {candidate['layers']} / {candidate['k']} | {scores} | {differences} | FUNCTIONAL_REACCESS_ONLY / {fmt(mean_kv)} MiB真实全视觉KV / 速度N/A | {stages} |")
    for name in sorted({r['condition'] for r in full_physical}):
        own={r['task']:r for r in full_physical if r['condition']==name and r['reference']=='SVH'}
        scores=' / '.join(fmt(own.get(t,{}).get('score')) for t in TASKS)
        differences='; '.join(t+': '+fmt(own.get(t,{}).get('delta')) for t in TASKS)
        lines.append(f"| Physical {name} | {grid[name]['layers']} / {grid[name]['k']} | {scores} | ΔSVH: {differences} | 真实选定层KV；额外构造计费；STRICT状态独列 | 全量计划，覆盖见下表 |")
    lines += ['', '## 冻结选择与结论边界',
        f"- 最佳质量候选：{selection['quality']}；质量/K预算折中：{selection['tradeoff']}；预声明最少层对照：{selection['minimum_budget']}。这些名字在Held-out前冻结，不依据Held-out重新排名。",
        '- 层预算K仅是访问权限；功能性Dense仍保留36层完整视觉KV，不能把较小K称作真实KV节省。上述表格各行阶段不同，不能跨阶段的不同样本集直接比绝对分数。',
        '- 同一阶段所有功能候选采用相同共同成功、完整MME双题及MMBench Circular组；逐任务配对CI见各阶段accuracy_summary.csv。与功能性K0的差值隔离BF16物理路径差异，与冻结Physical SVH的差值另列。',
        '- 所有区间为图像整簇探索性区间，未作确认性多重检验。Held-out图片隔离且不参与本轮层选择，但这仍不是完全独立外部数据集。',
        '- TextVQA主表严格Paper-OCR。Image-only未开展，不能从主分数推断没有OCR时的行为。Middle8与K8同K可比较；更大中间窗口与K8同时改变层数，不能单独归因于连续性。',
        f"- 物理移植状态：{physical_status}；追加固定候选全量状态：{full_status}。原SVH源码、退出点、R权限、L1-L6答案跳算均没有被修改。",
        '- 物理原型从实际L22输入建立独立的图像结束位置之前因果前缀，继续必要的视觉计算来构造选定层K/V；不包含R/P−1，不二次运行Vision或完整Prefill。主文本路径仍在L22前删除视觉行。辅助视觉Hidden仅在Prefill临时存在，只有选定层视觉KV持久存在；额外计算已包含TTFT/E2E，仍可能更慢。',
        '- 不将功能Dense分数替代Physical分数；物理Held-out成绩、Strict FAIL保留、构造成本和同机cap32计时在Physical/下分别报告。只有Physical-Full全部计划进入终态且覆盖完整才可称为完整成绩；OOM缺失时仍是共同有效子集。',
        f"- 用于排序的任务：{json.loads((OUT/'protocol.json').read_text())['ranking_tasks']}。如果预冻结共同有效数据无法构成某项完整指标，只报告明确标注的缩减任务目标，不冒称六任务最优。",
        '- 最少几层足够恢复质量必须结合每项逐题CI，不用等权平均掩盖个别任务损害；Pareto前沿见pareto_frontier.csv。',
        '- 历史LLaVA textvqa500-early-middle-late-budget本地ref/目标路径未找到，没有补造证据。',
        f'- 文件位置：{OUT}；原始预测仅保存在本地SQLite/CSV，不上传公共仓库。',
        '- 两阶段无训练、无权重更新、无最佳层穷举。方案B不是作者官方复现，方案A保持BLOCKED。59例和24例严格全通过状态均未获得；原门禁没有放宽。',
        '', '## 物理Held-out结果', '| 任务 | 方法 | 题数/计划 | 分数 | ΔPhysical SVH [95%CI] | 实际Image KV MiB | 构造ms |',
        '|---|---|---|---|---|---|---|']
    for row in physical:
        if row['reference']=='SVH':
            lines.append(f"| {row['task']} | {row['condition']} | {row['n']}/{row['planned']} | {fmt(row['score'])} | {fmt(row['delta'])} [{fmt(row['low'])}, {fmt(row['high'])}] | {fmt(row['actual_image_KV_MiB'])} | {fmt(row['mean_extra_construction_ms'])} |")
    lines+=['','## 固定质量候选Physical全量（若已放行）',
        '| 任务 | 方法 | 配对有效/计划 | 分数 | ΔSVH [95%CI] | 完整指标 |', '|---|---|---|---|---|---|']
    for row in full_physical:
        if row['reference']=='SVH':
            lines.append(f"| {row['task']} | {row['condition']} | {row['n']}/{row['planned']} | {fmt(row['score'])} | {fmt(row['delta'])} [{fmt(row['low'])}, {fmt(row['high'])}] | {row['full_metric']} |")
    lines+=['','## 同机工程计时（cap32，不是更改质量协议）','| 方法 | 对照 | 有效输入 | 同完整序列子集 | E2E ms | TTFT ms | TPOT ms | 实测E2E比 |',
        '|---|---|---|---|---|---|---|---|']
    for row in timing:
        lines.append(f"| {row['condition']} | {row['reference']} | {row['n']} | {row['same_complete_sequence_only']} | {fmt(row['e2e_ms'])} | {fmt(row['ttft_ms'])} | {fmt(row['decode_ms_per_token'])} | {fmt(row['e2e_speedup'])} |")
    text='\n'.join(lines)+'\n'
    (OUT/'report_zh.md').write_text(text)
    (ROOT/'overnight_report_zh.md').write_text(text)
    from ..qwen3vl_svh_preflight48.run import resource
    release=resource()
    from ..qwen3vl_overnight.common import GPUS
    release['authorized_gpu_processes_remaining']=[u for u in GPUS if u in release['processes']]
    write(OUT/'gpu_release.json',release)
    from ..qwen3vl_overnight.archive import commit
    commit('SVH-R frozen held-out and physical attempt',OUT)
