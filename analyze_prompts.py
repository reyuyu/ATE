"""Paired image-cluster statistics, discovery-only selection and final reporting."""
import argparse
import statistics
from collections import Counter, defaultdict

import numpy as np

from prompt_common import *


def load_records(stage):
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    index={r['key']:r for r in manifest if r['stage']==stage}
    records=[]
    with readonly(OUT/'p0_reused.sqlite') as db:
        for blob,expected in db.execute('SELECT payload,sha256 FROM records'):
            raw=zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest()==expected
            rec=json.loads(raw)
            if rec['key'] not in index:
                continue
            row=index[rec['key']]
            s=scored(row,rec['text'],rec['truncated'])
            assert s['score']==rec['scoring']['score']
            rec.update(s,stage=stage,prompt='P0',source='STRICTLY_BOUND_HISTORICAL_P0',circular_group=row.get('circular_group'))
            rec.pop('input_binding',None)
            rec.pop('scoring',None)
            records.append(rec)
    for shard in [0,1]:
        with readonly(OUT/f'worker{shard}.sqlite') as db:
            for blob,expected in db.execute('SELECT payload,sha256 FROM records'):
                raw=zlib.decompress(blob)
                assert hashlib.sha256(raw).hexdigest()==expected
                rec=json.loads(raw)
                if rec['stage']==stage:
                    records.append(rec)
    keys=[(r['key'],r['prompt'],r['condition']) for r in records]
    assert len(keys)==len(set(keys))
    choice=None if stage=='Discovery' else json.loads((OUT/'selection_frozen.json').read_text())['selected']
    for row in index.values():
        prompts=conditions(row['benchmark']) if stage=='Discovery' else list(dict.fromkeys(['P0',choice[row['benchmark']]]))
        for prompt in prompts:
            for method in ['Native','SVH']:
                assert (row['key'],prompt,method) in keys, ('Missing',stage,row['key'],prompt,method)
    return records


def units(records, metric='score'):
    if not records:
        return {}
    if records[0]['benchmark']!='MMBench':
        return {r['key']:(r['image_sha256'],float(r[metric])) for r in records}
    groups=defaultdict(list)
    for r in records:
        groups[r['circular_group']].append(r)
    return {str(key):(rows[0]['image_sha256'],float(all(r[metric] for r in rows))) for key,rows in groups.items()}


def bootstrap(a,b=None):
    assert b is None or set(a)==set(b)
    clusters=defaultdict(list)
    for key,(image,value) in a.items():
        if b is not None:
            assert image==b[key][0]
            value-=b[key][1]
        clusters[image].append(value)
    vals=np.array([sum(clusters[k]) for k in sorted(clusters)])
    counts=np.array([len(clusters[k]) for k in sorted(clusters)])
    rng=np.random.default_rng(SEED)
    samples=[]
    for start in range(0,20000,1000):
        ix=rng.integers(0,len(vals),size=(min(1000,20000-start),len(vals)))
        samples.extend((vals[ix].sum(1)/counts[ix].sum(1)).tolist())
    return dict(estimate_pp=100*vals.sum()/counts.sum(),ci_low_pp=100*float(np.quantile(samples,.025)),
                ci_high_pp=100*float(np.quantile(samples,.975)),n_units=int(counts.sum()),image_clusters=len(counts))


def summary(records):
    results=[]
    for task in TASKS:
        for prompt in conditions(task):
            for method in ['Native','SVH']:
                pool=[r for r in records if r['benchmark']==task and r['prompt']==prompt and r['condition']==method]
                if not pool:
                    continue
                stats=bootstrap(units(pool))
                lengths=np.array([r['generated_tokens'] for r in pool])
                result=dict(benchmark=task,prompt=prompt,condition=method,n_rows=len(pool),
                    accuracy=stats.pop('estimate_pp'),**stats,mean_tokens=float(lengths.mean()),p50_tokens=float(np.median(lengths)),
                    p95_tokens=float(np.quantile(lengths,.95)),eos_rate=float(np.mean([r['eos'] for r in pool])),
                    truncated_rate=float(np.mean([r['truncated'] for r in pool])),parse_failure_rate=float(np.mean([r['parse_failed'] for r in pool])),
                    empty=sum(r['empty'] for r in pool),max_tokens=int(lengths.max()),
                    explanatory_heuristic_rate=float(np.mean([len(r['text'].strip().split())>6 or '\n' in r['text'].strip() for r in pool])),
                    timing_source='HISTORICAL_NOT_FRESH' if prompt=='P0' else 'NEW_NATURAL_EOS')
                if task=='ScienceQA-IMG':
                    result['diagnostic_accuracy']=100*np.mean([r['diagnostic_score'] for r in pool])
                    result['content_correct_format_failed']=sum(r['format_only_failure'] for r in pool)
                results.append(result)
    return results


def select():
    assert not (OUT/'selection_frozen.json').exists()
    recs=load_records('Discovery')
    scores=summary(recs)
    table(OUT/'discovery_scores.csv',scores)
    winners={}
    ranking={}
    for task in TASKS:
        baseline={r['condition']:r['accuracy'] for r in scores if r['benchmark']==task and r['prompt']=='P0'}
        candidates=[]
        for prompt in conditions(task):
            rows=[r for r in scores if r['benchmark']==task and r['prompt']==prompt]
            candidates.append(dict(prompt=prompt,mean_accuracy=float(np.mean([r['accuracy'] for r in rows])),
                worst_delta=min(r['accuracy']-baseline[r['condition']] for r in rows),mean_tokens=float(np.mean([r['mean_tokens'] for r in rows]))))
        candidates.sort(key=lambda r:(-round(r['mean_accuracy'],10),-round(r['worst_delta'],10),r['mean_tokens'],r['prompt']))
        winners[task]=candidates[0]['prompt']
        ranking[task]=candidates
    write(OUT/'selection_frozen.json',dict(selected=winners,ranking=ranking,
        selection_rule=json.loads((OUT/'protocol.json').read_text())['discovery_selection'],
        discovery_scores_sha256=sha(OUT/'discovery_scores.csv'),validation_candidate_scores_accessed=False))
    print('FROZEN_SELECTION',json.dumps(winners),flush=True)


def paired(records,stage):
    output=[]
    for task in TASKS:
        prompts=sorted({r['prompt'] for r in records if r['benchmark']==task})
        for prompt in prompts:
            native=[r for r in records if r['benchmark']==task and r['prompt']==prompt and r['condition']=='Native']
            svh=[r for r in records if r['benchmark']==task and r['prompt']==prompt and r['condition']=='SVH']
            pairs=[(f'{prompt}:SVH-Native',svh,native)]
            if prompt!='P0':
                for method in ['Native','SVH']:
                    cur=svh if method=='SVH' else native
                    base=[r for r in records if r['benchmark']==task and r['prompt']=='P0' and r['condition']==method]
                    pairs.append((f'{method}:{prompt}-P0',cur,base))
            for label,a,b in pairs:
                ua,ub=units(a),units(b)
                stats=bootstrap(ua,ub)
                values=[(ua[k][1],ub[k][1]) for k in ua]
                output.append(dict(stage=stage,benchmark=task,comparison=label,metric='CIRCULAR_ACCURACY' if task=='MMBench' else 'FROZEN_STRICT_ACCURACY',
                    **stats,harmed=sum(x<y for x,y in values),rescued=sum(x>y for x,y in values),
                    both_correct=sum(x==y==1 for x,y in values),both_wrong=sum(x==y==0 for x,y in values),
                    exploratory=True,unit='Circular group' if task=='MMBench' else 'question'))
                if task=='ScienceQA-IMG':
                    output.append(dict(stage=stage,benchmark=task,comparison=label,metric='CONSERVATIVE_DIAGNOSTIC',
                        **bootstrap(units(a,'diagnostic_score'),units(b,'diagnostic_score')),exploratory=True))
    return output


def performance():
    records=[]
    for shard in [0,1]:
        with (OUT/f'performance_raw{shard}.csv').open() as stream:
            records.extend(csv.DictReader(stream))
    output=[]
    for task in TASKS:
        for mode in ['natural_EOS','fixed32']:
            for prompt in conditions(task):
                for method in ['Native','SVH']:
                    pool=[r for r in records if r['benchmark']==task and r['mode']==mode and r['prompt']==prompt and r['condition']==method]
                    baseline=[r for r in records if r['benchmark']==task and r['mode']==mode and r['prompt']=='P0' and r['condition']==method]
                    assert len(pool)==len(baseline)==6
                    fields=['vision_ms','prefill_ms','ttft_ms','decode_ms_per_token','e2e_ms','generated_tokens']
                    means={k:float(np.mean([float(r[k]) for r in pool])) for k in fields}
                    mean_e2e=np.mean([float(r['e2e_ms']) for r in pool])
                    p0_e2e=np.mean([float(r['e2e_ms']) for r in baseline])
                    sameprompt_native=[r for r in records if r['benchmark']==task and r['mode']==mode and r['prompt']==prompt and r['condition']=='Native']
                    native_e2e=np.mean([float(r['e2e_ms']) for r in sameprompt_native])
                    output.append(dict(benchmark=task,mode=mode,prompt=prompt,condition=method,n_images=2,repeats_per_image=3,**means,
                        tokens_per_s=1000*sum(float(r['generated_tokens']) for r in pool)/sum(float(r['e2e_ms']) for r in pool),
                        e2e_speedup_vs_same_method_P0=float(p0_e2e/mean_e2e),e2e_reduction_pct=float(100*(1-mean_e2e/p0_e2e)),
                        native_over_svh_same_prompt=float(native_e2e/mean_e2e),peak_allocated_mib=max(float(r['peak_allocated']) for r in pool)/2**20))
    table(OUT/'performance_comparison.csv',output)
    return output


def finish():
    selected=json.loads((OUT/'selection_frozen.json').read_text())['selected']
    discovery=load_records('Discovery')
    validation=load_records('Validation')
    ds,vs=summary(discovery),summary(validation)
    table(OUT/'discovery_scores.csv',ds)
    table(OUT/'validation_scores.csv',vs)
    table(OUT/'output_length_comparison.csv',[dict(stage=stage,**r) for stage,rows in [('Discovery',ds),('Validation',vs)] for r in rows])
    comparisons=paired(discovery,'Discovery')+paired(validation,'Validation')
    table(OUT/'paired_comparisons.csv',comparisons)
    negative=[]
    for stage,records in [('Discovery',discovery),('Validation',validation)]:
        by_key=defaultdict(dict)
        for r in records:
            by_key[r['key']][r['prompt'],r['condition']]=r
        for key,pool in by_key.items():
            for (prompt,method),r in pool.items():
                p0=pool['P0',method]
                native=pool[prompt,'Native']
                if (r['score']<p0['score'] or r['score']<native['score'] or r['truncated'] or r['parse_failed']):
                    negative.append(dict(stage=stage,key=key,benchmark=r['benchmark'],prompt=prompt,condition=method,
                        gold=r['gold'],answer=r['text'],score=r['score'],p0_answer=p0['text'],p0_score=p0['score'],
                        native_same_prompt_answer=native['text'],native_same_prompt_score=native['score'],
                        parse_failed=r['parse_failed'],truncated=r['truncated'],image_sha256=r['image_sha256']))
    table(OUT/'negative_cases.csv',negative)
    flat=[]
    keep=['stage','key','benchmark','prompt','condition','image_sha256','circular_group','gold','text','token_ids','score',
        'diagnostic_score','format_only_failure','parse_failed','eos','truncated','empty','generated_tokens','ttft_ms','decode_ms_per_token','e2e_ms',
        'input_binding_sha256','source','gpu_uuid']
    for r in discovery+validation:
        flat.append({k:r.get(k) for k in keep})
    table(OUT/'per_question.csv',flat)
    perf=performance()
    report=['# Qwen3-VL短答案Prompt独立实验','',
        '**独立协议：QWEN3VL_SHORT_ANSWER_PROMPT_ABLATION。方案B，非作者官方复现。SVH算法、Processor、BF16/Eager、视觉预算及评分器未变。原始ATE结果未覆盖。**','',
        '## 1. 当前Prompt',
        'GQA已有 `Answer the question using a single word or phrase.`；ScienceQA/MMBench只有 `Please select the correct answer from the options above. `，并未要求仅答字母。15个真实完整模板和Token绑定见 current_prompt_audit_zh.md / prompt_examples.jsonl。','',
        '## 2. Validation结果',
        '以下只报告预先隔离的Validation。每任务Prompt由Discovery冻结，未用Validation调参。MMBench为150个完整Circular组（全部旋转必须正确），不是逐行准确率。','',
        '|任务|Prompt|方法|严格分数%|增强诊断%（仅SQA）|均值/P50/P95 Token|解析失败%|截断%|',
        '|---|---|---|---:|---:|---|---:|---:|']
    for r in vs:
        report.append(f"|{r['benchmark']}|{r['prompt']}|{r['condition']}|{r['accuracy']:.2f}|{r.get('diagnostic_accuracy',float('nan')):.2f}|{r['mean_tokens']:.2f}/{r['p50_tokens']:.0f}/{r['p95_tokens']:.1f}|{100*r['parse_failure_rate']:.2f}|{100*r['truncated_rate']:.2f}|")
    report += ['', '## 3. 同Prompt下SVH−Native及Prompt收益', '', '|任务|比较|指标|变化pp|图片簇95%CI|损害/纠正|', '|---|---|---|---:|---|---|']
    for r in comparisons:
        if r['stage']!='Validation':
            continue
        report.append(f"|{r['benchmark']}|{r['comparison']}|{r['metric']}|{r['estimate_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|{r.get('harmed','N/A')}/{r.get('rescued','N/A')}|")
    report += ['', '## 4. 每任务冻结选择与解释','']
    for task in TASKS:
        prompt=selected[task]
        pools={(r['prompt'],r['condition']):r for r in vs if r['benchmark']==task}
        b_n,b_s=pools['P0','Native'],pools['P0','SVH']
        n,s=pools[prompt,'Native'],pools[prompt,'SVH']
        gap0=b_s['accuracy']-b_n['accuracy']
        gap=s['accuracy']-n['accuracy']
        report += [f"- {task}冻结选择{prompt}。Native严格分数变化{n['accuracy']-b_n['accuracy']:+.2f}pp，SVH变化{s['accuracy']-b_s['accuracy']:+.2f}pp。同Prompt差距由{gap0:+.2f}pp变为{gap:+.2f}pp（差距变化{gap-gap0:+.2f}pp）。Native/SVH平均长度分别由{b_n['mean_tokens']:.2f}/{b_s['mean_tokens']:.2f}变为{n['mean_tokens']:.2f}/{s['mean_tokens']:.2f}。"]
        if task=='ScienceQA-IMG':
            report += [f"  增强解析诊断：Native {b_n['diagnostic_accuracy']:.2f}→{n['diagnostic_accuracy']:.2f}%，SVH {b_s['diagnostic_accuracy']:.2f}→{s['diagnostic_accuracy']:.2f}%。截断输出仍判不确定；增强分数不替代严格正式分数。"]
    report += ['', 'Prompt带来的格式/长度收益属于通用输出协议改善，不代表SVH视觉交接能力本身提高。即使差距变小，也要检查Native是否同时受损；CI跨0不等于无损或等价。本轮不做非劣效认定。', '',
        '## 5. 同卡配对真实性能',
        '每任务2个预冻结Discovery输入（每卡1个），每条件1次预热、3次交错正式重复，冷KV、每请求重算Vision。以下为小样本工程诊断，不代表全部150题平均。自然EOS保留原cap；fixed32只在测速中忽略EOS，恰好生成32 Token，不参与质量评分。P0全体长度来自严格绑定历史结果，性能表P0为本轮重新测量。', '',
        '|任务|Prompt|方法|模式|TTFT ms|TPOT ms|E2E ms|相对同方法P0加速|同Prompt Native/SVH|',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in perf:
        if r['prompt'] not in ['P0',selected[r['benchmark']]]:
            continue
        report.append(f"|{r['benchmark']}|{r['prompt']}|{r['condition']}|{r['mode']}|{r['ttft_ms']:.1f}|{r['decode_ms_per_token']:.2f}|{r['e2e_ms']:.1f}|{r['e2e_speedup_vs_same_method_P0']:.3f}|{r['native_over_svh_same_prompt']:.3f}|")
    report += ['', '自然EOS时间下降不能当作SVH算子加速；fixed32比较剥离生成长度差异。完整字段含输出吞吐、峰值显存、Vision/Prefill，见 performance_comparison.csv。', '',
        '## 6. 有效性与限制',
        '按Image SHA整簇配对Bootstrap 20,000次；MMBench先计算完整Circular组再按图像簇重采样。所有区间均为探索性，未宣称多重比较确认性发现。三任务Discovery/Validation全局图片SHA隔离。P0复用逐记录验证Prompt、Token IDs、输入Tensor、Gold、模型Revision、cap与协议哈希。',
        '历史严格Physical数值验收仍为 FAIL_RETAINED、FULL_59未完成。新GPU Smoke仅验证Native/No-op严格相等、SVH重复运行严格相等、输入/Mask/位置/KV/MLP/Hook结构守恒；不将其冒充历史Dense/Physical严格门禁通过。',
        '本轮TextVQA无新增GPU生成，保留之前CPU审计结论。main、历史源码、原始预测、Checkpoint、评分器均只读；未自动推送。',
        'GPU资源释放见 gpu_release_audit.json。原始逐题文件 per_question.csv、worker0/1.sqlite、bindings0/1.sqlite；完整P0绑定 p0_reused.sqlite。', '']
    OUT.joinpath('report_zh.md').write_text('\n'.join(report))
    write(OUT/'completion.json',dict(status='COMPLETE',selected=selected,discovery_rows=len(discovery),validation_rows=len(validation),
        source_hashes_unchanged=bool(check_sources()),original_numerical_status='FAIL_RETAINED',new_prompt_protocol_only=True,
        validation_opened_once_after_selection=True,full_six_benchmark_re_evaluation_started=False))
    print('COMPLETE',len(discovery),len(validation),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['select','finish'])
    args=parser.parse_args()
    select() if args.action=='select' else finish()
