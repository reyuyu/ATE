"""Additional CPU-only paired gap/cell reporting; no candidate selection."""
from collections import defaultdict

from analyze_prompts import bootstrap, load_records, units
from prompt_common import *


def main():
    selected=json.loads((OUT/'selection_frozen.json').read_text())['selected']
    gaps=[]
    cells=[]
    quality_timing=[]
    for stage in ['Discovery','Validation']:
        records=load_records(stage)
        for task in TASKS:
            prompts=sorted({r['prompt'] for r in records if r['benchmark']==task})
            indexed={(r['key'],r['prompt'],r['condition']):r for r in records if r['benchmark']==task}
            keys=sorted({k for k,_,_ in indexed})
            for prompt in prompts:
                pairs=[(indexed[k,prompt,'Native'],indexed[k,prompt,'SVH']) for k in keys]
                for method in ['Native','SVH']:
                    pool=[indexed[k,prompt,method] for k in keys]
                    fields=['ttft_ms','decode_ms_per_token','e2e_ms','vision_ms','prefill_ms','generated_tokens']
                    quality_timing.append(dict(stage=stage,benchmark=task,prompt=prompt,condition=method,n_rows=len(pool),
                        source='HISTORICAL_QUALITY_LOGS_NOT_FRESH_TIMING' if prompt=='P0' else 'NEW_QUALITY_RUNS',
                        **{k:sum(r[k] for r in pool)/len(pool) for k in fields},
                        peak_allocated_mib=max(r['peak_allocated'] for r in pool)/2**20))
                cells.append(dict(stage=stage,benchmark=task,prompt=prompt,n_rows=len(pairs),
                    native_correct_svh_wrong=sum(a['score']==1 and b['score']==0 for a,b in pairs),
                    native_wrong_svh_correct=sum(a['score']==0 and b['score']==1 for a,b in pairs),
                    both_correct=sum(a['score']==b['score']==1 for a,b in pairs),
                    both_wrong=sum(a['score']==b['score']==0 for a,b in pairs),
                    native_content_correct_format_failed=sum(a.get('format_only_failure',False) for a,b in pairs),
                    svh_content_correct_format_failed=sum(b.get('format_only_failure',False) for a,b in pairs)))
                if prompt=='P0':
                    continue
                for metric in ['score']+(['diagnostic_score'] if task=='ScienceQA-IMG' else []):
                    pools={(p,m):units([r for r in records if r['benchmark']==task and r['prompt']==p and r['condition']==m],metric)
                           for p in ['P0',prompt] for m in ['Native','SVH']}
                    transformed={}
                    for p in ['P0',prompt]:
                        transformed[p]={k:(v[0],v[1]-pools[p,'Native'][k][1]) for k,v in pools[p,'SVH'].items()}
                    gaps.append(dict(stage=stage,benchmark=task,prompt=prompt,metric=metric,
                        definition='(SVH_new-Native_new)-(SVH_P0-Native_P0); positive = closes SVH deficit',
                        **bootstrap(transformed[prompt],transformed['P0'])))
    table(OUT/'gap_change_comparisons.csv',gaps)
    table(OUT/'native_svh_contingency_rows.csv',cells)
    table(OUT/'quality_timing_summary.csv',quality_timing)
    consistency=[]
    old=Store(OUT/'p0_reused.sqlite')
    for shard in [0,1]:
        with (OUT/f'performance_raw{shard}.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['mode']=='natural_EOS' and row['prompt']=='P0':
                    reference=old.get(row['key']+'|'+row['condition'])
                    consistency.append(dict(key=row['key'],condition=row['condition'],repeat=row['repeat'],
                        token_ids_identical=json.loads(row['token_ids'])==reference['token_ids'],
                        input_binding_identical=row['input_binding_sha256']==reference['input_binding_sha256']))
    old.conn.close()
    write(OUT/'p0_fresh_sequence_regression.json',dict(records=consistency,
        all_token_ids_identical=all(r['token_ids_identical'] for r in consistency),
        all_input_bindings_identical=all(r['input_binding_identical'] for r in consistency)))
    with (OUT/'performance_comparison.csv').open() as stream:
        perf=list(csv.DictReader(stream))
    report=(OUT/'report_zh.md').read_text()
    report+='\n## 7. 差距缩小的直接配对检验\n\n'
    report+='差中之差定义：(新Prompt SVH−Native)−(P0 SVH−Native)。正值表示SVH相对差距缩小；按相同图片簇配对Bootstrap，不是用两条独立CI相减。\n\n'
    report+='|任务|指标|差距变化pp|探索性95%CI|\n|---|---|---:|---|\n'
    for r in gaps:
        if r['stage']=='Validation':
            report+=f"|{r['benchmark']}|{r['metric']}|{r['estimate_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|\n"
    report+='\nGQA的P1在Discovery中，Native为68.00%（P0 70.00%），SVH为57.33%（P0 58.00%），因此按冻结规则保留P0。Validation没有再运行GQA P1；不能将保留P0的验证分数解释为P1已获独立验证。\n'
    report+='\n每条旋转记录的Native/SVH四格表及ScienceQA“内容正确、格式失败”数量见 native_svh_contingency_rows.csv；MMBench正式主分数与CI仍采用完整Circular组，不能混用分母。\n'
    report+='\n## 8. 自然EOS时间改善与计时边界\n\n'
    for task in TASKS:
        prompt=selected[task]
        if prompt=='P0':
            report+=f'- {task}保留原P0，无Prompt变更收益主张。\n'
            continue
        for method in ['Native','SVH']:
            row=next(r for r in perf if r['benchmark']==task and r['prompt']==prompt and r['condition']==method and r['mode']=='natural_EOS')
            base=next(r for r in perf if r['benchmark']==task and r['prompt']=='P0' and r['condition']==method and r['mode']=='natural_EOS')
            report+=f"- {task} {method}，两个固定输入的平均自然EOS E2E：{float(base['e2e_ms']):.1f}→{float(row['e2e_ms']):.1f}ms，减少{float(row['e2e_reduction_pct']):.1f}%，加速{float(row['e2e_speedup_vs_same_method_P0']):.2f}倍。\n"
    report+='\n计时沿用原引擎统一边界：预处理完的CPU输入开始，Vision及Language Prefill/真实Decode都重新计算，冷KV；不包含磁盘图片读取、Processor、模型载入及输入H2D初始化。E2E为该生成引擎实际墙钟，不是完整服务请求延迟。TTFT为首轮前向CUDA事件时间，TPOT不含首Token；tokens/s包含EOS，所有Token长度统计都沿用相同口径。\n'
    report+='\nP1/P2增加了原生Prefill文本位置，在冻结SVH规则下这些位置也可能接收视觉信息。因此本轮不能把收益全部归因于解析器，或断言视觉信息交接能力已修复；未做区分格式指令与新增接收位置的额外机制消融。\n'
    report+='\n总体建议：ScienceQA/MMBench可将已冻结并留出验证的P2作为新的独立短答协议候选；GQA保留P0。原ATE统一HF成绩保持原值。历史物理数值FAIL未解决，本轮结果不构成该门禁的重新通过。\n'
    (OUT/'report_zh.md').write_text(report)
    write(OUT/'postprocess_provenance.json',dict(source_sha256=sha(__file__),read_only_quality_analysis=True,
        no_selection_change=True,selection_sha256=sha(OUT/'selection_frozen.json')))


if __name__=='__main__':
    main()
