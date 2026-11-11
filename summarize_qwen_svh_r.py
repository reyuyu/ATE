"""Task-native scores, explicit missing coverage, image-cluster paired CIs."""
import argparse
import gzip
import json
from collections import Counter, defaultdict

import qwen_svh_r_full as run
from vlm_research.qwen3vl_overnight.metrics import complete_keys, metric, paired, pope


def main(partial=False):
    protocol = run.check()
    rows = run.manifest()
    specs = {r['key']: r for r in rows}
    mapping = {r['key']: {'cluster': r['image_sha256']} for r in rows}
    pools = {name: {} for name in ['Native', 'SVH', run.NAME]}
    for _, record in run.read_store(run.OUT/'baselines.sqlite'):
        pools[record['condition']][record['key']] = record
    for _, record in run.read_store(run.OUT/'reused.sqlite'):
        pools[run.NAME][record['key']] = record
    for shard in [0, 1]:
        path = run.OUT/f'worker{shard}.sqlite'
        if not path.exists():
            continue
        for key, record in run.read_store(path):
            assert key not in pools[run.NAME], ('DUPLICATE', key)
            assert record['protocol_sha256'] == run.sha(run.OUT/'protocol.json')
            assert record['layers'] == record['prefill_layers'] == record['decode_layers'] == run.LAYERS
            pools[run.NAME][key] = record
    for condition, pool in pools.items():
        for key, record in pool.items():
            spec = specs[key]
            for field in ['image_sha256', 'prompt_sha256', 'input_binding_sha256', 'gold']:
                assert record[field] == spec[field], ('BINDING', condition, key, field)
            assert record['gpu_uuid'] == run.UUIDS[spec['shard']]
            if run.state(record) == 'COMPLETE':
                assert record['token_ids'] and record['score'] is not None
                record['scoring'] = record.get('scoring', {k: record.get(k) for k in ['score', 'parsed', 'parse_failed']})
            for field in ['mme_task', 'pair_id', 'strategy', 'circular_group']:
                record[field] = spec.get(field)
    summaries, comparisons, strategies, mme_tasks = [], [], [], []
    for task, planned in run.COUNTS.items():
        task_rows = [r for r in rows if r['benchmark'] == task]
        keys = {r['key'] for r in task_rows}
        available = {name: {k for k in keys & set(pool) if run.state(pool[k]) == 'COMPLETE'} for name, pool in pools.items()}
        common = complete_keys(task, task_rows, set.intersection(*available.values()))
        for name, pool in pools.items():
            valid = complete_keys(task, task_rows, available[name])
            selected = [pool[k] for k in sorted(valid)]
            states = Counter(run.state(pool[k]) for k in keys & set(pool))
            summaries.append(dict(benchmark=task, condition=name, planned=planned, attempted=sum(states.values()),
                complete_predictions=len(available[name]), complete_group_predictions=len(valid),
                blocked_oom=sum(v for k,v in states.items() if 'OOM' in k), missing=planned-sum(states.values()),
                coverage_status='COMPLETE_FULL_TASK' if valid == keys else 'PARTIAL_NOT_FULL_SCORE',
                full_official_score=metric(task, selected, mapping) if valid == keys else None,
                available_subset_score=metric(task, selected, mapping), paired_common_questions=len(common),
                paired_common_score=metric(task, [pool[k] for k in sorted(common)], mapping),
                units=('MME Acc+Acc+ total' if task == 'MME' else 'MMBench Circular percent' if task == 'MMBench' else 'percent'),
                eos=sum(bool(r['eos']) for r in selected), truncated=sum(bool(r['truncated']) for r in selected),
                empty=sum(bool(r['empty']) for r in selected), parse_failed=sum(bool(r['scoring'].get('parse_failed')) for r in selected),
                mean_generated_tokens=sum(r['generated_tokens'] for r in selected)/len(selected) if selected else None))
            if task == 'POPE':
                for strategy in sorted({r['strategy'] for r in selected}):
                    strategies.append(dict(condition=name, strategy=strategy, **pope([r for r in selected if r['strategy'] == strategy])))
            if task == 'MME':
                for category in sorted({r['mme_task'] for r in selected}):
                    cat = [r for r in selected if r['mme_task'] == category]
                    pairs = defaultdict(list)
                    for r in cat:
                        pairs[r['pair_id']].append(r)
                    assert all(len(v) == 2 for v in pairs.values())
                    acc = sum(r['score'] for r in cat)/len(cat)
                    accplus = sum(all(r['score'] == 1 for r in v) for v in pairs.values())/len(pairs)
                    mme_tasks.append(dict(condition=name, mme_task=category, n=len(cat), pairs=len(pairs),
                        accuracy=100*acc, accuracy_plus=100*accplus, score=100*(acc+accplus), scope='Available complete pairs'))
        for baseline in ['Native', 'SVH']:
            a = [pools[baseline][k] for k in sorted(common)]
            b = [pools[run.NAME][k] for k in sorted(common)]
            stats = paired(task, a, b, mapping, repeats=0 if partial else 20000, seed=run.SEED)
            comparisons.append(dict(benchmark=task, candidate=run.NAME, baseline=baseline, n=len(common),
                scope='FULL' if common == keys else 'PAIRED_COMPLETE_COMMON_SUBSET_ONLY',
                harm=sum(y['score'] < x['score'] for x,y in zip(a,b)), rescue=sum(y['score'] > x['score'] for x,y in zip(a,b)),
                first_token_flip=sum(y['token_ids'][0] != x['token_ids'][0] for x,y in zip(a,b)),
                sequence_changed=sum(y['token_ids'] != x['token_ids'] for x,y in zip(a,b)), **stats))
    suffix = '_partial' if partial else ''
    run.table(run.OUT/f'accuracy_summary{suffix}.csv', summaries)
    run.table(run.OUT/f'paired_comparisons{suffix}.csv', comparisons)
    run.table(run.OUT/f'pope_strategy_scores{suffix}.csv', strategies)
    run.table(run.OUT/f'mme_task_scores{suffix}.csv', mme_tasks)
    flat = []
    for key, spec in sorted(specs.items()):
        r = pools[run.NAME].get(key)
        item = {field: spec.get(field) for field in ['key', 'benchmark', 'question_id', 'question', 'gold', 'image_sha256', 'prompt_sha256', 'input_binding_sha256', 'strategy', 'mme_task', 'pair_id', 'circular_group']}
        for name, pool in pools.items():
            value = pool.get(key, {})
            for field in ['text', 'token_ids', 'score', 'eos', 'truncated', 'empty', 'generated_tokens', 'gpu_uuid', 'error']:
                item[name+'_'+field] = value.get(field)
            item[name+'_state'] = run.state(value) if value else 'PENDING'
        for baseline in ['Native', 'SVH']:
            old = pools[baseline][key]
            item['delta_vs_'+baseline] = r['score']-old['score'] if r and run.state(r) == run.state(old) == 'COMPLETE' else None
        flat.append(item)
    run.table(run.OUT/f'per_question{suffix}.csv', flat)
    run.table(run.OUT/f'negative_cases{suffix}.csv', [r for r in flat if r['delta_vs_Native'] is not None and r['delta_vs_Native'] < 0])
    run.table(run.OUT/f'failures{suffix}.csv', [r for r in flat if any('OOM' in r[name+'_state'] for name in pools)])
    full_coverage = all(len(pool) == 35346 and all(run.state(r) == 'COMPLETE' for r in pool.values()) for pool in pools.values())
    lines = ['# qwen-SVH-R L19-L35 六项固定配置评测', '',
        '## 协议与边界', '',
        '- 方案B：ATE统一HF；不是作者官方复现。方案A继续BLOCKED。',
        '- P−1与全部真实Decode统一在0-based L19–L35共17层回看。R写入L0–L21，主路径视觉Hidden在L22前退出，L1–L6答案Attention跳算不变。',
        '- ScienceQA/MMBench采用冻结P2短答案提示；TextVQA保留Paper-OCR；POPE9000；MMBench为ATE内部legacy Dev Circular，不称V1.1。',
        '- 沿用选层时的功能性质量参考，包含真实因果前缀sidecar。未验证本配置物理加速，不宣称17层持久KV或E2E收益。',
        '- 历史严格数值FAIL_RETAINED未被改判。按用户近期指示不重复Smoke；CPU绑定通过不等于GPU数值门禁通过。',
        '- 固定候选来自重复探索的Discovery，扩展队列不是完全独立外部确认，CI跨0不等于无损。', '',
        '## 分数与覆盖', '',
        '| 任务 | Native完整/计划 | SVH完整/计划 | qwen-SVH-R完整/计划 | Native | SVH | qwen-SVH-R |',
        '|---|---:|---:|---:|---:|---:|---:|']
    lookup = {(r['benchmark'], r['condition']): r for r in summaries}
    for task in run.COUNTS:
        entries = [lookup[task,name] for name in pools]
        coverage = [f"{e['complete_predictions']}/{e['planned']}" for e in entries]
        scores = [f"{e['full_official_score']:.4f}" if e['full_official_score'] is not None else 'BLOCKED/PARTIAL' for e in entries]
        lines.append('| '+task+' | '+' | '.join(coverage+scores)+' |')
    lines += ['', 'MME单位为14任务Acc+Acc+总分，其余为百分比，MMBench按完整Circular组。',
        '全量分数仅在全部必需题/组完成时给出。OOM不计错误、不更换分母；部分结果仅可见available/common subset字段，不冒充正式完整分数。', '',
        '## 同题配对变化', '', '| 任务 | 对照 | 共同题数 | 差值 | 图片簇95%CI | 损害/纠正 | 范围 |', '|---|---|---:|---:|---|---:|---|']
    for c in comparisons:
        delta = 'N/A' if c['delta'] is None else f"{c['delta']:+.4f}"
        ci = 'NOT_AVAILABLE' if c['low'] is None else f"[{c['low']:.4f}, {c['high']:.4f}]"
        lines.append(f"| {c['benchmark']} | {c['baseline']} | {c['n']} | {delta} | {ci} | {c['harm']}/{c['rescue']} | {c['scope']} |")
    lines += ['', '最终统计按Image SHA整簇、问题等权、20,000次配对Bootstrap，固定种子20261111；MME保留双题、MMBench保留完整Circular组。',
        'TextVQA损害/纠正表示软分下降/上升；MME子任务与POPE三策略另存CSV。不同任务不平均。', '', '## 完成状态', '',
        f"- {'PARTIAL_SNAPSHOT_RUNNING' if partial else 'COMPLETE_FULL_SIX' if full_coverage else 'ALL_INPUTS_ATTEMPTED_WITH_BLOCKED_COVERAGE'}。",
        '- 原始数据、模型和历史结果只读复用；新逐题检查点写入本结果目录。',
        '- 本轮没有新层搜索、性能Benchmark、缩图、精度/后端切换。',
        '- 全部原始失败保留；运行日志、协议与源码SHA可复核。', '',
        '## 路径', '', str(run.OUT), '',
        '冻结配置：QWEN_SVH_R_FIXED.md。复现：qwen_svh_r_full.py --freeze / --launch；summarize_qwen_svh_r.py。']
    (run.OUT/f'report_zh{suffix}.md').write_text('\n'.join(lines)+'\n')
    if not partial:
        assert len(pools[run.NAME]) == 35346, 'Cannot finalize incomplete attempts'
        assert all((run.OUT/f'done{i}.json').exists() for i in [0, 1])
        with gzip.open(run.OUT/'full_records.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for name, pool in pools.items():
                for key, record in sorted(pool.items()):
                    stream.write(json.dumps(dict(key=key, condition=name, record=record), ensure_ascii=True)+'\n')
        artifacts = {p.name: dict(sha256=run.sha(p), bytes=p.stat().st_size) for p in run.OUT.iterdir()
            if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
        run.write(run.OUT/'completion.json', dict(status='COMPLETE_FULL_SIX' if full_coverage else 'ATTEMPTED_ALL_WITH_BLOCKED_COVERAGE',
            planned=35346, actual_attempts=len(pools[run.NAME]), numerical_status='FAIL_RETAINED',
            new_gpu_numerical_validation='NOT_RUN', physical_speedup='NOT_MEASURED', counts=summaries, artifacts=artifacts))
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
