"""Paired Discovery quality summary for the fixed seventeen-layer window."""
import gzip
import json
import math
from collections import Counter
from l19_l35 import BASE_OUT, EXPECTED, LAYERS, NAME, OUT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired

BASELINES = ['Native', 'SVH', 'L19_L33', 'L19_L32', 'L19_L31', 'L20_L34', 'Wide29_L7_35']


def main():
    frozen()
    generated = json.loads((OUT/'generation_complete.json').read_text())
    assert generated['actual'] == EXPECTED == 585
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    pools = {name: {} for name in BASELINES + [NAME]}
    for shard in [0, 1]:
        assert (OUT/f'done{shard}.json').exists()
        for _, r in read_store(OUT/f'worker{shard}.sqlite'):
            key = r['key']
            assert key in specs and key not in pools[NAME] and r['condition'] == NAME
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == LAYERS
            assert r['protocol_sha256'] == sha(OUT/'protocol.json')
            assert r['gpu_uuid'] == UUIDS[specs[key]['shard']]
            assert math.isfinite(r['score']) and r['token_ids']
            pools[NAME][key] = r
    references = {}
    previous = json.loads((BASE_OUT/'completion.json').read_text())
    for filename in ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']:
        expected = previous['artifacts'][filename]['sha256']
        assert sha(BASE_OUT/filename) == expected
        references[str(BASE_OUT/filename)] = expected
        with gzip.open(BASE_OUT/filename, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)['record']
                key, name = r['key'], r['condition']
                if name not in BASELINES:
                    continue
                assert key in specs and key not in pools[name]
                r['score'] = r.get('score', r.get('scoring', {}).get('score'))
                assert math.isfinite(r['score'])
                pools[name][key] = r
    assert len(specs) == 585 and all(set(pool) == set(specs) for pool in pools.values())
    for key, spec in specs.items():
        for pool in pools.values():
            r = pool[key]
            assert r['image_sha256'] == spec['image_sha256'] and r['prompt_sha256'] == spec['prompt_sha256']
            assert r['input_binding_sha256'] == pools[NAME][key]['input_binding_sha256']
    summary, comparisons, flat = [], [], []
    for task in TASKS:
        keys = sorted(r['key'] for r in rows if r['analysis_task'] == task)
        for name, pool in pools.items():
            selected = [pool[k] for k in keys]
            vals = contribution(task, selected)
            summary.append(dict(task=task, condition=name, rows=len(keys), units=len(vals),
                image_clusters=len({x[0] for x in vals}), accuracy=100*sum(x[1] for x in vals)/len(vals),
                eos=sum(r['eos'] for r in selected), truncated=sum(r['truncated'] for r in selected),
                empty=sum(r['empty'] for r in selected),
                mean_generated_tokens=sum(r['generated_tokens'] for r in selected)/len(selected)))
            if name != NAME:
                comparisons.append(dict(task=task, candidate=NAME, baseline=name,
                    **paired(task, selected, [pools[NAME][k] for k in keys])))
    for key, spec in sorted(specs.items()):
        r = pools[NAME][key]
        item = dict(key=key, question_id=spec['question_id'], task=spec['analysis_task'], question=spec['question'],
            condition=NAME, layers=LAYERS, gold=spec['gold'], image_sha256=spec['image_sha256'], image_path=spec['image_path'],
            prompt_sha256=spec['prompt_sha256'], input_binding_sha256=r['input_binding_sha256'],
            answer=r['text'], token_ids=r['token_ids'], score=r['score'], eos=r['eos'], truncated=r['truncated'],
            empty=r['empty'], generated_tokens=r['generated_tokens'], gpu_uuid=r['gpu_uuid'])
        for name in BASELINES:
            old = pools[name][key]
            item.update({name+'_answer': old['text'], name+'_score': old['score'], name+'_token_ids': old['token_ids'],
                         'delta_vs_'+name: r['score']-old['score']})
        flat.append(item)
    table(OUT/'accuracy_summary.csv', summary)
    table(OUT/'paired_comparisons.csv', comparisons)
    table(OUT/'per_question.csv', flat)
    table(OUT/'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT/'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, labels in [('full_records.jsonl.gz', [NAME]), ('paired_baselines.jsonl.gz', BASELINES)]:
        with gzip.open(OUT/filename, 'wt', encoding='utf-8') as f:
            for name in labels:
                for key, r in sorted(pools[name].items()):
                    f.write(json.dumps(dict(key=key, condition=name, record=r), ensure_ascii=False)+'\n')
    lookup = {(r['task'], r['condition']): r for r in summary}
    tv = lambda name: lookup['TextVQA', name]['accuracy']
    lines = ['# Qwen3-VL SVH-R: L19-L35 Discovery', '', '## 冻结范围', '',
        '- 仅新增0-based L19-L35（含首尾17层）；P−1与全部真实Decode统一同层回看。',
        '- 原585条Discovery：TextVQA 250题/236图片簇、GQA Query 120题、Non-Query 60题、ScienceQA P2 40题、MMBench P2 115旋转记录/30完整Circular组。',
        '- 继续按用户要求不重复Smoke。没有新增CPU回归/GPU数值PASS，只做语法、冻结源码/协议哈希与逐题输入绑定校验，历史FAIL_RETAINED保留。',
        '- 仅复用上一轮worker并配置层集合；模型BF16/Eager、Prompt、视觉预算、R写入L0-L21、L22前退出、L1-L6答案Attention跳算不变。',
        '- 方案B（ATE统一HF），不称作者官方复现；未使用Validation/Held-out评分或生成，未测速、扩样或增加其他候选。',
        '- 全量质量为功能参考结果，不将回看层数等同于物理KV/FLOPs节省或真实E2E加速。', '', '## 同题结果', '',
        '| 任务 | 题/组 | Native | SVH | L19-L33 | L19-L32 | L19-L31 | L20-L34 | L7-L35 | L19-L35 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('| '+task+' | '+str(lookup[task, NAME]['units'])+' | '+
                     ' | '.join(f"{lookup[task, name]['accuracy']:.2f}%" for name in pools)+' |')
    lines += ['', '## TextVQA配对差', '', '| 对照 | L19-L35减对照pp | 图片簇95%CI | 软分下降/上升题数 |', '|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA':
            lines.append(f"| {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '20,000次Image SHA整簇配对Bootstrap，固定种子20261111，题目等权；MMBench按完整Circular组。',
        '重复Discovery探索，未做多重比较校正，不宣称确认性显著、等效或非劣效；CI跨0不是无损证据。TextVQA损害/改善为软分下降/上升。', '', '## 主要结果', '',
        f"- L19-L35 TextVQA {tv(NAME):.2f}%，相对Native {tv(NAME)-tv('Native'):+.2f}pp，相对SVH {tv(NAME)-tv('SVH'):+.2f}pp。",
        f"- 相比L19-L33，增加L34/L35后的同题净变化为{tv(NAME)-tv('L19_L33'):+.2f}pp。只能讨论该背景下两层联合开放的效果，不拆成固定单层贡献。",
        f"- 相比L19-L32为{tv(NAME)-tv('L19_L32'):+.2f}pp，相比L19-L31为{tv(NAME)-tv('L19_L31'):+.2f}pp。", '', '## 完整性与资源', '']
    fresh = list(pools[NAME].values())
    lines += [f"- 完成{len(fresh)}/585；EOS {sum(r['eos'] for r in fresh)}，截断 {sum(r['truncated'] for r in fresh)}，空输出 {sum(r['empty'] for r in fresh)}。",
        f"- 双卡阶段墙钟{generated['wall_seconds']:.2f}秒，含加载及生成、不含CPU统计；不是方法E2E时延。",
        '- GPU1/3本任务进程已结束，未使用GPU0/2，未终止其他任务；无失败标记。',
        '- 分支codex/qwen3vl-svhr-l19-l35-discovery；本地提交，不自动推送。', '', '## 复现', '',
        '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
        '入口：l19_l35.py --freeze；l19_l35.py --launch；summarize_l19_l35.py。',
        '源码与协议运行前提交；同一输出目录已有launch时拒绝重复启动。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0] and not any(uuid in release['processes'] for uuid in UUIDS)
    assert not any(OUT.glob('*failure*.json')) and not any(OUT.glob('smoke*.json'))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT/'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=EXPECTED, actual=len(fresh),
        condition=NAME, layers=LAYERS, tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
        smoke='NOT_RUN_USER_REQUEST', new_gpu_numerical_validation='NOT_RUN', performance='NOT_RUN',
        validation_and_heldout='NOT_USED_OR_RUN', gpu_contexts_released=True,
        summary_source_sha256=sha(__file__), paired_reference_sha256=references, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
