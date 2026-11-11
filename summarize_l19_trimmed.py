"""Summarize two fixed Discovery windows against input-bound old predictions."""
import gzip
import json
import math
from collections import Counter
from l19_trimmed import CONDITIONS, EXPECTED, OUT, PRIOR_OUT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired

BASELINES = ['Native', 'SVH', 'L19_L33', 'L20_L34', 'L19_L25', 'Wide29_L7_35']


def main():
    frozen()
    generated = json.loads((OUT/'generation_complete.json').read_text())
    assert generated['actual'] == EXPECTED == 1170
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    pools = {name: {} for name in BASELINES + list(CONDITIONS)}
    for shard in [0, 1]:
        assert (OUT/f'done{shard}.json').exists()
        for _, r in read_store(OUT/f'worker{shard}.sqlite'):
            key, name = r['key'], r['condition']
            assert key in specs and name in CONDITIONS and key not in pools[name]
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == CONDITIONS[name]
            assert r['protocol_sha256'] == sha(OUT/'protocol.json')
            assert r['gpu_uuid'] == UUIDS[specs[key]['shard']]
            assert math.isfinite(r['score']) and r['token_ids']
            pools[name][key] = r
    references = {}
    previous = json.loads((PRIOR_OUT/'completion.json').read_text())
    for filename in ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']:
        expected = previous['artifacts'][filename]['sha256']
        assert sha(PRIOR_OUT/filename) == expected
        references[str(PRIOR_OUT/filename)] = expected
        with gzip.open(PRIOR_OUT/filename, 'rt', encoding='utf-8') as f:
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
            assert r['input_binding_sha256'] == pools['Native'][key]['input_binding_sha256']
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
        for name in CONDITIONS:
            for baseline in pools:
                if baseline != name:
                    comparisons.append(dict(task=task, candidate=name, baseline=baseline,
                        **paired(task, [pools[baseline][k] for k in keys], [pools[name][k] for k in keys])))
    for name, layers in CONDITIONS.items():
        for key, spec in sorted(specs.items()):
            r = pools[name][key]
            item = dict(key=key, question_id=spec['question_id'], task=spec['analysis_task'], question=spec['question'],
                condition=name, layers=layers, gold=spec['gold'], image_sha256=spec['image_sha256'], image_path=spec['image_path'],
                prompt_sha256=spec['prompt_sha256'], input_binding_sha256=r['input_binding_sha256'],
                answer=r['text'], token_ids=r['token_ids'], score=r['score'], eos=r['eos'], truncated=r['truncated'],
                empty=r['empty'], generated_tokens=r['generated_tokens'], gpu_uuid=r['gpu_uuid'])
            for baseline in pools:
                if baseline != name:
                    old = pools[baseline][key]
                    item.update({baseline+'_answer': old['text'], baseline+'_score': old['score'],
                                 baseline+'_token_ids': old['token_ids'], 'delta_vs_'+baseline: r['score']-old['score']})
            flat.append(item)
    table(OUT/'accuracy_summary.csv', summary)
    table(OUT/'paired_comparisons.csv', comparisons)
    table(OUT/'per_question.csv', flat)
    table(OUT/'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT/'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, labels in [('full_records.jsonl.gz', CONDITIONS), ('paired_baselines.jsonl.gz', BASELINES)]:
        with gzip.open(OUT/filename, 'wt', encoding='utf-8') as f:
            for name in labels:
                for key, r in sorted(pools[name].items()):
                    f.write(json.dumps(dict(key=key, condition=name, record=r), ensure_ascii=False)+'\n')
    lookup = {(r['task'], r['condition']): r for r in summary}
    tv = lambda name: lookup['TextVQA', name]['accuracy']
    lines = ['# Qwen3-VL SVH-R: L19-L32 and L19-L31', '', '## 范围与执行', '',
        '- 仅新增L19-L32（14层）及L19-L31（13层），均为0-based、含首尾；P−1与真实Decode统一同层回看。',
        '- 原585条Discovery：TextVQA 250题/236图片簇、GQA Query 120题、Non-Query 60题、ScienceQA P2 40题、MMBench P2 115旋转记录/30完整Circular组。',
        '- 本轮按用户要求不重复CPU回归或GPU Smoke；只做语法、冻结源码/协议哈希和逐题输入绑定检查。未生成新的数值PASS，历史FAIL_RETAINED不变。',
        '- 复用不变的Request引擎，仅增加无Smoke调度入口；原模型BF16/Eager、Prompt、视觉预算、R写入L0-L21、L22前退出、L1-L6答案Attention跳算不变。',
        '- 方案B（ATE统一HF），不称作者官方复现。没有使用Validation/Held-out成绩或生成，没有测速、扩样或额外候选。',
        '- 全量质量采用功能参考，不能把层数或分数当作物理KV节省、FLOPs节省或真实加速。', '', '## 同题结果', '',
        '| 任务 | 题/组 | Native | SVH | L19-L33 | L20-L34 | L19-L25 | L7-L35 | L19-L32 | L19-L31 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('| '+task+' | '+str(lookup[task, 'Native']['units'])+' | '+
                     ' | '.join(f"{lookup[task, name]['accuracy']:.2f}%" for name in pools)+' |')
    lines += ['', '## TextVQA配对差', '', '| 候选 | 对照 | 差值pp | 图片簇95%CI | 软分下降/上升题数 |', '|---|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA':
            lines.append(f"| {r['candidate']} | {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '20,000次Image SHA整簇配对Bootstrap，固定种子20261111，题目等权；MMBench按完整Circular组。',
        '全部为重复Discovery探索性结果，未经多重比较校正，不据此宣称非劣效、显著性确认或无损。TextVQA损害/改善为软分下降/上升。', '', '## 结论', '']
    for name in CONDITIONS:
        lines.append(f"- {name} TextVQA {tv(name):.2f}%，相对Native {tv(name)-tv('Native'):+.2f}pp，相对SVH {tv(name)-tv('SVH'):+.2f}pp，相对L19-L33 {tv(name)-tv('L19_L33'):+.2f}pp。")
    lines += [f"- L19-L32相对L19-L31为{tv('L19_L32')-tv('L19_L31'):+.2f}pp；该同题对比只改变L32回看权限，不应泛化为独立于其他层的固定贡献。", '', '## 完整性与资源', '']
    fresh = [r for name in CONDITIONS for r in pools[name].values()]
    lines += [f"- 完成{len(fresh)}/1170，每条件585；EOS {sum(r['eos'] for r in fresh)}，截断 {sum(r['truncated'] for r in fresh)}，空输出 {sum(r['empty'] for r in fresh)}。",
        f"- 双卡阶段墙钟{generated['wall_seconds']:.2f}秒，含加载及生成、不含CPU统计；不是方法E2E时延。",
        '- GPU1/3本任务进程已结束，无GPU0/2操作，无他人进程终止；无失败标记。',
        '- 分支codex/qwen3vl-svhr-l19-trimmed-discovery；本地提交，不自动推送。', '', '## 复现', '',
        '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
        '入口：l19_trimmed.py --freeze；l19_trimmed.py --launch；summarize_l19_trimmed.py。',
        '运行前冻结源码和输入；同一输出目录已有launch时拒绝重复启动。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0] and not any(uuid in release['processes'] for uuid in UUIDS)
    assert not any(OUT.glob('*failure*.json')) and not any(OUT.glob('smoke*.json'))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT/'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=EXPECTED, actual=len(fresh),
        conditions=CONDITIONS, per_condition={n: len(pools[n]) for n in CONDITIONS},
        tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
        smoke='NOT_RUN_USER_REQUEST', new_gpu_numerical_validation='NOT_RUN', performance='NOT_RUN',
        validation_and_heldout='NOT_USED_OR_RUN', gpu_contexts_released=True,
        summary_source_sha256=sha(__file__), paired_reference_sha256=references, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
