"""Paired Discovery-only summaries for four fixed shifted windows."""
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from shifted15 import BASE_OUT, CONDITIONS, EXPECTED, OUT, ROOT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired

LAST_OUT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-l22-l35-discovery/vlm_research/results/qwen3vl_svhr_l22_l35_discovery')
BASELINES = ['Native', 'SVH', 'L20_L34', 'L24_L34', 'L24_L35', 'L22_L35', 'Wide29_L7_35', 'Wide30_AllActive']


def main():
    frozen()
    assert (OUT/'generation_complete.json').exists()
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    pools = {name: {} for name in BASELINES + list(CONDITIONS)}
    for shard in [0, 1]:
        assert (OUT/f'done{shard}.json').exists()
        for _, r in read_store(OUT/f'worker{shard}.sqlite'):
            key, name = r['key'], r['condition']
            assert name in CONDITIONS and key in specs and key not in pools[name]
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == CONDITIONS[name]
            assert r['protocol_sha256'] == sha(OUT/'protocol.json')
            assert r['gpu_uuid'] == UUIDS[specs[key]['shard']]
            assert math.isfinite(r['score']) and r['token_ids']
            pools[name][key] = r
    references = {}
    for directory, filenames in [(BASE_OUT, ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']),
                                 (LAST_OUT, ['full_records.jsonl.gz'])]:
        completion = json.loads((directory/'completion.json').read_text())
        for filename in filenames:
            expected = completion['artifacts'][filename]['sha256']
            assert sha(directory/filename) == expected
            references[str(directory/filename)] = expected
            with gzip.open(directory/filename, 'rt', encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)['record']
                    key, name = r['key'], r['condition']
                    assert name in BASELINES and key in specs and key not in pools[name]
                    r['score'] = r.get('score', r.get('scoring', {}).get('score'))
                    assert math.isfinite(r['score'])
                    pools[name][key] = r
    assert len(specs) == 585 and all(set(pool) == set(specs) for pool in pools.values())
    for key, spec in specs.items():
        for pool in pools.values():
            r = pool[key]
            assert r['image_sha256'] == spec['image_sha256'] and r['prompt_sha256'] == spec['prompt_sha256']
            assert r['input_binding_sha256'] == pools['Native'][key]['input_binding_sha256']
    summary, comparisons = [], []
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
                if name == baseline:
                    continue
                stats = paired(task, [pools[baseline][k] for k in keys], [pools[name][k] for k in keys])
                comparisons.append(dict(task=task, candidate=name, baseline=baseline, **stats))
    table(OUT/'accuracy_summary.csv', summary)
    table(OUT/'paired_comparisons.csv', comparisons)
    flat, fresh = [], []
    for name in CONDITIONS:
        for key, spec in sorted(specs.items()):
            r = pools[name][key]
            item = dict(key=key, question_id=spec['question_id'], task=spec['analysis_task'], question=spec['question'],
                condition=name, layers=r['layers'], gold=spec['gold'], image_sha256=spec['image_sha256'],
                image_path=spec['image_path'], prompt_sha256=spec['prompt_sha256'], input_binding_sha256=r['input_binding_sha256'],
                answer=r['text'], token_ids=r['token_ids'], score=r['score'], eos=r['eos'], truncated=r['truncated'],
                empty=r['empty'], generated_tokens=r['generated_tokens'], gpu_uuid=r['gpu_uuid'])
            for baseline, pool in pools.items():
                if baseline == name:
                    continue
                old = pool[key]
                item.update({baseline+'_answer': old['text'], baseline+'_score': old['score'], baseline+'_token_ids': old['token_ids'],
                             'delta_vs_'+baseline: r['score']-old['score']})
            flat.append(item)
            fresh.append(dict(key=key, condition=name, record=r))
    assert len(fresh) == EXPECTED
    table(OUT/'per_question.csv', flat)
    table(OUT/'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT/'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, records in [('full_records.jsonl.gz', fresh),
        ('paired_baselines.jsonl.gz', [dict(condition=n, record=r) for n in BASELINES for r in pools[n].values()])]:
        with gzip.open(OUT/filename, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
    lookup = {(r['task'], r['condition']): r for r in summary}
    tv = lambda name: lookup['TextVQA', name]['accuracy']
    displayed = ['Native', 'SVH', 'L20_L34', 'L22_L35', 'Wide29_L7_35', *CONDITIONS]
    lines = ['# Qwen3-VL-SVH-R：四个15层平移窗口', '',
        '**COMPLETE_DISCOVERY_ONLY：4条件各585条，共2,340条完整真实生成；没有Validation、Held-out或性能测试。**', '',
        '## 预冻结实验', '',
        '- 仅新增L19–L33、L18–L32、L17–L31、L16–L30四个15层窗口；0-based，含两端。',
        '- 原始P−1和全部真实Decode使用相同回看集合；其他答案层禁图，独立Cache、自然EOS及原逐题生成上限。',
        '- R写入L0–L21、主视觉Hidden在L22前退出、L1–L6答案Attention跳算不变。晚层视觉KV来自冻结因果视觉sidecar。',
        '- 复用不可变历史worker，只重绑当前进程配置；不修改原SVH源码、权重、输入、评分器、BF16/eager、mRoPE或DeepStack。',
        '- 质量测试使用功能参考，全量保留活跃层视觉KV；少量物理存储Smoke不构成全量物理质量或性能验收。', '',
        '## 同题得分', '',
        '| 任务 | 题/组 | Native | SVH | L20–L34 | L22–L35 | L7–L35 | L19–L33 | L18–L32 | L17–L31 | L16–L30 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('| '+task+' | '+str(lookup[task, 'Native']['units'])+' | '+
                     ' | '.join(f"{lookup[task, name]['accuracy']:.2f}%" for name in displayed)+' |')
    lines += ['', 'TextVQA为250题/236图片簇；GQA Query 120题、Non-Query 60题；ScienceQA P2 40题；',
        'MMBench P2为115旋转记录/30完整Circular组，整组全对才得分。所有历史对照严格同题、同图片SHA、同Prompt及处理后输入绑定。',
        '完整12条件汇总见accuracy_summary.csv。没有读取或使用Validation/Held-out输出。', '',
        '## TextVQA：相对Native、SVH、同预算L20–L34及宽回看', '',
        '| 候选 | 对照 | 差值pp | 图片簇95%CI | 软分下降/上升题数 |', '|---|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA' and r['baseline'] in ['Native', 'SVH', 'L20_L34', 'Wide29_L7_35']:
            lines.append(f"| {r['candidate']} | {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '固定种子20261111，20,000次Image SHA整簇配对Bootstrap；题目等权，MMBench保留完整Circular组。',
        'TextVQA下降/上升为软分变化，不冒充二值纠错。本轮重复使用Discovery，所有区间为探索性，',
        '没有用未经校正区间宣称确认性优胜；CI跨0不等于无损或等效。所有新窗口间比较也保存在paired_comparisons.csv。', '',
        '## 实际观察', '']
    for name in CONDITIONS:
        lines.append(f"- {name}：TextVQA {tv(name):.2f}%；相对Native {tv(name)-tv('Native'):+.2f}pp；相对同预算L20–L34 {tv(name)-tv('L20_L34'):+.2f}pp。")
    lines += ['- 仅描述已测试集合，不把较少层数、较低KV或接近分数宣称为真实E2E加速、非劣效或全局最优。',
        '- 结果可能非单调；不因某个窗口分数好坏追加层搜索、筛题或调整原算法。', '', '## 验证与资源', '',
        '- 17项CPU检查通过。保留旧缓存测试未初始化备用容量的原FAIL，使用活跃行、原始字节、独立指针与源缓存严格不变检查。',
        '- 每卡3题、每题4候选Smoke；Native/No-op严格torch.equal，功能参考/选层物理存储沿用0.02门槛及Token一致要求。',
        '- 检查禁止Image概率为0、非目标Prefill不变、输入/权重不变、正常/异常清理与独立KV。',
        '- 历史严格数值状态仍为FAIL_RETAINED，不把当前Smoke通过改写为历史全面验收通过。']
    for shard in [0, 1]:
        smoke = json.loads((OUT/f'smoke{shard}.json').read_text())
        lines.append(f"- GPU {UUIDS[shard]}：12项候选Smoke通过，有效Logits最大误差{max(r['functional_physical_max_logit_error'] for r in smoke['cases']):.6f}。")
    for name in CONDITIONS:
        selected = list(pools[name].values())
        lines.append(f"- {name}：{len(selected)}/585；EOS {sum(r['eos'] for r in selected)}，截断 {sum(r['truncated'] for r in selected)}，空输出 {sum(r['empty'] for r in selected)}。")
    wall = json.loads((OUT/'generation_complete.json').read_text())['wall_seconds']
    lines += [f'- 双卡执行墙钟{wall:.2f}秒（加载、Smoke及生成，不含CPU开发统计），不能当作方法每请求E2E时延。',
        '- GPU1/3已释放；GPU0/2未使用，未终止外部进程；没有性能测试或自动推送GitHub。', '', '## 复现', '',
        '分支：codex/qwen3vl-svhr-shifted15-discovery。',
        '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
        '命令：shifted15_cpu.py；shifted15.py --freeze；shifted15.py --launch；summarize_shifted15.py。',
        '源码、协议和输入运行前冻结；原子逐题逐条件检查点独立保存，已有launch时拒绝重复启动，不覆盖历史结果。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0] and not any(uuid in release['processes'] for uuid in UUIDS)
    assert not any(OUT.glob('*failure*.json'))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT/'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=EXPECTED, actual=len(fresh),
        conditions=CONDITIONS, per_condition={name: len(pools[name]) for name in CONDITIONS},
        tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
        performance='NOT_RUN', validation_and_heldout='NOT_READ_OR_RUN', gpu_contexts_released=True,
        summary_source_sha256=sha(__file__), paired_reference_sha256=references, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
