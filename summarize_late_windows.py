"""Summarize only the paired Discovery cohort, without selecting a new winner."""
import gzip
import json
import math
from collections import Counter
from late_windows import CONDITIONS, OUT, ROOT, TASKS, UUIDS, frozen, prior, read_store, sha, table, write
from analysis import contribution, paired


def main():
    frozen()
    assert (OUT / 'generation_complete.json').exists()
    rows = json.loads((OUT / 'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    pools = {name: {} for name in ['Native', 'SVH', 'L24_L34', 'Wide29_L7_35', 'Wide30_AllActive', *CONDITIONS]}
    for shard in [0, 1]:
        assert (OUT / f'done{shard}.json').exists()
        for _, r in read_store(OUT / f'worker{shard}.sqlite'):
            name, key = r['condition'], r['key']
            assert name in CONDITIONS and key in specs and key not in pools[name]
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == CONDITIONS[name]
            assert r['image_sha256'] == specs[key]['image_sha256']
            assert r['gpu_uuid'] == UUIDS[specs[key]['shard']]
            assert r['protocol_sha256'] == sha(OUT / 'protocol.json')
            assert math.isfinite(r['score']) and r['token_ids']
            pools[name][key] = r
    old_complete = json.loads((prior.OUT / 'completion.json').read_text())
    for filename in ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']:
        assert sha(prior.OUT / filename) == old_complete['artifacts'][filename]['sha256']
        with gzip.open(prior.OUT / filename, 'rt', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                r = entry['record']
                name, key = r['condition'], r['key']
                assert name not in CONDITIONS and name in pools and key in specs and key not in pools[name]
                r['score'] = r.get('score', r.get('scoring', {}).get('score'))
                assert math.isfinite(r['score'])
                pools[name][key] = r
    assert len(specs) == 585 and all(set(pool) == set(specs) for pool in pools.values())
    for key, spec in specs.items():
        reference = pools['Native'][key]
        for pool in pools.values():
            r = pool[key]
            assert r['input_binding_sha256'] == reference['input_binding_sha256']
            assert r['image_sha256'] == spec['image_sha256']
            assert r['prompt_sha256'] == spec['prompt_sha256']
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
    table(OUT / 'accuracy_summary.csv', summary)
    table(OUT / 'paired_comparisons.csv', comparisons)
    flat, fresh_records = [], []
    for name in CONDITIONS:
        for key in sorted(specs):
            spec, r = specs[key], pools[name][key]
            item = dict(key=key, question_id=spec['question_id'], task=spec['analysis_task'], question=spec['question'],
                        condition=name, layers=r['layers'], gold=spec['gold'], image_sha256=spec['image_sha256'],
                        image_path=spec['image_path'], prompt_sha256=spec['prompt_sha256'],
                        input_binding_sha256=r['input_binding_sha256'], answer=r['text'], token_ids=r['token_ids'],
                        score=r['score'], eos=r['eos'], truncated=r['truncated'], empty=r['empty'],
                        generated_tokens=r['generated_tokens'], gpu_uuid=r['gpu_uuid'])
            for baseline, pool in pools.items():
                if baseline == name:
                    continue
                old = pool[key]
                item.update({baseline+'_answer': old['text'], baseline+'_token_ids': old['token_ids'],
                             baseline+'_score': old['score'], 'delta_vs_'+baseline: r['score']-old['score']})
            flat.append(item)
            fresh_records.append(dict(key=key, condition=name, record=r))
    table(OUT / 'per_question.csv', flat)
    table(OUT / 'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT / 'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, records in [('full_records.jsonl.gz', fresh_records),
                              ('paired_baselines.jsonl.gz', [dict(condition=name, record=r)
                               for name, pool in pools.items() if name not in CONDITIONS for r in pool.values()])]:
        with gzip.open(OUT / filename, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
    lookup = {(r['task'], r['condition']): r for r in summary}
    lines = ['# Qwen3-VL-SVH-R：L24–L35 与 L20–L34', '',
        '**COMPLETE_DISCOVERY_ONLY：两条件各585条，共1,170条真实生成完成；未运行Validation、Held-out或性能测试。**', '',
        '## 冻结定义', '',
        '- L24–L35为12层；L20–L34为15层；层号均0-based。',
        '- 原始P−1和全部真实Decode使用同一回看集合，正常Greedy至EOS，沿用逐题上限与原Prompt。',
        '- R写入L0–L21、主视觉Hidden在L22前退出、L1–L6答案Attention跳算均未修改。',
        '- 两候选使用冻结功能参考做质量评价。少量选层物理存储Smoke不能代替完整物理性能验收。',
        '- 不修改原SVH、权重、视觉预算、Processor、DeepStack、mRoPE、BF16/eager、评分器或数值阈值。', '',
        '## 同题成绩', '',
        '| 任务 | 题/组 | Native | SVH | L24–L34 | L7–L35 | L0+L7–L35 | L24–L35 | L20–L34 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        scores = [lookup[task, name]['accuracy'] for name in pools]
        lines.append('| '+task+' | '+str(lookup[task, 'Native']['units'])+' | '+' | '.join(f'{v:.2f}%' for v in scores)+' |')
    lines += ['', 'TextVQA为250题、236图片簇；GQA为Query 120题/Non-Query 60题；ScienceQA 40题；',
              'MMBench为115旋转记录、30完整Circular组，必须整组全对。ScienceQA/MMBench沿用P2输入。',
              '所有历史对照均严格绑定同题、同Prompt、同图片SHA及处理后输入SHA；不读取Validation或Held-out输出。', '',
              '## TextVQA配对结果', '',
              '| 候选 | 对照 | 差值pp | 图片簇95%CI | 软分下降/上升题数 |',
              '|---|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA':
            lines.append(f"| {r['candidate']} | {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '固定种子20261111、20,000次Image SHA整簇配对Bootstrap，题目等权；MMBench保持Circular组。',
              '均为重复使用Discovery的探索性区间，未作确认性多重比较判断；CI包含0不等于无损或等效。',
              '软分下降/上升不是二值准确率翻转，不能混称纠错。', '', '## 主要发现', '']
    tv = lambda name: lookup['TextVQA', name]['accuracy']
    for name in CONDITIONS:
        lines.append(f"- {name}：{tv(name):.2f}%，相对SVH {tv(name)-tv('SVH'):+.2f}pp；相对Native {tv(name)-tv('Native'):+.2f}pp；相对11层L24–L34 {tv(name)-tv('L24_L34'):+.2f}pp。")
    lines += [f"- 两个新候选直接比较，L24–L35减L20–L34为{tv('L24_L35')-tv('L20_L34'):+.2f}pp。",
              '- 这些结果只能用于已测试回看集合的探索性比较；层数更少不保证E2E更快或KV更少。',
              '- 未经独立Validation，不宣称最优、非劣效或完全恢复；本轮到此停止，不自动增搜。', '',
              '## 正确性与资源', '',
              '- 20项CPU检查通过。旧缓存测试的未初始化容量问题及原FAIL保留在上一轮目录；本轮使用严格活跃行、原始字节与独立指针检查。',
              '- 每卡3题、每题2候选真实Smoke；Native/No-op要求torch.equal，功能/选层存储沿用0.02，生成Token一致。',
              '- Smoke核查非目标Prefill未变、禁止视觉概率为0、输入/权重未改、Hook清理与独立Cache。',
              '- 历史严格数值状态继续为FAIL_RETAINED，本次小Smoke通过不改写历史全面数值失败。']
    for shard in [0, 1]:
        smoke = json.loads((OUT / f'smoke{shard}.json').read_text())
        error = max(r['functional_physical_max_logit_error'] for r in smoke['cases'])
        lines.append(f"- GPU {UUIDS[shard]}：6项候选Smoke通过，功能/物理有效Logits最大误差{error:.6f}。")
    for name in CONDITIONS:
        selected = list(pools[name].values())
        lines.append(f"- {name}：{len(selected)}/585，EOS {sum(r['eos'] for r in selected)}，截断 {sum(r['truncated'] for r in selected)}，空输出 {sum(r['empty'] for r in selected)}。")
    wall = json.loads((OUT / 'generation_complete.json').read_text())['wall_seconds']
    lines += [f'- 双卡执行墙钟{wall:.2f}秒（包含加载模型及Smoke，不含CPU开发与统计），不是方法E2E延迟。',
              '- GPU1/3已退出并释放；不使用GPU0/2，不终止外部任务。', '',
              '## 复现与保存', '',
              '分支：codex/qwen3vl-svhr-late-windows-discovery；仅本地提交，未自动推送。',
              '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
              '命令：cpu_checks.py；late_windows.py --freeze；late_windows.py --launch；summarize_late_windows.py。',
              '执行源码和协议、输入清单运行前冻结；所有引用源码均有绝对路径及SHA256。',
              '再次运行时不能覆盖已有launch或协议；检查点保持独立，不复制模型、图片或历史大数组。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT / 'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0]
    assert not any(uuid in release['processes'] for uuid in UUIDS)
    assert all(not (OUT / f'failure{i}.json').exists() for i in [0, 1])
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT / 'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=1170, actual=len(fresh_records),
          conditions=CONDITIONS, per_condition={name: len(pools[name]) for name in CONDITIONS},
          tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
          performance='NOT_RUN', validation_and_heldout='NOT_READ_OR_RUN', gpu_contexts_released=True,
          summary_source_sha256=sha(__file__), artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
