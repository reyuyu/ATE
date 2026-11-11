"""Discovery-only paired summary for the corrected final-layer window."""
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from l22_l35 import LAYERS, NAME, ORIGINAL_OUT, OUT, ROOT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired

LATE_OUT = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-late-windows-discovery/vlm_research/results/qwen3vl_svhr_late_windows_discovery')


def main():
    frozen()
    assert (OUT/'generation_complete.json').exists()
    rows = json.loads((OUT/'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    pools = {name: {} for name in ['Native', 'SVH', 'L24_L34', 'L24_L35', 'L20_L34', 'Wide29_L7_35', 'Wide30_AllActive', NAME]}
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
    for directory, filenames in [(ORIGINAL_OUT, ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']),
                                 (LATE_OUT, ['full_records.jsonl.gz'])]:
        completion = json.loads((directory/'completion.json').read_text())
        for filename in filenames:
            expected = completion['artifacts'][filename]['sha256']
            assert sha(directory/filename) == expected
            references[str(directory/filename)] = expected
            with gzip.open(directory/filename, 'rt', encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)['record']
                    name, key = r['condition'], r['key']
                    assert name in pools and name != NAME and key in specs and key not in pools[name]
                    r['score'] = r.get('score', r.get('scoring', {}).get('score'))
                    assert math.isfinite(r['score'])
                    pools[name][key] = r
    assert len(specs) == 585 and all(set(p) == set(specs) for p in pools.values())
    for key, spec in specs.items():
        for pool in pools.values():
            r = pool[key]
            assert r['image_sha256'] == spec['image_sha256'] and r['prompt_sha256'] == spec['prompt_sha256']
            assert r['input_binding_sha256'] == pools[NAME][key]['input_binding_sha256']
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
            if name != NAME:
                stats = paired(task, selected, [pools[NAME][k] for k in keys])
                comparisons.append(dict(task=task, candidate=NAME, baseline=name, **stats))
    table(OUT/'accuracy_summary.csv', summary)
    table(OUT/'paired_comparisons.csv', comparisons)
    flat = []
    for key, spec in sorted(specs.items()):
        r = pools[NAME][key]
        item = dict(key=key, question_id=spec['question_id'], task=spec['analysis_task'], question=spec['question'],
            condition=NAME, layers=LAYERS, gold=spec['gold'], image_sha256=spec['image_sha256'], image_path=spec['image_path'],
            prompt_sha256=spec['prompt_sha256'], input_binding_sha256=r['input_binding_sha256'],
            answer=r['text'], token_ids=r['token_ids'], score=r['score'], eos=r['eos'], truncated=r['truncated'],
            empty=r['empty'], generated_tokens=r['generated_tokens'], gpu_uuid=r['gpu_uuid'])
        for name, pool in pools.items():
            if name == NAME:
                continue
            old = pool[key]
            item.update({name+'_answer': old['text'], name+'_score': old['score'], name+'_token_ids': old['token_ids'],
                         'delta_vs_'+name: r['score']-old['score']})
        flat.append(item)
    table(OUT/'per_question.csv', flat)
    table(OUT/'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT/'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, records in [('full_records.jsonl.gz', [dict(key=k, record=r) for k, r in pools[NAME].items()]),
        ('paired_baselines.jsonl.gz', [dict(condition=n, record=r) for n, p in pools.items() if n != NAME for r in p.values()])]:
        with gzip.open(OUT/filename, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
    lookup = {(r['task'], r['condition']): r for r in summary}
    tv = lambda name: lookup['TextVQA', name]['accuracy']
    lines = ['# Qwen3-VL-SVH-R：L22–L35探索测试', '',
        '**COMPLETE_DISCOVERY_ONLY：585条完整生成；不含Validation、Held-out或性能测试。**', '',
        '## 层号说明', '',
        '用户写作L22–L36。本模型实际为36层，沿用0-based L0–L35，不存在L36。',
        '本轮已预先明确按L22–L35（含两端，共14层）执行，没有切换为1-based，没有运行虚构的第37层。', '',
        '## 冻结定义', '',
        '- P−1与全部真实Decode在相同L22–L35集合回看，其他答案层禁图，独立Cache、自然EOS。',
        '- 原SVH的R写入L0–L21、主视觉Hidden在L22前退出、L1–L6答案Attention跳算不变。',
        '- 晚层视觉KV由冻结的因果视觉前缀sidecar计算，不复制其他层KV、不使用额外Native前向来选答案。',
        '- 本轮全量质量使用功能参考保留所有活跃层视觉KV；不能把14层数直接当作物理KV节省或加速。',
        '- 仅配置适配器复用历史worker，冻结引擎、权重、视觉预算、Prompt、mRoPE、DeepStack和评分器不变。', '',
        '## 同题结果', '',
        '| 任务 | 题/组 | Native | SVH | L24–L34 | L24–L35 | L20–L34 | L7–L35 | L0+L7–L35 | L22–L35 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('| '+task+' | '+str(lookup[task, NAME]['units'])+' | '+
                     ' | '.join(f"{lookup[task, name]['accuracy']:.2f}%" for name in pools)+' |')
    lines += ['', 'TextVQA 250题/236图片簇；GQA Query 120题、Non-Query 60题；ScienceQA P2 40题；',
        'MMBench P2 115旋转记录/30完整Circular组，组内全对才得分。全部输入与历史同题逐字节绑定。', '',
        '## TextVQA配对差', '', '| 对照 | 新候选−对照pp | 图片簇95%CI | 软分下降/上升题数 |', '|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA':
            lines.append(f"| {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '20,000次Image SHA整簇配对Bootstrap，固定种子20261111；题目等权，Circular保留整组。',
        '所有区间均为探索性描述，重复Discovery不冒充独立验证；CI跨0不等于无损或等效。', '', '## 主要结论', '',
        f"- L22–L35得到{tv(NAME):.2f}%，相对Native {tv(NAME)-tv('Native'):+.2f}pp，相对SVH {tv(NAME)-tv('SVH'):+.2f}pp。",
        f"- 相比L24–L35，仅额外开放L22、L23，新候选变化{tv(NAME)-tv('L24_L35'):+.2f}pp。",
        f"- 相比L20–L34，新候选变化{tv(NAME)-tv('L20_L34'):+.2f}pp；两者同时差在L20/L21与L35，不能单归因于某一层。",
        f"- 相比L7–L35宽回看，新候选变化{tv(NAME)-tv('Wide29_L7_35'):+.2f}pp。",
        '- 未做独立Validation或性能测试，不宣称最优、非劣效、真实加速或物理KV减少。', '', '## 验证与资源', '',
        '- 17项CPU检查通过；旧缓存未初始化容量的原FAIL仍保留，使用活跃行、原始字节和指针严格检查。',
        '- 两GPU各3题Smoke；Native/No-op要求torch.equal，功能参考/选层存储沿用0.02门槛。',
        '- Smoke检查禁止视觉概率0、非目标Prefill不变、输入/权重未改、正常和异常清理与独立Cache。',
        '- 历史严格数值状态继续FAIL_RETAINED，小Smoke通过不改判历史全面验收。']
    for shard in [0, 1]:
        smoke = json.loads((OUT/f'smoke{shard}.json').read_text())
        lines.append(f"- GPU {UUIDS[shard]} Smoke有效Logits最大误差{max(r['functional_physical_max_logit_error'] for r in smoke['cases']):.6f}，Token一致。")
    fresh = list(pools[NAME].values())
    wall = json.loads((OUT/'generation_complete.json').read_text())['wall_seconds']
    lines += [f"- 完成{len(fresh)}/585；EOS {sum(r['eos'] for r in fresh)}，截断 {sum(r['truncated'] for r in fresh)}，空输出 {sum(r['empty'] for r in fresh)}。",
        f'- 双卡墙钟{wall:.2f}秒（模型加载、Smoke及生成；不含CPU开发统计），不作为方法E2E时延。',
        '- GPU1/3释放；未使用GPU0/2或终止他人任务；不自动扩样、增搜或推送GitHub。', '', '## 复现', '',
        '分支：codex/qwen3vl-svhr-l22-l35-discovery。',
        '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
        '入口：l22_l35_cpu.py；l22_l35.py --freeze；l22_l35.py --launch；summarize_l22_l35.py。',
        '源码、原样本及协议运行前已冻结；同目录已有launch时拒绝重复启动，不覆盖历史结果。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0] and not any(uuid in release['processes'] for uuid in UUIDS)
    assert not any(OUT.glob('*failure*.json'))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT/'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=585, actual=len(fresh), layers=LAYERS,
        condition=NAME, user_label='L22-L36', actual_valid_zero_based='L22-L35',
        tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
        performance='NOT_RUN', validation_and_heldout='NOT_READ_OR_RUN', gpu_contexts_released=True,
        summary_source_sha256=sha(__file__), paired_reference_sha256=references, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
