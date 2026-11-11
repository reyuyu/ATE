"""Image-cluster paired summaries for the requested seven-layer candidate."""
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from l19_l25 import LAYERS, NAME, OUT, ROOT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired

REFERENCE = Path('/data/lk/ATE-worktrees/qwen3vl-svhr-shifted15-discovery/vlm_research/results/qwen3vl_svhr_shifted15_discovery')
BASELINES = ['Native', 'SVH', 'L19_L33', 'L20_L34', 'L24_L35', 'Wide29_L7_35']


def main():
    frozen()
    assert (OUT/'generation_complete.json').exists()
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
    completion = json.loads((REFERENCE/'completion.json').read_text())
    for filename in ['paired_baselines.jsonl.gz', 'full_records.jsonl.gz']:
        expected = completion['artifacts'][filename]['sha256']
        assert sha(REFERENCE/filename) == expected
        references[str(REFERENCE/filename)] = expected
        with gzip.open(REFERENCE/filename, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)['record']
                name, key = r['condition'], r['key']
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
                comparisons.append(dict(task=task, candidate=NAME, baseline=name,
                    **paired(task, selected, [pools[NAME][k] for k in keys])))
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
        for name in BASELINES:
            old = pools[name][key]
            item.update({name+'_answer': old['text'], name+'_score': old['score'],
                         name+'_token_ids': old['token_ids'], 'delta_vs_'+name: r['score']-old['score']})
        flat.append(item)
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
    lines = ['# Qwen3-VL SVH-R: L19-L25 Discovery', '', '## 范围与协议', '',
        '- 仅新增0-based L19-L25（含首尾共7层），P−1与全部真实Decode使用同一回看集合。',
        '- 复用原585条Discovery；TextVQA 250题/236图片簇、GQA Query 120题、Non-Query 60题、ScienceQA P2 40题、MMBench P2 115旋转记录/30完整组。',
        '- 没有读取或运行Validation/Held-out，没有测速或额外搜索。模型、Prompt、视觉预算、R写入L0-L21、L22前退出和L1-L6答案Attention跳算不变。',
        '- 主质量生成采用功能参考，不能把回看层数直接解释成真实KV、FLOPs或E2E节省。',
        '- 方案B（ATE统一HF），不称作者官方复现。', '', '## 同题结果', '',
        '| 任务 | 题/组 | Native | SVH | L19-L33 | L20-L34 | L24-L35 | L7-L35 | L19-L25 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        lines.append('| '+task+' | '+str(lookup[task, NAME]['units'])+' | '+
                     ' | '.join(f"{lookup[task, name]['accuracy']:.2f}%" for name in pools)+' |')
    lines += ['', '## TextVQA配对差', '', '| 对照 | L19-L25减对照pp | 图片簇95%CI | 软分下降/上升题数 |', '|---|---:|---|---:|']
    for r in comparisons:
        if r['task'] == 'TextVQA':
            lines.append(f"| {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '20,000次Image SHA整簇配对Bootstrap，固定种子20261111，题目等权；MMBench维持完整Circular组。',
        '重复Discovery扫描均为探索性描述，CI跨0不是等效或无损证据。TextVQA损害/改善指软分下降/上升。', '', '## 主要结果', '',
        f"- L19-L25 TextVQA {tv(NAME):.2f}%，相对Native {tv(NAME)-tv('Native'):+.2f}pp，相对SVH {tv(NAME)-tv('SVH'):+.2f}pp。",
        f"- 相比L19-L33，移除L26-L33后的同题变化为{tv(NAME)-tv('L19_L33'):+.2f}pp；不能据此把联合效应拆成单层贡献。",
        f"- 相比L20-L34变化{tv(NAME)-tv('L20_L34'):+.2f}pp，两窗口边界不同，不能单归因某层。", '', '## 验证与资源', '',
        '- 17项CPU检查通过；历史未初始化缓存容量的测试FAIL保留，使用活跃行、原始字节及独立存储的严格替代检查；未放宽生产数值门槛。',
        '- 各GPU三题真实Smoke：Native/No-op全部有效Logits严格torch.equal，功能参考与选层存储误差门槛0.02、生成Token一致；禁止连接概率0、非目标Prefill不变。',
        '- 历史严格数值状态继续FAIL_RETAINED，本轮小Smoke通过不改判历史全面验收。']
    for shard in [0, 1]:
        smoke = json.loads((OUT/f'smoke{shard}.json').read_text())
        lines.append(f"- GPU {UUIDS[shard]} Smoke最大有效Logits误差 {max(r['functional_physical_max_logit_error'] for r in smoke['cases']):.6f}。")
    fresh = list(pools[NAME].values())
    wall = json.loads((OUT/'generation_complete.json').read_text())['wall_seconds']
    lines += [f"- 完成{len(fresh)}/585；EOS {sum(r['eos'] for r in fresh)}，截断 {sum(r['truncated'] for r in fresh)}，空输出 {sum(r['empty'] for r in fresh)}。",
        f'- GPU阶段墙钟{wall:.2f}秒，含加载、Smoke及生成，不含CPU统计，不代表方法E2E时延。',
        '- GPU1/3本任务上下文已释放，未使用GPU0/2，未终止其他任务；本地提交、不自动推送。', '', '## 复现', '',
        '分支：codex/qwen3vl-svhr-l19-l25-discovery。',
        '环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
        '依次执行：l19_l25_cpu.py；l19_l25.py --freeze；l19_l25.py --launch；summarize_l19_l25.py。',
        '源码和协议在运行前分别提交；相同输出目录已有launch时拒绝重复启动。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    release = json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['exit_codes'] == [0, 0] and not any(uuid in release['processes'] for uuid in UUIDS)
    assert not any(OUT.glob('*failure*.json'))
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and not p.name.endswith(('.sqlite', '.sqlite-wal', '.sqlite-shm')) and p.name != 'completion.json'}
    write(OUT/'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=585, actual=len(fresh), layers=LAYERS,
        condition=NAME, tasks=dict(Counter(r['analysis_task'] for r in rows)), historical_strict_status='FAIL_RETAINED',
        performance='NOT_RUN', validation_and_heldout='NOT_READ_OR_RUN', gpu_contexts_released=True,
        summary_source_sha256=sha(__file__), paired_reference_sha256=references, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
