"""Paired Discovery-only summary; never reads validation or held-out outputs."""
import gzip
import hashlib
import json
import math
import sqlite3
import zlib
from collections import Counter
from l24_l34 import LAYERS, NAME, OUT, PREVIOUS, ROOT, TASKS, UUIDS, frozen, read_store, sha, table, write
from analysis import contribution, paired


def main():
    frozen()
    assert (OUT / 'generation_complete.json').exists()
    rows = json.loads((OUT / 'sample_manifest.json').read_text())
    specs = {r['key']: r for r in rows}
    fresh = {}
    for shard in [0, 1]:
        assert (OUT / f'done{shard}.json').exists()
        for _, r in read_store(OUT / f'worker{shard}.sqlite'):
            assert r['key'] not in fresh
            assert r['layers'] == r['prefill_layers'] == r['decode_layers'] == LAYERS
            assert r['image_sha256'] == specs[r['key']]['image_sha256']
            assert math.isfinite(r['score'])
            fresh[r['key']] = r
    assert set(fresh) == set(specs) and len(fresh) == 585
    pools = {'Native': {}, 'SVH': {}, 'Wide29_L7_35': {}, 'Wide30_AllActive': {}, NAME: fresh}
    for _, r in read_store(OUT / 'paired_baselines.sqlite'):
        r['score'] = r.get('score', r.get('scoring', {}).get('score'))
        assert r['score'] is not None
        pools[r['condition']][r['key']] = r
    for shard in [0, 1]:
        db = sqlite3.connect(f'file:{(PREVIOUS / f"worker{shard}.sqlite").resolve()}?mode=ro', uri=True)
        try:
            for name in ['Wide29_L7_35', 'Wide30_AllActive']:
                for key, blob, expected in db.execute('SELECT id,payload,sha256 FROM records WHERE id LIKE ?',
                                                      ('Discovery|Unified|%|' + name,)):
                    raw = zlib.decompress(blob)
                    assert hashlib.sha256(raw).hexdigest() == expected
                    r = json.loads(raw)
                    assert r['key'] in specs and r['key'] not in pools[name]
                    assert r['input_binding_sha256'] == fresh[r['key']]['input_binding_sha256']
                    pools[name][r['key']] = r
        finally:
            db.close()
    assert all(set(pool) == set(specs) for pool in pools.values())
    comparisons, summary = [], []
    for task in TASKS:
        keys = sorted(r['key'] for r in rows if r['analysis_task'] == task)
        for name, pool in pools.items():
            selected = [pool[key] for key in keys]
            vals = contribution(task, selected)
            summary.append(dict(task=task, condition=name, rows=len(selected), units=len(vals),
                                image_clusters=len({x[0] for x in vals}),
                                accuracy=100*sum(x[1] for x in vals)/len(vals),
                                eos=sum(r['eos'] for r in selected),
                                truncated=sum(r['truncated'] for r in selected),
                                empty=sum(r['empty'] for r in selected),
                                mean_generated_tokens=sum(r['generated_tokens'] for r in selected)/len(selected)))
        for name in ['Native', 'SVH', 'Wide29_L7_35', 'Wide30_AllActive']:
            stats = paired(task, [pools[name][key] for key in keys], [fresh[key] for key in keys])
            comparisons.append(dict(task=task, candidate=NAME, baseline=name, **stats))
    table(OUT / 'accuracy_summary.csv', summary)
    table(OUT / 'paired_comparisons.csv', comparisons)
    flat = []
    raw_records = []
    for key in sorted(specs):
        r, spec = fresh[key], specs[key]
        item = dict(key=key, question_id=spec['question_id'], question=spec['question'],
                    benchmark=spec['benchmark'], task=spec['analysis_task'], gold=spec['gold'],
                    image_sha256=spec['image_sha256'], image_path=spec['image_path'],
                    prompt_sha256=spec['prompt_sha256'], input_binding_sha256=r['input_binding_sha256'],
                    answer=r['text'], token_ids=r['token_ids'], score=r['score'], eos=r['eos'],
                    truncated=r['truncated'], empty=r['empty'], generated_tokens=r['generated_tokens'],
                    first_token_flip_vs_svh=r['first_token_flip_vs_svh'], gpu_uuid=r['gpu_uuid'])
        for name in ['Native', 'SVH', 'Wide29_L7_35', 'Wide30_AllActive']:
            old = pools[name][key]
            assert old['image_sha256'] == r['image_sha256'] and old['prompt_sha256'] == r['prompt_sha256']
            item.update({name+'_answer': old['text'], name+'_score': old['score'],
                         name+'_token_ids': old['token_ids'], 'delta_vs_'+name: r['score']-old['score']})
        flat.append(item)
        raw_records.append(dict(key=key, record=r))
    table(OUT / 'per_question.csv', flat)
    table(OUT / 'negative_cases.csv', [r for r in flat if r['delta_vs_Native'] < 0])
    table(OUT / 'improvements_vs_svh.csv', [r for r in flat if r['delta_vs_SVH'] > 0])
    for filename, records in [('full_records.jsonl.gz', raw_records),
                              ('paired_baselines.jsonl.gz', [dict(condition=name, record=row) for name, pool in pools.items()
                                                            if name != NAME for row in pool.values()])]:
        with gzip.open(OUT / filename, 'wt', encoding='utf-8') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    lines = ['# Qwen3-VL-SVH-R：L24–L34 单候选结果', '',
             '**COMPLETE：585条Discovery生成；不含Validation/Held-out，不测速。**', '',
             '只新增11层集合L24–L34（0-based，不含L35）。P−1与全部真实Decode同层回看；',
             'R写入L0–L21、主视觉Hidden在L22前退出、答案L1–L6跳算均保持冻结。',
             '本轮是功能性参考质量测试，不能据此宣称物理加速或KV节省。', '',
             '## 同题结果', '', '| 任务 | 题目/组 | Native | SVH | L7–L35 | L0＋L7–L35 | L24–L34 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    lookup = {(r['task'], r['condition']): r for r in summary}
    for task in TASKS:
        base = lookup[task, NAME]
        scores = [lookup[task, name]['accuracy'] for name in pools]
        lines.append('| '+task+' | '+str(base['units'])+' | '+' | '.join(f'{v:.2f}%' for v in scores)+' |')
    tv = lookup['TextVQA', NAME]['accuracy']
    native = lookup['TextVQA', 'Native']['accuracy']
    svh = lookup['TextVQA', 'SVH']['accuracy']
    wide = lookup['TextVQA', 'Wide29_L7_35']['accuracy']
    wall = json.loads((OUT / 'generation_complete.json').read_text())['wall_seconds']
    lines += ['', '## 中文结论', '',
              f'L24–L34的11层回看达到{tv:.2f}%，较原SVH提高{tv-svh:.2f}pp，',
              f'仍低于同题Native {native-tv:.2f}pp，比L7–L35低{wide-tv:.2f}pp。',
              '这说明缩减到11层仍可恢复大部分原始损失，但不能维持宽回看的全部质量。',
              '该候选值得作为质量/层数折中保留；没有独立Validation或实际测速，',
              '不能宣称满足−1pp质量目标、严格无损或真实计算加速。',
              f'本轮双卡执行墙钟{wall:.2f}秒，含模型加载和GPU Smoke，不含CPU开发与统计。']
    lines += ['', 'ScienceQA / MMBench沿用原冻结P2输入；MMBench按完整Circular组计分。',
              'TextVQA为原Discovery的250题，不能与先前498题Validation百分比混作同一对照。', '',
              '## L24–L34 配对差', '', '| 任务 | 对照 | 差值pp | 图片簇95%CI | 损害/恢复 |', '|---|---|---:|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['task']} | {r['baseline']} | {r['delta_pp']:+.2f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['harm']}/{r['rescue']} |")
    lines += ['', '固定种子20261111，20,000次Image SHA整簇配对Bootstrap；题目等权，Circular整组。',
              'TextVQA损害/恢复表示软分下降/上升，不是假定二值纠错。区间仅作探索性描述。', '',
              '## 验证与局限', '',
              '- 585条记录完整，无重复、无跳样、无本轮OOM；原始视觉预算/权重/Prompt/评分器不变。',
              '- 两卡各3道真实Smoke，Native/No-op严格torch.equal；禁止视觉概率为0；非目标Prefill保持一致。',
              '- 功能参考与选层存储的少量Smoke采用原0.02门槛；生成Token一致。不是全量物理质量验收。',
              '- 历史严格数值状态继续为FAIL_RETAINED，不把新Smoke改写成旧算法全部数值PASS。',
              '- 首次CPU复查有一个旧缓存用例失败，保留cpu_validation.json；该用例比较未初始化备用容量，',
              '  可被NaN填充构造出假失败，但最初失败矩阵未捕获，不能事后证明它就是当时的原因。',
              '  cpu_cache_checks.py以活跃行严格相等、全部原始字节严格相等、独立指针与源缓存不变进行补充。',
              '  修正测试夹具后17项CPU检查通过；未修改原引擎、旧测试文件或任何GPU数值阈值。',
              '- 使用物理GPU1/3，未触碰GPU0/2；释放记录见gpu_release_audit.json。',
              '- 原Held-out的MME OOM历史及其PARTIAL状态不变，本轮没有续跑或删除其中样本。',
              '- 不自动扩大条件或样本，不进行测速，不自动推送GitHub。', '',
              '## 复现', '',
              '代码分支：codex/qwen3vl-svhr-l24-l34-discovery。',
              '运行环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。',
              '冻结入口：l24_l34.py --freeze；执行：l24_l34.py --launch；汇总：summarize_l24_l34.py。',
              '历史运行时由protocol.json绑定绝对路径及SHA256；不复制模型或历史大数组。',
              '同目录已有launch.json时入口拒绝重复启动，故障时保留检查点，需明确恢复而非覆盖。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    processes = json.loads((OUT / 'gpu_release_audit.json').read_text())['processes']
    assert not any(uuid in processes for uuid in UUIDS)
    artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in OUT.iterdir()
                 if p.is_file() and p.suffix not in ['.sqlite', '.log'] and p.name != 'completion.json'}
    write(OUT / 'completion.json', dict(status='COMPLETE_DISCOVERY_ONLY', expected=585, actual=len(fresh),
                                        tasks=dict(Counter(r['analysis_task'] for r in rows)), layers=LAYERS,
                                        benchmark_or_algorithm_changed=False, historical_strict_status='FAIL_RETAINED',
                                        performance='NOT_RUN', validation_and_heldout='NOT_READ_OR_RUN',
                                        gpu_contexts_released=True, artifacts=artifacts))
    print(json.dumps([r for r in summary if r['task'] == 'TextVQA'], indent=2))


if __name__ == '__main__':
    main()
