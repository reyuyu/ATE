"""Verify completed checkpoints and preserve unavailable controls as missing."""
from collections import Counter, defaultdict
import csv
import json
import sqlite3

import numpy as np

from .common import *


def csv_file(name, records):
    p = OUT / name
    fields = list(dict.fromkeys(k for row in records for k in row))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in records:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def cluster_ci(samples, scores):
    if not samples:
        return None
    units = sorted({s['image_sha256'] for s in samples})
    index = {key: i for i, key in enumerate(units)}
    sums, counts = np.zeros(len(units)), np.zeros(len(units))
    for s, v in zip(samples, scores):
        j = index[s['image_sha256']]
        sums[j] += v
        counts[j] += 1
    rng = np.random.default_rng(20261110)
    values = []
    for _ in range(200):
        w = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=100)
        values.extend((100 * (w @ sums) / (w @ counts)).tolist())
    return np.percentile(values, [2.5, 97.5]).tolist()


def main():
    p = verify()
    samples = {s['key']: s for s in rows(OUT / 'sample_manifest.jsonl.gz')}
    records, seen = [], set()
    for path in sorted(OUT.glob('worker*/results.sqlite')):
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        for key, raw, h in db.execute('SELECT key,payload,sha256 FROM results'):
            r = json.loads(raw)
            assert key not in seen and digest(r) == h and r['sample_sha256'] == digest(samples[key])
            assert r['protocol_sha256'] == digest(p)
            seen.add(key)
            records.append(r)
        db.close()
    records.sort(key=lambda r: (list(COUNTS).index(r['benchmark']), r['key']))
    counts = Counter(r['benchmark'] for r in records)
    if records:
        csv_file('per_question.csv', [dict(key=r['key'], benchmark=r['benchmark'], image_sha256=r['image_sha256'],
                    gold=r['gold'], text=r['generation']['text'], token_ids=r['generation']['token_ids'],
                    score=r['scoring']['score'], parsed=r['scoring']['parsed'],
                    eos=r['generation']['eos'], truncated=r['generation']['truncated'],
                    empty=r['generation']['empty'], generated_tokens=len(r['generation']['token_ids']),
                    first_logits_sha256=r['generation']['first_logits_sha256']) for r in records])
    old = read(OUT / 'references' / PAPER_REF / 'score_summary.json')['benchmarks']
    statuses = read(OUT / 'data_acceptance.json')['benchmarks']
    summary, details = [], []
    for bench, expected in COUNTS.items():
        subset = [r for r in records if r['benchmark'] == bench]
        ss = [samples[r['key']] for r in subset]
        values = [r['scoring']['score'] for r in subset]
        state = 'COMPLETE' if len(subset) == expected else statuses[bench]['status'] if statuses[bench]['status']=='BLOCKED' else 'PARTIAL'
        value = 100 * float(np.mean(values)) if values else None
        metric = 'Soft Accuracy (%)' if bench == 'TextVQA' else 'Accuracy (%)'
        if bench == 'MMBench' and len(subset) == expected:
            from ..information_horizon_alignment.analyze import circular_rows
            ss, scores = circular_rows(ss, values)
            values = scores.tolist()
            assert len(values) == 1176
            value = 100 * float(np.mean(values))
            metric = 'Circular Accuracy (%)'
        old_value = (59.25425345841946 if bench == 'GQA' else
                     old[bench].get('value', old[bench].get('scores', {}).get('Total')))
        if bench == 'MME':
            metric = 'MME points'
            value = None
        if bench == 'OCRBench-v1':
            metric = 'Official raw score /1000'
            value = None
        summary.append(dict(benchmark=bench, completed=len(subset), expected=expected, status=state,
                            metric=metric, middle16=value, middle16_ci95=cluster_ci(ss, values) if values else None,
                            old_svh_r=old_value, native=61.91763396406424 if bench=='GQA' else None,
                            svh=57.59262203847989 if bench=='GQA' else None,
                            delta_old_svh_r=(value-old_value) if value is not None and state=='COMPLETE' else None,
                            paired_ci95=None, paired_status='BLOCKED_MISSING_BOUND_SAME_PROTOCOL_PER_QUESTION_CONTROLS',
                            reference_commit=GQA_REF if bench=='GQA' else PAPER_REF,
                            anomalies=dict(Counter(flag for r in subset for flag in ['eos','empty','truncated','repeated_3gram'] if r['generation'].get(flag)))))
        if bench == 'POPE' and subset:
            for strategy in ['random', 'popular', 'adversarial']:
                group = [r for r in subset if samples[r['key']]['strategy'] == strategy]
                tp = sum(r['gold']=='yes' and r['scoring']['parsed']=='yes' for r in group)
                tn = sum(r['gold']=='no' and r['scoring']['parsed']=='no' for r in group)
                fp = sum(r['gold']=='no' and r['scoring']['parsed']=='yes' for r in group)
                fn = sum(r['gold']=='yes' and r['scoring']['parsed']=='no' for r in group)
                details.append(dict(benchmark=bench, group=strategy, count=len(group), tp=tp, tn=tn, fp=fp, fn=fn,
                                    accuracy=(tp+tn)/len(group) if group else None))
    csv_file('score_summary.csv', summary)
    csv_file('task_details.csv', details)
    complete = all(r['status']=='COMPLETE' for r in summary)
    atomic(OUT / 'completion.json', dict(status='COMPLETE' if complete else 'PARTIAL',
           completed=len(records), requested=36256, benchmarks=summary,
           no_unmatched_paired_inference=True, data_blockers=p['data_blockers']))
    lines = ['# SVH-R Middle16 七项评测', '', '**状态：' + ('COMPLETE' if complete else 'PARTIAL') + '**', '',
             '本轮只新增物理 Middle16（L8–L23回看，L24前删除视觉Hidden）。旧源码未修改。', '',
             '| 数据集 | 完成/目标 | Middle16 | 原SVH-R | 状态 |', '|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f"| {r['benchmark']} | {r['completed']}/{r['expected']} | {r['middle16'] if r['middle16'] is not None else '未测'} | {r['old_svh_r']} | {r['status']} |")
    lines += ['', '## 限制', '',
              '- MME指定恢复图像SHA与当前本地版本不同；整项BLOCKED，未替换或少测两题。',
              '- OCRBench官方标注/评分脚本已核验，但官方原始图片ZIP未恢复；没有用HF重编码图片冒充。',
              '- 原版完整逐题控制记录不在本机，只有参考提交的汇总。配对翻转和配对差值CI保持缺失；上表旧值不当作本轮实测。',
              '- 0414中的旧GQA协议不复用；GQA仅引用71a5762官方对齐的历史分数。',
              '- 当前均值CI为本轮同图簇Bootstrap，不是相对历史算法的配对差值CI。',
              '- 同时改变了层位置和层数，不能将变化单独归因于中间位置。', '',
              '数值Smoke、真实视觉KV布局及资源释放记录分别见worker*/smoke.json和gpu_release_audit.json。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print('VERIFIED_RESULTS', len(records), 'OF', 36256)


if __name__ == '__main__':
    main()
