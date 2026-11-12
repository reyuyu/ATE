"""Independent rescore and completeness audit, with explicit missing controls."""
from collections import Counter
from datetime import datetime
import gzip
import json
import sqlite3

import numpy as np

from ..fastv_svh.analyze import write_csv
from ..information_horizon_alignment.analyze import mme_score, circular_rows, paired_boot
from .checkpoint_audit import audit_sample, score_immutable
from .common import *

HISTORICAL = {'GQA': 57.592622, 'ScienceQA-IMG': 68.963808, 'POPE': 87.205387,
              'MME': 1827.751801, 'MMBench': 63.010204, 'TextVQA': 50.4}


def collect(stage):
    samples = {s['key']: s for s in rows(MANIFEST)}
    assignment = read(PARTITION)['assignment']
    verify('sparse')
    roots = [(OUT, digest(verify('sparse')), set(COUNTS))] if stage == 'sparse' else [
        (PRIOR, digest(read(PRIOR / 'protocol.json')), set(COUNTS) - {'MME'})]
    if stage == 'mme':
        roots.append((MME_OUT, digest(verify('mme')), {'MME'}))
    expected_counts = {k: v for k, v in COUNTS.items() if stage != 'fastv-ready' or k != 'MME'}
    records, legacy_audits = {}, []
    for root, expected_protocol, allowed in roots:
        for rank in (0, 1):
            target = root / f'worker{rank}'
            complete = read(target / 'completion.json')
            assert complete['status'] == 'COMPLETE_SHARD' and complete['completed'] == complete['expected']
            db = sqlite3.connect((target / 'results.sqlite').as_uri() + '?mode=ro', uri=True)
            assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            for key, raw, checksum in db.execute('SELECT key,payload,sha256 FROM results'):
                record = json.loads(raw)
                assert key not in records and key in samples
                sample = samples[key]
                assert record['benchmark'] in allowed
                assert record['protocol_sha256'] == expected_protocol
                assert digest(record) == checksum, key
                legacy = audit_sample(record, sample, allow_legacy_scoring_mutation=root == PRIOR)
                if legacy:
                    legacy_audits.append(legacy)
                assert record['rank'] == assignment[key] == rank
                assert record['image_sha256'] == sample['image_sha256']
                assert record['input_binding']['prompt'] == sample['prompt']
                assert record['gold'] == sample['answer'] and record['question'] == sample['question']
                assert score_immutable(sample, record['generation']['text']) == record['scoring'], key
                assert record['generation']['kv']['persistent_original_visual_tokens'] == 0
                if stage == 'sparse':
                    assert [s['visual_rows'] for s in record['generation']['physical_rows']] == (
                        [576]*3 + [309]*4 + [204]*9 + [113]*4 + [0]*12)
                records[key] = record
            db.close()
    assert len(records) == sum(expected_counts.values())
    assert Counter(r['benchmark'] for r in records.values()) == expected_counts
    return samples, [records[k] for k in sorted(records)], legacy_audits


def analyze(stage):
    from ..information_horizon_alignment import score as scorer
    scorer.AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
    samples, output, legacy_audits = collect(stage)
    target = OUT if stage == 'sparse' else PRIOR / (
        'recovered_five_benchmarks' if stage == 'fastv-ready' else 'complete_with_mme_authorized')
    target.mkdir(parents=True, exist_ok=True)
    atomic(target / 'checkpoint_binding_audit.json', dict(records_verified=len(output),
        raw_record_checksums_strict=True, sample_binding_exceptions=legacy_audits,
        original_checkpoints_unchanged=True, scoring_input_deep_copied=True))
    flat = []
    for r in output:
        g = r['generation']
        flat.append(dict(key=r['key'], benchmark=r['benchmark'], image_sha256=r['image_sha256'],
            gold=r['gold'], text=g['text'], token_ids=g['token_ids'], score=r['scoring']['score'],
            parsed=r['scoring']['parsed'], eos=g['eos'], truncated=g['truncated'], empty=g['empty'],
            protocol_sha256=r['protocol_sha256'], input_binding_sha256=digest(r['input_binding']),
            selection_sha256=digest(g.get('stages', g.get('selector'))),
            persistent_visual_kv=g['kv']['persistent_original_visual_tokens']))
    write_csv(target / 'per_question.csv', flat)
    compressed = target / 'per_question.jsonl.gz'
    temporary = compressed.with_suffix('.tmp')
    with gzip.GzipFile(filename=str(temporary), mode='wb', mtime=0) as stream:
        for record in output:
            stream.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
    temporary.replace(compressed)
    summary, details = [], []
    for benchmark, expected in COUNTS.items():
        if stage == 'fastv-ready' and benchmark == 'MME':
            continue
        subset = [r for r in output if r['benchmark'] == benchmark]
        assert len(subset) == expected
        ss = [samples[r['key']] for r in subset]
        values = [r['scoring']['score'] for r in subset]
        ci = None
        if benchmark == 'MME':
            totals, tasks = mme_score(ss, values)
            value, metric = totals['Total'], 'MME Accuracy+Accuracy+ points'
            details.extend(dict(benchmark=benchmark, **t) for t in tasks)
        else:
            if benchmark == 'MMBench':
                ss, values = circular_rows(ss, values)
                assert len(values) == 1176
            value = 100 * float(np.mean(values))
            boot = paired_boot(ss, np.zeros(len(values)), values, seed=20261110, draws=20000)
            ci = [boot['ci_low'], boot['ci_high']]
            metric = 'Circular Accuracy (%)' if benchmark == 'MMBench' else (
                'VQA Soft Accuracy (%)' if benchmark == 'TextVQA' else 'Accuracy (%)')
        if benchmark == 'POPE':
            for strategy in ('random', 'popular', 'adversarial'):
                group = [r for r in subset if samples[r['key']]['strategy'] == strategy]
                tp = sum(r['gold'] == 'yes' and r['scoring']['parsed'] == 'yes' for r in group)
                tn = sum(r['gold'] == 'no' and r['scoring']['parsed'] == 'no' for r in group)
                fp = sum(r['gold'] == 'no' and r['scoring']['parsed'] == 'yes' for r in group)
                fn = sum(r['gold'] == 'yes' and r['scoring']['parsed'] == 'no' for r in group)
                precision = tp / (tp + fp) if tp + fp else 0.
                recall = tp / (tp + fn) if tp + fn else 0.
                details.append(dict(benchmark=benchmark, strategy=strategy, questions=len(group),
                    accuracy=100*(tp + tn)/len(group), precision=precision, recall=recall,
                    f1=2*precision*recall/(precision + recall) if precision + recall else 0.,
                    yes_ratio=(tp + fp)/len(group), tp=tp, tn=tn, fp=fp, fn=fn))
        summary.append(dict(benchmark=benchmark, questions=len(subset), status='COMPLETE', metric=metric,
            fusion=value, own_score_ci95=ci, svh_bound_control=None, delta_svh_paired=None,
            paired_ci95=None, paired_status='MISSING_BOUND_ROW_LEVEL_CONTROLS',
            historical_svh_reference=HISTORICAL[benchmark],
            aggregate_reference_difference=value-HISTORICAL[benchmark],
            reference_difference_is_not_verified_paired_comparison=True,
            eos=sum(r['generation']['eos'] for r in subset),
            truncated=sum(r['generation']['truncated'] for r in subset),
            empty=sum(r['generation']['empty'] for r in subset)))
    write_csv(target / 'score_summary.csv', summary)
    write_csv(target / 'task_details.csv', details)
    acceptance = read(OUT / 'mme_authorization.json')
    atomic(target / 'data_acceptance.json', acceptance)
    condition = 'SparseVLM-192-prefix + llava-SVH' if stage == 'sparse' else 'FastV-K3 + llava-SVH'
    completion = dict(status='VERIFIED_FIVE_AWAITING_MME' if stage == 'fastv-ready' else
        'COMPLETE_GENERATION_UNPAIRED_CONTROLS', completed=len(output), requested=35256,
        independent_rescore_passed=True, method=condition, benchmarks=summary,
        complete_predictions=stage != 'fastv-ready', paired_controls_complete=False,
        mme_byte_exception_authorized=True, historical_mme_strictly_comparable=False,
        no_efficiency_claim=True, no_github_push=True, finished_at=datetime.now().astimezone().isoformat())
    atomic(target / 'completion.json', completion)
    text = ['# 六项质量评测', '', f'方法：{condition}。{len(output)}/35256 条，已保存预测独立重新评分完成。', '',
            '| Benchmark | 分数 | 历史SVH参考 | 参考差值（非严格配对） |', '|---|---:|---:|---:|']
    for s in summary:
        text.append(f"| {s['benchmark']} | {s['fusion']:.6f} | {s['historical_svh_reference']:.6f} | {s['aggregate_reference_difference']:+.6f} |")
    text += ['', '## 有效性与局限',
        '- 原FastV汇总失败的原因已精确复现：1条MMBench评分函数原地小写化选项，存下评分后的样本哈希。原记录校验全部严格通过，未改Prompt、输入或预测。复核凭据独立保存，未覆盖原检查点。',
        '- MME完整2374题均保留；1张图片/2题使用本机已授权字节版本，未冒充历史SHA一致。',
        '- 历史Native/SVH逐题原始输出及Pixel/Input绑定未恢复。表中参考差值不是已验收的配对比较，不能提供伪造的损害/纠正数或配对CI。',
        '- 空答案和截断保留在分母。普通自身分数区间按图片簇重采样；MMBench保持Circular组。',
        '- 没有同机配对效率实验，本轮不宣称端到端加速。',
        '- 全部结果仅本地保存与提交，不推送GitHub，不改main。']
    if stage == 'sparse':
        means = [r['generation']['mean_visual_rows'] for r in output]
        text += ['', '## 融合语义',
            f'- 实测逐层视觉Hidden平均：{float(np.mean(means)):.6f}；完整独立SparseVLM的206.5不能用作本融合预算。',
            '- 官方192工作点Top-k和Recycling/Clustering AST不变；L2/L6/L15 score-only使用当层实际Hidden及SVH合法来源Mask。P−1图像概率仍为0，评分不注入答案残差。',
            '- 原始文本RoPE位置不压紧；新合并视觉Token使用协议声明的图像区锚点。这是有意混合定义，不声称与独立SparseVLM的compact位置实现完全等价。',
            '- 视觉Hidden为576→309→204→113→0，L20起完全退出；持久原始Visual KV为0。']
    (target / 'report_zh.md').write_text('\n'.join(text) + '\n')
    files = [target / f for f in ('per_question.csv', 'per_question.jsonl.gz', 'score_summary.csv',
                                  'task_details.csv', 'report_zh.md', 'completion.json', 'data_acceptance.json',
                                  'checkpoint_binding_audit.json')]
    atomic(target / 'artifact_sha256.json', {str(p.relative_to(target)): sha(p) for p in files})
    print('INDEPENDENTLY_VERIFIED', condition, len(output), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sparse', 'mme', 'fastv-ready'], required=True)
    analyze(parser.parse_args().stage)
