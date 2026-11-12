"""CPU-only rescore and explicit coverage; no missing-answer zero imputation."""
import argparse
import gzip
import hashlib
import json
import sqlite3
import zlib
from collections import Counter, defaultdict

from .common import OUT, ROOT, TASKS, check, inventory, scorers, sha, table, write, digest
from .metrics import complete_keys, metric, paired, pope


def records():
    result, states = [], Counter()
    for path in sorted(OUT.glob('shard*.sqlite')):
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        states.update(dict(db.execute('SELECT state,count(*) FROM predictions GROUP BY state')))
        for blob, expected in db.execute('SELECT payload,payload_sha256 FROM predictions WHERE payload IS NOT NULL'):
            raw = zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == expected
            result.append(json.loads(raw))
        db.close()
    assert len(result) == len({(r['key'], r['condition']) for r in result})
    return result, states


def format_number(value):
    return 'PENDING' if value is None else f'{value:.4f}'


def numeric_summary():
    import numpy as np
    observations = []
    for path in sorted((OUT/'canaries').glob('*.json')):
        item = json.loads(path.read_text())
        for step in item.get('comparisons', []):
            observations.append(dict(key=item['key'], source=str(path), **step))
    for path in sorted(OUT.glob('smoke[01].json')):
        for item in json.loads(path.read_text()).get('rows', []):
            for step in item.get('comparisons', []):
                observations.append(dict(key=item['key'], source=str(path), **step))
    table(OUT/'sampled_numerical_checks.csv', observations)
    checks = [r for r in observations if r.get('mean_abs') is not None]
    by_request = defaultdict(list)
    for row in observations:
        by_request[(row['source'], row['key'])].append(row)
    failures = sum(not all(s['pass'] for s in steps) for steps in by_request.values())
    summary = dict(strict_status='FAIL_RETAINED', evaluated_steps=len(observations),
        evaluated_requests=len(by_request), failed_requests=failures,
        sampled_failure_rate=failures/max(len(by_request), 1),
        mean_abs_quantiles=np.quantile([r['mean_abs'] for r in checks],[0,.5,.95,1]).tolist() if checks else [],
        max_abs_quantiles=np.quantile([r['max_abs'] for r in checks],[0,.5,.95,1]).tolist() if checks else [],
        full_59_validated=False, full_24_validated=False,
        note='Periodic sample only, not a numerical failure estimate for every quality request. After token divergence prefixes differ.')
    write(OUT/'numerical_summary.json', summary)
    return summary


def legacy_pope(planned, successful, mapping):
    from ..qwen3vl_svh_fasttrack.freeze import AUDIT
    from pathlib import Path
    registry=json.loads((AUDIT/'dataset_manifest.json').read_text())
    path=Path(registry['manifest_path'])
    assert sha(path)==registry['manifest_sha256']
    with gzip.open(path,'rt') as stream:
        old=[json.loads(line) for line in stream]
    old=[r for r in old if r['benchmark']=='POPE']
    current={r['question_id']:r for r in planned if r['benchmark']=='POPE'}
    common=[]
    for row in old:
        now=current[row['question_id']]
        assert (now['image_sha256'],now['question'],now['gold'])==(row['image_sha256'],row['question'],row['answer'])
        common.append(now['key'])
    assert len(common)==len(set(common))==8910
    valid=sorted(k for k in common if set(successful[k])=={'Native','SVH'})
    output=[]
    for name in ['Native','SVH']:
        selected=[successful[k][name] for k in valid]
        output.append(dict(view='POPE_LEGACY_8910',condition=name,planned=8910,paired_effective=len(valid),
            full_metric=len(valid)==8910,**pope(selected)))
    table(OUT/'pope_8910_auxiliary.csv',output)
    write(OUT/'pope_8910_binding_audit.json',dict(status='PASS',manifest=str(path),sha256=sha(path),
        matched_questions=8910,additional_9000_questions=sorted(set(r['key'] for r in current.values())-set(common)),
        dataset_binding_only=True,old_llava_predictions_not_reused=True))


def generate(partial=False):
    protocol = check()
    if partial and (OUT/'completion.json').exists():
        saved=json.loads((OUT/'completion.json').read_text())
        if saved['status']=='TERMINAL':
            print('READ_ONLY_TERMINAL_STATUS',json.dumps(saved,sort_keys=True),flush=True)
            return saved
    rows, states = records()
    planned = inventory()
    by_id = {r['key']: r for r in planned}
    mapping = json.loads((OUT / 'shards.json').read_text())
    scoring = scorers()
    successful = defaultdict(dict)
    for r in rows:
        expected = by_id[r['key']]
        assert r['image_sha256'] == expected['image_sha256'] and r['gold'] == expected['gold']
        assert r['protocol_sha256'] == sha(OUT / 'protocol.json')
        if r['status'] == 'COMPLETE':
            assert r['scoring'] == scoring.score(expected, r['text'])
            assert r['input_binding_sha256'] == digest(r['input_binding'])
            successful[r['key']][r['condition']] = r
    summaries, groups, circular = [], [], []
    for task in TASKS:
        pool = [r for r in planned if r['benchmark'] == task]
        paired_keys = {r['key'] for r in pool if set(successful[r['key']]) == {'Native', 'SVH'}}
        paired_keys = complete_keys(task, pool, paired_keys)
        for key in paired_keys:
            assert successful[key]['Native']['input_binding_sha256'] == successful[key]['SVH']['input_binding_sha256']
        a = [successful[k]['Native'] for k in sorted(paired_keys)]
        b = [successful[k]['SVH'] for k in sorted(paired_keys)]
        ci = paired(task, a, b, mapping, repeats=0 if partial else 20000)
        completed = {name: sum(name in successful[r['key']] for r in pool) for name in ['Native', 'SVH']}
        oom = {name: sum(r['status']=='SKIPPED_OOM' and r['condition']==name and r['benchmark']==task for r in rows) for name in ['Native', 'SVH']}
        summary = dict(benchmark=task, planned=len(pool), native_success=completed['Native'], svh_success=completed['SVH'],
            native_oom=oom['Native'], svh_oom=oom['SVH'], paired_effective=len(paired_keys),
            native=metric(task, a, mapping), svh=metric(task, b, mapping), **ci,
            harm=sum(y['scoring']['score']<x['scoring']['score'] for x, y in zip(a,b)),
            rescue=sum(y['scoring']['score']>x['scoring']['score'] for x, y in zip(a,b)),
            full_metric=len(paired_keys)==len(pool), mode='RELAXED_NUMERICAL_QUALITY_ONLY', structural='PASS_FOR_COMPLETE_REQUESTS')
        summaries.append(summary)
        for name, selected in [('Native', a), ('SVH', b)]:
            if task == 'POPE':
                for strategy in ['random', 'popular', 'adversarial']:
                    groups.append(dict(task=task, condition=name, group=strategy, **pope([r for r in selected if r['strategy']==strategy])))
            if task == 'MME':
                for category in sorted({r['mme_task'] for r in selected}):
                    values = [r for r in selected if r['mme_task']==category]
                    pairs = defaultdict(list)
                    for r in values:
                        pairs[r['pair_id']].append(r['scoring']['score'])
                    accuracy = 100*sum(r['scoring']['score'] for r in values)/len(values)
                    plus = 100*sum(all(v==1 for v in p) for p in pairs.values())/len(pairs)
                    groups.append(dict(task=task, condition=name, group=category, n=len(values), accuracy=accuracy, accuracy_plus=plus, score=accuracy+plus))
            if task == 'MMBench':
                by_group=defaultdict(list)
                for row in selected:
                    by_group[row['circular_group']].append(row)
                for group, values in by_group.items():
                    circular.append(dict(group=group,condition=name,n=len(values),
                        correct=all(r['scoring']['score']==1 for r in values),keys=[r['key'] for r in values]))
    terminal = len(rows)==70692 and not states.get('PENDING',0) and not states.get('RUNNING',0)
    table(OUT / 'accuracy_summary.csv', summaries)
    table(OUT / 'subgroup_summary.csv', groups)
    table(OUT / 'mmbench_circular_groups.csv', circular)
    numerical = numeric_summary()
    if not partial:
        try:
            legacy_pope(planned,successful,mapping)
        except (AssertionError,FileNotFoundError,KeyError) as exc:
            write(OUT/'pope_8910_binding_audit.json',dict(status='UNRESOLVED',error=repr(exc),
                main_9000_view_not_substituted=True,auxiliary_scores_unavailable=True))
    anomalies=[]
    for task in TASKS:
        for name in ['Native','SVH']:
            selected=[r for r in rows if r['benchmark']==task and r['condition']==name and r['status']=='COMPLETE']
            anomalies.append(dict(benchmark=task,condition=name,n=len(selected),empty=sum(r['empty'] for r in selected),
                parse_failed=sum(r['scoring']['parse_failed'] for r in selected),
                truncated=sum(r['truncated'] for r in selected),eos=sum(r['eos'] for r in selected),
                max_generated_tokens=max((r['generated_tokens'] for r in selected),default=0),
                mean_generated_tokens=sum(r['generated_tokens'] for r in selected)/max(len(selected),1),
                peak_allocated_bytes=max((r['peak_allocated'] for r in selected),default=0)))
    table(OUT/'output_anomalies.csv',anomalies)
    failed = [{k:v for k,v in r.items() if k!='input_binding'} for r in rows if r['status']!='COMPLETE']
    table(OUT / 'failure_cases.csv', failed)
    if not partial:
        table(OUT / 'per_question.csv', [{k:v for k,v in r.items() if k!='input_binding'} for r in rows])
        write(OUT / 'cpu_scoring_audit.json', dict(status='PASS', rescored=len([r for r in rows if r['status']=='COMPLETE']),
            unique_records=len(rows), protocol_sha256=sha(OUT/'protocol.json'), mapping_sha256=sha(OUT/'shards.json'),
            scoring_report_source_sha256=sha(__file__),metrics_source_sha256=sha(ROOT/'vlm_research/qwen3vl_overnight/metrics.py')))
        write(OUT / 'completion.json', dict(status='TERMINAL' if terminal else 'PARTIAL', states=dict(states),
            completed_records=len(rows), expected_records=70692, strict_numerical='FAIL_RETAINED',
            mode='RELAXED_NUMERICAL_QUALITY_ONLY', structural='PASS_FOR_COMPLETE_REQUESTS', stage2_allowed=terminal))
    lines = ['# Qwen3-VL / SVH 连续研究报告', '',
        '| Benchmark | Native | SVH | Native/SVH成功 | OOM N/S | 配对有效/计划 | ΔSVH [95%CI] | 完整指标 | 数值/结构 |',
        '|---|---:|---:|---|---|---|---|---|---|']
    for r in summaries:
        ci = f"{format_number(r['delta'])} [{format_number(r['low'])}, {format_number(r['high'])}]"
        lines.append(f"| {r['benchmark']} | {format_number(r['native'])} | {format_number(r['svh'])} | {r['native_success']}/{r['svh_success']} | {r['native_oom']}/{r['svh_oom']} | {r['paired_effective']}/{r['planned']} | {ci} | {r['full_metric']} | RELAXED / 结构PASS |")
    lines += ['', '| SVH-R候选 | 层集合/K | 六项得分 | 类型/KV/速度 | 阶段 |', '|---|---|---|---|---|',
        '| 尚未开始 | 冻结SVH后才执行 | PENDING | 不宣称物理节省或加速 | 等待第一项终态 |', '',
        f'当前检查点：{dict(states)}。第一项状态：{"TERMINAL" if terminal else "PARTIAL"}。',
        '表中配对分数只使用共同成功且MME双题/MMBench Circular组完整的集合。覆盖不足时不是正式全量指标，缺失样本没有按错题补零。',
        '方案B，非Qwen作者官方复现；方案A仍BLOCKED。原0.02严格数值FAIL保留，质量结果不能证明严格数值等价或实际加速。',
        f"周期数值检查：{numerical['failed_requests']}/{numerical['evaluated_requests']}次严格FAIL；这是抽检分母，不是全部质量题目的失败率。",
        '本轮原视觉/语言Eager、BF16、原像素预算；OOM跳过后保留全部计划分母。Bootstrap按图像簇及必要完整组，不把随机层集合当作独立图片。',
        'MME为14任务Acc+Acc+总分；MMBench为冻结ATE Circular离线版本，不是作者V1.1。POPE主视图9000，历史8910辅助视图另行校验。',
        f'原始数据：{OUT}；逐条完整输入绑定和预测在分卡SQLite（zlib无损JSON+SHA），不上传公共仓库。', '']
    path = ROOT / 'overnight_report_zh.md'
    path.write_text('\n'.join(lines))
    (OUT / 'report_zh.md').write_text('\n'.join(lines))
    if not partial and terminal:
        from ..qwen3vl_svh_preflight48.run import resource
        write(OUT/'gpu_release.json',resource())
        from .archive import commit
        commit('Native-SVH six benchmarks terminal',OUT)
    print('CPU REPORT', len(rows), dict(states), 'TERMINAL' if terminal else 'PARTIAL', flush=True)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    generate(args.partial)
