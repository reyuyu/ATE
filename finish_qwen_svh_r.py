"""CPU-only final reporting; no model loading or new quality conditions."""
import csv
import gzip
import importlib.metadata
import json
import platform
import subprocess
from collections import defaultdict

import numpy as np
import qwen_svh_r_full as run
from vlm_research.qwen3vl_overnight.metrics import pope


def read_csv(name):
    with (run.OUT/name).open() as stream:
        return list(csv.DictReader(stream))


def f1(counts):
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    return 200*tp/np.maximum(2*tp+fp+fn, 1)


def f1_paired(a, b):
    assert set(a) == set(b)
    clusters = sorted({r['image_sha256'] for r in a.values()})
    ix = {c: i for i,c in enumerate(clusters)}
    arrays = []
    for pool in [a,b]:
        values = np.zeros((len(clusters), 3))
        for r in pool.values():
            gold, pred = r['gold'], r['scoring']['parsed']
            values[ix[r['image_sha256']]] += [gold == pred == 'yes', gold == 'no' and pred == 'yes', gold == 'yes' and pred == 'no']
        arrays.append(values)
    rng = np.random.default_rng(run.SEED)
    draws = []
    for start in range(0, 20000, 64):
        weights = rng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=min(64, 20000-start))
        draws.extend((f1(weights@arrays[1])-f1(weights@arrays[0])).tolist())
    low, high = np.quantile(draws, [.025,.975])
    return dict(delta=float(f1(arrays[1].sum(0))-f1(arrays[0].sum(0))), low=float(low), high=float(high),
        image_clusters=len(clusters), bootstrap_replicates=20000, seed=run.SEED)


def main():
    run.check()
    completion = json.loads((run.OUT/'completion.json').read_text())
    generation = json.loads((run.OUT/'generation_complete.json').read_text())
    assert completion['actual_attempts'] == 35346
    release = json.loads((run.OUT/'gpu_release_audit.json').read_text())
    launch = json.loads((run.OUT/'launch.json').read_text())
    assert release['exit_codes'] == [0,0]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    live_pids = {int(line.split(',')[1].strip()) for line in processes.splitlines() if ',' in line}
    assert not set(launch['worker_pids']) & live_pids, 'Own workers still own CUDA'
    pools = {name: {} for name in ['Native','SVH',run.NAME]}
    with gzip.open(run.OUT/'full_records.jsonl.gz', 'rt') as stream:
        for item in map(json.loads, stream):
            r = item['record']
            if r['benchmark'] == 'POPE':
                assert run.state(r) == 'COMPLETE'
                pools[item['condition']][r['key']] = r
    assert all(len(pool) == 9000 for pool in pools.values())
    scores, intervals = [], []
    strategies = ['ALL','random','popular','adversarial']
    for strategy in strategies:
        selected = {name: {k:r for k,r in pool.items() if strategy == 'ALL' or r['strategy'] == strategy}
                    for name,pool in pools.items()}
        assert all(len(pool) == (9000 if strategy == 'ALL' else 3000) for pool in selected.values())
        for name, pool in selected.items():
            scores.append(dict(condition=name, strategy=strategy, **pope(list(pool.values()))))
        for baseline in ['Native','SVH']:
            intervals.append(dict(strategy=strategy, candidate=run.NAME, baseline=baseline, metric='F1_pp',
                **f1_paired(selected[baseline], selected[run.NAME])))
    run.table(run.OUT/'pope_overall_and_strategy_scores.csv', scores)
    run.table(run.OUT/'pope_f1_paired_comparisons.csv', intervals)
    env = dict(python=platform.python_version(), platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ['torch','transformers','numpy','Pillow']},
        model_revision=run.MODEL.name, model_loaded_by_this_script=False,
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version,memory.used,utilization.gpu', '--format=csv'], text=True))
    run.write(run.OUT/'runtime_environment.json', env)
    run.write(run.OUT/'final_gpu_release_audit.json', dict(status='OWN_WORKERS_RELEASED', worker_pids=launch['worker_pids'],
        live_processes=processes, external_processes_untouched=True, source_audit_sha256=run.sha(run.OUT/'gpu_release_audit.json')))
    summaries = read_csv('accuracy_summary.csv')
    comparisons = read_csv('paired_comparisons.csv')
    lookup = {(r['benchmark'],r['condition']): r for r in summaries}
    lines = ['# qwen-SVH-R L19–L35：六项固定配置质量结果', '',
        '**配置已固定：P−1及全部真实Decode统一在0-based L19–L35回看，共17层。**', '',
        '采用方案B（ATE统一HF），不是作者官方复现。ScienceQA/MMBench为冻结P2；其余Prompt与视觉预算不变。',
        'R写入L0–L21、主路径L22前视觉退出、L1–L6答案Attention跳算均保持不变。', '',
        '## 分数', '', '| Benchmark | Native | SVH | qwen-SVH-R | 覆盖 |', '|---|---:|---:|---:|---|']
    for task in run.COUNTS:
        entries = [lookup[task,name] for name in pools]
        common = int(entries[0]['paired_common_questions'])
        complete = all(e['coverage_status'] == 'COMPLETE_FULL_TASK' for e in entries)
        def text(entry):
            field = 'full_official_score' if complete else 'paired_common_score'
            if entry[field]:
                return f"{float(entry[field]):.4f}"+('' if complete else '（共同子集）')
            return 'BLOCKED：无完整总分'
        coverage = '完整' if complete else f"不完整；三方共同有效{common}题"
        lines.append('| '+task+' | '+' | '.join(text(e) for e in entries)+' | '+coverage+' |')
    lines += ['', '表中POPE为Accuracy，MMBench为完整Circular准确率；TextVQA为十答案软分。MME应为14任务Acc+Acc+总分，不用普通准确率代替。', '',
        '### POPE F1', '', '| 策略 | Native | SVH | qwen-SVH-R |', '|---|---:|---:|---:|']
    for strategy in strategies:
        values = {r['condition']: r for r in scores if r['strategy'] == strategy}
        lines.append('| '+strategy+' | '+' | '.join(f"{values[n]['f1']:.4f}" for n in pools)+' |')
    lines += ['', '## 逐题配对变化', '', '| 任务 | 对照 | R减对照 | 图片簇95%CI | 损害/纠正 | 同题数 |', '|---|---|---:|---|---:|---:|']
    for r in comparisons:
        delta = f"{float(r['delta']):+.4f}" if r['delta'] else 'N/A'
        ci = f"[{float(r['low']):.4f}, {float(r['high']):.4f}]" if r.get('low') else 'N/A'
        lines.append(f"| {r['benchmark']} | {r['baseline']} | {delta} | {ci} | {r['harm']}/{r['rescue']} | {r['n']} |")
    lines += ['', '全部为20,000次Image SHA整簇配对Bootstrap；保持MME双题、MMBench完整Circular。TextVQA损害/纠正为软分下降/上升。',
        '普通探索性区间，不据CI跨0宣称无损、等效或非劣效；本队列包含选层时重复使用的Discovery图片。', '',
        '## 完整性与局限', '',
        '- 所有35,346个固定输入均已核验并处理，含292条严格复用；实际成功、OOM与分组覆盖逐项保存在accuracy_summary.csv。',
        '- 原Eager Native在TextVQA已有5条OOM、MME已有406条OOM；历史失败未删除或改判。新配置失败同样保留。',
        '- 未缩图、换精度/后端、改评分器或丢弃失败题。缺失不会记错后混入分数，也不会用可用子集冒充完整测试。',
        '- 选层时与本轮使用同一功能质量参考，含因果视觉前缀sidecar；参考保留30个active层视觉KV，不能据17层回看声称17层持久KV或真实加速。',
        '- 历史严格数值FAIL_RETAINED继续保留；没有新增数值PASS，没有运行性能Benchmark。',
        '- GQA的Query/Non-Query见gqa_query_nonquery.csv；POPE各项指标及F1区间、MME子任务、全部逐题答案与负例另存CSV。',
        '- 回看改善属于答案直接访问视觉的收益，不能称原SVH已完成无损信息交接。单一固定配置结果不证明层集合全局最优。', '',
        '## 资源与复现', '',
        '- 本任务GPU1/GPU3的两个Worker已退出；GPU0/GPU2及其他进程未干扰。',
        f"- 双卡阶段墙钟{generation['elapsed_s']:.2f}秒，含模型加载和生成，不含CPU准备/统计；不是方法E2E延迟或加速比。",
        '- 冻结代码Commit：706a36edc8d90ae98d1bc07c9d7849524c9f2993；分支codex/qwen-svh-r-l19-l35-sixbench。',
        '- CPU入口：summarize_qwen_svh_r.py；gqa_svh_r_subgroups.py；finish_qwen_svh_r.py。',
        '- 全部结果：'+str(run.OUT), '', '到此停止，不自动搜索新层集合或追加实验。']
    (run.OUT/'final_report_zh.md').write_text('\n'.join(lines)+'\n')
    files = {str(p.relative_to(run.ROOT)): dict(sha256=run.sha(p), bytes=p.stat().st_size)
        for p in run.OUT.iterdir() if p.is_file() and not p.name.endswith(('.sqlite','.sqlite-wal','.sqlite-shm')) and p.name != 'final_artifact_index.json'}
    for name in ['qwen_svh_r_full.py','summarize_qwen_svh_r.py','gqa_svh_r_subgroups.py','finish_qwen_svh_r.py','test_qwen_svh_r_full.py','QWEN_SVH_R_FIXED.md']:
        path = run.ROOT/name
        files[name] = dict(sha256=run.sha(path), bytes=path.stat().st_size)
    run.write(run.OUT/'final_artifact_index.json', dict(status='FROZEN_QUALITY_EVALUATION_FINISHED',
        coverage=completion['status'], numerical_status='FAIL_RETAINED', own_gpu_workers_released=True, artifacts=files))
    print('Final report:', run.OUT/'final_report_zh.md')


if __name__ == '__main__':
    main()
