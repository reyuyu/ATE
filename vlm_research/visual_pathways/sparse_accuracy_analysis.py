"""Image-cluster paired summaries for the frozen six-condition experiment."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .common import ROOT, TASKS, read, write, sha, digest, command
from .analyze import csv_write, boot_mean

OUT = ROOT/'vlm_research/results/qwen_internvl_sparse_visual_3000'
NAMES = ('Full', 'Only Text', 'Sparse K1', 'Sparse K3', 'Sparse K8', 'Low-8 OFF')
LLAVA = Path('/data/lk/ATE-worktrees/rq2-gqa-text-vs-sparse-3000/vlm_research/results/rq2_gqa_text_vs_sparse_3000')


def csv_read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def aggregate(model):
    samples = read(OUT/'sample_manifest.json')['samples']
    protocol = read(OUT/'protocol.json')
    all_records, flat = [], []
    gpu_owners, commits = {}, set()
    for sample in samples:
        path = OUT/model/'checkpoints'/f'{sample["task"]}__{sample["question_id"]}.json'
        record = read(path)
        assert record['complete'] and set(record['conditions']) == set(NAMES)
        assert record['sample_sha256'] == digest(sample) and record['protocol_sha256'] == sha(OUT/'protocol.json')
        assert record['image_sha256'] == sample['image_sha256']
        assert gpu_owners.setdefault(sample['image_sha256'], record['physical_gpu']) == record['physical_gpu']
        assert record['physical_gpu'] in (0, 2, 3)
        commits.add(record['code_commit'])
        for name in NAMES:
            val = record['conditions'][name]
            assert val['correct'] == int(val['prediction'] == sample['answer'])
            assert val['fresh_cache'] and val['token_ids'] and len(val['token_ids']) == val['generated_tokens']
            assert len(val['all_step_logits_sha256']) == val['generated_tokens']
            assert set(map(int, val['mask_audits'])) == set(protocol['conditions'][model][name])
            for audit in val['mask_audits'].values():
                assert audit['prefill_calls'] == 1 and audit['decode_calls'] == val['generated_tokens']-1
                assert audit['masked_probability_max'] == 0 and audit['row_sum_max_error'] <= 2**-8+2e-6
            flat.append(dict(model=model, task=sample['task'], question_id=sample['question_id'],
                image_id=sample['image_id'], image_sha256=sample['image_sha256'], gold=sample['answer'],
                condition=name, correct=val['correct'], prediction=val['prediction'], text=val['text'],
                token_ids=json.dumps(val['token_ids']), first_token_flip=val['top1_flip'],
                signed_logp_loss=val['signed_logp_loss'], absolute_logp_change=val['absolute_logp_change'],
                js_divergence=val['js_divergence'],
                token_sequence_changed=int(val['token_ids'] != record['conditions']['Full']['token_ids']),
                text_changed=int(val['text'] != record['conditions']['Full']['text']),
                eos=val['eos'], truncated=val['truncated'], empty=val['empty'],
                generated_tokens=val['generated_tokens'], physical_gpu=record['physical_gpu']))
        all_records.append(record)
    assert len(samples) == 3000 and len(gpu_owners) == 2409 and len(commits) == 1
    checks = []
    for gpu in (0, 2, 3):
        folder = OUT/model/f'gpu{gpu}'
        v, c = read(folder/'validation.json'), read(folder/'complete.json')
        assert v['passed'] and c['passed'] and c['questions'] == 1000
        assert c['code_commit'] in commits and v['code_commit'] in commits
        assert c['weights_unchanged'] and c['hooks_clean'] and c['no_nan_inf_oom']
        checks.append(c)
    scores = np.array([[r['conditions'][n]['correct'] for n in NAMES] for r in all_records], float)
    # A single paired resample carries all six scores and five Full differences.
    matrix = np.concatenate([scores*100, (scores[:, 1:]-scores[:, :1])*100,
        np.stack([(scores[:, j]-scores[:, 1])*100 for j in (2, 3, 4)], axis=1)], axis=1)
    summary, tasks, paired = [], [], []
    for task in ('Overall',)+TASKS:
        ids = [i for i, s in enumerate(samples) if task == 'Overall' or s['task'] == task]
        selected = [samples[i] for i in ids]
        mean, low, high, _, groups = boot_mean(matrix[ids], selected)
        for j, name in enumerate(NAMES):
            values = [all_records[i]['conditions'][name] for i in ids]
            base = scores[ids, 0]
            current = scores[ids, j]
            k = 5+j
            row = dict(model=model, task=task, condition=name, n=len(ids), image_clusters=groups,
                correct=int(current.sum()), accuracy_percent=float(mean[j]), accuracy_ci_low=float(low[j]), accuracy_ci_high=float(high[j]),
                delta_full_pp=float(mean[k]) if j else 0., delta_ci_low=float(low[k]) if j else 0., delta_ci_high=float(high[k]) if j else 0.,
                full_correct_to_wrong=int(((base == 1)&(current == 0)).sum()),
                full_wrong_to_correct=int(((base == 0)&(current == 1)).sum()),
                first_token_flip_count=sum(v['top1_flip'] for v in values),
                token_sequence_changed_count=sum(v['token_ids'] != all_records[i]['conditions']['Full']['token_ids'] for i, v in zip(ids, values)),
                text_changed_count=sum(v['text'] != all_records[i]['conditions']['Full']['text'] for i, v in zip(ids, values)),
                mean_signed_logp_loss=float(np.mean([v['signed_logp_loss'] for v in values])),
                mean_generated_tokens=float(np.mean([v['generated_tokens'] for v in values])),
                eos=sum(v['eos'] for v in values), truncated=sum(v['truncated'] for v in values), empty=sum(v['empty'] for v in values),
                exploratory=True)
            (summary if task == 'Overall' else tasks).append(row)
        for k, j in enumerate((2, 3, 4)):
            a, b = scores[ids, j], scores[ids, 1]
            paired.append(dict(model=model, task=task, comparison=NAMES[j]+' - Only Text', n=len(ids),
                image_clusters=groups, delta_accuracy_pp=float(mean[11+k]), ci_low=float(low[11+k]), ci_high=float(high[11+k]),
                reference_correct_to_wrong=int(((b == 1)&(a == 0)).sum()), reference_wrong_to_correct=int(((b == 0)&(a == 1)).sum()),
                exploratory=True))
    folder = OUT/model
    csv_write(folder/'per_question.csv', flat)
    csv_write(folder/'accuracy_summary.csv', summary)
    csv_write(folder/'task_accuracy.csv', tasks)
    csv_write(folder/'paired_comparisons.csv', paired)
    write(folder/'integrity.json', dict(passed=True, questions=3000, generations=18000, image_clusters=2409,
        code_commit=next(iter(commits)), all_mask_probabilities_zero=True, all_decode_mask_calls_verified=True,
        same_image_single_gpu=True, hooks_clean=True, weights_unchanged=True, gpu_runs=checks))
    print(model, [(r['condition'], r['accuracy_percent'], r['delta_full_pp']) for r in summary], flush=True)


def llava_reference():
    samples = read(OUT/'sample_manifest.json')['samples']
    lookup = {(s['task'], s['question_id']): s for s in samples}
    rows = csv_read(LLAVA/'per_question.csv')
    assert len(rows) == 3000
    for row in rows:
        sample = lookup[(row['task'], row['question_id'])]
        assert all(row[k] == sample[k] for k in ('image_id', 'image_sha256'))
        assert row['gold'] == sample['answer']
    protocol = read(LLAVA/'protocol.json')
    assert protocol['model_revision'] == 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    assert protocol['conditions']['Only_Text'] == 'Native relay; original P-1 and all Decode Image OFF all32'
    assert protocol['conditions']['Sparse'] == 'Native relay; original P-1 and all Decode Image OFF except L14/L19'
    summary = csv_read(LLAVA/'accuracy_summary.csv')
    csv_write(OUT/'llava_historical_accuracy.csv', summary)
    scores = np.array([[float(row[c+'_score']) for c in ('Full', 'Only_Text', 'Sparse')] for row in rows])
    paired_samples = [lookup[(r['task'], r['question_id'])] for r in rows]
    mean, low, high, _, groups = boot_mean(np.concatenate([scores*100, (scores[:, 1:]-scores[:, :1])*100], axis=1), paired_samples)
    reference = []
    for j, name in enumerate(('Full', 'Only_Text', 'Sparse')):
        reference.append(dict(condition=name, n=3000, image_clusters=groups,
            accuracy_percent=float(mean[j]), ci_low=float(low[j]), ci_high=float(high[j]),
            delta_full_pp=float(mean[j+2]) if j else 0.,
            delta_ci_low=float(low[j+2]) if j else 0., delta_ci_high=float(high[j+2]) if j else 0.))
    csv_write(OUT/'llava_cluster_reference.csv', reference)
    write(OUT/'llava_historical_binding.json', dict(questions=3000, passed=True,
        original_worktree=str(LLAVA), commit=command('git', '-C', str(LLAVA), 'rev-parse', 'HEAD'),
        files={n: sha(LLAVA/n) for n in ('protocol.json', 'per_question.csv', 'accuracy_summary.csv')},
        caveat='Historical LLaVA native prompt/FP16/576 image tokens; sparse keeps L14/L19 (K2), not a measured K1/K3/K8. No new LLaVA inference.'))


def figures_and_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    summaries = {m: csv_read(OUT/m/'accuracy_summary.csv') for m in ('qwen', 'internvl')}
    selection = {}
    protocol = read(OUT/'protocol.json')
    for model, root in [('qwen', ROOT/'vlm_research/results/visual_pathways_seven_hour'),
                        ('internvl', ROOT/'vlm_research/results/internvl_native_replay_recovery')]:
        path = root/model/'layer_summary.csv'
        rows = [r for r in csv_read(path) if r['cohort'] == 'all' and r['task'] == 'Overall']
        order = sorted(rows, key=lambda r: (-float(r['mean_absolute_logp_change']), int(r['layer'])))
        rank = [int(r['layer']) for r in order]
        for count in (1, 3, 8):
            retained = set(range(len(rank)))-set(protocol['conditions'][model][f'Sparse K{count}'])
            assert retained == set(rank[:count])
        assert set(protocol['conditions'][model]['Low-8 OFF']) == set(rank[-8:])
        selection[model] = dict(passed=True, ranking=rank, historical_csv_sha256=sha(path),
            based_on_same_3000=True, no_current_accuracy_selection=True)
    write(OUT/'layer_selection_audit.json', selection)
    allrows = [row for rows in summaries.values() for row in rows]
    csv_write(OUT/'accuracy_summary.csv', allrows)
    csv_write(OUT/'task_accuracy.csv', [r for m in summaries for r in csv_read(OUT/m/'task_accuracy.csv')])
    csv_write(OUT/'paired_comparisons.csv', [r for m in summaries for r in csv_read(OUT/m/'paired_comparisons.csv')])
    llava_reference()
    llava = csv_read(OUT/'llava_cluster_reference.csv')
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), layout='constrained', sharey=True)
    for ax, model, depth, color in zip(axes[:2], summaries, (36, 28), ('#236995', '#248064')):
        data = {r['condition']: r for r in summaries[model]}
        names = ['Only Text', 'Sparse K1', 'Sparse K3', 'Sparse K8', 'Low-8 OFF', 'Full']
        x = np.array([0, 1, 3, 8, depth-8, depth])
        y = np.array([float(data[n]['accuracy_percent']) for n in names])
        low = np.array([float(data[n]['accuracy_ci_low']) for n in names])
        high = np.array([float(data[n]['accuracy_ci_high']) for n in names])
        ax.errorbar(x, y, yerr=[y-low, high-y], color=color, marker='o', markersize=4, capsize=3, lw=1.4)
        ax.axhline(y[-1], color=color, ls='--', lw=.8, alpha=.65)
        for a, b, label, offset in zip(x, y, ('Text', 'K1', 'K3', 'K8', 'Low-8 OFF', 'Full'),
                                      [(2, -15), (0, 11), (0, -15), (0, 11), (0, -15), (0, 11)]):
            ax.annotate(label, (a, b), xytext=offset, textcoords='offset points', ha='left' if label == 'Text' else 'center', fontsize=7)
        ax.set_xticks(x)
        ax.get_xticklabels()[0].set_ha('right')
        ax.get_xticklabels()[1].set_ha('left')
        ax.set_title('Qwen3-VL-8B' if model == 'qwen' else 'InternVL3-8B')
    vals = {r['condition']: r for r in llava}
    llava_y = np.array([float(vals[n]['accuracy_percent']) for n in ('Only_Text', 'Sparse', 'Full')])
    llava_low = np.array([float(vals[n]['ci_low']) for n in ('Only_Text', 'Sparse', 'Full')])
    llava_high = np.array([float(vals[n]['ci_high']) for n in ('Only_Text', 'Sparse', 'Full')])
    axes[2].errorbar([0, 2, 32], llava_y, yerr=[llava_y-llava_low, llava_high-llava_y], marker='s', capsize=3, color='#b55240', label='Historical LLaVA')
    axes[2].set_xticks([0, 2, 32])
    axes[2].set_title('LLaVA-1.5-7B (historical)')
    axes[2].text(.04, .88, 'K2 = L14 / L19\nNo K1/K3/K8 measurements', transform=axes[2].transAxes, fontsize=8)
    for ax in axes:
        ax.set_xlabel('Layers retaining direct Image access')
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Full-answer GQA accuracy (%)')
    fig.suptitle('P-1 + all Decode queries | 3,000 same questions | exploratory, selected on same cohort', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'accuracy_vs_visual_layers.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), layout='constrained', sharey=True)
    for ax, (model, rows) in zip(axes, summaries.items()):
        y = np.array([float(r['delta_full_pp']) for r in rows])
        low = np.array([float(r['delta_ci_low']) for r in rows])
        high = np.array([float(r['delta_ci_high']) for r in rows])
        ax.errorbar(range(6), y, yerr=[y-low, high-y], fmt='o', capsize=3, color='#354c5b')
        ax.axhline(0, color='#777', lw=.8)
        ax.set_xticks(range(6), ['Full', 'Text', 'K1', 'K3', 'K8', 'Low-8\nOFF'])
        ax.set_title(model)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Accuracy change vs paired Full (pp)')
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'paired_accuracy_changes.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Qwen / InternVL 视觉来源稀疏化 Accuracy（3,000题探索验证）', '',
        '完整覆盖六类 GQA 各500题、2,409个图片簇。两个模型顺序执行；每个模型物理 GPU0/2/3 按整图片簇分片各1,000题，未使用GPU1。',
        '固定历史 Revision、官方 Prompt、BF16/eager。Qwen DeepStack 原生；InternVL max_num=1、256视觉Token、logits_to_keep=1。',
        '仅在原始 P−1 与每个真实增量 Decode Query 上对指定 Image Key 施加 Softmax 前 -inf；其他 Prefill、MLP、残差、视觉计算原生。每条件独立Cache。Greedy最多32Token。',
        'Full 全部重新配对生成，并逐题精确校验首步全词表logits哈希及原始输入与历史来源扫描一致。未复用 P−1-only 或 Relay OFF 答案。', '',
        '## 总体结果', '', '|模型|条件|Accuracy %|ΔFull pp（配对95%CI）|原对→错|原错→对|首Token翻转|EOS/截断/空|',
        '|---|---|---:|---|---:|---:|---:|---|']
    for r in allrows:
        lines.append(f"|{r['model']}|{r['condition']}|{float(r['accuracy_percent']):.4f}|{float(r['delta_full_pp']):+.4f} [{float(r['delta_ci_low']):+.4f}, {float(r['delta_ci_high']):+.4f}]|{r['full_correct_to_wrong']}|{r['full_wrong_to_correct']}|{r['first_token_flip_count']}|{r['eos']}/{r['truncated']}/{r['empty']}|")
    lines += ['', '## Sparse 相对 Only Text', '', '|模型|比较|ΔAccuracy pp（配对95%CI）|损害|纠正|', '|---|---|---|---:|---:|']
    for model in summaries:
        for r in csv_read(OUT/model/'paired_comparisons.csv'):
            if r['task'] == 'Overall':
                lines.append(f"|{model}|{r['comparison']}|{float(r['delta_accuracy_pp']):+.4f} [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]|{r['reference_correct_to_wrong']}|{r['reference_wrong_to_correct']}|")
    lines += ['', '## 六任务 Accuracy %', '', '|模型|任务|Full|Only Text|K1|K3|K8|Low-8 OFF|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in summaries:
        taskrows = csv_read(OUT/model/'task_accuracy.csv')
        for task in TASKS:
            values = {r['condition']: r['accuracy_percent'] for r in taskrows if r['task'] == task}
            lines.append('|'+model+'|'+task+'|'+'|'.join(f'{float(values[n]):.2f}' for n in NAMES)+'|')
    lines += ['', '## 核心结论', '']
    for model, rows in summaries.items():
        data = {r['condition']: r for r in rows}
        one, eight, low8 = [data[n] for n in ('Sparse K1', 'Sparse K8', 'Low-8 OFF')]
        lines.append(f"- {model}：K1相对Full为{float(one['delta_full_pp']):+.4f} pp（95%CI [{float(one['delta_ci_low']):+.4f}, {float(one['delta_ci_high']):+.4f}]），损害{one['full_correct_to_wrong']}题、纠正{one['full_wrong_to_correct']}题。K8相对Full为{float(eight['delta_full_pp']):+.4f} pp（[{float(eight['delta_ci_low']):+.4f}, {float(eight['delta_ci_high']):+.4f}]）。不能仅凭总体点估计接近就说K1已经充分或严格无损。")
        lines.append(f"- {model}：Low-8 OFF相对Full为{float(low8['delta_full_pp']):+.4f} pp（95%CI [{float(low8['delta_ci_low']):+.4f}, {float(low8['delta_ci_high']):+.4f}]），损害{low8['full_correct_to_wrong']}题、纠正{low8['full_wrong_to_correct']}题；完整生成文本改变{low8['text_changed_count']}题。联合低敏感来源关闭的净得分不能替代个体损害审查。")
    lines += ['- K1/K3/K8相对Only Text的增益及区间见上表：区间跨0不能被解释为已证明有益或严格无作用；点估计接近Full也不是等价检验。',
        '- 仅保留Qwen L26并未精确恢复Full；本轮不支持“只需L26即可严格无损”。InternVL的K3点估计低于K1，K8虽更接近Full，但本轮不能证明它必然需要比Qwen更多视觉层。',
        '- 不预设保留更多层必然单调改善。各模型只测了固定层集合，不能据此声称已找到最少视觉层数或全局最优组合。',
        '', '## 判断与边界', '',
        '图和CSV保留所有负结果。单层 OFF 敏感性排序不是 Keep 层充分性证明，也不能保证多个低敏感层联合关闭安全。Only Text 仅表示当前答案 Query 不直接读图：其他 Prefill 文本已融合视觉，不能理解为无视觉模型。',
        '本批3,000题也用于选择层集合；全部区间为图像SHA256整簇配对Bootstrap（10,000次、seed=20261102）的普通探索性95%CI，保持题目加权估计。不作独立非劣效、等价或严格无损声明，不用CI跨0解释为无损害。',
        '历史LLaVA使用同题及SHA256逐题绑定的Full/Only Text/K2结果，仅作描述性对照。不同原生Prompt、精度、图像处理与模型架构使模型间差异不可直接归因于层数。',
        '准确率曲线的横轴只表示保留视觉来源访问的层数，连接线是离散冻结条件的展示，不表示未测量层数或加速收益。',
        '完整生成文本、Token IDs、首Token LogP/JS、文本及序列变化、六任务和每题结果均保存在CSV；逐层逐Decode Mask零概率与行和审计保存在逐题检查点。', '']
    if (OUT/'postprocessing_recovery.json').exists():
        lines += ['离线绘图曾因隔离模型环境缺少Matplotlib退出；随后使用已有AFAC环境完成，未安装或修改模型环境、未重跑模型。失败日志pipeline.log与恢复记录postprocessing_recovery.json均保留。', '']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def delivery_audit():
    from .sparse_accuracy import verify
    verify()
    assert read(OUT/'pipeline_complete.json')['passed']
    assert not list(OUT.glob('**/error.json')) and not list(OUT.glob('**/*.tmp'))
    commits, pids, seconds, environments = set(), [], {}, {}
    for model in ('qwen', 'internvl'):
        integrity = read(OUT/model/'integrity.json')
        assert integrity['passed'] and integrity['generations'] == 18000
        commits.add(integrity['code_commit'])
        timings, envs = [], []
        for gpu in (0, 2, 3):
            folder = OUT/model/f'gpu{gpu}'
            assert read(folder/'validation.json')['passed']
            pids.append(read(folder/'preflight.json')['pid'])
            timings.append(read(folder/'complete.json')['formal_seconds'])
            envs.append(read(folder/'environment.json'))
        assert envs[0] == envs[1] == envs[2]
        assert envs[0]['versions']['transformers'] == ('4.57.6' if model == 'qwen' else '4.49.0')
        seconds[model] = max(timings)
        environments[model] = envs[0]
    assert len(commits) == 1
    for name in ('cpu_qwen.log', 'cpu_internvl.log', 'cpu_statistics.log'):
        assert '\nOK\n' in (OUT/name).read_text()
    active = command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader')
    active_pids = set(int(p.strip()) for p in active.splitlines() if p.strip())
    assert not (active_pids & set(pids)), 'Owned GPU process still alive'
    now = datetime.now().astimezone()
    start = datetime.fromisoformat('2026-11-02T12:49:42+08:00')
    audit = dict(passed=True, code_execution_commit=next(iter(commits)), model_questions=3000,
        generations_total=36000, cpu_tests_passed=True, gpu_smokes_passed=6,
        no_nan_inf_oom=True, hooks_clean=True, model_versions=environments,
        image_clusters=2409, physical_gpus=[0, 2, 3], prohibited_gpu=1,
        owned_pids=pids, active_owned_pids=[], owned_gpu_processes_released=True,
        gpu_inventory=command('nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'),
        started=start.isoformat(), finished=now.isoformat(), total_wall_minutes=(now-start).total_seconds()/60,
        formal_parallel_seconds=seconds,
        analysis_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            Path(__file__).with_name('sparse_accuracy_finish.py'), Path(__file__).with_name('test_sparse_statistics.py')]})
    write(OUT/'validation.json', audit)
    report = OUT/'report_zh.md'
    if '## 执行与资源' not in report.read_text():
        report.write_text(report.read_text()+f"\n## 执行与资源\n\n执行源码Commit：`{next(iter(commits))}`。两个模型共36,000次完整生成，六个真实GPU Smoke、两环境CPU回归和统计测试通过。\n\n正式GPU并行阶段：Qwen {seconds['qwen']/60:.2f}分钟，InternVL {seconds['internvl']/60:.2f}分钟。含数据/权重校验、实现、测试与交付前检查的墙钟约{audit['total_wall_minutes']:.2f}分钟（精确时间见validation.json）。\n\n本任务GPU0/2/3进程已全部退出，GPU1未使用或干预。main未修改。结果为同批选层后的探索验证，不声称独立非劣效或严格无损。\n")
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'result_hashes.json')
    write(OUT/'result_hashes.json', dict(count=len(files), files={str(p.relative_to(OUT)): sha(p) for p in files}))
    print('DELIVERY AUDIT PASS', len(files), 'artifacts;', audit['total_wall_minutes'], 'minutes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['qwen', 'internvl', 'report', 'audit'])
    args = parser.parse_args()
    if args.action == 'audit':
        delivery_audit()
    elif args.action == 'report':
        figures_and_report()
    else:
        aggregate(args.action)
