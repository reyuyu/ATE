"""Complete29-window accuracy inference, always including the reused L2-L5."""
import csv
import json
import numpy as np
from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.analyze import anomaly
from paper_rq1_early_joint_attention.run import checkpoint_valid
from paper_rq1_early_joint_attention.statistics import bootstrap
from decode_five_vs_six import ENV_KEYS
from .run import OUT, CONDITIONS, NEW, TASKS, SEED, cohort, rank_for, read, reused, reused_path, verify

KEYS = ['Full', *CONDITIONS]


def write_csv(name, records):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def summarize(values, gens, label, images, intervals=None):
    rows = []
    for k, key in enumerate(KEYS):
        s = intervals[k - 1] if k and intervals is not None else None
        flags = [anomaly(g[k]['token_ids'], g[k]['text'], g[k]['eos']) for g in gens]
        rows.append(dict(group=label, condition=key, start_layer=k - 1 if k else '',
            end_layer=k + 2 if k else '', n=len(values), images=len(set(images)),
            correct=int(values[:, k].sum()), accuracy_pct=100 * float(values[:, k].mean()),
            delta_pp=100 * float((values[:, k] - values[:, 0]).mean()),
            ci95_low_pp=100 * s['ci95_low'] if s else '', ci95_high_pp=100 * s['ci95_high'] if s else '',
            corrected95_low_pp=100 * s['family95_low'] if s else '', corrected95_high_pp=100 * s['family95_high'] if s else '',
            noninferiority=bool(s['family95_low'] > -.01) if s else '',
            degenerate_bootstrap=s['degenerate_bootstrap'] if s else '',
            correct_to_wrong=int(((values[:, 0] == 1) & (values[:, k] == 0)).sum()),
            wrong_to_correct=int(((values[:, 0] == 0) & (values[:, k] == 1)).sum()),
            first_top1_flip_count=sum(g[k]['token_ids'][0] != g[0]['token_ids'][0] for g in gens),
            first_top1_flip_pct=100 * float(np.mean([g[k]['token_ids'][0] != g[0]['token_ids'][0] for g in gens])),
            eos=sum(g[k]['eos'] for g in gens), truncated=sum(g[k]['truncated'] for g in gens),
            empty=sum(f['empty_answer'] for f in flags), repetition=sum(f['repetition_flag'] for f in flags),
            text_changed=sum(g[k]['text'] != g[0]['text'] for g in gens),
            mean_generated_tokens=float(np.mean([g[k]['length'] for g in gens])),
            source='reused-Figure3' if key == 'Full' else 'reused-E' if key == 'W02' else 'new'))
    return rows


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10})
    rows = summary[1:]
    x = np.arange(29)
    y = np.array([r['delta_pp'] for r in rows])
    low = np.array([r['corrected95_low_pp'] for r in rows])
    high = np.array([r['corrected95_high_pp'] for r in rows])
    ordinary_low = np.array([r['ci95_low_pp'] for r in rows])
    ordinary_high = np.array([r['ci95_high_pp'] for r in rows])
    zoom = high.max() - low.min() > 12
    if zoom:
        fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.9), sharex=True,
            gridspec_kw={'height_ratios': [1.8, 1], 'hspace': .18})
        axes[1].set_ylim(-3, 1.5)
        axes[1].set_title('Detail near the -1 pp boundary (full range above)', loc='left', fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(7.1, 3.6))
        axes = [ax]
    for ax in axes:
        ax.fill_between(x, ordinary_low, ordinary_high, color='#b8c7d2', alpha=.5, label='Ordinary95%CI')
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), color='#3b5567',
            ecolor='#76858f', fmt='o-', markersize=2.8, linewidth=1.15, elinewidth=.85,
            capsize=1.7, label='29-comparison corrected95%CI')
        ax.axhline(0, color='#888888', linewidth=.7)
        ax.axhline(-1, color='#b24e43', linestyle='--', linewidth=1, label='Noninferiority: -1 pp')
        ax.plot(2, y[2], marker='*', color='#087e75', markersize=9, zorder=5, label='L2-L5 (reused)')
        ax.set_xlim(-.5, 28.5)
        ax.set_ylabel('Accuracy change (pp)')
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18, linewidth=.5)
        ax.set_axisbelow(True)
    axes[0].set_title('Four-layer joint Attention ablation during P-1 + Decode', pad=9)
    axes[0].legend(loc='best', frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel('Window start layer s (jointly ablate Ls-Ls+3)')
    fig.text(.5, .015, '3,000 questions / 2,409 images; paired image-cluster bootstrap; exploratory, not independent confirmation',
        ha='center', fontsize=7)
    fig.subplots_adjust(left=.10, right=.99, bottom=.13, top=.91)
    fig.savefig(OUT / 'four_layer_window_curve.pdf')
    fig.savefig(OUT / 'four_layer_window_curve.png', dpi=300)
    plt.close(fig)


def report(summary, tasks, completed):
    rows = summary[1:]
    e = rows[2]
    passing = [f"L{r['start_layer']}–L{r['end_layer']}" for r in rows if r['noninferiority']]
    best = sorted(rows, key=lambda r: (-r['accuracy_pct'], r['start_layer']))[:3]
    worst = sorted(rows, key=lambda r: (r['accuracy_pct'], r['start_layer']))[:3]
    lines = ['# Research 1：四层 Attention 滑动窗口探索', '',
        '完整完成29个连续四层窗口，全部使用P−1+Decode。原六类GQA各500题，共3,000题、2,409个图像簇。',
        '复用Full与L2–L5各3,000条；其余28个窗口新增84,000次完整自由生成，逐条件独立真实前缀与KV Cache。', '',
        f"Full Accuracy：{summary[0]['accuracy_pct']:.3f}%。主图：`four_layer_window_curve.pdf` / `.png`。", '',
        '## 核心结论', '',
        f"- L2–L5：{e['accuracy_pct']:.3f}%，ΔFull {e['delta_pp']:+.3f}pp；29项校正95%CI [{e['corrected95_low_pp']:+.3f},{e['corrected95_high_pp']:+.3f}]pp；非劣效{'达到' if e['noninferiority'] else '未达到'}。本次必须重新使用family29区间，而非旧family5区间。",
        '- 达到总体−1pp非劣效标准的窗口：' + ('、'.join(passing) if passing else '无') + '。',
        '- 观察Accuracy最高三项：' + '；'.join(f"L{r['start_layer']}–L{r['end_layer']} {r['accuracy_pct']:.3f}% ({r['delta_pp']:+.3f}pp)" for r in best) + '。',
        '- 最低三项：' + '；'.join(f"L{r['start_layer']}–L{r['end_layer']} {r['accuracy_pct']:.3f}% ({r['delta_pp']:+.3f}pp)" for r in worst) + '。',
        f"- L2–L5仍有{e['correct_to_wrong']}题原对变错、{e['wrong_to_correct']}题原错变对，首Token翻转率{e['first_top1_flip_pct']:.3f}%；达到非劣效不意味着零损害，也不表示显著优于Full。",
        '- 这只是原3,000题上的探索性层段比较，不是L2–L5独立确认；不能把探索排序当成独立测试成绩，也不能把相关性或输出置零解释为真实计算加速。', '',
        '## 全部窗口', '',
        '| 窗口 | Accuracy% | ΔFull pp | 校正95%CI pp | 对→错 | 错→对 | 首Token翻转% | 非劣效 |',
        '|---|---:|---:|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| L{r['start_layer']}–L{r['end_layer']} | {r['accuracy_pct']:.3f} | {r['delta_pp']:+.3f} | [{r['corrected95_low_pp']:+.3f},{r['corrected95_high_pp']:+.3f}] | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_top1_flip_pct']:.3f} | {'达到' if r['noninferiority'] else '未达到'} |")
    lines += ['', '## 协议与有效性', '',
        '- LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager、Transformers4.49.0；原Prompt、Greedy、KV Cache及官方GQA精确答案评分均保持。',
        '- 仅指定层o_proj输出置零：Prefill仅原始最后Query，Decode仅当前Query；全部其他Prefill Query及MLP正常。复用原三卡图片分配，不干扰GPU0或其他GPU进程；GPU忙时先等待。',
        '- CPU测试及每卡真实Smoke先于该卡正式生成；Native Full及E的历史Token/首Logits精确复现；No-op有效Logits逐步精确一致；其他Prefill Query在全部32层保持原值；Hook清理与输入/参数版本检查通过。',
        '- 按图像SHA256整簇配对Bootstrap1,000,000次，seed20261031；题目加权Accuracy；普通95%与Bonferroni29校正95%CI。非劣效要求校正下界严格大于−1pp；CI含0不是等价证据。退化Bootstrap另记标志，不解释为总体零风险。',
        '- 子任务只报告冻结500题上的分数/翻转等描述性结果，不另作174项同时显著性声明。未增加BOS、ROI或其他机制分析。',
        '- 生成上限沿用原32新Token，遇EOS提前停止；截断、空输出和重复输出全部保留计分，绝不因异常或表现删除题目。',
        '- 各新窗口EOS/截断/空输出/重复标记：' + '；'.join(f"L{r['start_layer']}–L{r['end_layer']} {r['eos']}/{r['truncated']}/{r['empty']}/{r['repetition']}" for r in rows) + '。',
        '- 分片正式运行秒数：' + ' / '.join(f"{d['seconds']:.1f}" for d in completed) + '（资源等待和Smoke另见日志）。',
        '- `accuracy_summary.csv`为全部主结果；`task_accuracy.csv`包含六类分数；`per_question.csv`包含90,000条生成答案及Token/逐题评分，其中6,000条注明复用来源。检查点、模型/数据/源码绑定、CPU/GPU记录及哈希均保留。',
        '- 基线Commit：' + 'e495adf4fd93337a3638981bbb66c2bbac81242d。独立分支codex/four-layer-p1-decode-scan，不修改main。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    c = verify()
    binding = digest(c)
    completed = [read(OUT / f'shard{k}/completion.json') for k in range(3)]
    accepted = [read(OUT / f'shard{k}/smoke.json') for k in range(3)]
    assert all(d['status'] == 'complete' and d['binding'] == binding and d['hooks_clean'] and d['parameters_unchanged'] for d in completed)
    assert all(d['binding'] == binding and d['validation']['passed'] for d in accepted)
    shared = [k for k in ENV_KEYS if k != 'cuda_visible_devices']
    assert len({digest({k: a['environment'][k] for k in shared}) for a in accepted}) == 1
    bindings = {(r['task'], r['question_id']): r for r in read(OUT / 'reuse_bindings.json')}
    samples = cohort()
    scores, images, gens, per_question = [], [], [], []
    for sample in samples:
        current = []
        directory = OUT / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}"
        for key in KEYS:
            if key in ('Full', 'W02'):
                assert sha256(reused_path(sample, key)) == bindings[sample['task'], sample['question_id']][key + '_sha256']
                g = reused(sample, key, c)
            else:
                path = directory / f'{key}.json'
                assert checkpoint_valid(path, binding, sample, key)
                g = read(path)['generation']
            ids = g['token_ids']
            assert 0 < len(ids) <= 32 and 2 not in ids[:-1]
            assert g['eos'] == (ids[-1] == 2) and g['truncated'] == (len(ids) == 32 and ids[-1] != 2)
            assert g['prediction'] == gqa_prediction(g['text']) and g['score'] == int(g['prediction'] == sample['answer'])
            assert g['edits'] == (0 if key == 'Full' else 4 * len(ids))
            assert g['own_prefill'] and g['own_prefix']
            if key != 'Full':
                assert g['first_token'] == ids[0] and g['full_first_token'] == current[0]['token_ids'][0]
                assert g['first_top1_flip'] == (ids[0] != current[0]['token_ids'][0])
            flags = anomaly(ids, g['text'], g['eos'])
            current.append(g)
            per_question.append(dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], gold=sample['answer'], condition=key,
                start_layer='' if key == 'Full' else int(key[1:]),
                end_layer='' if key == 'Full' else int(key[1:]) + 3,
                text=g['text'], prediction=g['prediction'], token_ids=json.dumps(ids), score=g['score'],
                full_score=current[0]['score'], delta_score=g['score'] - current[0]['score'],
                first_token=ids[0], full_first_token=current[0]['token_ids'][0],
                first_top1_flip=ids[0] != current[0]['token_ids'][0], eos=g['eos'], truncated=g['truncated'],
                generated_tokens=len(ids), empty=flags['empty_answer'], repetition=flags['repetition_flag'],
                source='reused-Figure3' if key == 'Full' else 'reused-E' if key == 'W02' else 'new'))
        gens.append(current)
        images.append(sample['image_sha256'])
        scores.append([g['score'] for g in current])
    assert sum(len(list((OUT / f'shard{k}/checkpoints').glob('*/*.json'))) for k in range(3)) == 84000
    values, identities = np.asarray(scores, dtype=float), np.asarray(images)
    assert values.shape == (3000, 30) and int(values[:, 0].sum()) == 2627 and int(values[:, 3].sum()) == 2630
    print('Coverage and reuse verified; bootstrap1m/family29 starting', flush=True)
    intervals = bootstrap(values[:, 1:] - values[:, [0]], identities, [29] * 29, repetitions=1000000, seed=SEED)
    summary = summarize(values, gens, 'Overall', identities, intervals)
    tasks = []
    for task in TASKS:
        selected = np.asarray([i for i, s in enumerate(samples) if s['task'] == task])
        tasks += summarize(values[selected], [gens[i] for i in selected], task, identities[selected])
    write_csv('accuracy_summary.csv', summary)
    write_csv('task_accuracy.csv', tasks)
    write_csv('per_question.csv', per_question)
    figure(summary)
    report(summary, tasks, completed)
    artifacts = ['accuracy_summary.csv', 'task_accuracy.csv', 'per_question.csv',
        'four_layer_window_curve.pdf', 'four_layer_window_curve.png', 'report.md']
    freeze_json(OUT / 'result.json', dict(status='complete', binding=binding, questions=3000, images=2409,
        windows=29, new_generations=84000, reused_generations=6000, summary=summary,
        validation=dict(cpu=True, all_gpu_smoke=True, source_and_environment_match=True,
            reuse_hashes_verified=True, coverage_complete=True), artifact_sha256={n: sha256(OUT / n) for n in artifacts}))
    print('Complete', json.dumps(summary), flush=True)


if __name__ == '__main__': main()
