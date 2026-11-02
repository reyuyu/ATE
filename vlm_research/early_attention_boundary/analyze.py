"""Question-weighted image-cluster paired intervals and a concise pilot report."""
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import *


def paired_interval(delta, images):
    clusters = sorted(set(images))
    sums = np.array([sum(d for d, image in zip(delta, images) if image == c) for c in clusters])
    sizes = np.array([sum(image == c for image in images) for c in clusters])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(clusters), (20000, len(clusters)))
    draws = 100*sums[indices].sum(1)/sizes[indices].sum(1)
    ordinary = np.quantile(draws, [.025, .975])
    simultaneous = np.quantile(draws, [.025/9, 1-.025/9])
    return [float(v) for v in (*ordinary, *simultaneous)]


def main():
    verify()
    samples = read(OUT / 'sample_manifest.json')['samples']
    assert len(samples) == 120 and all(sha256(s['image_path']) == s['image_sha256'] for s in samples)
    rows, summaries, tasks, comparisons, completions = [], [], [], [], {}
    for name, conditions in CONDITIONS.items():
        completions[name] = read(OUT / f'{name}_completion.json')
        assert completions[name]['complete'] and read(OUT / f'{name}_smoke.json')['passed']
        by_label = {label: [] for label in conditions}
        for sample in samples:
            checkpoint = read(OUT / 'checkpoints' / name / (key(sample)+'.json'))
            assert checkpoint['sample'] == sample
            assert checkpoint['protocol_sha256'] == sha256(OUT / 'protocol.json')
            assert set(checkpoint['conditions']) == set(conditions)
            full = checkpoint['conditions']['Full']
            for label, result in checkpoint['conditions'].items():
                assert sha256(result['raw_path']) == result['raw_sha256']
                row = dict(model=name, condition=label, task=sample['task'], question_id=sample['question_id'],
                    image_id=sample['image_id'], image_sha256=sample['image_sha256'], question=sample['question'],
                    gold=sample['answer'], full_text=full['text'], text=result['text'],
                    full_correct=int(full['correct']), correct=int(result['correct']),
                    score_change=int(result['correct'])-int(full['correct']),
                    harm=int(full['correct'] and not result['correct']), repair=int(not full['correct'] and result['correct']),
                    first_token_flip=int(result['token_ids'][0] != full['token_ids'][0]),
                    sequence_changed=int(result['token_ids'] != full['token_ids']),
                    answer_text_changed=int(result['text'] != full['text']),
                    eos=int(result['eos']), eos_changed=int(result['eos'] != full['eos']),
                    truncated=int(result['truncated']), empty=int(result['empty']), anomaly=int(result['anomaly']),
                    token_ids=json.dumps(result['token_ids']), full_token_ids=json.dumps(full['token_ids']),
                    non_eos_tokens=result['non_eos_tokens'], full_non_eos_tokens=full['non_eos_tokens'],
                    length=result['length'], length_change=result['length']-full['length'],
                    signed_logp_loss=result['signed_logp_loss'], js_divergence=result['js_divergence'],
                    input_binding=result['input_binding'], raw_path=result['raw_path'], raw_sha256=result['raw_sha256'])
                rows.append(row)
                by_label[label].append(row)
        for label, selected in by_label.items():
            lo, hi, clo, chi = paired_interval([r['score_change'] for r in selected], [r['image_sha256'] for r in selected])
            summary = dict(model=name, condition=label, n=len(selected), image_clusters=len({r['image_sha256'] for r in selected}),
                accuracy=100*np.mean([r['correct'] for r in selected]),
                delta_pp=100*np.mean([r['score_change'] for r in selected]),
                ci95_low_pp=lo, ci95_high_pp=hi, corrected_ci_low_pp=clo, corrected_ci_high_pp=chi,
                harm=sum(r['harm'] for r in selected), repair=sum(r['repair'] for r in selected),
                first_token_flips=sum(r['first_token_flip'] for r in selected),
                sequence_changed=sum(r['sequence_changed'] for r in selected),
                answer_text_changed=sum(r['answer_text_changed'] for r in selected),
                eos=sum(r['eos'] for r in selected), truncated=sum(r['truncated'] for r in selected),
                empty=sum(r['empty'] for r in selected), anomalies=sum(r['anomaly'] for r in selected),
                signed_logp_loss=float(np.mean([r['signed_logp_loss'] for r in selected])),
                bootstrap_degenerate=bool(label != 'Full' and not any(r['harm'] or r['repair'] for r in selected)),
                exploratory=True, noninferiority_claim=False)
            summaries.append(summary)
            if label != 'Full':
                comparisons.append({k: v for k, v in summary.items() if k not in ('accuracy',)})
            for task in TASKS:
                group = [r for r in selected if r['task'] == task]
                tasks.append(dict(model=name, condition=label, task=task, n=len(group),
                    accuracy=100*np.mean([r['correct'] for r in group]),
                    delta_pp=100*np.mean([r['score_change'] for r in group]),
                    harm=sum(r['harm'] for r in group), repair=sum(r['repair'] for r in group)))
    assert len(rows) == 1320 and len({(r['model'], r['condition'], r['task'], r['question_id']) for r in rows}) == 1320
    csv_write(OUT / 'per_question.csv', rows)
    csv_write(OUT / 'accuracy_summary.csv', summaries)
    csv_write(OUT / 'task_accuracy.csv', tasks)
    csv_write(OUT / 'paired_comparisons.csv', comparisons)

    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), sharey=True)
    for axis, name in zip(axes, CONDITIONS):
        group = [r for r in summaries if r['model'] == name and r['condition'] != 'Full']
        for i, row in enumerate(group):
            primary = row['condition'] == ('L1-L8' if name == 'qwen' else 'L1-L4')
            axis.errorbar(i, row['delta_pp'], yerr=[[row['delta_pp']-row['corrected_ci_low_pp']],
                          [row['corrected_ci_high_pp']-row['delta_pp']]], fmt='o', capsize=4,
                          color='#b53338' if primary else '#367b77', markersize=7 if primary else 5)
        axis.axhline(0, color='#555555', linewidth=.8)
        axis.axhline(-1, color='#888888', linestyle='--', linewidth=1, label='-1 pp reference')
        axis.set_xticks(range(len(group)), [r['condition'].replace('-', '\u2013') for r in group])
        axis.set_xlabel('Attention output ablation window')
        axis.set_title('Qwen3-VL-8B' if name == 'qwen' else 'InternVL3-8B')
        axis.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Accuracy change vs Full (pp)')
    fig.tight_layout()
    fig.savefig(OUT / 'figure_boundary.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'figure_boundary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    release = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not release, ('GPU1 has a compute process; cannot certify release', release)
    write(OUT / 'gpu_release.json', dict(no_compute_process=True, snapshot=gpu_snapshot(), gpu_uuid=UUID))
    write(OUT / 'validation.json', dict(complete=True, sample_count=120, same_samples_both_models=True,
          image_clusters=len({s['image_sha256'] for s in samples}), result_rows=len(rows),
          no_duplicate_rows=True, all_checkpoint_raw_hashes_verified=True, no_dropped_samples=True,
          cpu=read(OUT / 'cpu_tests.json'), smoke={n: read(OUT / f'{n}_smoke.json') for n in CONDITIONS},
          only_physical_gpu1=True, models_ran_sequentially=True, gpu_released=True,
          no_noninferiority_claim=True, bootstrap_zero_flip_not_safety=True))
    lines = ['# 早层 Attention 低敏感区边界：120题探索', '',
        '六类GQA各20题，固定输入哈希抽样，两个模型共用样本。只用物理GPU1顺序执行。',
        '干预仅将原Prompt P−1与后续真实Decode Query的指定层o_proj输出置零；不跳过Block，不代表物理加速。', '',
        '| 模型 | 窗口 | Accuracy | ΔFull (pp) | 损害/纠正 | 普通95%CI | 九项校正95%CI |',
        '|---|---|---:|---:|---:|---|---|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['condition']} | {r['accuracy']:.3f}% | {r['delta_pp']:+.3f} | {r['harm']}/{r['repair']} | [{r['ci95_low_pp']:.3f}, {r['ci95_high_pp']:.3f}] | [{r['corrected_ci_low_pp']:.3f}, {r['corrected_ci_high_pp']:.3f}] |")
    lines += ['', '## 边界与结论', '',
        '−1pp仅是预先固定的参考线。本轮数据参与过发现阶段，不是独立确认；120题也不能可靠排除1pp总体损失。',
        '零准确率翻转时的经验Bootstrap [0,0]仅描述本样本，不是总体安全保证。CI跨0不能证明等价。', '']
    for name in CONDITIONS:
        group = [r for r in summaries if r['model'] == name and r['condition'] != 'Full']
        eligible = [r for r in group if r['delta_pp'] >= -1]
        if eligible:
            maximum = max(len(CONDITIONS[name][r['condition']]) for r in eligible)
            longest = ', '.join(r['condition'] for r in eligible if len(CONDITIONS[name][r['condition']]) == maximum)
            lines.append(f'- {name}：预设候选中，按点估计损失不超过1pp的最长窗口为 {longest}（{maximum}层）；仅作为后续候选，未确认非劣效或“死亡区”。')
        else:
            lines.append(f'- {name}：预设窗口均未达到点估计损失不超过1pp的参考线，不指定低损害边界。')
        changed = '; '.join(f"{r['condition']}：首Token翻转{r['first_token_flips']}、序列变化{r['sequence_changed']}、文本变化{r['answer_text_changed']}、EOS {r['eos']}/120、截断{r['truncated']}、空答案{r['empty']}" for r in group)
        lines.append(f'- {name}生成诊断：{changed}。')
        harmful = sorted((r for r in tasks if r['model'] == name and r['condition'] != 'Full'), key=lambda r: r['delta_pp'])[:3]
        lines.append('- 最明显的任务下降（每组仅20题，描述性）：'+ '; '.join(f"{r['condition']}/{r['task']} {r['delta_pp']:+.1f}pp" for r in harmful)+'。')
        c = completions[name]
        lines.append(f"- {name}正式生成耗时 {c['formal_seconds']/60:.2f}分钟；加载及Smoke {c['smoke_and_load_seconds']/60:.2f}分钟；进程显存监测峰值 {c['nvidia_smi_peak_mib']}MiB。")
    lines += ['', '## 验证与范围', '',
        'CPU回归、两题真实Smoke、原生/No-op与重复Full精确一致；首Token与历史同题Full逐词表精确一致。',
        '所有非P−1 Prefill Query的逐层Attention/输出前缀哈希保持一致；视觉编码和DeepStack特征哈希不变；MLP、残差与每步Cache长度检查通过。',
        '目标层P−1输出严格为0，每个实际Decode Query均执行同一既有Hook；异常路径Hook清理通过。模型参数版本和输入哈希保持不变。',
        '无额外窗口、无来源Mask、无训练，无新增3,000题实验。GPU1已释放。', '',
        '普通与Bonferroni九项校正区间均按Image SHA256整簇配对Bootstrap（20,000次），保留题目加权Accuracy；六任务仅作描述。',
        '完整逐题答案/Token、损益、首Token LogP Loss和任务分组见CSV与检查点；原始词表Logits保存在本机外部路径，不上传权重或大缓存。', '',
        f"冻结协议SHA256：`{sha256(OUT / 'protocol.json')}`。"]
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT / 'result_hashes.json', {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*'))
          if p.is_file() and p.name != 'result_hashes.json'})
    print('\n'.join(lines[:20]), flush=True)


if __name__ == '__main__':
    main()
