import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .run import OUT, read, sha256, verify, write


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, task):
    result = []
    for layer in range(36):
        subset = [r for r in rows if r['layer'] == layer and (task == 'All' or r['task'] == task)]
        losses = np.array([r['signed_logp_loss'] for r in subset])
        flips = sum(r['top1_flip'] for r in subset)
        result.append(dict(task=task, layer=layer, n=len(subset), mean_signed_logp_loss=float(losses.mean()),
            mean_absolute_logp_change=float(np.abs(losses).mean()), signed_loss_std=float(losses.std(ddof=1)),
            top1_flip_count=flips, top1_flip_rate=flips/len(subset),
            min_signed_loss=float(losses.min()), max_signed_loss=float(losses.max()),
            negative_loss_count=int((losses < 0).sum())))
    return result


def plot(summary, filename, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    keys = [('mean_signed_logp_loss', 'Mean signed LogP loss (nat)', '#bb3d35'),
            ('mean_absolute_logp_change', 'Mean absolute LogP change (nat)', '#147d83'),
            ('top1_flip_rate', 'Top-1 flip rate (%)', '#4a62a2')]
    for ax, (key, label, color) in zip(axes, keys):
        y = [r[key] * (100 if key == 'top1_flip_rate' else 1) for r in summary]
        ax.plot(range(36), y, color=color, linewidth=1.8, marker='.', markersize=4)
        ax.axhline(0, color='#888888', linewidth=.6)
        ax.set(xlabel='Decoder layer', ylabel=label, xlim=(0, 35), xticks=[0, 5, 10, 15, 20, 25, 30, 35])
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'{filename}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def aggregate():
    protocol = verify()
    samples = read(OUT / 'sample_manifest.json')['samples']
    expected = {(s['task'], s['question_id']): s for s in samples}
    rows, seen, completed = [], set(), []
    for shard in range(3):
        complete = read(OUT / f'shard_{shard}/complete.json')
        assert complete['passed'] and complete['count'] == 80
        assert complete['protocol_sha256'] == sha256(OUT / 'protocol.json')
        completed.append(complete)
        files = sorted((OUT / f'shard_{shard}/checkpoints').glob('*.json'))
        assert len(files) == 80
        for path in files:
            record = read(path)
            assert record['protocol_sha256'] == sha256(OUT / 'protocol.json')
            assert len(record['rows']) == 36
            sample_key = (record['rows'][0]['task'], record['rows'][0]['question_id'])
            sample = expected[sample_key]
            assert sample['shard'] == shard and record['input_binding'] == sample['input_binding']
            assert sample_key not in seen
            seen.add(sample_key)
            assert [r['layer'] for r in record['rows']] == list(range(36))
            for row in record['rows']:
                assert row['image_sha256'] == sample['image_sha256']
                assert row['image_probability_max'] == 0 and row['observed_native_probabilities']
                assert row['mask_changed_entries'] == row['image_tokens']
                assert row['absolute_logp_change'] == abs(row['signed_logp_loss'])
            rows.extend(record['rows'])
    assert seen == set(expected) and len(rows) == 8640
    assert len({r['code_commit'] for r in completed}) == 1
    tasks = sorted({s['task'] for s in samples})
    overall = summarize(rows, 'All')
    by_task = [row for task in tasks for row in summarize(rows, task)]
    csv_write(OUT / 'per_question.csv', rows)
    csv_write(OUT / 'layer_summary.csv', overall)
    csv_write(OUT / 'task_layer_summary.csv', by_task)
    plot(overall, 'overall_sensitivity', 'Qwen3-VL-8B: first-answer Image-source knockout (N=240)')
    for task in tasks:
        plot([r for r in by_task if r['task'] == task], f'task_{task}', f'{task}: first-answer Image-source knockout (N=40)')
    signed = sorted(overall, key=lambda r: r['mean_signed_logp_loss'], reverse=True)[:5]
    absolute = sorted(overall, key=lambda r: r['mean_absolute_logp_change'], reverse=True)[:5]
    flip = sorted(overall, key=lambda r: r['top1_flip_count'], reverse=True)[:5]
    integrity = dict(passed=True, questions=240, conditions_per_question=36, intervention_forwards=8640,
        native_full_forwards=240, qid_image_overlap_with_llava_rq2_1=240,
        task_counts=dict(Counter(s['task'] for s in samples)),
        zero_image_probability_in_all_8640_conditions=True, source_code_commit=completed[0]['code_commit'],
        formal_wall_seconds_max_shard=max(r['seconds'] for r in completed),
        shard_timings=completed, protocol_sha256=sha256(OUT / 'protocol.json'),
        model_revision=protocol['model_revision'], no_nan_inf_oom=True)
    write(OUT / 'integrity.json', integrity)
    def desc(values, key):
        return ', '.join(f'L{r["layer"]}={r[key]:.6f}' for r in values)
    report = f'''# Qwen3-VL 首答案 Token 视觉 Attention 敏感性

## 实际完成
240/240题，六任务各40；完整36层，共8,640/8,640个单层 Image OFF 条件。Question ID、Image ID和图片SHA256与LLaVA RQ2.1的240题全部一致。原始图片与问题复用，输入使用已经冻结的Qwen官方Chat Template，不冒充LLaVA的Token位置。

- 模型：Qwen/Qwen3-VL-8B-Instruct，revision `{protocol['model_revision']}`；BF16/eager；Transformers 4.57.6。
- 物理GPU0/2/3，每卡80题；未使用GPU1。冻结源码Commit：`{completed[0]['code_commit']}`。
- 正式运行最长Shard：{integrity['formal_wall_seconds_max_shard']/60:.2f}分钟；各卡Smoke耗时见integrity.json。
- 仅原始Prefill P−1的目标层到实际Image Key施加Softmax前`-inf`，其他Query、MLP、视觉Token和DeepStack保持原生。没有自由生成、训练或其他消融。

## 核心观察
- Mean Signed LogP Loss最高的五层（nat）：{desc(signed, 'mean_signed_logp_loss')}。
- Mean Absolute LogP Change最高的五层（nat）：{desc(absolute, 'mean_absolute_logp_change')}。
- 首Token翻转最多的五层：{', '.join(f'L{r["layer"]}: {r["top1_flip_count"]}/240 ({r["top1_flip_rate"]*100:.2f}%)' for r in flip)}。
- Signed均值允许负值；Absolute先逐题取绝对值再平均，不是Signed均值的绝对值。全词表FP32 log_softmax，目标固定为本题Native Full首Top-1，不使用Gold答案替换。

## 正确性与动态视觉位置
实际Image Key由当前Processor展开后的image_token_id及image_grid_thw/merge_size交叉核验，不假定576个Token。每个正式干预均从原生Attention返回值检查全部Head的P−1→Image概率为0；保存实际Token数、Mask改变的元素数与BF16行和误差。

各GPU先通过真实Smoke：No-op和Full全部有效Prompt Logits逐元素精确一致；复用视觉特征与Native Full全部有效Logits精确一致；L0/L2/L17/L35的目标Mask有效，其他Prefill Query的Attention输出、层输出与最终Logits精确一致；更早层不变；三个DeepStack特征与实际注入精确核验；输入和权重不变；正常及异常退出Hook均清理。全部GPU通过后才开始正式扫描。

首次Smoke将全部位置LM Head投影和仅末位置投影跨形状比较，触发BF16精确一致断言，正式扫描未启动；原始失败日志保存在attempt0_projection_shape_check。修正为各自相同投影形状的Native/No-op严格逐元素比较，不放宽容差。两种投影形状的实测差异单列在validation.json，正式Full与Image OFF始终使用相同的仅末位置投影。

每题只计算一次原生视觉编码，将最终视觉特征和全部三个DeepStack特征一起原样克隆复用到各层条件；没有删除或遗漏DeepStack。每次使用独立无Cache Prefill，防止条件间KV污染。

## 解释边界
这是240题首Token的探索性直接路径扫描，不测量完整答案Accuracy。Image路径低敏感不能证明不需要视觉信息，其他Token残差及DeepStack仍可能携带视觉信息。Signed损失、绝对变化和Top1翻转为不同指标，负值/抵消可能使Signed曲线弱化；本轮重新计算全部数据，不复用旧图或旧消融结果。

## 文件
`per_question.csv`为8,640行逐题逐层数据；`layer_summary.csv`为36层总体统计；`task_layer_summary.csv`为六任务统计。`overall_sensitivity.png/pdf`和`task_*.png/pdf`均由本轮真实数据直接绘制，无平滑插值。`sample_manifest.json`保存全部动态输入映射及三卡分片，`shard_*/validation.json`、日志及`integrity.json`保存验证和执行记录。
'''
    (OUT / 'report_zh.md').write_text(report, encoding='utf-8')
    print(json.dumps(integrity, ensure_ascii=False), flush=True)
