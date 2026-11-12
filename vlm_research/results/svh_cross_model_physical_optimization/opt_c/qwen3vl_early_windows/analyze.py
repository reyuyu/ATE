"""Whole-image paired intervals and two-panel figure for the fixed six windows."""
import csv
import json
from pathlib import Path

import numpy as np

from .experiment import CONDITIONS, OUT, SEED, sha256, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(delta, identities, repetitions=200000):
    unique = sorted(set(identities))
    sums = np.stack([delta[np.array(identities) == key].sum(0) for key in unique])
    counts = np.array([identities.count(key) for key in unique])
    rng = np.random.default_rng(SEED)
    draws = np.empty((repetitions, delta.shape[1]))
    for first in range(0, repetitions, 500):
        last = min(first+500, repetitions)
        index = rng.integers(0, len(unique), size=(last-first, len(unique)))
        draws[first:last] = sums[index].sum(1) / counts[index].sum(1)[:, None]
    return np.quantile(draws, [.025, .975, .025/6, 1-.025/6], axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = read(OUT / "sample_manifest.json")
    records = []
    for sample in manifest["samples"]:
        path = OUT / "checkpoints" / f"{sample['task']}__{sample['question_id']}.json"
        row = read(path)
        assert row["input_binding"] == sample["input_binding"]
        assert row["protocol_sha256"] == sha256(OUT / "protocol.json")
        assert list(row["conditions"]) == list(CONDITIONS)
        records.append(row)
    assert len(records) == 240
    names = list(CONDITIONS)
    scores = np.array([[r["conditions"][c]["correct"] for c in names] for r in records])
    delta = (scores[:, 1:]-scores[:, :1])*100.
    intervals = bootstrap(delta, [r["image_sha256"] for r in records])
    summary, first_summary, paired, flat, task_rows = [], [], [], [], []
    for j, condition in enumerate(names):
        rows = [r["conditions"][condition] for r in records]
        harm = int(((scores[:, 0] == 1) & (scores[:, j] == 0)).sum())
        gain = int(((scores[:, 0] == 0) & (scores[:, j] == 1)).sum())
        item = dict(condition=condition, layers=str(CONDITIONS[condition]), n=240,
            accuracy_percent=float(scores[:, j].mean()*100), delta_pp=float((scores[:, j]-scores[:, 0]).mean()*100),
            correct_to_wrong=harm, wrong_to_correct=gain, eos=sum(x["eos"] for x in rows),
            truncated=sum(x["truncated"] for x in rows), empty=sum(x["empty"] for x in rows),
            anomalies=sum(x["anomaly"] for x in rows), mean_generated_tokens=float(np.mean([x["generated_tokens"] for x in rows])),
            ci95_low_pp=float(intervals[0,j-1]) if j else 0., ci95_high_pp=float(intervals[1,j-1]) if j else 0.,
            bonferroni95_low_pp=float(intervals[2,j-1]) if j else 0., bonferroni95_high_pp=float(intervals[3,j-1]) if j else 0.)
        summary.append(item)
        first_summary.append(dict(condition=condition, n=240, first_token_flips=sum(x["first_top1_flip"] for x in rows),
            first_token_flip_percent=float(np.mean([x["first_top1_flip"] for x in rows])*100),
            mean_full_token_logp_loss=float(np.mean([x["first_logp_loss"] for x in rows])),
            median_full_token_logp_loss=float(np.median([x["first_logp_loss"] for x in rows])),
            answer_text_changes=sum(x["text"] != r["conditions"]["Full"]["text"] for x,r in zip(rows,records))))
        if j:
            paired.append(dict(comparison=condition+" - Full", clusters=manifest["unique_images"], bootstrap_repetitions=200000,
                               family_size=6, **{k:v for k,v in item.items() if k not in ("condition", "layers")}))
        for r, result in zip(records, rows):
            full = r["conditions"]["Full"]
            flat.append(dict(task=r["task"], question_id=r["question_id"], image_id=r["image_id"], image_sha256=r["image_sha256"],
                question=r["question"], answer=r["answer"], condition=condition, text=result["text"], prediction=result["prediction"],
                token_ids=json.dumps(result["token_ids"]), correct=result["correct"], full_correct=full["correct"],
                correct_to_wrong=int(full["correct"] and not result["correct"]), wrong_to_correct=int(not full["correct"] and result["correct"]),
                first_token=result["first_token_id"], first_top1_flip=result["first_top1_flip"], first_logp_loss=result["first_logp_loss"],
                full_first_token_logp=result["full_first_token_logp"], changed_full_token_logp=result["changed_full_token_logp"],
                generated_tokens=result["generated_tokens"], non_eos_tokens=result["non_eos_tokens"], eos=result["eos"],
                truncated=result["truncated"], empty=result["empty"], anomaly=result["anomaly"], seconds=result["seconds"],
                input_binding=r["input_binding"], protocol_sha256=r["protocol_sha256"]))
    tasks = sorted({r["task"] for r in records})
    heat = np.zeros((6,6))
    for i, task in enumerate(tasks):
        subset = np.array([r["task"] == task for r in records])
        assert subset.sum() == 40
        for j, condition in enumerate(names):
            task_rows.append(dict(task=task, condition=condition, n=40,
                accuracy_percent=float(scores[subset,j].mean()*100),
                delta_pp=float((scores[subset,j]-scores[subset,0]).mean()*100),
                correct_to_wrong=int(((scores[subset,0] == 1)&(scores[subset,j] == 0)).sum()),
                wrong_to_correct=int(((scores[subset,0] == 0)&(scores[subset,j] == 1)).sum())))
            if j:
                heat[i,j-1] = task_rows[-1]["delta_pp"]
    for name, rows in (("accuracy_summary.csv", summary), ("task_accuracy.csv", task_rows),
                       ("first_token_summary.csv", first_summary), ("paired_comparisons.csv", paired), ("per_question.csv", flat)):
        csv_write(name, rows)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.8), gridspec_kw={"width_ratios":[1.05,1.25]}, constrained_layout=True)
    labels = [f"W{i}\nL{min(CONDITIONS[f'W{i}'])}-{max(CONDITIONS[f'W{i}'])}" for i in range(6)]
    means = delta.mean(0)
    axes[0].errorbar(range(6), means, yerr=np.array([means-intervals[2],intervals[3]-means]),
                     fmt="o", color="#215f8b", capsize=4, linewidth=1.2, markersize=5)
    axes[0].axhline(0, color="#555555", linewidth=.8)
    axes[0].axhline(-1, color="#b53038", linestyle="--", linewidth=1, label="-1 pp reference")
    axes[0].set_xticks(range(6), labels)
    axes[0].set_ylabel("Accuracy change vs Full (pp)")
    axes[0].set_title("(a) Fixed-window impact")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=.15)
    vmax = max(1., abs(heat).max())
    image = axes[1].imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_yticks(range(6), tasks)
    axes[1].set_xticks(range(6), [f"W{i}" for i in range(6)])
    axes[1].set_title("(b) Task-wise accuracy change")
    for i in range(6):
        for j in range(6):
            axes[1].text(j, i, f"{heat[i,j]:+.1f}", ha="center", va="center",
                         color="white" if abs(heat[i,j]) > .6*vmax else "black", fontsize=8)
    fig.colorbar(image, ax=axes[1], label="Change (pp)", shrink=.85)
    fig.savefig(OUT / "figure_early_windows.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure_early_windows.png", dpi=350, bbox_inches="tight")
    plt.close(fig)
    execution = read(OUT / "execution.json")
    lines = ["# Qwen3-VL 早层 Attention 固定窗口探索", "", "## 设计与有效性", "",
        f"完成240题、{manifest['unique_images']}个Image SHA256簇、7条件，共1680次生成。六类任务各40题，固定seed={SEED}，不按模型输出选题。使用物理GPU1、官方BF16模型、Eager、正常独立KV Cache、最多32 Token。",
        "只置零原P−1与后续真实Decode Query的Attention o_proj输出；不是来源Mask、整层物理跳过或加速实验。其他Prefill Query、MLP、Residual、Q/K/V及DeepStack保留原生计算。",
        "CPU及两张真实图片（最短/最长输入）的Smoke通过；No-op有效文本Logits最大差0，Full重复Token精确一致。逐步Hook、36层Cache长度、残差加法及三处DeepStack注入均检查通过。", "",
        "## 总体结果", "", "|条件|Accuracy|ΔFull (pp)|普通95%CI|六项校正95%CI|原对→错|原错→对|首Token翻转|平均LogP Loss|",
        "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for item, first in zip(summary, first_summary):
        lines.append(f"|{item['condition']}|{item['accuracy_percent']:.3f}%|{item['delta_pp']:+.3f}|[{item['ci95_low_pp']:.3f}, {item['ci95_high_pp']:.3f}]|[{item['bonferroni95_low_pp']:.3f}, {item['bonferroni95_high_pp']:.3f}]|{item['correct_to_wrong']}|{item['wrong_to_correct']}|{first['first_token_flips']}|{first['mean_full_token_logp_loss']:.6f}|")
    lines += ["", "## 六类任务（描述性）", "", "|任务|Full|W0|W1|W2|W3|W4|W5|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in tasks:
        lines.append("|"+task+"|"+"|".join(f"{r['accuracy_percent']:.1f}%" for r in task_rows if r["task"] == task)+"|")
    lines += ["", "## 结论与边界", "",
        "全部窗口均保留，包括负向结果。上表的点估计不能替代区间及错误翻转；−1pp只是与LLaVA可比的参考线。本240题用于探索，不是独立非劣效确认，区间跨0不表示安全或等价。",
        "逐窗口第一Token LogP Loss、Top-1翻转与最终Accuracy见first_token_summary.csv；两类指标不等同，概率损害不直接等于准确率损害。",
        "同题跨模型的正式比较需要冻结共同样本及不同官方Chat Template绑定；此处不混用LLaVA的Full。任何低损害候选若进入下一阶段，应先冻结，再使用新的图像隔离样本确认；本轮不自行启动36层扫描。",
        "统计采用按Image SHA256整簇配对Bootstrap（200000次），保留题目加权Accuracy；六项Bonferroni校正。六任务分组仅描述，不额外宣称确认性发现。", "",
        "## 运行记录", "",
        f"正式运行GPU进程耗时 {execution['seconds']/60:.2f} 分钟；生成耗时逐题保存，不报告推理加速比。CUDA allocated峰值 {execution['cuda_peak_allocated_mib']:.1f} MiB，reserved {execution['cuda_peak_reserved_mib']:.1f} MiB，nvidia-smi采样进程峰值 {execution['sampled_process_peak_mib']} MiB。",
        f"EOS/截断/空答案/异常：{[(r['condition'],r['eos'],r['truncated'],r['empty'],r['anomalies']) for r in summary]}。",
        "所有逐题答案、Token IDs、首Token指标、错误翻转及输入绑定保存在per_question.csv和原子检查点中。资源释放另见gpu_release.json。", ""]
    (OUT / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    write(OUT / "result_integrity.json", dict(questions=240, rows=len(flat), conditions=7,
        complete=True, unique_images=manifest["unique_images"], files={p.name:sha256(p) for p in OUT.iterdir()
            if p.is_file() and p.suffix in (".csv", ".pdf", ".png")}))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
