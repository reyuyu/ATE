"""Descriptive cluster intervals and unsmoothed, architecture-aware pilot figures."""
import csv
import json
from pathlib import Path

import numpy as np

from .engine import SOURCES
from .run import LLAVA_MASS,LLAVA_SENS,OLD,OUT,SEED,read,sha256,source_hashes,write


def csv_write(name,rows):
    with (OUT / name).open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap(values,identities,repetitions=10000,seed=SEED):
    unique = sorted(set(identities))
    mask = np.array(identities)
    sums = np.stack([values[mask == key].sum(0) for key in unique])
    counts = np.array([(mask == key).sum() for key in unique])
    shape = values.shape[1:]
    sums = sums.reshape(len(sums),-1)
    draws = np.empty((repetitions,sums.shape[1]),dtype=np.float64)
    rng = np.random.default_rng(seed)
    for first in range(0,repetitions,128):
        last = min(first+128,repetitions)
        index = rng.integers(0,len(unique),(last-first,len(unique)))
        draws[first:last] = sums[index].sum(1)/counts[index].sum(1)[:,None]
    return np.quantile(draws,[.025,.975],axis=0).reshape((2,)+shape)


def ranks(values):
    unique,inverse,counts = np.unique(values,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-.5*(counts-1))[inverse]


def correlation(x,y,rank=False):
    x,y = np.asarray(x),np.asarray(y)
    if rank:
        x,y = ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def runs(mask,minimum=3):
    answer,start = [],None
    for i,value in enumerate(list(mask)+[False]):
        if value and start is None:
            start = i
        if not value and start is not None:
            if i-start >= minimum:
                answer.append([start,i-1])
            start = None
    return answer


def assess(mass,metrics,tasks,task_metrics):
    anchor = mass[:,0]
    signed,absolute,flip = metrics[:,0],metrics[:,1],metrics[:,2]
    middle = np.arange(13,25)
    early = np.arange(2,13)
    late = np.arange(25,36)
    early_majority = runs((anchor>=.5)&np.isin(np.arange(36),early))
    late_majority = runs((anchor>=.5)&np.isin(np.arange(36),late))
    selected = [i for a,b in early_majority for i in range(a,b+1)]
    hypotheses = dict(
        H1=bool(all(anchor[i] < anchor[early].mean() and mass[i,1]+mass[i,5] > anchor[i] for i in (0,1))),
        H2=bool(selected and absolute[selected].mean() < absolute[middle].mean()),
        H3=bool(int(absolute.argmax()) in middle and len({int(task_metrics[t][:,1].argmax()) for t in tasks})>1),
        H4=bool(late_majority and anchor[late].mean()>anchor[middle].mean()))
    return dict(scope="Descriptive 120-question pilot, not significance or discovered four-stage boundaries",
        four_stage_status="descriptive_full_pattern" if all(hypotheses.values()) else "descriptive_partial_or_different_pattern" if any(hypotheses.values()) else "not_supported",
        probe_rule_results=hypotheses,probe_ranges=dict(early=[2,12],middle=[13,24],late=[25,35]),
        no_fitted_boundaries=True,no_smoothing=True,
        actual_majority_anchor_runs=runs(anchor>=.5),early_majority_anchor_runs=early_majority,
        late_majority_anchor_runs=late_majority,
        first_two_layers=[dict(layer=i,anchor=float(anchor[i]),image=float(mass[i,1]),prompt=float(mass[i,5])) for i in (0,1)],
        probe_anchor_means=dict(early=float(anchor[early].mean()),middle=float(anchor[middle].mean()),late=float(anchor[late].mean())),
        early_platform_mean_absolute_loss=float(absolute[selected].mean()) if selected else None,
        middle_mean_absolute_loss=float(absolute[middle].mean()),
        global_signed_peak=dict(layer=int(signed.argmax()),mean=float(signed.max())),
        global_absolute_peak=dict(layer=int(absolute.argmax()),mean=float(absolute.max())),
        global_flip_peak=dict(layer=int(flip.argmax()),rate=float(flip.max())),
        task_peaks={t:dict(signed_layer=int(task_metrics[t][:,0].argmax()),signed_peak=float(task_metrics[t][:,0].max()),
                           absolute_layer=int(task_metrics[t][:,1].argmax()),absolute_peak=float(task_metrics[t][:,1].max())) for t in tasks},
        cross_layer_correlations={source:dict(pearson=correlation(mass[:,j],absolute),spearman=correlation(mass[:,j],absolute,True)) for j,source in enumerate([*SOURCES,"prompt"])},
        caveats=["Probe ranges are not claimed phase boundaries", "Task differences have only 20 questions each; no task specificity confirmation", "Mass, attention update magnitude and functional sensitivity are distinct; update magnitude was not measured", "No-op exactness and historical Top1 binding must pass; no NI claim"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    protocol = read(OUT / "protocol.json")
    assert protocol["code_sha256"] == source_hashes()
    assert read(OUT / "execution.json")["status"] == "completed"
    samples = read(OUT / "sample_manifest.json")["samples"]
    native,scans = [],[]
    for sample in samples:
        key = f"{sample['task']}__{sample['question_id']}"
        n,s = read(OUT / "native_checkpoints" / f"{key}.json"),read(OUT / "scan_checkpoints" / f"{key}.json")
        assert n["input_binding"] == s["input_binding"] == sample["input_binding"]
        assert n["protocol_sha256"] == s["protocol_sha256"] == sha256(OUT / "protocol.json")
        assert [r["layer"] for r in s["rows"]] == list(range(36))
        native.append(n)
        scans.append(s)
    assert len(native) == len(scans) == 120
    basic_mass = np.array([r["native_mass"] for r in native])
    mass = np.concatenate([basic_mass,(basic_mass[:,:,2]+basic_mass[:,:,3])[:,:,None]],axis=-1)
    metrics = np.array([[[r[k] for k in ("signed_logp_loss","absolute_logp_change","top1_flip","js_divergence")] for r in s["rows"]] for s in scans])
    identities = [s["image_sha256"] for s in samples]
    tasks = sorted({s["task"] for s in samples})
    features = np.concatenate([mass,metrics],axis=-1)
    group_data = {}
    mass_rows,sens_rows,per_mass,per_sens,within = [],[],[],[],[]
    for group in ["Overall",*tasks]:
        indices = [i for i,s in enumerate(samples) if group == "Overall" or s["task"] == group]
        values = features[indices]
        ci = bootstrap(values,[identities[i] for i in indices])
        means = values.mean(0)
        group_data[group] = (means,ci)
        for layer in range(36):
            for j,source in enumerate([*SOURCES,"prompt"]):
                mass_rows.append(dict(group=group,layer=layer,source=source,n_questions=len(indices),
                    n_images=len({identities[i] for i in indices}),mean=float(means[layer,j]),
                    ci95_low=float(ci[0,layer,j]),ci95_high=float(ci[1,layer,j])))
            losses = metrics[indices,layer,0]
            row = dict(group=group,layer=layer,n_questions=len(indices),n_images=len({identities[i] for i in indices}),
                mean_signed_logp_loss=float(means[layer,6]),signed_ci95_low=float(ci[0,layer,6]),signed_ci95_high=float(ci[1,layer,6]),
                mean_absolute_logp_change=float(means[layer,7]),absolute_ci95_low=float(ci[0,layer,7]),absolute_ci95_high=float(ci[1,layer,7]),
                median_signed_loss=float(np.median(losses)),p90_signed_loss=float(np.quantile(losses,.90)),p95_signed_loss=float(np.quantile(losses,.95)),
                p95_absolute_change=float(np.quantile(abs(losses),.95)),minimum_signed_loss=float(losses.min()),maximum_signed_loss=float(losses.max()),
                positive_loss_count=int((losses>0).sum()),negative_loss_count=int((losses<0).sum()),zero_loss_count=int((losses==0).sum()),
                top1_flip_count=int(metrics[indices,layer,2].sum()),top1_flip_rate=float(means[layer,8]),
                flip_ci95_low=float(ci[0,layer,8]),flip_ci95_high=float(ci[1,layer,8]),mean_js_divergence=float(means[layer,9]))
            sens_rows.append(row)
    for i,sample in enumerate(samples):
        for layer in range(36):
            base = dict(task=sample["task"],question_id=sample["question_id"],image_sha256=sample["image_sha256"],layer=layer)
            per_mass.append(dict(**base,**{source:float(mass[i,layer,j]) for j,source in enumerate([*SOURCES,"prompt"])},
                image_tokens=sample["layout"]["image_tokens"],row_error=native[i]["mass_row_max_error"]))
            per_sens.append(dict(**base,**{k:v for k,v in scans[i]["rows"][layer].items() if k != "layer"},
                input_binding=sample["input_binding"],native_raw_path=native[i]["raw_path"],native_raw_sha256=native[i]["raw_sha256"],
                ablated_raw_path=scans[i]["raw_path"],ablated_raw_sha256=scans[i]["raw_sha256"]))
    for layer in range(36):
        for j,source in enumerate([*SOURCES,"prompt"]):
            within.append(dict(layer=layer,source=source,n=120,pearson=correlation(mass[:,layer,j],metrics[:,layer,1]),
                               spearman=correlation(mass[:,layer,j],metrics[:,layer,1],True),scope="within-layer descriptive; image clustered sampling, not causal"))
    csv_write("attention_mass_layer_summary.csv",[r for r in mass_rows if r["group"] == "Overall"])
    csv_write("attention_mass_task_summary.csv",[r for r in mass_rows if r["group"] != "Overall"])
    csv_write("single_layer_sensitivity.csv",[r for r in sens_rows if r["group"] == "Overall"])
    csv_write("task_layer_sensitivity.csv",[r for r in sens_rows if r["group"] != "Overall"])
    csv_write("attention_mass_per_question.csv",per_mass)
    csv_write("single_layer_per_question.csv",per_sens)
    csv_write("within_layer_mass_sensitivity_correlations.csv",within)
    means,ci = group_data["Overall"]
    assessment = assess(means[:,:6],means[:,6:],tasks,{t:group_data[t][0][:,6:] for t in tasks})
    write(OUT / "four_stage_assessment.json",assessment)
    csv_write("mass_sensitivity_alignment.csv",[dict(layer=i,**{source:float(means[i,j]) for j,source in enumerate([*SOURCES,"prompt"])},
        signed_loss=float(means[i,6]),absolute_change=float(means[i,7]),top1_flip_rate=float(means[i,8])) for i in range(36)])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
    colors = {"Start Anchor":"#9e3e75","Image":"#277f8e","Prompt":"#be782c","Self":"#666666"}
    def save(fig,name):
        fig.savefig(OUT / (name+".pdf"),bbox_inches="tight")
        fig.savefig(OUT / (name+".png"),dpi=350,bbox_inches="tight")
        plt.close(fig)
    def clean(ax):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
        ax.set_xlabel("Language decoder layer")
    def mass_plot(ax,group):
        m,c = group_data[group]
        for label,j in (("Start Anchor",0),("Image",1),("Prompt",5),("Self",4)):
            ax.plot(range(36),m[:,j],label=label,color=colors[label],linewidth=1.4,linestyle="--" if label == "Self" else "-")
            ax.fill_between(range(36),c[0,:,j],c[1,:,j],color=colors[label],alpha=.12)
        ax.set_ylim(0,1)
        ax.set_ylabel("P-1 attention mass")
        clean(ax)
    fig,ax = plt.subplots(figsize=(7.1,3.15),constrained_layout=True)
    mass_plot(ax,"Overall")
    ax.set_title("Qwen3-VL: 36-layer native attention sources (n=120)")
    ax.legend(frameon=False,ncol=4,fontsize=8)
    save(fig,"figure1_attention_mass")
    fig,axes = plt.subplots(2,3,figsize=(11,5.5),constrained_layout=True)
    for ax,task in zip(axes.flat,tasks):
        mass_plot(ax,task)
        ax.set_title(task+" (n=20)")
    axes[0,0].legend(frameon=False,fontsize=7)
    save(fig,"supplement_task_attention_mass")
    fig,axes = plt.subplots(1,3,figsize=(11,3.1),constrained_layout=True)
    for ax,j,title,ylabel,factor in zip(axes,(6,7,8),("Signed LogP loss","Absolute LogP change","Top-1 flip rate"),("Loss (nat)","Absolute change (nat)","Flip rate (%)"),(1,1,100)):
        ax.plot(range(36),means[:,j]*factor,color="#245b84",linewidth=1.5)
        ax.fill_between(range(36),ci[0,:,j]*factor,ci[1,:,j]*factor,color="#245b84",alpha=.18)
        peak = int(means[:,j].argmax())
        ax.scatter([peak],[means[peak,j]*factor],s=20,color="#ae3548",zorder=3)
        ax.annotate(f"L{peak}",(peak,means[peak,j]*factor),xytext=(3,7),textcoords="offset points",fontsize=8)
        ax.axhline(0,color="#666666",linewidth=.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        clean(ax)
    save(fig,"figure2_attention_sensitivity")
    fig,ax = plt.subplots(figsize=(11,3.2),constrained_layout=True)
    heat = np.stack([group_data[t][0][:,6] for t in tasks])
    limit = max(float(abs(heat).max()),1e-6)
    image = ax.imshow(heat,cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
    ax.set_yticks(range(6),tasks)
    ax.set_xticks([0,5,10,15,20,25,30,35])
    ax.set_xlabel("Language decoder layer")
    ax.set_title("Task-wise signed Full-token LogP loss (20 questions/task)")
    fig.colorbar(image,ax=ax,label="Loss (nat)",shrink=.9)
    save(fig,"figure2_task_sensitivity_heatmap")
    llava_mass = [r for r in csv_read(LLAVA_MASS) if r["group"] == "Overall"]
    llava_sens = csv_read(LLAVA_SENS)
    llava_curve = np.array([-np.mean([float(r["mean_delta_logp"]) for r in llava_sens if int(r["layer"]) == i]) for i in range(32)])
    fig,axes = plt.subplots(2,2,figsize=(10,6),constrained_layout=True,sharey="row")
    for source,label in (("bos_mass","BOS"),("image_mass","Image"),("prompt_mass","Prompt")):
        rr = sorted([r for r in llava_mass if r["source"] == source],key=lambda r:int(r["layer"]))
        color = colors["Start Anchor" if label == "BOS" else label]
        axes[0,0].plot([int(r["layer"]) for r in rr],[float(r["mean"]) for r in rr],label=label,color=color)
        axes[0,0].fill_between([int(r["layer"]) for r in rr],[float(r["ci95_low"]) for r in rr],[float(r["ci95_high"]) for r in rr],color=color,alpha=.12)
    axes[0,0].set_ylim(0,1)
    axes[0,0].set_title("LLaVA-1.5-7B (32 layers, historical n=3000)")
    axes[0,0].set_ylabel("P-1 attention mass")
    axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
    clean(axes[0,0])
    mass_plot(axes[0,1],"Overall")
    axes[0,1].set_title("Qwen3-VL-8B (36 layers, pilot n=120)")
    axes[0,1].legend(frameon=False,ncol=2,fontsize=8)
    axes[1,0].plot(range(32),llava_curve,color="#245b84")
    axes[1,0].set_title("Historical mean; sign converted")
    axes[1,1].plot(range(36),means[:,6],color="#245b84")
    axes[1,1].fill_between(range(36),ci[0,:,6],ci[1,:,6],color="#245b84",alpha=.18)
    axes[1,1].set_title("Pilot mean and descriptive 95% CI")
    for ax in axes[1]:
        ax.set_ylabel("Signed Full-token LogP loss (nat)")
        ax.axhline(0,color="#666666",linewidth=.6)
        clean(ax)
    save(fig,"figure3_llava_vs_qwen")
    csv_write("llava_comparison_source.csv",[dict(layer=i,signed_logp_loss=float(llava_curve[i]),
        **{source:next(float(r["mean"]) for r in llava_mass if r["source"] == source and int(r["layer"]) == i) for source in ("bos_mass","image_mass","prompt_mass")}) for i in range(32)])
    report = ["# Qwen3-VL 四阶段Attention快速探索（120题）","",
        "## 首页结论","",
        f"1. 四阶段结构：预声明描述性检查结果为 `{assessment['four_stage_status']}`，H1–H4分别为 `{assessment['probe_rule_results']}`；不是统计确认，详见原始曲线及下文人工核读。",
        f"2. 敏感峰：Signed Loss最高在L{assessment['global_signed_peak']['layer']}（{assessment['global_signed_peak']['mean']:.6f} nat）；Absolute Change最高在L{assessment['global_absolute_peak']['layer']}（{assessment['global_absolute_peak']['mean']:.6f} nat）。任务峰见下表；每任务仅20题，不确认任务特异性。",
        f"3. 早层高Anchor与低敏感：多数Anchor连续区域为{assessment['actual_majority_anchor_runs']}，早层探查区域中的连续段为{assessment['early_majority_anchor_runs']}。仅报告描述性重合，不将Mass当作功能必要性或更新幅度。",
        "4. 跨模型背景：历史Qwen W1–W4没有评分翻转，尤其W2（L2–L5）与LLaVA候选相同，支持进一步研究低敏感现象；不是独立确认，不宣称非劣效或物理跳层加速。","",
        "## 设计与有效性","",
        "从已冻结Qwen 240题中按seed与输入哈希选择六任务各20题，共120个不同Image SHA256簇。全部输入与历史Prompt、Processor、Token布局精确一致。BF16/Eager、Transformers4.57.6、固定模型Revision，只使用物理GPU1。",
        "正式运行120次Native Prefill及4320次单层P−1 o_proj OFF Prefill；没有答案生成、Decode、来源Mask、新窗口或确认集。Native Full只运行一次并复用于36个单层比较。",
        "Start Anchor是第一个真实<|im_start|>用户角色起始符，不是BOS。五来源互斥完备；Prompt=Question+Template，排除Anchor和Self。所有原生概率来自未经改变的官方Eager Softmax，逐层只复制P−1行到CPU。",
        "CPU测试与三题GPU Smoke通过；有效首答案完整词表Logits的No-op/Mass Observer差值均为0。单层只置零目标P−1，其他Prefill Query精确不变，MLP调用、BF16残差加法、DeepStack三次注入及36层KV长度正确。原生Full首Token与历史结果逐题一致。","",
        "## 任务敏感峰","","|任务|Signed峰层|Mean Signed Loss|Absolute峰层|Mean Absolute Change|","|---|---:|---:|---:|---:|"]
    for task,p in assessment["task_peaks"].items():
        report.append(f"|{task}|L{p['signed_layer']}|{p['signed_peak']:.6f}|L{p['absolute_layer']}|{p['absolute_peak']:.6f}|")
    report += ["","## 统计与解释边界","",
        "区间为Image SHA256整簇配对Bootstrap，10000次，普通描述性95%区间；题目加权。Median/P90/P95、负Loss及极端样本保留在CSV。不因均值接近0声称每个样本都低敏感，不用CI跨0作安全证明。",
        "未拟合变点、未平滑、未强制四段。预声明的早/中/深探查范围仅用于描述比较，不是真实阶段边界；高Anchor连续段以Mass≥0.5、至少3层记录，完整曲线优先。",
        "LLaVA对照来自已有3000题统计，Qwen是120题子集；模板、精度、架构和样本规模不同。LLaVA历史delta_logp=OFF−Full，此处取负统一为Full−OFF。LLaVA BOS与Qwen Start Anchor并非同一种Token，图不能证明跨架构普遍规律。",
        "Mass与敏感性的跨层和层内相关均仅描述；层内相关另存CSV，不能以跨层相关推断局部因果。更新范数未在本轮测量，因此不能把高Mass解释为大Attention更新。","",
        "## 历史窗口背景（不重新生成）","","|窗口|历史ΔAccuracy|损害/纠正|","|---|---:|---:|"]
    for r in csv_read(OLD / "accuracy_summary.csv"):
        if r["condition"] != "Full":
            report.append(f"|{r['condition']} {r['layers']}|{float(r['delta_pp']):+.3f}pp|{r['correct_to_wrong']}/{r['wrong_to_correct']}|")
    execution = read(OUT / "execution.json")
    report += ["","## 产物与资源","",
        "Figure1：36层来源Mass（含Self），附六任务来源图。Figure2：Signed/Absolute/Flip三面板及六任务热图。Figure3：LLaVA与Qwen并排对照；全部提供PDF/PNG和源CSV。",
        f"正式GPU耗时{execution['seconds']/60:.2f}分钟；CUDA allocated峰值{execution['cuda_peak_allocated_mib']:.1f}MiB，reserved {execution['cuda_peak_reserved_mib']:.1f}MiB，nvidia-smi采样峰值{execution['sampled_process_peak_mib']}MiB。",
        f"原始Native/单层完整词表Logits及每Head Mass存放于服务器 `{read(OUT / 'protocol.json')['raw_tensor_directory']}`，路径和SHA256绑定在逐题检查点及CSV，不上传Git。资源释放见gpu1_release.json。",""]
    (OUT / "report_zh.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(assessment,indent=2),flush=True)


if __name__ == "__main__":
    main()
