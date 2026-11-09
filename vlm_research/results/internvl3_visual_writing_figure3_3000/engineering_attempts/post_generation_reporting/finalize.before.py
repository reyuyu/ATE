"""CPU-only final report and evidence checksums; no model and no remote Git writes."""
import csv,json,subprocess
from pathlib import Path
from PIL import Image
from .engine import CONDITIONS,TASKS
from .prepare import ROOT,OUT,MODEL,sha,save

def rows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def main():
    assert json.loads((OUT/"analysis_validation.json").read_text())["passed"]
    assert json.loads((OUT/"persistence_complete.json").read_text())["passed"]
    assert not (OUT/"checkpoints").is_symlink()
    p=json.loads((OUT/"protocol.json").read_text())
    summary=rows("accuracy_summary.csv");tasks=rows("task_accuracy.csv")
    overall=[r for r in rows("layerwise_accuracy.csv") if r["subset"]=="Overall"]
    assert len(summary)==31 and len(tasks)==186 and len(overall)==28
    assert {r["condition"] for r in summary}==set(CONDITIONS)
    base=next(r for r in summary if r["condition"]=="All-Writer")
    no=next(r for r in summary if r["condition"]=="No-Writer")
    deep=next(r for r in summary if r["condition"]=="Keep-L0-L16")
    worst=sorted(overall,key=lambda r:(-float(r["delta_accuracy_pp"]),int(r["layer"])))[:5]
    figs={}
    for name in ["figure3_internvl_row","figure3_three_models"]:
        image=Image.open(OUT/(name+".png"));dpi=image.info.get("dpi");assert dpi and min(dpi)>=300
        pdf=OUT/(name+".pdf");pdf_bytes=pdf.read_bytes()
        assert len(pdf_bytes)>1000 and pdf_bytes.startswith(b"%PDF") and b"/Subtype /Image" not in pdf_bytes
        figs[name]=dict(PNG_dimensions=list(image.size),PNG_DPI=dpi,PDF_bytes=pdf.stat().st_size)
    assert json.loads((OUT/"figure_render_validation.json").read_text())["PDF_vector_heatmap"]
    assert all(sha(path)==h for path,h in json.loads((OUT/"source_manifest.json").read_text()).items())
    runtime={}
    cache=Path("/home/featurize/.cache/huggingface/modules/transformers_modules/InternVL3-8B")
    for path in cache.glob("*.py"):
        original=MODEL/path.name
        if original.exists():
            assert sha(path)==sha(original)
            runtime[str(path)]=dict(sha256=sha(path),pinned_source=str(original))
    save("runtime_remote_code_audit.json",dict(passed=True,files=runtime))
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"],text=True).strip()
    assert not processes,"A GPU compute process remains; do not claim released"
    uuid=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    assert uuid==p["gpu_uuid"]
    save("gpu_final_release.json",dict(passed=True,GPU_UUID=uuid,compute_processes="",server_not_shutdown=True))
    plot=rows("plot_layerwise.csv");concentration=[]
    for model in ["LLaVA-1.5-7B","Qwen3-VL-8B","InternVL3-8B"]:
        selected=[r for r in plot if r["model"]==model];count=int(selected[0]["layers"])
        harm=[sum(int(r["correct_to_wrong"]) for r in selected if int(r["layer"])==i) for i in range(count)]
        delta=[sum(float(r["delta_accuracy_pp"]) for r in selected if int(r["layer"])==i)/6 for i in range(count)]
        fraction=sum(sorted(harm,reverse=True)[:5])/sum(harm) if sum(harm) else 0
        concentration.append(dict(model=model,top5_damage_event_share=fraction,
            layers_loss_ge_0_5pp=sum(x>=.5-1e-10 for x in delta),layers=count))
    from .analyze import write_csv
    write_csv("descriptive_concentration.csv",concentration)
    closest=min(concentration[:2],key=lambda r:abs(r["top5_damage_event_share"]-concentration[-1]["top5_damage_event_share"]))
    lines=["# InternVL3-8B Visual Writing Figure 3","",
      f"完成31×3000＝93,000次生成；GPU {uuid}（RTX3090），BF16/Eager，Revision {p['revision']}，Transformers4.49.0，logits_to_keep=1。",
      "3000题与四份历史清单的QID、顺序、Gold、任务和Image SHA严格一致，2409图片簇均匹配；没有重新抽样或筛掉错误答案。",
      "",
      f"**All-Writer {float(base['accuracy_pct']):.4f}%（{base['correct']}/3000）；No-Writer {float(no['accuracy_pct']):.4f}%；L17–L27 OFF {float(deep['accuracy_pct']):.4f}%（{deep['correct']}/3000）。**",
      f"深层联合关闭的损失（All-Writer减限制条件）**{float(deep['delta_accuracy_pp']):+.4f} pp**；整簇配对普通95%CI [{float(deep['delta_ci95_low']):+.4f}, {float(deep['delta_ci95_high']):+.4f}] pp；原对变错{deep['correct_to_wrong']}题，原错变对{deep['wrong_to_correct']}题。",
      "CI跨0不证明等价或无损；L17边界在GPU执行前固定，没有按结果调整。",
      "",
      "## 最敏感的五个单层","",
      "| Layer | ACC % | 损失 pp | 普通95%CI pp | Bonferroni-28 CI pp | 对→错 / 错→对 |",
      "| --- | ---: | ---: | --- | --- | ---: |"]
    for r in worst:
        lines.append(f"| L{r['layer']} | {float(r['accuracy_pct']):.4f} | {float(r['delta_accuracy_pp']):+.4f} | [{float(r['delta_ci95_low']):+.4f}, {float(r['delta_ci95_high']):+.4f}] | [{float(r['Bonferroni28_low']):+.4f}, {float(r['Bonferroni28_high']):+.4f}] | {r['correct_to_wrong']} / {r['wrong_to_correct']} |")
    lines+=["","Top5按观察到的整体净损失排序；校正区间用于区分峰值与统计证据。",
       "","## 六任务差异","","| Task | All-Writer % | 最大净损害层 | 损失 pp | L17–L27 OFF % |","| --- | ---: | --- | ---: | ---: |"]
    for task in TASKS:
        ts=[r for r in tasks if r["subset"]==task];b=next(r for r in ts if r["condition"]=="All-Writer")
        d=next(r for r in ts if r["condition"]=="Keep-L0-L16")
        peak=max((r for r in ts if r["condition"].startswith("Drop-")),key=lambda r:float(r["delta_accuracy_pp"]))
        lines.append(f"| {task} | {float(b['accuracy_pct']):.2f} | {peak['condition']} | {float(peak['delta_accuracy_pp']):+.3f} | {float(d['accuracy_pct']):.2f} |")
    lines+=["","## 集中性和统计限制","",
       "以下是事后探索性描述：Top5层占各层原对变错事件总数的比例。同一题可在不同层重复计数，不是互斥概率质量。"]
    for r in concentration:
        lines.append(f"- {r['model']}：Top5损害事件占比 {100*r['top5_damage_event_share']:.2f}%；净损失≥0.5pp的层 {r['layers_loss_ge_0_5pp']}/{r['layers']}。")
    negatives=[r["condition"] for r in overall if float(r["delta_accuracy_pp"])<0]
    lines += [f"按上述探索性集中度距离，InternVL更接近{closest['model']}。不能由此把模型差异归因于架构。",
       "整体负ΔACC（关闭后改善）的层："+("、".join(negatives) if negatives else "无")+"。负值、损害及恢复案例全部保留。",
       "Image SHA整簇Bootstrap20,000次、题数加权、同题配对。28项单层比较在Overall和各任务内分别给Bonferroni-28区间；六任务分析是探索性的，不声称覆盖全部196项比较的统一家族错误率。",
       "","## 正确性与历史复用","",
       "- 最短/典型/最长题官方Chat、独立Native/No-op全部有效Logits严格torch.equal，Token与视觉注入完全一致。",
       "- Smoke及全部正式请求的被禁概率为0；非目标Mask、Image Query、RoPE、Norm/MLP/Residual及28层真实Cache通过检查。",
       "- Drop-L27在全部3000题上与All-Writer Token和全部有效Logits哈希严格一致；正常/异常Hook清理通过。",
       "- All-Writer不是Native Full：所有条件的原始P−1和全部Decode在28层均不能直接读原始Image Key。",
       "- 历史InternVL Only Text=90.3333%与本轮All-Writer权限、输入和评分同义；只作参考，本轮未复用生成。",
       "- LLaVA/Qwen历史曲线从固定CSV读取；LLaVA深层窗口证据Commit b13a21b07e84a0095731d10dc1e8854f514a7877；Qwen窗口证据Commit dfe0cbacd09de9dfeacff980b911e6aec22a7665。历史数据未重跑、未修改。",
       "- 两次正式生成前的加载/官方Chat参数包装错误保存在engineering_attempts，没有放宽数值门槛，也未排除失败正式样本。",
       "- Checkpoint逐请求写SSD、异步镜像工作盘；迁移SHA及协议哈希一致。最终所有Checkpoint恢复为工作盘实体目录，SHA已核验。",
       "","## 物理移除的结论","",
       "**本轮结果尚不能单独支持在L17入口物理移除视觉Hidden。** 本实验仅改Dense访问权限，视觉Hidden、视觉Query、Norm/MLP/Residual与真实视觉KV仍计算/保留。小损失不能证明删除Hidden、KV或视觉Query路径后等价；需要另行授权的物理路径验证，本轮未执行。",
       "","## 图与证据","",
       f"- {OUT/'figure3_internvl_row.png'} / .pdf",
       f"- {OUT/'figure3_three_models.png'} / .pdf",
       f"- 原始绘图数据：{OUT/'plot_layerwise.csv'}；{OUT/'plot_deep.csv'}。",
       f"- 英文图注：{OUT/'figure_caption_en.md'}；脚本：{OUT/'plot_figure3.py'}。",
       f"- 逐题预测：{OUT/'per_question.csv'}；Checkpoint：{OUT/'checkpoints'}。",
       "PNG≥300DPI；PDF文字、线条、热图矢量化。热图共享0–10%色标，超范围用三角及真实数值显式标注；没有平滑、删负值或裁掉峰值。",
       "本轮附件只有任务文字，没有独立参考Figure3图片；按指定3×3排版和冻结Qwen六任务配色制图，不宣称逐像素复刻未提供的参考图。",
       "各模型Prompt、视觉Token预算、架构及历史软件协议存在差异，已在英文图注及references披露；这是描述性功能比较，不是效率或跨模型因果结论。",
       "","## 结束状态","",
       "独立分支 codex/internvl3-visual-writing-figure3-3000。GitHub未推送（未获本轮授权）。GPU无计算进程，服务器未关闭。",""]
    (OUT/"report_zh.md").write_text("\n".join(lines))
    save("validation.json",dict(passed=True,requests=93000,conditions=31,questions=3000,image_clusters=2409,
        scorer_exact=True,source_fingerprints_unchanged=True,persistent_checkpoints_SHA_exact=True,
        figures=figs,GPU_released=True,no_push=True,no_training=True,no_physical_experiment=True))
    entries={}
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name not in {"SHA256SUMS","evidence_manifest.json"}:
            entries[str(f.relative_to(OUT))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    save("evidence_manifest.json",dict(status="PASS",files=entries,external_model_path=str(MODEL),
        external_image_path="/home/featurize/data/internvl3-visual-writing-3000/images",
        external_assets_verified_by="model_manifest.json and data_audit.json"))
    entries["evidence_manifest.json"]=dict(sha256=sha(OUT/"evidence_manifest.json"))
    (OUT/"SHA256SUMS").write_text("".join(v["sha256"]+"  "+n+"\n" for n,v in sorted(entries.items())))
    print("ALL_93000_FINAL_VALIDATION_PASS",flush=True)

if __name__=="__main__":main()
