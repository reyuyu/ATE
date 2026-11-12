"""Describe the measured V2 matrix without conflating it with legacy Image bypass."""
from .common import OUT, ROOT, read


def report(p, accuracy, paired, latency, flops, worker, history, smoke):
    a = {r['condition']: r for r in accuracy}
    t = {r['condition']: r for r in latency}
    f = {r['condition']: r for r in flops}
    d = {(r['candidate'], r['baseline']): r for r in paired}
    old = read(ROOT / 'vlm_research/results/ate_fulljoint_vqav2_quick/worker_completion.json')
    lines = ['# ATE-FullJoint V2：Answer-only Attention Skip', '',
        '**COMPLETE**：同一60题、60张不同图片，五条件全部重新生成300条答案；同一20题每条件3次预热、5次正式计时。仅VQAv2工程Pilot。',
        '', '## 算法与绑定',
        'C0是真Native Full；C1是冻结Fast-P2 All-Writer；C2在C1上采用Fixed-Q-Conditional-Assistant-15；C3-V2在C1上增加Answer-only物理跳算；C4-V2组合Conditional Writer与Answer-only跳算。旧C3/C4没有重命名或进入新推理路径。',
        'L2–L5的Prefill仍为全部Token执行Norm、K/V和MLP；Q、QK/Softmax/AV、o_proj仅省去最后P−1一行。全部576个Image Query及此前Prefix/Question/R Query真实计算Attention。保留完整临时K/V，使用原逻辑RoPE；P−1不能影响此前因果Query。Decode这四层QKVO/Attention不执行，因此不分配其持久KV。',
        'Conditional Writer仅控制L0–L19的R→Image Softmax前Hard Mask：全部Question Subword；首词what/which或Question Token数≥15时，加完整ASSISTANT四个Subword。其他R/P−1禁止直接读图，不能将这一Mask解释成跳过R计算。L20前物理移除视觉Hidden，后12层只计算文本，保留原位置。其余28层持久Cache原地存文本KV，不复制历史。',
        '同旧实验完全相同的图片SHA、Question ID、Prompt IDs、输入Tensor、10个官方答案；Manifest和Input Specs的文件SHA逐位一致。使用原LLaVA revision、FP16/eager、Transformers4.49.0、Greedy32、自然EOS。数据来自历史本地冻结500题，不是全VQAv2的无偏随机测试。',
        '', '## 官方VQAv2评分',
        '采用冻结官方VQAEval归一化和10标注leave-one-out共识软评分；不是字符串单答案准确率。',
        '|条件|软Accuracy %|ΔNative pp|EOS|截断|空答案|',
        '|---|---:|---:|---:|---:|---:|']
    for r in accuracy:
        lines.append(f"|{r['condition']}|{r['score_pct']:.3f}|{r['delta_native_pp']:+.3f}|{r['eos']}/60|{r['truncated']}|{r['empty']}|")
    lines += ['', '|比较|Δ软分 pp|图片配对95%CI|软分下降/上升题数|完整序列改变|',
              '|---|---:|---|---:|---:|']
    for r in paired:
        lines.append(f"|{r['candidate']} − {r['baseline']}|{r['delta_pp']:+.3f}|[{r['paired95_low']:+.3f}, {r['paired95_high']:+.3f}]|{r['score_down']}/{r['score_up']}|{r['token_sequence_changes']}|")
    lines += ['', '固定seed、20,000次Image SHA整簇配对Bootstrap；60题为重复使用的探索性Pilot，CI包含0不是等价或无损。软分损益不能全称为二值正确翻转；满分丢失/获得及0↔1另存CSV。负面案例不删。',
        '', '## 实测时间',
        '|条件|Language Prefill ms|GPU TTFT ms|自然E2E ms|Decode ms/token|E2E vs Native|E2E vs C1|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in latency:
        dec = f"{r['decode_ms_per_token']:.3f}" if r['decode_ms_per_token'] else 'N/A'
        lines.append(f"|{r['condition']}|{r['language_prefill_ms']:.3f}|{r['ttft_ms']:.3f}|{r['latency_ms']:.3f}|{dec}|{r['E2E_speedup_vs_C0']:.3f}×|{r['E2E_speedup_vs_C1']:.3f}×|")
    lines += ['', '每题5次重复先取中位数，再跨20题取中位数；加速比为同题配对时延比中位数。预热3次/条件，正式条件交错；每请求独立冷KV，Vision原生重算。CPU图片I/O、Processor和输入位置准备统一不计入；同步墙钟E2E包括控制器、Mask和Cache构造。GPU TTFT为首Forward CUDA Events，不包含全部CPU准备。自然生成长度会影响E2E，同完整Token序列子集加速另存CSV。没有固定16/64测试。',
        '', '|条件|Prefill TFLOPs|Decode FLOPs/token|估算Prefill TFLOP/s|Live KV MiB|容量KV MiB|Peak allocated/reserved MiB|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in flops:
        lines.append(f"|{r['condition']}|{r['prefill_matrix_flops']/1e12:.6f}|{r['decode_matrix_flops_per_token']:.0f}|{r['estimated_prefill_tflops']:.3f}|{r['live_kv_MiB']:.3f}|{r['capacity_kv_MiB']:.3f}|{r['peak_allocated_MiB']:.1f}/{r['peak_reserved_MiB']:.1f}|")
    lines += ['', '实际执行Linear/Conv/QK/AV/bmm Shape，2 FLOPs/MAC；不含Norm、Softmax、RoPE等非矩阵操作。Vision FLOPs单列CSV。估算TFLOP/s不是硬件实测计数器，硬件TFLOPS/带宽N/A。独立公式逐60题核对：C1到V2的Prefill节省恰为4×(4H²+4PH)，H=4096；没有把Image Query误算成跳过。C1/C2及C3-V2/C4-V2逐题Prefill FLOPs相同，全部MLP FLOPs相同，Conditional Writer没有减少Dense算子形状。',
        '', '## 十项直接结论',
        '1. L2–L5视觉Attention完整保留：实际Q/o_proj输入均为P−1之前全部行，K/V输入为全部Prompt；三题逐层Attention/Hidden及L6输入的Image、Question、Other R、Prefix分组与Dense参考通过冻结数值门槛。visual_prefill_attention_preserved不是手写结论，逐项证据见CSV和Smoke算子Shape。',
        '2. 仅P−1及真实Decode跳算；P−1 Attention更新严格为零，Hidden/Residual并不置零；全部32层MLP保留。',
        f"3. C3-V2−C1={d['C3-V2','C1']['delta_pp']:+.3f}pp，95%CI [{d['C3-V2','C1']['paired95_low']:+.3f}, {d['C3-V2','C1']['paired95_high']:+.3f}]；软分损害/纠正{d['C3-V2','C1']['score_down']}/{d['C3-V2','C1']['score_up']}题。",
        f"4. C3-V2相对C1：Prefill {t['C3-V2']['language_prefill_ms_speedup_vs_C1']:.3f}×，Decode/token {t['C3-V2']['decode_ms_per_token_speedup_vs_C1']:.3f}×，E2E {t['C3-V2']['E2E_speedup_vs_C1']:.3f}×；相对Native E2E {t['C3-V2']['E2E_speedup_vs_C0']:.3f}×。",
        f"5. C3-V2相对旧C3的软分变化{d['C3-V2','Legacy-C3']['delta_pp']:+.3f}pp，95%CI [{d['C3-V2','Legacy-C3']['paired95_low']:+.3f}, {d['C3-V2','Legacy-C3']['paired95_high']:+.3f}]，序列改变{d['C3-V2','Legacy-C3']['token_sequence_changes']}题。旧C3跳过Image更新，本轮保留，故算法不同；旧耗时只作历史参考，没有同轮重测旧C3，不能声称严格配对新旧加速。",
        f"6. C4-V2−C3-V2={d['C4-V2','C3-V2']['delta_pp']:+.3f}pp，95%CI [{d['C4-V2','C3-V2']['paired95_low']:+.3f}, {d['C4-V2','C3-V2']['paired95_high']:+.3f}]，额外软分损害/纠正{d['C4-V2','C3-V2']['score_down']}/{d['C4-V2','C3-V2']['score_up']}题。",
        f"7. Conditional Writer没有额外Dense FLOPs或KV容量收益；C4-V2相对C3-V2 E2E={t['C4-V2']['E2E_speedup_vs_C3-V2']:.3f}×，不能把微小时延波动视为新增物理算量节省。",
        f"8. C3-V2容量KV={f['C3-V2']['capacity_kv_MiB']:.3f}MiB，C1={f['C1']['capacity_kv_MiB']:.3f}MiB，同题容量恒减12.5%（四层不持久存储）。Live KV依实际序列长度变化，不能当作额外Token压缩。",
        '9. 论文不应再使用旧I+A A2充当“视觉写入不变”的算法。优先以C1作可复现基础，C3-V2作语义清晰的Answer-only优化候选；C4-V2的新增Writer仅减少访问连接，没有额外算量收益，其质量差异需如实保留。60题不能最终选定质量保持的论文主方法。',
        '10. 存在FP16 Shape舍入与小样本风险。未放宽历史Logits atol=.10/rtol=.003、Hidden相对RMSE≤.01/Cosine距离≤1e−4。单GPU交错配对仍有系统/时钟波动，不能对很小的时延差作强结论。',
        '', '## 审计、失败与资源',
        f"9项CPU测试、3/3真实GPU Smoke通过。Native/No-op全部有效Logits及官方generate严格一致。两条件Dense参考共有{len(smoke['numerical'])}组数值对照、{len(smoke['preservation'])}项局部保留审计。禁止概率严格0、L20实际删除视觉行、RoPE位置、KV独立/无历史拷贝、权重和输入哈希、正常及异常Hook清理全部检查。",
        '首次Smoke在正式矩阵前因零向量Cosine误判停止：两边P−1 Attention均精确零，但通用Cosine返回距离1。改为严格零值检查，添加回归测试；非零向量仍使用原冻结容差。模型算法未改，失败日志和原协议保留failed_smoke_1，不隐藏失败。',
        f"C0/C1/C2历史绑定审计：{sum(r['tokens_equal'] for r in history)}/{len(history)}条完整序列一致，{sum(r['first_logits_sha256_equal'] for r in history)}/{len(history)}条首Logits哈希一致；本轮质量和计时仍全部重新运行。",
        f"正式矩阵耗时{worker['elapsed_seconds']/60:.2f}分钟，不含CPU准备、模型加载、Smoke和汇总；旧矩阵耗时{old['elapsed_seconds']/60:.2f}分钟仅作记录，不当作方案加速指标。",
        '仅物理GPU2；GPU0/1外部任务未触碰，GPU3未使用。GPU进程结束并检查释放后生成resource_audit.json。main及所有历史实验均未修改。本轮完成即停止，不扩样或继续搜索。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    (OUT / 'algorithm_definition.md').write_text('\n'.join(lines[4:9])+'\n')
