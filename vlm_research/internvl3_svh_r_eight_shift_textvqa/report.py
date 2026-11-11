"""Measured Chinese report for the three user-requested eight-layer windows."""
from .common import *
def main():
 d=read(OUT/"analysis_digest.json");q={x["method"]:x for x in d["quality"]};meta={x["method"]:x for x in d["meta"]};length={x["method"]:x for x in d["length"]}
 lines=["# InternVL3-SVH-R：三个八层回看窗口补测","",
 "**L18–L25：43.9063%；L17–L24：41.0156%；L16–L23：36.2500%。** 同128题、同当前GPU、8层窗口中，此前L19–L26的49.6094%仍是最高观测值。所有比较均为历史同题探索，不是独立Holdout或通用最优证明。",
 "",
 "状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。结构、输入完整性与独立评分通过；原NUMERICAL_EQUIVALENCE_FAILED保留，不宣称BF16 Dense/Physical严格等价。",
 "",
 "## 冻结协议与来源","",
 f"模型OpenGVLab/InternVL3-8B，Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0、Python3.11.8，batch1 greedy，正常EOS、max_new_tokens=32、logits_to_keep=1；单图448×448、256视觉Token。当前GPU {GPU}，与Deep6/7/8同一卡。未下载已有权重/数据、未改环境或算法。",
 "",
 "原128题及顺序保持不变。原生InternVL Chat Template与ATE_SVH_1TILE_SCORER_ALIGNED短答指令，Image-only：只有原始图片和问题，无OCR辅助、Gold、Caption或Hint。131个缓存模型输入（128主题+3 Smoke）的Token IDs、BF16 pixels、Prompt与roles绑定在CPU重验，Gold未进入模型输入。",
 "",
 "只新增3×128=384次正式生成；逐图片轮换三方法，独立Cache，模型加载一次。896条旧Main记录（七条件×128）按原始记录SHA、输入/Gold/源码/协议绑定与独立评分核验复用，没有重生成对照。父Commit "+PARENT_COMMIT+"。Native/R0/R6/R14原始记录来自已明确授权的历史不同GPU；Deep6/7/8和本轮三条件来自同一当前GPU，此区别继续注明。没有新增控制、测速、其他窗口、GQA/OCRBench、训练或扩样。",
 "",
 "新窗口是用户查看前轮结果后追加，128题已用于相关探索，不能称作独立确认样本。窗口移动也改变Hidden退出位置，不能把结果仅归因于某一层。",
 "",
 "## 质量与真实EOS末端缓存","",
 "| 条件 | Soft Accuracy % | 距离Native pp | 追回Native−R0差距 % | EOS末端KV Live=Capacity MiB | 来源 |",
 "|---|---:|---:|---:|---:|---|"]
 for x in d["quality"]:lines.append(f"| {x['method']} | {x['soft_accuracy_pct']:.4f} | {x['native_gap_pp']:.4f} | {x['recovery_of_native_minus_R0_pct']:.4f} | {x['mean_KV_live_MiB']:.4f} | {x['source']} |")
 lines += ["","EOS末端缓存受各方法生成长度影响，不能冒充统一KV@32。没有为新窗口测量或补造E2E/TTFT/TPOT，也不引用旧窗口时间代替。",
 "",
 "## 相同8层预算的配对比较","",
 "Image SHA整簇配对Bootstrap5000次，seed20261112；128题恰有128图片簇。普通未校正95%CI，保留多项比较及追加候选的探索性限制。下表聚焦L19–L26与三个新窗口，完整24项对照见paired_soft_changes.csv。",
 "",
 "| 候选−基准 | 差值 pp | 95%CI pp | 提高/下降/不变 |","|---|---:|---|---|"]
 for p in d["pairs"]:
  if p["base"]=="Deep8_L19_26" or p["base"] in CONFIGS:lines.append(f"| {p['candidate']} − {p['base']} | {p['difference_pp']:+.4f} | [{p['CI95_low_pp']:.4f}, {p['CI95_high_pp']:.4f}] | {p['improved']}/{p['declined']}/{p['unchanged']} |")
 lines += ["","这些固定窗口每个均保留8层原始视觉KV，但cutoff是各自最后KEEP层+1：L18–25/17–24/16–23分别在L26/25/24入口退出；L19–26在L27退出。因此比较同时涉及答案读图深度与完整视觉Hidden生存期。R→Image从L20禁止始终不变。CI跨0不代表等价，CI不跨0也不免除多重比较、历史样本与BF16数值豁免限制。",
 "",
 "## 真实恢复、异常与损害","",
 "| 新窗口 | 相对R0 pp | 零分→正分 | 得分提高/下降/不变 | 元回答 | unanswerable |","|---|---:|---:|---|---:|---:|"]
 for m in CONFIGS:
  p=next(p for p in d["pairs"] if p["base"]=="R0" and p["candidate"]==m);x=meta[m];lines.append(f"| {m} | {p['difference_pp']:+.4f} | {p['zero_to_positive']} | {p['improved']}/{p['declined']}/{p['unchanged']} | {x['meta']} | {x['unanswerable']} |")
 lines += ["","官方十答案Soft Score与逐题实际匹配是恢复依据，不以元回答减少代替OCR恢复。完整答案、Gold、Tokens、首Token/全文变化与逐题损害见per_question.csv和per_question_changes.csv。错误数字/品牌/词语/复述沿用前轮明确的重叠规则诊断，只作辅助，不作为人工OCR错误归因或新评分器。",
 "",
 "| 新窗口 | Token P50/P95/最大值 | EOS/128 | 空输出 | 截断 |","|---|---|---:|---:|---:|"]
 for m in CONFIGS:
  x=length[m];lines.append(f"| {m} | {x['P50']:.2f}/{x['P95']:.2f}/{x['maximum']} | {x['EOS']} | {x['empty']} | {x['truncated']} |")
 lines += ["","全部1280条保存记录经过SHA、输入/Gold/源码/协议绑定、Token上限与EOS核验，正式官方评分与独立leave-one-out十答案评分逐条一致；未过滤任何零分、错误、异常形式或截断。",
 "",
 "## 真实物理实现与数值审计","",
 "直接复用已验证参数化引擎，算法源码逐字节不变，只改变用户指定keep/cutoff。每个窗口在各自cutoff入口真实删除整256视觉Hidden行，之前真实计算，之后只处理文本，未重新插入视觉或退回Dense。持久视觉KV只存在于对应8层，每层256；非KEEP只有文本，L2–L5无持久Cache。",
 "",
 "P−1/Decode仅KEEP可读取原始Image Key，R→Image仍从L20禁止。L2–L5答案QKV/QK/AV/o_proj真实不执行，所有Norm/MLP/Residual保留。原始RoPE和逻辑KV位置不重编号，临时零Scratch按原位置补齐、禁止位置Softmax前−inf，不计持久KV。逐层实Tensor shape/dtype/length、Live bytes、底层Storage及算子实际输入行数均保存核验；缓存节省不等于FLOPs缩减。",
 "",
 "CPU官方28层Qwen2 Float64，每配置1 Prefill+8 Decode，对齐Hidden和Logits MaxAbs均0；Mask、RoPE、真实Storage、独立Cache和异常清理通过。GPU3题覆盖短/中/长，每配置Dense/Physical Smoke；Native/No-op有效Logits严格torch.equal。被禁视觉概率0，允许层>0，未发生结构错误、非法读图、非有限值或OOM。",
 "",
 "| 新窗口 | 完整Token一致/3 | 标准化答案一致/3 | 对齐Logits MaxAbs | 对齐步MeanAbs均值 |","|---|---:|---:|---:|---:|"]
 for x in d["numerical"]:lines.append(f"| {x['condition']} | {x['full_tokens_equal']} | {x['normalized_answers_equal']} | {x['logits_max_abs']:.8f} | {x['mean_step_mean_abs']:.8f} |")
 lines += ["","仅在相同生成前缀比较Logits/Hidden；逐层Max/Mean差异和全部物理形状/概率在原始审计。答案一致不替代数值验收，本轮数值豁免仍保留，未调整阈值、模型、Mask或Kernel来获得正面结果。",
 "",
 "cutoff26/25/24分别只有后2/3/4层享有视觉Hidden删行，计512/768/1024视觉layer-rows，占Native28×256的7.14%/10.71%/14.29%。这是实际输入行数计数，不是FLOPs或延迟；临时Scratch保持Attention reduction宽度。所有新窗口未测速。",
 "",
 "## 存储、同步与结束","",
 f"SSD执行目录：{RUN}。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-eight-shift-textvqa-3090/vlm_research/results/internvl3_svh_r_eight_shift_textvqa_3090/。完整SQLite、原始audit及CPU/GPU日志在server_only；模型、图片、准备张量继续本地SSD复用，无Windows传输。",
 "",
 "GPU前冻结78个Python文件SHA全部未变，后处理/归档/发布脚本单列analysis_source_manifest.json。独立分支codex/internvl3-svh-r-eight-shift-textvqa-3090，不改main、不强推、不建PR、不覆盖旧结果。artifact_manifest.json、SHA256SUMS列出上传与服务器原件；固定Commit HTTPS逐文件回读及SSH分支头一致性回执保留server_only/git_publish_receipt.json。",
 "",
 "384次新增正式生成已完成，GPU释放，无遗留计算进程，服务器保持运行，不自动进入下一窗口或扩大样本。",
 ""]
 (OUT/"report_zh.md").write_text("\n".join(lines));print("SHIFT_REPORT_READY",flush=True)
if __name__=="__main__":main()
