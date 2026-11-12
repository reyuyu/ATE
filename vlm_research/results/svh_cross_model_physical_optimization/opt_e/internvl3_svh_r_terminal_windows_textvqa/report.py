"""Truthful post-run report for three accepted late-window candidates."""
from .common import *
def main():
 d=read(OUT/"analysis_digest.json");q={x["method"]:x for x in d["quality"]};meta={x["method"]:x for x in d["meta"]};length={x["method"]:x for x in d["length"]}
 lines=["# InternVL3-SVH-R：末层回看候选TextVQA补测","",
 "本轮只测试用户接受的三个候选，全部为同一128题TextVQA Image-only：","",
 "| 新候选 | 回看层数 | Hidden退出位置 | Soft Accuracy % | 距离历史Native pp |",
 "|---|---:|---|---:|---:|"]
 for m,c in CONFIGS.items():
  x=q[m];stage="L27结束后、最终tokenwise Norm前" if c["cutoff"]==28 else "L27入口";lines.append(f"| {m} | {len(c['keep'])} | {stage} | {x['soft_accuracy_pct']:.4f} | {x['native_gap_pp']:.4f} |")
 lines += ["","状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER；输入、结构、实际KV与独立评分通过，原NUMERICAL_EQUIVALENCE_FAILED保留。不同预算不能冒充相同Token预算比较。",
 "",
 "## 冻结协议与复用","",
 f"模型OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、Python3.11.8，batch1 greedy，max_new_tokens32、自然EOS、logits_to_keep1，单图448×448与256视觉Token。GPU固定{GPU}，与前几轮新窗口相同。",
 "",
 "原128题及顺序、Prompt、Gold、原生Chat Template与ATE_SVH_1TILE_SCORER_ALIGNED短答指令不变。Image-only，仅原始图片/问题，无OCR辅助或Gold输入。131个准备模型输入（含3 Smoke）的BF16 pixels/Token IDs/Prompt/roles绑定全部CPU重验；权重、图片、环境和准备张量均本地SSD复用。",
 "",
 "新增384次正式质量生成，每窗口128，逐图片轮换，独立Cache，模型只加载一次。1280条旧Main预测（十条件×128）经原始SHA、输入/Gold/源码/协议绑定及官方/独立评分核验后复用，未重跑旧条件，未新增控制、质量全量或速度实验。父Commit "+PARENT_COMMIT+"。Native/R0/R6/R14依然来自用户明确授权的历史不同GPU；其余旧窗口及本轮三项同当前GPU。",
 "",
 "这是反复使用的历史128题，在前轮结果后追加候选，不是独立Holdout、预注册最优搜索或非劣效证明。所有失败形式与零分输出均保留，不按结果重新选题或调整窗口。",
 "",
 "## 全部同题质量与EOS末端实际缓存","",
 "| 条件 | Soft Accuracy % | 追回Native−R0差距 % | EOS末端KV Live=Capacity MiB | 来源 |","|---|---:|---:|---:|---|"]
 for x in d["quality"]:lines.append(f"| {x['method']} | {x['soft_accuracy_pct']:.4f} | {x['recovery_of_native_minus_R0_pct']:.4f} | {x['mean_KV_live_MiB']:.4f} | {x['source']} |")
 lines += ["","EOS末端缓存受生成长度影响，不是固定32 Token KV@32。本轮未测任何新E2E/TTFT/TPOT，不用旧窗口延迟替代。",
 "",
 "## 重点配对比较","",
 "Image SHA整簇配对Bootstrap5000次，seed20261112；128题有128图片簇。普通未校正95%CI，多重比较/历史样本选择限制保留。完整33项比较与逐题变化见CSV。",
 "",
 "| 候选−基准 | 差值 pp | 95%CI pp | 得分提高/下降/不变 |","|---|---:|---|---|"]
 focus={("Deep8_L19_26","Late8_L20_27"),("Deep8_L19_26","Late9_L19_27"),("Deep6_L20_25","Late6_L21_26"),("Deep7_L20_26","Late6_L21_26"),("Deep7_L20_26","Late8_L20_27"),("Late8_L20_27","Late9_L19_27")}
 for p in d["pairs"]:
  if (p["base"],p["candidate"]) in focus:lines.append(f"| {p['candidate']} − {p['base']} | {p['difference_pp']:+.4f} | [{p['CI95_low_pp']:.4f}, {p['CI95_high_pp']:.4f}] | {p['improved']}/{p['declined']}/{p['unchanged']} |")
 lines += ["","L20–27 vs L19–26为8层同预算，L21–26 vs L20–25为6层同预算，但二者也移动Hidden退出位置；不能只归因于某一层。L19–27 vs L19–26增加1个回看层且延长视觉Hidden到L27。L19–27 vs L20–27则同为terminal cutoff28，额外开放L19，可更直接比较该层的答案视觉权限。所有R→Image的冻结L20边界不变。CI跨0不证明等价，未校正CI不证明独立最优。",
 "",
 "## 真实得分恢复与损害","",
 "| 候选 | 相对R0 pp | 零分→正分 | 提高/下降/不变 | 元回答 | unanswerable |","|---|---:|---:|---|---:|---:|"]
 for m in CONFIGS:
  p=next(p for p in d["pairs"] if p["base"]=="R0" and p["candidate"]==m);x=meta[m];lines.append(f"| {m} | {p['difference_pp']:+.4f} | {p['zero_to_positive']} | {p['improved']}/{p['declined']}/{p['unchanged']} | {x['meta']} | {x['unanswerable']} |")
 lines += ["","恢复由官方十答案Soft Score与实际Gold匹配支持，不以元回答减少代替OCR恢复。所有原始答案/Gold/Tokens/逐题提高及损害/首Token变化均保留。错误数字/品牌/词语/复述为沿用的重叠规则诊断，不是人工OCR归因或额外评分器。",
 "",
 "| 候选 | Token P50/P95/最大值 | EOS/128 | 空输出 | 截断 |","|---|---|---:|---:|---:|"]
 for m in CONFIGS:
  x=length[m];lines.append(f"| {m} | {x['P50']:.2f}/{x['P95']:.2f}/{x['maximum']} | {x['EOS']} | {x['empty']} | {x['truncated']} |")
 lines += ["","全部1664条保存记录独立检查SHA、输入/Gold/源码/协议、Token cap和EOS，官方评分与独立leave-one-out十答案评分逐条一致；没有丢弃错误输出。",
 "",
 "## Terminal cutoff28的必要兼容实现","",
 "原引擎声明接受cutoff28，但只在Transformer循环i==cutoff时执行删行，28层循环没有L28，且审计要求确实发生退出事件。因此在独立engine.py中新增唯一terminal分支：所有L0–L27运行结束后，对Hidden按原文本位置index_select，删除256视觉行，再执行tokenwise最终RMSNorm。逐层Norm/MLP/Residual与最后答案位置保持原生。未修改历史引擎、模型权重、Prompt或Attention连接；完整最小差异见engine_delta.patch。",
 "",
 "这两个terminal配置在28个语言Block中都保留完整视觉Hidden，**没有Prefill语言Block的视觉删行节省**。最后的删行只是明确结束Hidden生命周期，不冒充提前退出。L21–26仍在L27入口真实删行，因此只有最后1层节省256视觉layer-rows（28×256的3.57%）；不是FLOPs或延迟测量。",
 "",
 "L2–L5的P−1/Decode真实不执行QKV/QK/AV/o_proj，所有Norm/MLP/Residual保留。持久原始视觉KV只在相应8/9/6个KEEP层，每层256；非KEEP仅文本、SKIP无持久KV。RoPE/原始逻辑位置不重编号，零Scratch仅瞬时原位置补齐，禁止位置Softmax前−inf。逐层实Tensor形状、dtype、Live/Storage、原始位置及实际操作输入行数保存核验。",
 "",
 "CPU官方28层Qwen2 Float64每配置1 Prefill+8 Decode，对齐Logits/Hidden MaxAbs均0。另有cutoff26/27旧路径CPU回归：修改前后Logits严格torch.equal，KV逻辑位置相同。三个GPU短/中/长Smoke覆盖三候选，Native/No-op有效Logits严格torch.equal；被禁视觉概率0，允许层>0，实际退出事件、答案跳算、Cache与非目标连接通过，未发生NaN/Inf、OOM或结构错误。",
 "",
 "| 候选 | 完整Token一致/3 | 标准化答案一致/3 | 对齐Logits MaxAbs | 对齐步MeanAbs均值 |","|---|---:|---:|---:|---:|"]
 for x in d["numerical"]:lines.append(f"| {x['condition']} | {x['full_tokens_equal']} | {x['normalized_answers_equal']} | {x['logits_max_abs']:.8f} | {x['mean_step_mean_abs']:.8f} |")
 lines += ["","只比较相同生成前缀的Logits/Hidden。答案一致不替代严格数值验收；继续保留数值豁免，不将Dense/Physical标为严格等价，不做BF16内核搜索或事后阈值放宽。",
 "",
 "## 存储、同步与停止","",
 f"SSD执行目录：{RUN}。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-terminal-windows-textvqa-3090/vlm_research/results/internvl3_svh_r_terminal_windows_textvqa_3090/。完整SQLite、原始物理审计与CPU/GPU日志保留server_only，权重/图片/准备张量留服务器本地SSD，无Windows传输。",
 "",
 "GPU前冻结88个Python文件SHA保持不变；所有历史父文件逐字节保留。后处理/归档/发布代码另列analysis_source_manifest.json。独立分支codex/internvl3-svh-r-terminal-windows-textvqa-3090，不改main、不强推、不建PR、不覆盖旧结果。artifact_manifest.json与SHA256SUMS列出上传及服务器原件，固定Commit HTTPS逐文件回读和SSH分支头验证回执位于server_only/git_publish_receipt.json。",
 "",
 "384次新增正式生成完成，GPU释放、无遗留计算进程，服务器保持运行；不自动再选窗口、扩大样本或启动下一阶段。",
 ""]
 (OUT/"report_zh.md").write_text("\n".join(lines));print("TERMINAL_REPORT_READY",flush=True)
if __name__=="__main__":main()
