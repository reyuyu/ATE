# InternVL3-SVH-R：末层回看候选TextVQA补测

本轮只测试用户接受的三个候选，全部为同一128题TextVQA Image-only：

| 新候选 | 回看层数 | Hidden退出位置 | Soft Accuracy % | 距离历史Native pp |
|---|---:|---|---:|---:|
| Late8_L20_27 | 8 | L27结束后、最终tokenwise Norm前 | 50.4688 | 8.0469 |
| Late9_L19_27 | 9 | L27结束后、最终tokenwise Norm前 | 50.7031 | 7.8125 |
| Late6_L21_26 | 6 | L27入口 | 46.9531 | 11.5625 |

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER；输入、结构、实际KV与独立评分通过，原NUMERICAL_EQUIVALENCE_FAILED保留。不同预算不能冒充相同Token预算比较。

## 冻结协议与复用

模型OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、Python3.11.8，batch1 greedy，max_new_tokens32、自然EOS、logits_to_keep1，单图448×448与256视觉Token。GPU固定GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，与前几轮新窗口相同。

原128题及顺序、Prompt、Gold、原生Chat Template与ATE_SVH_1TILE_SCORER_ALIGNED短答指令不变。Image-only，仅原始图片/问题，无OCR辅助或Gold输入。131个准备模型输入（含3 Smoke）的BF16 pixels/Token IDs/Prompt/roles绑定全部CPU重验；权重、图片、环境和准备张量均本地SSD复用。

新增384次正式质量生成，每窗口128，逐图片轮换，独立Cache，模型只加载一次。1280条旧Main预测（十条件×128）经原始SHA、输入/Gold/源码/协议绑定及官方/独立评分核验后复用，未重跑旧条件，未新增控制、质量全量或速度实验。父Commit 84790bb7ce9904126054cfff39ef54d39a76f25a。Native/R0/R6/R14依然来自用户明确授权的历史不同GPU；其余旧窗口及本轮三项同当前GPU。

这是反复使用的历史128题，在前轮结果后追加候选，不是独立Holdout、预注册最优搜索或非劣效证明。所有失败形式与零分输出均保留，不按结果重新选题或调整窗口。

## 全部同题质量与EOS末端实际缓存

| 条件 | Soft Accuracy % | 追回Native−R0差距 % | EOS末端KV Live=Capacity MiB | 来源 |
|---|---:|---:|---:|---|
| Native | 58.5156 | 100.0000 | 18.3472 | REUSED_OLD_GPU |
| R0 | 8.5156 | 0.0000 | 3.7859 | REUSED_OLD_GPU |
| R6 | 7.8125 | -1.4062 | 6.7727 | REUSED_OLD_GPU |
| R14 | 7.0312 | -2.9687 | 10.7474 | REUSED_OLD_GPU |
| Deep6_L20_25 | 43.2031 | 69.3750 | 6.6991 | REUSED_SAME_CURRENT_GPU |
| Deep7_L20_26 | 47.5781 | 78.1250 | 7.1826 | REUSED_SAME_CURRENT_GPU |
| Deep8_L19_26 | 49.6094 | 82.1875 | 7.6910 | REUSED_SAME_CURRENT_GPU |
| Shift8_L18_25 | 43.9062 | 70.7812 | 7.6929 | REUSED_SAME_CURRENT_GPU |
| Shift8_L17_24 | 41.0156 | 65.0000 | 7.7126 | REUSED_SAME_CURRENT_GPU |
| Shift8_L16_23 | 36.2500 | 55.4688 | 7.7145 | REUSED_SAME_CURRENT_GPU |
| Late8_L20_27 | 50.4688 | 83.9062 | 7.6764 | NEW_CURRENT_GPU |
| Late9_L19_27 | 50.7031 | 84.3750 | 8.1965 | NEW_CURRENT_GPU |
| Late6_L21_26 | 46.9531 | 76.8750 | 6.6852 | NEW_CURRENT_GPU |

EOS末端缓存受生成长度影响，不是固定32 Token KV@32。本轮未测任何新E2E/TTFT/TPOT，不用旧窗口延迟替代。

## 重点配对比较

Image SHA整簇配对Bootstrap5000次，seed20261112；128题有128图片簇。普通未校正95%CI，多重比较/历史样本选择限制保留。完整33项比较与逐题变化见CSV。

| 候选−基准 | 差值 pp | 95%CI pp | 得分提高/下降/不变 |
|---|---:|---|---|
| Late8_L20_27 − Deep7_L20_26 | +2.8906 | [-1.2500, 7.3438] | 6/3/119 |
| Late8_L20_27 − Deep8_L19_26 | +0.8594 | [-3.9062, 5.7812] | 6/7/115 |
| Late9_L19_27 − Deep8_L19_26 | +1.0938 | [-3.3594, 5.4688] | 6/6/116 |
| Late6_L21_26 − Deep6_L20_25 | +3.7500 | [-1.0938, 8.8281] | 11/6/111 |
| Late6_L21_26 − Deep7_L20_26 | -0.6250 | [-3.2832, 2.0312] | 2/3/123 |
| Late9_L19_27 − Late8_L20_27 | +0.2344 | [-2.8906, 3.3594] | 3/2/123 |

L20–27 vs L19–26为8层同预算，L21–26 vs L20–25为6层同预算，但二者也移动Hidden退出位置；不能只归因于某一层。L19–27 vs L19–26增加1个回看层且延长视觉Hidden到L27。L19–27 vs L20–27则同为terminal cutoff28，额外开放L19，可更直接比较该层的答案视觉权限。所有R→Image的冻结L20边界不变。CI跨0不证明等价，未校正CI不证明独立最优。

## 真实得分恢复与损害

| 候选 | 相对R0 pp | 零分→正分 | 提高/下降/不变 | 元回答 | unanswerable |
|---|---:|---:|---|---:|---:|
| Late8_L20_27 | +41.9531 | 60 | 61/3/64 | 0 | 1 |
| Late9_L19_27 | +42.1875 | 61 | 62/3/63 | 0 | 1 |
| Late6_L21_26 | +38.4375 | 57 | 58/3/67 | 4 | 2 |

恢复由官方十答案Soft Score与实际Gold匹配支持，不以元回答减少代替OCR恢复。所有原始答案/Gold/Tokens/逐题提高及损害/首Token变化均保留。错误数字/品牌/词语/复述为沿用的重叠规则诊断，不是人工OCR归因或额外评分器。

| 候选 | Token P50/P95/最大值 | EOS/128 | 空输出 | 截断 |
|---|---|---:|---:|---:|
| Late8_L20_27 | 3.00/5.65/10 | 128 | 0 | 0 |
| Late9_L19_27 | 3.00/8.00/12 | 128 | 0 | 0 |
| Late6_L21_26 | 3.00/6.00/12 | 128 | 0 | 0 |

全部1664条保存记录独立检查SHA、输入/Gold/源码/协议、Token cap和EOS，官方评分与独立leave-one-out十答案评分逐条一致；没有丢弃错误输出。

## Terminal cutoff28的必要兼容实现

原引擎声明接受cutoff28，但只在Transformer循环i==cutoff时执行删行，28层循环没有L28，且审计要求确实发生退出事件。因此在独立engine.py中新增唯一terminal分支：所有L0–L27运行结束后，对Hidden按原文本位置index_select，删除256视觉行，再执行tokenwise最终RMSNorm。逐层Norm/MLP/Residual与最后答案位置保持原生。未修改历史引擎、模型权重、Prompt或Attention连接；完整最小差异见engine_delta.patch。

这两个terminal配置在28个语言Block中都保留完整视觉Hidden，**没有Prefill语言Block的视觉删行节省**。最后的删行只是明确结束Hidden生命周期，不冒充提前退出。L21–26仍在L27入口真实删行，因此只有最后1层节省256视觉layer-rows（28×256的3.57%）；不是FLOPs或延迟测量。

L2–L5的P−1/Decode真实不执行QKV/QK/AV/o_proj，所有Norm/MLP/Residual保留。持久原始视觉KV只在相应8/9/6个KEEP层，每层256；非KEEP仅文本、SKIP无持久KV。RoPE/原始逻辑位置不重编号，零Scratch仅瞬时原位置补齐，禁止位置Softmax前−inf。逐层实Tensor形状、dtype、Live/Storage、原始位置及实际操作输入行数保存核验。

CPU官方28层Qwen2 Float64每配置1 Prefill+8 Decode，对齐Logits/Hidden MaxAbs均0。另有cutoff26/27旧路径CPU回归：修改前后Logits严格torch.equal，KV逻辑位置相同。三个GPU短/中/长Smoke覆盖三候选，Native/No-op有效Logits严格torch.equal；被禁视觉概率0，允许层>0，实际退出事件、答案跳算、Cache与非目标连接通过，未发生NaN/Inf、OOM或结构错误。

| 候选 | 完整Token一致/3 | 标准化答案一致/3 | 对齐Logits MaxAbs | 对齐步MeanAbs均值 |
|---|---:|---:|---:|---:|
| Late8_L20_27 | 3 | 3 | 0.00000000 | 0.00000000 |
| Late9_L19_27 | 3 | 3 | 0.00000000 | 0.00000000 |
| Late6_L21_26 | 3 | 3 | 0.25000000 | 0.01686925 |

只比较相同生成前缀的Logits/Hidden。答案一致不替代严格数值验收；继续保留数值豁免，不将Dense/Physical标为严格等价，不做BF16内核搜索或事后阈值放宽。

## 存储、同步与停止

SSD执行目录：/home/featurize/ate-runs/internvl3-svh-r-terminal-windows-textvqa-3090。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-terminal-windows-textvqa-3090/vlm_research/results/internvl3_svh_r_terminal_windows_textvqa_3090/。完整SQLite、原始物理审计与CPU/GPU日志保留server_only，权重/图片/准备张量留服务器本地SSD，无Windows传输。

GPU前冻结88个Python文件SHA保持不变；所有历史父文件逐字节保留。后处理/归档/发布代码另列analysis_source_manifest.json。独立分支codex/internvl3-svh-r-terminal-windows-textvqa-3090，不改main、不强推、不建PR、不覆盖旧结果。artifact_manifest.json与SHA256SUMS列出上传及服务器原件，固定Commit HTTPS逐文件回读和SSH分支头验证回执位于server_only/git_publish_receipt.json。

384次新增正式生成完成，GPU释放、无遗留计算进程，服务器保持运行；不自动再选窗口、扩大样本或启动下一阶段。
