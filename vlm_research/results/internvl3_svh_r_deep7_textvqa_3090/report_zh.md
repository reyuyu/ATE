# InternVL3-SVH-R：L20–L26 七层回看补测

**Deep7：TextVQA Image-only Soft Accuracy 47.5781%（128题，soft score sum=60.9）。** 相对Deep6 +4.3750pp，相对Deep8 -2.0312pp；距离历史Native 10.9375pp，追回Native−R0差距78.1250%。

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER；结构、输入完整性及独立评分通过，原NUMERICAL_EQUIVALENCE_FAILED保留，不声称Dense/Physical严格数值等价。

## 范围与协议

这是用户查看Deep6/Deep8结果后明确请求的单一追加窗口，非预先独立确认。只新增L20–L26固定配置的128次正式质量生成；原128题、顺序、图片、Prompt、Gold、处理器、BF16输入Tensor和评分规则完全冻结，未扩展其他窗口、GQA、OCRBench或训练，未新增效率测量。

模型OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9，BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0、Python3.11.8，batch1/greedy，单图448×448、256视觉Token，原生EOS、max_new_tokens=32、logits_to_keep=1。GPU固定GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，与前轮Deep6/Deep8相同卡。Native/R0/R6/R14原始记录仍来自已明确授权的历史不同GPU，限制沿用。

沿用ATE_SVH_1TILE_SCORER_ALIGNED原生InternVL Chat及短答指令：Answer with the text visible in the image, using a word or short phrase. Do not explain. 仅原始图片和原问题，不输入OCR辅助、Gold或Hint。

131个缓存模型输入（128主题+3 Smoke）重算input_ids、BF16 pixels、Prompt与roles绑定通过，Gold从未包含在模型输入文件。816条旧记录（6方法×128+Exit27对照48）逐条原始SHA、输入/源码/协议绑定与官方评分核验后复用，未生成旧条件。父Commit c5a1036abccc1ecf2d77d1b187bde55d64b8231a；本128题是历史同题探索样本，不是新独立Holdout。

## 主质量

| 条件 | N | Soft Accuracy % | EOS末端KV Live=Capacity MiB | 来源 |
|---|---:|---:|---:|---|
| Native | 128 | 58.5156 | 18.3472 | REUSED_OLD_GPU |
| R0 | 128 | 8.5156 | 3.7859 | REUSED_OLD_GPU |
| R6 | 128 | 7.8125 | 6.7727 | REUSED_OLD_GPU |
| R14 | 128 | 7.0312 | 10.7474 | REUSED_OLD_GPU |
| Deep6_L20_25 | 128 | 43.2031 | 6.6991 | REUSED_SAME_CURRENT_GPU |
| Deep7_L20_26 | 128 | 47.5781 | 7.1826 | NEW_CURRENT_GPU |
| Deep8_L19_26 | 128 | 49.6094 | 7.6910 | REUSED_SAME_CURRENT_GPU |

EOS末端KV受输出长度影响；该列不是固定32 Token KV@32，本轮没有给Deep7编造E2E或KV@32。

## 同题配对统计

按Image SHA整簇Bootstrap5000次，seed20261112；128题恰有128个图片簇。普通未校正95%CI，属于多项、事后追加窗口的探索性比较。

| Deep7−基准 | 差值 pp | 95%CI pp | Soft Score提高/下降/不变 |
|---|---:|---|---|
| Native | -10.9375 | [-18.2812, -3.6699] | 10/27/91 |
| R0 | +39.0625 | [30.4668, 47.7344] | 58/3/67 |
| R6 | +39.7656 | [31.2500, 48.2812] | 59/4/65 |
| R14 | +40.5469 | [32.2656, 48.9844] | 59/3/66 |
| Deep6_L20_25 | +4.3750 | [-0.7812, 9.6875] | 11/5/112 |
| Deep8_L19_26 | -2.0312 | [-5.9375, 1.7188] | 3/6/119 |

Deep7与Deep8在同一卡上、同cutoff27、同视觉Hidden生存期，区别是Deep8额外开放L19的答案视觉读取；该配对最直接反映此窗口差异。Deep6的cutoff26与Deep7的cutoff27不同，二者差异还包含L26回看及一层视觉Hidden生存期变化。均不把CI跨0解释成等价，也不从此样本宣称通用最优。

## 48题已存在退出层对照

固定同48题：Exit27_NoLateRead 5.4167%，Deep7 45.8333%。Deep7−Exit27=+40.4167pp，95%CI [28.1250, 53.3333]；提高/下降/不变=23/0/25。复用前轮同GPU原始Exit27记录，未重跑控制。该小样本仅用于机制探索。

## 真实得分恢复与答案形式

相对R0：57题由零分变正分，47题由未满分到满分，21题由元回答变为官方有得分的非元回答；58题Soft Score提高，3题下降。结论基于官方十答案Soft Score，不以元回答数量代替OCR恢复。

Deep7元回答3题（answering前缀3），unanswerable 1题；错误数字候选15、品牌候选5、零分词语候选42、问题复述候选0题。后四类为重叠的规则诊断而非人工OCR归因，规则见继承的categories函数及per_question.csv。全部原始文本、Gold、Tokens、分数、逐题提高/损害已保留，未过滤。

Deep7正常EOS 128/128，空输出0，截断0；Token P50/P95/最大值=3.00/6.00/12。所有944条保存质量记录独立重算官方/leave-one-out十答案评分一致。

## 真实物理实现与数值状态

复用前轮已验证的参数化动态cutoff引擎，算法源码逐字节不变；本轮只传keep=L20–26与cutoff27。L0–L26视觉Hidden真实存在，进入L27删除整256行；L27只处理文本，未重插视觉。R→Image仍从L20禁止；P−1/Decode只在L20–26读原始视觉KV。L2–L5答案QKV/QK/AV/o_proj真实不执行，Norm/MLP/Residual保留。持久视觉KV只在7个KEEP层，每层256；非KEEP仅文本，SKIP层无持久Cache。零Scratch只是临时按原位置补齐，禁用位置Softmax前−inf，不能把Scratch宽度等同于FLOPs节省。

CPU官方28层Qwen2 Float64小模型：1 Prefill+8 Decode，Dense/Physical对齐Logits与Hidden MaxAbs均0；逻辑RoPE、mask、KV生命周期/实际Storage、独立Cache及异常清理均通过。GPU3题短/中/长Smoke与Native/No-op严格有效Logits一致检查通过，实际非目标连接正确屏蔽、被禁视觉概率0、允许层概率>0，未发生结构错误、NaN/Inf或OOM。

GPU Dense/Physical：完整Token一致3/3，标准化答案一致3/3；相同生成前缀Logits MaxAbs=0.25000000，各对齐步MeanAbs均值=0.02484108。这不是严格数值等价通过；原数值失败与豁免状态保留。逐层隐藏差、实际操作输入行数、KV真实Tensor与Storage、原始位置、允许/禁止视觉概率均有原始证据。

Deep7只有L27获得Prefill视觉Hidden删行，节省256个视觉layer-rows（Native28×256的3.57%）；这是实际行数计数，不是FLOPs或延迟。EOS末端缓存数值如主表，未补测固定32 Token速度，因此不声称Deep7更快或引用旧速度代替。

## 存储、核验与同步

SSD执行目录：/home/featurize/ate-runs/internvl3-svh-r-deep7-textvqa-3090。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-deep7-textvqa-3090/vlm_research/results/internvl3_svh_r_deep7_textvqa_3090/。所有原始Checkpoint、完整audit、GPU/CPU日志及复用证据保留server_only；模型、图片和准备张量继续本地SSD复用，没有下载到Windows。

GPU前70个源码SHA全部保持不变；后处理/归档/发布代码另列analysis_source_manifest.json。GitHub独立分支codex/internvl3-svh-r-deep7-textvqa-3090，不改main、不强推、不建PR、不覆盖前轮；SHA256SUMS与artifact_manifest.json列出上传和服务器保留范围。远端固定Commit逐文件回读SHA256及分支头核验回执在server_only/git_publish_receipt.json（避免Commit自引用）。

128次新增正式生成已完成，GPU已释放且无遗留计算进程，服务器保持运行。没有自动扩大样本或开启下一阶段。
