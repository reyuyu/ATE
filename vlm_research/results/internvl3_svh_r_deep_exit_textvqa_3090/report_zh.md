# InternVL3-SVH-R：晚层回看与动态物理退出

**Deep6（L20–L25，L26入口退出）43.2031%；Deep8（L19–L26，L27入口退出）49.6094%。** 两者明显优于历史R0/R6/R14，Deep8比Deep6高6.4063pp；仍分别落后历史Native 15.3125pp、8.9063pp。48题同GPU退出层对照支持晚层P−1/Decode读图是主要恢复来源，不能仅靠延迟Hidden退出解释。

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。结构、完整性、官方/独立评分已通过；历史NUMERICAL_EQUIVALENCE_FAILED不变，BF16 Dense/Physical不宣称数值等价。

## 冻结协议、换卡与复用

模型OpenGVLab/InternVL3-8B，Revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0，Python3.11.8，batch1，greedy，正常EOS，max_new_tokens=32。单图448×448、256视觉Token，原生Chat Template，ATE_SVH_1TILE_SCORER_ALIGNED短答指令，Image-only；Gold及OCR辅助文字从未进入模型输入。

当前GPU：GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，Driver610.57.04。用户明确授权从历史GPU GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e 换到本卡。128题、顺序、Prompt、图片SHA、BF16 Pixel Tensor、Token IDs、Gold、模型及Scorer绑定核验通过；Native/R0/R6/R14的512条真实历史记录来自父Commit 1885b03de8a396e178f39c6128c22a47999d3ef1，未重新正式生成。当前Native/No-op Smoke有效Logits严格torch.equal，单题历史Native完整Token也一致；此小量检查不能证明全部跨卡预测相同。主表是冻结历史同题探索比较，不是新独立Holdout，也不是本卡重测的Native质量基线。

新增正式质量352次：Deep6/Deep8各128，Exit26/Exit27无晚层读图对照各48。控制48题与速度8图均在GPU前按固定SHA冻结，无按结果选题。模型仅加载一次。

## 128题主质量结果

| 条件 | Soft Accuracy % | 与Native差距 pp | 追回Native−R0差距 % | 正常EOS末端KV Live/Capacity MiB | 来源 |
|---|---:|---:|---:|---:|---|
| Native | 58.5156 | 0.0000 | 100.0000 | 18.3472 / 18.3472 | 历史旧GPU复用 |
| R0 | 8.5156 | 50.0000 | 0.0000 | 3.7859 / 3.7859 | 历史旧GPU复用 |
| R6 | 7.8125 | 50.7031 | -1.4062 | 6.7727 / 6.7727 | 历史旧GPU复用 |
| R14 | 7.0312 | 51.4844 | -2.9687 | 10.7474 / 10.7474 | 历史旧GPU复用 |
| Deep6_L20_25 | 43.2031 | 15.3125 | 69.3750 | 6.6991 / 6.6991 | 本轮新GPU实测 |
| Deep8_L19_26 | 49.6094 | 8.9063 | 82.1875 | 7.6910 / 7.6910 | 本轮新GPU实测 |

正常EOS末端KV受答案长度影响，不能拿该列替代统一32 Token的KV@32。未筛除任何错误、元回答或未得分输出。

## 配对Soft Score变化

按Image SHA整簇配对Bootstrap 5000次，seed20261112；本128题恰有128图片簇。下表均为候选−基准，普通未校正95%CI，多项比较与历史样本探索性不确定性均保留。

| 候选−基准 | 差值 pp | 95%CI pp | 提高/下降/不变（题） |
|---|---:|---|---|
| Deep6_L20_25 − R0 | +34.6875 | [26.1719, 43.1250] | 52/3/73 |
| Deep8_L19_26 − R0 | +41.0937 | [32.4219, 49.7656] | 61/3/64 |
| Deep6_L20_25 − R6 | +35.3906 | [27.0312, 43.8281] | 52/2/74 |
| Deep8_L19_26 − R6 | +41.7969 | [33.3594, 50.1582] | 62/3/63 |
| Deep6_L20_25 − R14 | +36.1719 | [28.0469, 44.2207] | 52/1/75 |
| Deep8_L19_26 − R14 | +42.5781 | [34.4531, 50.6270] | 62/2/64 |
| Deep8_L19_26 − Deep6_L20_25 | +6.4062 | [0.8594, 11.9531] | 15/6/107 |
| Deep6_L20_25 − Native | -15.3125 | [-23.0469, -7.6563] | 10/34/84 |
| Deep8_L19_26 − Native | -8.9062 | [-15.7813, -2.2656] | 11/24/93 |

Deep8−Deep6的CI本轮为正，但未经多重比较校正且不是独立Holdout；不宣称最优层或非劣效。全部逐题损害、首Token变化、完整文本变化、Gold和原始Token IDs见per_question_changes.csv与per_question.csv。

## 48题退出层机制对照

| 条件 | Soft Accuracy % |
|---|---:|
| Native | 55.2083 |
| R0 | 5.4167 |
| Exit26_NoLateRead | 5.4167 |
| Deep6_L20_25 | 37.0833 |
| Exit27_NoLateRead | 5.4167 |
| Deep8_L19_26 | 49.1667 |

| 候选−基准 | 差值 pp | 95%CI pp | 提高/下降/不变 |
|---|---:|---|---|
| Deep6_L20_25 − Exit26_NoLateRead | +31.6667 | [20.0000, 44.5833] | 17/0/31 |
| Deep8_L19_26 − Exit27_NoLateRead | +43.7500 | [31.2500, 56.6667] | 25/0/23 |
| Exit26_NoLateRead − R0 | +0.0000 | [0.0000, 0.0000] | 0/0/48 |
| Exit27_NoLateRead − R0 | +0.0000 | [0.0000, 0.0000] | 0/0/48 |

Exit26与Exit27在48题上与历史R0逐题分数完全相同，虽分别有4/2题文本变化。对应Deep6/Deep8相对同GPU、同cutoff禁读图对照提高31.6667/43.7500pp；因此仅延长Hidden生命周期未产生本样本质量恢复，新增晚层答案视觉访问有直接支持。仅48题，不作独立统计确认；BF16路径差异仍存在，不能证明精确数值因果分解。

## 真实答案恢复与异常形式

R0的answering/元回答39题，Deep6降至6题，Deep8降至3题；unanswerable分别19/8/2题。相对R0，Deep6有51题从零分到正分、42题从未满分到满分，Deep8对应60/49题；各有22题由元回答变为有官方得分的非元回答。两者分别共有52/61题Soft Score提高，3/3题下降。恢复来自官方十答案匹配的真实得分，不只是元回答减少。

错误数字（文本含数字、零分且非元回答）Native/R0/R6/R14/Deep6/Deep8为14/22/27/27/15/15题；错误品牌候选（品牌类问题、零分字母回答）为5/9/10/10/6/4题；零分词语候选为30/54/55/56/45/39题；问题复述候选为0/1/1/2/0/0题。这些是重叠的、明确规则诊断标签，不是人工OCR错误分类或新的评分器，不能证明每个词语错误的视觉原因；规则及全部案例见analyze.py、meta_response_analysis.csv、per_question.csv。

全部864条质量记录均正常EOS、无空答案、无32 Token截断。Main128输出Token P50/P95/最大值：Deep6=3/7.65/13，Deep8=3/7/12；其余条件及48题对照长度统计见generation_length_summary.csv。

## 物理执行与数值审计

四配置CPU官方28层Qwen2 Float64小模型，1 Prefill+8 Decode，Dense/Physical对齐Hidden与Logits MaxAbs均0。独立Cache、异常清理、Mask、原始RoPE、KV位置、MLP/Norm/Residual及真实SKIP均通过。

真实GPU用3题覆盖短/中/长输入（独立于128题）对四配置做Dense/Physical Smoke。全部12对完整Tokens及标准化答案一致，但Logits不相等：

| 条件 | 完整Token一致/3 | 标准化答案一致/3 | 对齐步Logits MaxAbs | 对齐步MeanAbs均值 |
|---|---:|---:|---:|---:|
| Deep6_L20_25 | 3 | 3 | 1.70312500 | 0.19696917 |
| Deep8_L19_26 | 3 | 3 | 0.12890625 | 0.01679657 |
| Exit26_NoLateRead | 3 | 3 | 1.89843750 | 0.21107353 |
| Exit27_NoLateRead | 3 | 3 | 0.38281250 | 0.01989404 |

仅在相同生成前缀比较Logits/Hidden，完整逐层Max/Mean见smoke_hidden_differences.csv。历史数值失败状态仍保留，答案一致不能替代数值验收。本轮结构通过建立在预先明确的探索性数值豁免上，未放宽结构标准，未做BF16内核修复。

实际Hook记录确认Deep6在L26入口、Deep8在L27入口删除整256行；此前真实视觉Hidden继续存在，未重插。R→Image仍从L20禁止且实测概率0；P−1/Decode仅KEEP可读图且实测概率>0，非KEEP禁止概率0。L2–L5答案真实不调用QKV/QK/AV/o_proj，所有层Norm/MLP/Residual执行。持久KV只在Deep6的L20–25、Deep8的L19–26含256视觉Token；其余非SKIP层仅文本，L2–L5无持久Cache。逐层实Tensor形状、dtype、物理长度、底层Storage及原始逻辑位置均核验，无重复/重编号/跨请求共享。

非KEEP的必要零Scratch只是临时原位置补齐，禁用位置在Softmax前为−inf，不计持久KV；Attention reduction宽度仍保留，因此KV压缩率不代表Attention FLOPs缩减。

Deep6仅L26、27享有Prefill视觉删行（512个视觉layer-rows，占Native 28×256的7.14%）；Deep8仅L27（256行，3.57%）。这是输入行计数，不是FLOPs或延迟估算。实际Q/K/V/o_proj、Norm和MLP行数详见operator_input_rows.csv与完整物理审计。

## 同卡8图真实32 Token小型效率

两候选均满足预冻结门槛：Accuracy≥20%且相对R0≥10pp。新GPU同一进程与模型，每方法2次预热，8图×3重复、交错轮换Native/R0/Deep6/Deep8，共96次正式测速。1 Prefill+31真实Decode，忽略EOS，不含磁盘/Tokenizer/H2D，重型审计Hook关闭。E2E含视觉编码、Projector/注入、语言Prefill、Decode与Context清理的同步墙钟；KV布局审计与Token CPU拷贝在计时外。各请求Cache独立，未拼接历史时间。

| 方法 | KV@32 Live=Capacity MiB | E2E ms | TTFT ms | TPOT ms/token | Native/E2E | KV减少 % | E2E降低 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 19.8652 | 1131.2003 | 143.5041 | 31.8612 | 1.0000× | 0.00 | 0.00 |
| R0 | 5.0273 | 1102.0522 | 123.4315 | 31.5684 | 1.0264× | 74.69 | 2.58 |
| Deep6_L20_25 | 8.0273 | 1096.2015 | 138.7191 | 30.8865 | 1.0319× | 59.59 | 3.09 |
| Deep8_L19_26 | 9.0273 | 1093.4337 | 141.2121 | 30.7168 | 1.0345× | 54.56 | 3.34 |

上述均为24请求算术均值；每图重复次数相同，亦等于每图均值的8图平均。中位数、逐请求、GPU Event时间、Peak allocated/reserved见speed_summary.csv和speed_per_input.csv。KV以32输出Token产生完时的真实Cache计，最后输出Token尚未再次输入，Cache逻辑长度=input+31。

| 方法 | Vision GPU Event ms | Projector/注入增量 ms | Language Prefill Event ms | Decode Event ms | Peak allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|
| Native | 32.5345 | 0.5073 | 110.3852 | 987.7098 | 15286.63/15424.00 |
| R0 | 32.5276 | 0.5129 | 90.3039 | 978.6267 | 15286.63/15424.00 |
| Deep6_L20_25 | 32.5208 | 0.4998 | 105.6090 | 957.4910 | 15286.63/15424.00 |
| Deep8_L19_26 | 32.5171 | 0.5050 | 108.1004 | 952.2297 | 15286.63/15424.00 |

原CSV的vision_projector_injection_ms是从start到视觉+投影/注入完成的累计值，不能再与vision_encoder_ms相加；上表增量为两者相减。CUDA Event分解与同步墙钟不是相同测量，不能强行相加冒充墙钟E2E。Peak总显存几乎不变，不代表没有真实KV下降。

Deep6/Deep8相对Native节省11.8379/10.8379 MiB持久KV，约59.59%/54.56%；E2E仅降低3.09%/3.34%，加速约1.032×/1.035×。有缓存价值，但这8图无法支撑明显延迟加速的泛化结论，也不能将Deep8略快解释为稳定最佳效率点。没有额外完整Benchmark或跨实例拼接。

## 结论与限制

Deep8是本轮两个固定晚层窗口中的较高质量候选，值得在另行授权后做独立TextVQA样本验证；不自动扩样。它仍比历史Native低8.91pp，不能声称接近无损或严格非劣效。BF16 Dense/Physical数值不等价、跨GPU历史基线复用、历史同题探索、多重比较与仅8图测速均限制正式论文结论。只延迟退出的48题结果不支持质量改善；晚层原始视觉KV访问表现出真实Soft Score恢复，但不能从本实验推断通用最优层或物理加速保证。

## 复现、存储与同步

新源码：vlm_research/internvl3_svh_r_deep_exit_textvqa/；父源码Commit 1885b03de8a396e178f39c6128c22a47999d3ef1。GPU前冻结source_manifest的62文件SHA保持不变；后处理/发布代码单列analysis_source_manifest，不修改冻结运行源码。

SSD执行目录：/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-deep-exit-textvqa-3090/vlm_research/results/internvl3_svh_r_deep_exit_textvqa_3090/。完整SQLite、原始物理审计、日志、下载/缓存回执及复用原始记录在server_only/；模型、图片和张量仍在服务器本地SSD，不上传GitHub或Windows。

GitHub独立分支codex/internvl3-svh-r-deep-exit-textvqa-3090，不改main、不强推、不建PR。逐文件上传清单与SHA见artifact_manifest.json、SHA256SUMS，推送后本地/远端Commit与每个Git Blob一致性回执保存在server_only/git_publish_receipt.json，避免把Commit SHA递归写进自身Commit。完整物理审计压缩件上传，原始JSON保存在server_only/，轻量同名JSON引用该完整证据，不丢失审计。

352次正式质量+96次正式速度已完成，512条历史复用；无其他Benchmark、训练、层搜索或模型更改。GPU计算进程已退出，释放记录见gpu_release.json，服务器保持运行。
