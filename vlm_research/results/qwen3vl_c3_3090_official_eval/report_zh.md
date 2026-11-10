# Qwen3-VL 原始 Physical C3：新 RTX 3090 评测

**当前状态：PARTIAL / PAUSED_FOR_PROTOCOL_REVIEW。用户质疑评测协议后，已暂停 GPU。冻结标签 OFFICIAL_PROTOCOL 只保留作历史标识，完整 Qwen 作者推理协议尚未验证。请先阅读 [协议疑点与暂停证据](protocol_review/report_zh.md)。**

状态：**PARTIAL**。更新时间 UTC：2026-11-10T00:34:38.533268+00:00。

本轮历史 GQA1000 中，Physical-Original − Native 为 -10.400 pp（图片簇配对95%CI [-12.713, -8.114]）。这是历史1000题复现，不是完整官方GQA成绩。
固定32 Token的48题效率阶段尚未完成，KV节省和E2E@32加速待测，不使用旧卡1.174×填补。
已完成可比质量集合中差值最低的是 GQA -9.071 pp, ScienceQA-IMG -1.289 pp；不同单位的MME/OCRBench另表列示。

## 模型、输入与执行版本

实际GPU：NVIDIA GeForce RTX3090 24GB；UUID `GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6`。模型 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，16个资产与固定Revision及历史权重SHA完全一致。BF16/Eager、Transformers4.57.6、PyTorch2.6.0+cu124、batch1、greedy、logits_to_keep=1。

工作树：`/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval`；独立分支 `codex/qwen3vl-c3-3090-official-eval`。执行基线Commit `30d3e7a0c9318e71a5951aca9a05b6b78108db72`，新增入口由source_manifest.json逐文件绑定。冻结C3核心未改；没有Repaired、训练、层搜索、量化或Attention后端变化。

模型真实GPU加载及生成成功；当前停止记录未包含全程峰值字段，暂不填写该值；已保存的逐请求成功峰值不能替代OOM失败状态。OOM失败记录若存在另行保留，不以该值替代OOM时分配状态。

历史1000题/918图像簇的Question ID、Gold、图像SHA和全部输入Tensor绑定与冻结Manifest完全一致；原Manifest SHA256 `bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88`。历史图像按固定MMEB旧Revision恢复，918张JPEG全部精确匹配原SHA。七项数据来自已校验备份，仅对实际读取图片按需核对SHA。

## 物理与数值验收

结构检查：**PASS**；严格数值门禁：**NUMERICAL_EQUIVALENCE_FAILED**。max_abs≤0.5、mean_abs≤0.02、Top1与完整Token相同的原门槛未变。

最短、典型、最长输入均验证Native/no-op有效Logits严格相同、手动Native与HF generate严格相同、审计开关不改变Physical输出。L22前视觉Hidden退出，P−1/Decode的L1–L6 QKV/QK/AV/o_proj实际跳算；36层Norm/MLP/Residual仍执行，DeepStack三路注入正常，mRoPE绝对位置保留，禁用连接概率为0，Compact KV及请求隔离/Hook清理通过。

| Question ID | 输入Token | 视觉Token | 最大max_abs | 最大mean_abs | 完整输出Token相同 | 严格门禁 |
|---|---|---|---|---|---|---|
| 08149474 | 92 | 64 | 0.500 | 0.056 | True | NUMERICAL_EQUIVALENCE_FAILED |
| 101022347 | 217 | 192 | 0.703 | 0.137 | True | NUMERICAL_EQUIVALENCE_FAILED |
| 12731033 | 1105 | 1080 | 0.500 | 0.082 | True | NUMERICAL_EQUIVALENCE_FAILED |

因此本轮保留原始实现的负面数值结果，全部质量和效率结论均标为探索性，不能称为Dense/Physical数值等价。

## 历史 GQA1000 严格复现

| 条件 | 完成题数 | Accuracy % | 空输出 | 截断 | 旧参考 % |
|---|---|---|---|---|---|
| Native | 1000/1000 | 69.400 | 0 | 0 | 69.4 |
| Dense-F | 1000/1000 | 58.800 | 0 | 0 | 58.8 |
| Physical-Original | 1000/1000 | 59.000 | 0 | 0 | 59.0 |

| 条件 − 参考 | 差值 pp | 图片簇95%CI | 分数下降/上升 | 首Token改变 |
|---|---|---|---|---|
| Dense-F − Native | -10.600 | [-12.887, -8.358] | 126/20 | 270 |
| Physical-Original − Native | -10.400 | [-12.713, -8.114] | 127/23 | 274 |
| Physical-Original − Dense-F | 0.200 | [-0.296, 0.700] | 2/4 | 15 |

全部3000条新生成与历史参考的原始答案、完整Token IDs、首Token、解析答案与分数均完全一致；旧参考仅在本轮3000条生成完成后读取，不充当新卡生成。严格数值门禁仍保留FAIL，不以答案一致替代Logits等价。

## 七项质量评测

协议差异与固定作者版本见official_protocol_audit_zh.md。官方/ATE存在不同输入时独立生成、独立计分。TextVQA明确保留Paper-OCR；ScienceQA明确为2017道IMG子集；MMBench为1176个完整Circular组的exact_matching，不付费调用裁判；OCRBench原始总分0–1000，MME总分0–2800。

| Benchmark / 协议 | 条件 | 完成/应有 | 正式分数 | 单位 | 状态 |
|---|---|---|---|---|---|
| GQA / OFFICIAL_PROTOCOL | Native | 12578/12578 | 61.735 | Accuracy % | COMPLETE |
| GQA / OFFICIAL_PROTOCOL | Physical-Original | 12578/12578 | 52.663 | Accuracy % | COMPLETE |
| ScienceQA-IMG / ATE_PAPER_PROTOCOL_AND_OFFICIAL_IMG_SCORING | Native | 2017/2017 | 94.794 | Accuracy % | COMPLETE |
| ScienceQA-IMG / ATE_PAPER_PROTOCOL_AND_OFFICIAL_IMG_SCORING | Physical-Original | 2017/2017 | 93.505 | Accuracy % | COMPLETE |
| TextVQA / ATE_PAPER_OCR_AND_OFFICIAL_SOFT_SCORING | Native | 787/5000 | — | Accuracy % | BLOCKED_CUDA_OUT_OF_MEMORY |
| TextVQA / ATE_PAPER_OCR_AND_OFFICIAL_SOFT_SCORING | Physical-Original | 787/5000 | — | Accuracy % | BLOCKED_CUDA_OUT_OF_MEMORY |
| POPE / ATE_PAPER_PROTOCOL | Native | 8910/8910 | 88.732 | Accuracy % | COMPLETE |
| POPE / ATE_PAPER_PROTOCOL | Physical-Original | 8910/8910 | 88.709 | Accuracy % | COMPLETE |
| POPE / OFFICIAL_PROTOCOL | Native | 879/9000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| POPE / OFFICIAL_PROTOCOL | Physical-Original | 878/9000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MME / ATE_PAPER_PROTOCOL | Native | 0/2374 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MME / ATE_PAPER_PROTOCOL | Physical-Original | 0/2374 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MME / OFFICIAL_PROTOCOL | Native | 0/2374 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MME / OFFICIAL_PROTOCOL | Physical-Original | 0/2374 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MMBench / ATE_PAPER_INPUT_OFFICIAL_CIRCULAR_EXACT_MATCHING | Native | 0/4377 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| MMBench / ATE_PAPER_INPUT_OFFICIAL_CIRCULAR_EXACT_MATCHING | Physical-Original | 0/4377 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| OCRBench-v1 / ATE_PAPER_PROTOCOL | Native | 0/1000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| OCRBench-v1 / ATE_PAPER_PROTOCOL | Physical-Original | 0/1000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| OCRBench-v1 / OFFICIAL_PROTOCOL | Native | 0/1000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |
| OCRBench-v1 / OFFICIAL_PROTOCOL | Physical-Original | 0/1000 | — | Accuracy % | PARTIAL_NOT_A_FORMAL_SCORE |

| Benchmark / 协议 | C3 − Native | 图片簇95%CI | 单位 |
|---|---|---|---|
| GQA / OFFICIAL_PROTOCOL | -9.071 | [-10.109, -8.055] | Accuracy pp |
| ScienceQA-IMG / ATE_PAPER_PROTOCOL_AND_OFFICIAL_IMG_SCORING | -1.289 | [-1.937, -0.695] | Accuracy pp |
| POPE / ATE_PAPER_PROTOCOL | -0.022 | [-0.202, 0.157] | Accuracy pp |

逐题完整答案、Token IDs、Gold、解析、初始/独立复算分数及空输出/截断均保留。paired_cases.csv.gz给出每题损害/纠正及答案、首Token变化，stratified_results.csv保留问题类型/子任务；重复yes/no或选项答案只作描述，不作为删除样本的理由。七项阶段每个OOM会保留失败题、输入长度和协议不变记录，停止该视图，不降低图像分辨率、不更换后端。

MME保留 `landmark/03d5e3bfc958be38.jpg` 已知单图SHA版本差异及其两题，不替换或删除。因此本轮Native/C3使用相同恢复图片可配对，但不能声称与最早旧实例所有MME图片逐字节相同。

## 新卡真实效率

48题使用原冻结的48个不同图像SHA，Native、Dense-F、Physical-Original在同一新GPU循环轮换。每输入/条件/模式先2次预热、再5次测量，取各输入重复中位数；主加速为中位数之和的比值。自然EOS、固定16和固定32分别保存。固定32真实执行1次Prefill+31次Decode，遇EOS仍继续。

统一E2E边界包括H2D、Execution/Cache上下文创建、Vision/Projector、语言Prefill、真实Decode、greedy与EOS及上下文清理；不含图片磁盘读取、Processor、模型加载或计时后的文本解码。事件时间单列Vision、Prefill与Decode，KV Live/Capacity按实际Tensor元素/存储字节统计，与分配器峰值显存分开。

| 模式 / 条件 | 输入数 | KV Live MiB | KV Capacity MiB | E2E ms | TTFT wall ms | TPOT ms | Vision ms | 语言Prefill ms | Decode ms | 峰值Allocated MiB | Reserved MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|

| 模式 / 条件 | Native/条件 E2E加速 | 95%CI | KV Live节省 % | Capacity节省 % |
|---|---|---|---|---|

## 独立评分、异常及局限

CPU原始标注审计验证GQA全12578题、ScienceQA全部2017整数Gold与Hint/Choices、TextVQA全部5000组十答案、MME1187双题组、MMBench1176完整旋转组、POPE三策略以及OCRBench1000条官方顺序。32条ScienceQA Hint只有冻结Paper模板的连续空白压缩，信息完整，未改输入。

新增ScienceQA初版适配器存在字母与整数Gold比较错误，且通用选项解析比冻结ATE更宽松。两项差异均在ScienceQA生成前发现：离线v3从冻结e281754原score.py提取原ScienceQA解析分支，保留严格提取规则并按官方整数索引匹配；v2通用解析仅作明确对照。原始生成、Gold与初始分数全部保留，不重跑GPU。逐条修正见scienceqa_offline_score_corrections.csv，解析对照见scienceqa_parser_alternate_per_question.csv及summary.csv。MMBench解析函数/正则与历史源码AST相同；TextVQA规范化与十答案软评分除进度条包装外AST相同，原作者文件SHA亦匹配历史vendor。

第一次GPU进程在第96题写入非ASCII答案时遭遇ASCII文件编码错误。已保存的285条预测按同一模型/源码/输入/协议哈希复用；仅通过启动环境PYTHONUTF8/PYTHONIOENCODING恢复，未改冻结推理源码。未保存完整记录的请求需要重做，完整错误保留在execution.log及execution_interruption.json。

统计采用20000次Image SHA256整簇配对Bootstrap，固定种子20261106，问题加权；MME重算任务Acc+AccPlus，MMBench按完整Circular组，TextVQA只报告软分数变化，不冒充二值翻转。不显著不能证明等价。未完成集合不填写正式成绩；MMBench若需要不可用官方裁判，保留exact_matching分数及AWAITING_OFFICIAL_SCORING状态。

模型作者公开vLLM采样及长输出设置不同于本轮授权的BF16/Eager Greedy，因此本轮不宣称复现Qwen模型榜单。原始Physical算法的严格数值一致性失败限制其等价性主张；质量损失与效率收益必须同时保留。

## 保存与资源状态

当前GPU：`NVIDIA GeForce RTX 3090, GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6, 1 MiB, 24123 MiB, 0 %`。
计算进程：`无`。

全部预测与必要大文件当前在独立本地 ext4 目录；持续盘只保留此前迁移断点，完整新预测尚未全部备份；可恢复检查点绑定源码、模型、协议、输入及图片SHA。此前仅本地保存；用户最新消息已另行授权向独立 GitHub 分支同步轻量证据。推送核验回执另存，未改main、未创建PR。Windows备份只有真实传输并校验后才在backup_receipt.json中标注完成。

最新执行状态：`PAUSED_FOR_PROTOCOL_REVIEW` / `SEVEN_BENCHMARK_QUALITY`。本报告不自动启动其他实验，也不关闭服务器。

## 未完成视图的子集诊断（非正式成绩）


以下只描述按冻结原顺序、在OOM/停止前实际完成的同题预测，不代表完整Benchmark；失败题和缺失题仍保留，正式分数继续为空。95%CI仅对已完成子集按图片簇重采样，不覆盖OOM缺失偏差，不能推广到完整测试集。

| 视图 | 完成配对/全量 | Native子集均分 % | C3子集均分 % | 差值 pp | 子集图片簇95%CI | 分数下降/上升 |
|---|---:|---:|---:|---:|---|---|
| TextVQA / ATE_PAPER_OCR_AND_OFFICIAL_SOFT_SCORING | 787/5000 | 80.051 | 60.102 | -19.949 | [-23.160, -16.822] | 203/29 |
| POPE / OFFICIAL_PROTOCOL | 878/9000 | 92.597 | 90.774 | -1.822 | [-2.860, -0.797] | 20/4 |

运行观察：进程曾实测处于NFS RPC等待，造成总体耗时增长。磁盘读写在模型请求E2E计时之外；未启动额外下载或并发GPU模型，保留固定暖机、轮换及GPU状态，不据此修正测量值。

## 用户授权的本地盘迁移与断点恢复

用户于本轮运行中明确授权切换为本地盘执行、持续盘保留备份。迁移仅改变存储位置，未改变冻结输入、模型权重、生成参数、算法或评分规则。
迁移完成时间（UTC）：2026-11-09T23:11:12.061494+00:00。运行副本根目录：`/home/featurize/ate-local-runs/qwen3090-29535-20261110`。原绝对路径通过独立 bind mount 指向本地 ext4；因此协议、输入 Manifest 和检查点绑定的 SHA256 保持一致。

| 内容 | 本地运行位置 | 持续盘保留副本 |
|---|---|---|
| worktree | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/worktree` | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/persistent_work_disk/ATE-qwen3vl-c3-3090-official-eval` |
| data | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/data` | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/persistent_work_disk/ate-qwen3090-data` |
| model | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/model` | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/persistent_work_disk/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| control | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/control` | `/home/featurize/ate-local-runs/qwen3090-29535-20261110/persistent_work_disk/ate-qwen3090-control-29535` |

全部模型必需文件重新核验 SHA256；10,248 个引用图片路径完整存在。数据通过 rsync 传输校验，原有逐张读取时的图片 SHA256 检查继续启用，未重复扫描所有图片哈希。
本地校验通过：历史预测 3,000 条、请求预测 41,329 条、逐题索引 49,079 条；保留了 1 个中断时仅完成单条件的索引，其缺失条件按原规则补齐。
本次迁移新增一次模型加载。隔离的恢复入口只跳过已经完成的历史 Smoke/GQA1000 及完整 Benchmark 视图，并保留已因 OOM 停止的 TextVQA 视图；不重试该 OOM，不重新生成已完成预测。恢复入口与冻结原入口的差异及 SHA256 见 `resume_scripts/local_resume_orchestration.diff` 和 `local_resume_source_manifest.json`。
迁移前已确认旧 GPU 进程退出且显存释放，再启动唯一的新 GPU 进程。迁移前持续盘副本完整保留；新增进度先写本地，完整结果备份状态以最终归档与校验回执为准。
质量生成包含迁移前后的同卡结果，输入与数值协议一致。严格 Dense/Physical 数值门禁的失败记录保留，存储迁移不构成数值等价验收。
正式 48 题效率测量尚未完成；后续将按原有统一计时边界在本地盘环境执行，磁盘读取不计入计时。

迁移证据已在 Windows 完成 6 个文件的 SHA256 校验（2026-11-09T23:15:42.489905+00:00）；该回执仅覆盖迁移记录和恢复源码，不代表尚未结束的完整实验已完成备份。

## OCRBench 协议差异说明更正（实际输入未变）

`input_manifest.json.protocol_differences` 中“OCRBench 的 ATE 文本含字面反斜杠 n、官方文本使用真实换行”的说明不准确。这是说明字段的误判，冻结 Manifest 原件保留，未悄悄改写。
对可信备份 ZIP 的原始 `inputs.jsonl` 成员与恢复文件逐字节核验后，确认 1,000 道题的 ATE 输入和官方输入本来就都使用真实换行，Prompt 完全相同，且均符合此次核对的作者模板。
原始成员及恢复输入 SHA256：`9472ae40763151ad4d53c3f3e77d8df8427f83abcf5ab5beeb8cf00eedfdf98c`。审计过程没有重新扫描图片或重新哈希整个数据包。
两种标签保留各自索引及评分记录；同图、同 Prompt、同生成上限的请求可合法共用本轮新生成的预测，因此不能将两个标签视为两次独立推理实验。模型输入、Gold、Scorer、算法及生成参数均未修改，不需要 GPU 重跑。
完整证据见 `protocol_description_correction.json`，可复核脚本为 `analysis/protocol_metadata_audit.py`。
