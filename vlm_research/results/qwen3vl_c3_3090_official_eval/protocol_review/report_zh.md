# 新 RTX 3090 Qwen3-VL 评测：协议疑点与暂停证据

状态：**PARTIAL / PAUSED_FOR_PROTOCOL_REVIEW**。用户要求先同步证据并质疑评测协议后，已停止唯一 GPU 推理进程；未启动其他模型或恢复评测。完整预测、Gold、冻结输入和原始失败记录均保留。完成数以 [snapshot.json](snapshot.json) 和 [protocol_matrix.csv](protocol_matrix.csv) 为准；进度日志每十题更新，可能落后于实际逐题存档。

## 可以确认与尚不能确认的事

**此前 `OFFICIAL_PROTOCOL` 的命名过于宽泛。** 已核对官方数据划分、Gold 和评分器，不代表已经复现 Qwen 作者使用的完整推理、Prompt、图像分辨率和生成配置。冻结 JSON 保留原标签以便追溯；本次补充说明收窄其含义，不回改历史输入或重新解释成模型官方榜单成绩。

Qwen 作者的固定 [README](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction) 指定 vLLM，列举 VLMEvalKit/lmms-eval，并说明部分任务有定制 Prompt；Instruct 设置为采样、temperature 0.7、top_p 0.8、top_k 20、presence_penalty 1.5、输出上限 32768。本轮用户明确要求 BF16/Eager/HF/Greedy，且任务上限为 32/128/1024。这些是已知实验条件差异，不能宣称 Qwen 官方榜单复现，也不能据此单独归因当前准确率差距。

| 检查项 | 已验证事实 | 判定及限制 |
|---|---|---|
| 同题 Native / C3 输入 | 共享冻结样本、模型、Prompt、Tensor 与生成上限绑定；CPU 已逐条复算现有预测 | VERIFIED：本轮配对；不保证与外部榜单协议相同 |
| POPE 官方视图 Prompt | 9000 题仅使用原问题；ATE 8910 题含短答案指令 | VERIFIED：两视图不同，不能把耗时直接横比为模型速度差 |
| POPE 的“官方”完整运行协议 | 作者仓库给出题目模板和答案评分代码，尚未取得本模型的逐任务作者运行配置 | UNRESOLVED：原问题-only 不足以证明完整官方 Prompt 已对齐 |
| GQA | 12578 题；本轮 Prompt 附加 `Answer using a single word or short phrase.`；官方答案全字符串匹配 | VERIFIED：具体输入与规则；UNRESOLVED：是否等同 Qwen 作者该任务 Prompt |
| ScienceQA-IMG | 2017 道含图测试题，保留 Hint/选项，使用冻结 ATE 选项提取及官方整数 Gold | VERIFIED：不冒充 ScienceQA 全测试划分或模型作者完整协议 |
| TextVQA | Paper-OCR、十答案软评分；固定模型默认 Processor 未指定单独像素预算 | VERIFIED：不是 image-only；该分辨率配置无法完成当前 24GB/Eager 全集 |
| MME / MMBench / OCRBench | 未开始生成；Prompt 样例与计划配置已留存 | NOT_VALIDATED：不填写成绩 |

## 速度为何比此前慢

对同图、同题、同 Gold、同策略的 **39 对 POPE 题**，仅使用现有预测做 CPU 对照，没有新增 GPU 请求，也没有挑选正确答案。样本范围为诊断时最近 40 道中的可匹配 39 道，属于速度诊断便利样本，不是质量结论或完整性能评测。

| 视图 | 条件 | 平均生成 Token | 模型请求 E2E | Decode TPOT |
|---|---|---:|---:|---:|
| 原问题-only，冻结标签 OFFICIAL_PROTOCOL | Native | 71.36 | 3736.4 ms | 50.64 ms/Token |
| 带短答案指令，ATE_PAPER_PROTOCOL | Native | 2.00 | 224.7 ms | 50.83 ms/Token |
| 原问题-only，冻结标签 OFFICIAL_PROTOCOL | Physical-Original | 33.74 | 1552.0 ms | 42.97 ms/Token |
| 带短答案指令，ATE_PAPER_PROTOCOL | Physical-Original | 2.00 | 188.1 ms | 42.52 ms/Token |

示例：`Is there a dog in the image?` 与同问题后追加换行及 `Answer the question using a single word or phrase.`。逐题完整输出、Token IDs、Tensor 哈希、原评分与独立 Yes/No 解析见 [pope_matched_prompt_cases.jsonl](pope_matched_prompt_cases.jsonl)。结论是输出长度解释了大部分总耗时差；不是固定长度、交替顺序的受控加速测试，也不能以输出较短宣称 C3 更快。

另有真实存储问题：虽然模型、数据和结果的子目录已绑定本地 ext4，`/home/featurize/work` 上级仍是 NFS。只读系统调用跟踪捕获到一次对已存在预测目录执行 `mkdir` 的 **9.107173 秒**阻塞，并同步观察到主线程 `D / rpc_wait_bit_killable`。一般预测写入 fsync 为数毫秒；停顿不能全部归因于长答案。详见 [intermittent_pause_diagnosis.json](../intermittent_pause_diagnosis.json)。原迁移的隔离不完整，已记录为执行问题。

曾准备私有挂载命名空间以隔离 NFS 父路径，但 **CPU 预检失败，未启用，未重新加载模型**。不能将其报告为已修复。原始代码与失败说明另存 [namespace_fix_attempt.json](namespace_fix_attempt.json)。用户的其他持续盘 I/O 和主机挂载未被覆盖。

## 已有结果与失败必须如何解读

历史 GQA1000 新卡复现全部 3000 次生成完成：Native 69.4%、Dense-F 58.8%、Physical-Original 59.0%。它们与旧卡逐题输出及 Token 一致，仅代表该冻结历史集合；不能冒充完整官方 GQA。

完整 GQA、ScienceQA-IMG 和 ATE POPE 视图已经生成并完成现有输出的 CPU 评分复核，具体分数、配对 CI、损害/纠正和解析失败见 [accuracy_summary.csv](../accuracy_summary.csv)、[paired_comparisons.csv](../paired_comparisons.csv)。这些分数只适用于明确记录的本轮输入协议；在完整官方模型评测配置未核实前，不声称作者协议复现。

TextVQA 在第 788 道、Question ID **35389** 阻塞：输入 15591 Token，其中视觉 15552 Token；原生 Eager Vision Attention 尝试分配 **115.33 GiB**。已有 787 对输出保留，5000 题正式成绩留空。OOM 出现在共享原生视觉阶段，不是 C3 专属失败；本轮未缩图、未改后端。缺失偏差意味着这 787 题的子集均分不能当作完整成绩。

ScienceQA 初版后处理的字母与整数 Gold 匹配错误已在该任务生成前发现，v3 离线修正恢复冻结 ATE 严格提取规则，原始预测和原分数保留。通用选项解析的额外对照单独存档，不取较高分冒充冻结成绩。具体修正与解析差异 CSV 已发布，不能掩盖这一实际适配错误。

严格 Dense / Physical 数值门禁仍为 **NUMERICAL_EQUIVALENCE_FAILED**：不放宽阈值、不以短输出一致替代数值等价。结构、Cache、安全和 Native/No-op 检查通过，质量结果只能按已授权探索性条件解释。详见 [validation.json](../validation.json)。

OCRBench 冻结 Manifest 中关于“字面反斜杠 n”的描述曾错误；实际两视图 1000 条 Prompt 完全相同且均为真实换行。已有独立纠错证据，未改任何输入；两标签不应当作两种不同 Prompt 实验。详见 [protocol_description_correction.json](../protocol_description_correction.json)。

## 当前边界与下一步

当前没有 GPU 推理进程。48 题固定 16/32 Token 效率测试未开始，**无本轮正式 KV/TPOT/E2E 加速结论**。诊断跟踪可能影响少量质量阶段计时，不能拿那些记录充当正式性能结果。

本提交发布轻量源码、冻结配置/输入压缩 Manifest、评分统计、当前逐题轻量结果、Prompt 示例、速度及 I/O 证据，排除模型、图片、全部原始请求文件和大日志。现存完整原始输出仍保留服务器本地盘；此前 Windows 备份范围与最终全量结果备份必须区分，不能据此宣布服务器可关机。

后续应先锁定逐任务可复现的模型评测入口，核对短答案/选项指令与图像像素预算；将数据/Scorer 对齐和模型作者推理协议对齐分开标识。若需要变更 Prompt、像素预算或 Backend，须作为新版本保留旧条件；本次没有修改并重跑，也没有继续扩大评测。
