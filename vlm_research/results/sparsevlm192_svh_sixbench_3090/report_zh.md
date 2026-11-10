# 六项融合评测核查与发布

方法：**SparseVLM-192-prefix + llava-SVH**。六项完整生成与本机独立评分均已完成：**35,256 / 35,256**。
总体标记 **PARTIAL_PAIRED_COMPARISON**：缺少绑定输入的历史 SVH 逐题对照，不能把汇总分差当作严格配对结论。

| 基准 | 完成/要求 | 本次完整分数 | 原始 SVH 历史分数 | 历史参考差值 |
|---|---:|---:|---:|---:|
| GQA | 12578/12578 | 56.137701 | 57.592622 | -1.454921 |
| ScienceQA-IMG | 2017/2017 | 68.170550 | 68.963808 | -0.793258 |
| POPE | 8910/8910 | 85.656566 | 87.205387 | -1.548821 |
| MME | 2374/2374 | 1820.250000 | 1827.751801 | -7.501801 |
| MMBench | 4377/4377 | 63.010204 | 63.010204 | +0.000000 |
| TextVQA | 5000/5000 | 50.298000 | 50.400000 | -0.102000 |

## 评分与比较边界
- 除 MME 为总分点数外，其余均为百分比，差值单位为百分点。MME 为 14 类 Acc+Acc+ 总分；MMBench 为 1,176 个 Circular 组的组级正确率，不是 4,377 行的简单平均。
- MMBench 为冻结 VLMEvalKit exact-matching 的离线复现，不宣称新的官方服务器提交成绩。GQA/SQA-IMG/POPE 为冻结准确率；TextVQA 为官方词法 Soft Accuracy。
- POPE 严格保留冻结的 8,910 题，不扩成其他版本的 9,000 题。所有分母含空答案、截断与解析失败。
- 原始 SVH 仅有历史汇总分；缺少逐题预测以及 Pixel/Input 绑定。paired_comparisons.csv 的配对数为 0，CI、损害/纠正数留空，明确 PARTIAL。没有执行新的 Full/SVH 生成。
- MME 的 landmark/03d5e3bfc958be38.jpg（2 题）为历史授权的本机字节版本；保留全部 2,374 题，不宣称历史图片 SHA 一致或严格配对可比。预期 SHA 为 ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a；实际为 61d010ea9fabdc0bf930362d3cc1e4dc554789036f219feba927551effecd510。

## 物理与数值验证
- 历史 CPU 测试通过（20 项）；真实 GPU Smoke 6 张通过。当前发布只读取、校验记录，没有调用模型或 GPU。原始证据位于 evidence/。
- Native/No-op 有效 Logits 与关闭剪枝后的 SVH Logits 使用严格 torch.equal，保留 logits_to_keep=1。没有放宽冻结门槛。
- 全部 35,256 条重新核对 SQLite 完整性、逐记录 SHA256、题号/问题/Gold/Prompt/图片绑定、冻结协议、独立 Cache、有限 Logits 标志和逐层物理视觉行数。持久原始视觉 KV 均为 0。
- 原始图片没有重新跑 Processor；历史逐输入 Pixel/ID Hash 仅保留并绑定原记录，未将其描述为当前重算的验证。
- 历史物理/dense 参考：Token 一致，Logits 最大绝对误差 0.03125000，最大 Hidden relative RMSE 0.00181567；按原先 allclose(atol=0.10, rtol=0.003)、RMSE≤0.01、cosine distance≤1e-4 验收，不冒充逐位相等。
- 视觉 Hidden 为 576→309→204→113→0，L20 起退出，32 层平均 164.125。官方 192 Top-k、回收与聚类 AST 冻结；文本 RoPE 不压紧，合并视觉 Token 使用已声明锚点。这不是独立 SparseVLM compact-position 实现的完全等价版本。
- 曾有官方参考函数 CPU/CUDA bias 设备不一致的失败 Smoke；失败记录保留。随后仅限定工厂分配设备，生产数学与门槛不变，重新冻结并通过六图门槛。

## 负面结果
- 对历史 SVH 参考，五项分数下降；MMBench 为轻微上升或持平。该观察不是配对显著性结论，不证明安全无损。
- TextVQA 有 35 个空答案、4 个截断；全部保留于分母。各项解析失败和重复统计见 negative_results.csv。
- 未进行同机配对速度实验，不宣称端到端加速；物理视觉行减少不等于已证明实际加速。

## 冻结、复现与原始数据
- protocol.json 与 mme_protocol.json 是原始冻结字节副本；其中 PENDING、no_push 等为当时状态，不能替代 completion.json、validation.json 与 publication.json 的当前发布结论。
- mme_execution_source/ 保存 MME 补跑的精确冻结源码与小型运行时依赖；与后来 SparseVLM 正式源码不同的 4 个文件从原 Commit 9ba5d63ba00d74c14393cc81dee70ba3071f0bec 恢复并逐个核对 SHA256。未把较新代码冒充 MME 执行版本。
- 源码按原文件复制，附必要本地 import 依赖。模型 revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager，torch 2.6.0+cu124，Transformers 4.49.0；SparseVLM 额外固定 einops 0.6.1，其运行时依赖 SHA 保存在 source_sha256.json。
- 逐条 SQLite、完整 CSV/JSONL、样本 Manifest、图片和权重留在服务器原路径，raw_inventory.json 给出位置与校验值；不上传完整数据集或巨量日志。源码保留原服务器路径，跨机器复现需恢复这些冻结资源，不可自动改协议。
- 发布审计曾使用默认 Python 3.13，在 TextVQA:34875 的 0.6 与历史 0.5999999999999999 上被严格检查阻止。恢复原 Python 3.11 后原样复核；未放宽相等条件或更改原记录。Python 3.12 起内置浮点 sum 的算法变化不可混入冻结评分。
- CPU 审计复现：/data/lk/深度学习/.envs/ate-vlm-research/bin/python vlm_research/fusion_results_delivery/audit.py --method sparse --destination <新的独立交付worktree>。必须使用 Python 3.11。此入口只做已保存预测的审计/词法重评分，不生成答案。
- SHA256SUMS 覆盖本交付核心文件和源码；文件摘要不自包含。与 Git Commit 一起使用。未创建或合并 PR，未修改 main。
