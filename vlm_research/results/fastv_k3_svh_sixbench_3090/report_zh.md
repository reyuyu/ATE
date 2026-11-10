# 六项融合评测核查与发布

方法：**FastV-K3 + llava-SVH**。六项完整生成与本机独立评分均已完成：**35,256 / 35,256**。
总体标记 **PARTIAL_PAIRED_COMPARISON**：缺少绑定输入的历史 SVH 逐题对照，不能把汇总分差当作严格配对结论。

| 基准 | 完成/要求 | 本次完整分数 | 原始 SVH 历史分数 | 历史参考差值 |
|---|---:|---:|---:|---:|
| GQA | 12578/12578 | 55.167753 | 57.592622 | -2.424869 |
| ScienceQA-IMG | 2017/2017 | 67.922657 | 68.963808 | -1.041151 |
| POPE | 8910/8910 | 83.108866 | 87.205387 | -4.096521 |
| MME | 2374/2374 | 1752.638255 | 1827.751801 | -75.113546 |
| MMBench | 4377/4377 | 63.180272 | 63.010204 | +0.170068 |
| TextVQA | 5000/5000 | 50.020000 | 50.400000 | -0.380000 |

## 评分与比较边界
- 除 MME 为总分点数外，其余均为百分比，差值单位为百分点。MME 为 14 类 Acc+Acc+ 总分；MMBench 为 1,176 个 Circular 组的组级正确率，不是 4,377 行的简单平均。
- MMBench 为冻结 VLMEvalKit exact-matching 的离线复现，不宣称新的官方服务器提交成绩。GQA/SQA-IMG/POPE 为冻结准确率；TextVQA 为官方词法 Soft Accuracy。
- POPE 严格保留冻结的 8,910 题，不扩成其他版本的 9,000 题。所有分母含空答案、截断与解析失败。
- 原始 SVH 仅有历史汇总分；缺少逐题预测以及 Pixel/Input 绑定。paired_comparisons.csv 的配对数为 0，CI、损害/纠正数留空，明确 PARTIAL。没有执行新的 Full/SVH 生成。
- MME 的 landmark/03d5e3bfc958be38.jpg（2 题）为历史授权的本机字节版本；保留全部 2,374 题，不宣称历史图片 SHA 一致或严格配对可比。预期 SHA 为 ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a；实际为 61d010ea9fabdc0bf930362d3cc1e4dc554789036f219feba927551effecd510。

## 物理与数值验证
- 历史 CPU 测试通过（11 项）；真实 GPU Smoke 3 张通过。当前发布只读取、校验记录，没有调用模型或 GPU。原始证据位于 evidence/。
- Native/No-op 有效 Logits 与关闭剪枝后的 SVH Logits 使用严格 torch.equal，保留 logits_to_keep=1。没有放宽冻结门槛。
- 全部 35,256 条重新核对 SQLite 完整性、逐记录 SHA256、题号/问题/Gold/Prompt/图片绑定、冻结协议、独立 Cache、有限 Logits 标志和逐层物理视觉行数。持久原始视觉 KV 均为 0。
- 原始图片没有重新跑 Processor；历史逐输入 Pixel/ID Hash 仅保留并绑定原记录，未将其描述为当前重算的验证。
- 视觉 Hidden 为 L0–L2:576、L3–L19:192、L20–L31:0。三图 Smoke 核对官方选择器、位置、MLP 和答案旁路残差；没有保存融合开启时完整 physical/dense Logits 对照，不能声称这项额外检查已通过。
- 原 FastV 汇总曾因 MMBench:1737 评分器原地小写化选项而失败；本次精确重放其副作用，原记录 SHA 仍严格通过，未改 Prompt、输入、预测或原检查点。

## 负面结果
- 对历史 SVH 参考，五项分数下降；MMBench 为轻微上升或持平。该观察不是配对显著性结论，不证明安全无损。
- TextVQA 有 35 个空答案、3 个截断；全部保留于分母。各项解析失败和重复统计见 negative_results.csv。
- 未进行同机配对速度实验，不宣称端到端加速；物理视觉行减少不等于已证明实际加速。

## 冻结、复现与原始数据
- protocol.json 与 mme_protocol.json 是原始冻结字节副本；其中 PENDING、no_push 等为当时状态，不能替代 completion.json、validation.json 与 publication.json 的当前发布结论。
- mme_execution_source/ 保存 MME 补跑的精确冻结源码与小型运行时依赖；与后来 SparseVLM 正式源码不同的 4 个文件从原 Commit 9ba5d63ba00d74c14393cc81dee70ba3071f0bec 恢复并逐个核对 SHA256。未把较新代码冒充 MME 执行版本。
- 源码按原文件复制，附必要本地 import 依赖。模型 revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager，torch 2.6.0+cu124，Transformers 4.49.0；SparseVLM 额外固定 einops 0.6.1，其运行时依赖 SHA 保存在 source_sha256.json。
- 逐条 SQLite、完整 CSV/JSONL、样本 Manifest、图片和权重留在服务器原路径，raw_inventory.json 给出位置与校验值；不上传完整数据集或巨量日志。源码保留原服务器路径，跨机器复现需恢复这些冻结资源，不可自动改协议。
- 发布审计曾使用默认 Python 3.13，在 TextVQA:34875 的 0.6 与历史 0.5999999999999999 上被严格检查阻止。恢复原 Python 3.11 后原样复核；未放宽相等条件或更改原记录。Python 3.12 起内置浮点 sum 的算法变化不可混入冻结评分。
- CPU 审计复现：/data/lk/深度学习/.envs/ate-vlm-research/bin/python vlm_research/fusion_results_delivery/audit.py --method fastv --destination <新的独立交付worktree>。必须使用 Python 3.11。此入口只做已保存预测的审计/词法重评分，不生成答案。
- SHA256SUMS 覆盖本交付核心文件和源码；文件摘要不自包含。与 Git Commit 一起使用。未创建或合并 PR，未修改 main。
