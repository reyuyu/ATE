# L0–L1 Attention 与 L2–L5 BOS Sink：GPU2 小型机制实验

状态：**complete**。原始六类 GQA 各500题，共3,000题、2,409个图像 SHA256 聚类。只新增 L0–L1 OFF 的3,000次完整答案生成；Full、L2–L5 OFF、L0–L5 OFF 的9,000条历史答案直接复用。未重采样，未修改历史结果或 main。

## 核心答案

1. **L0–L1 单独关闭没有观察到总体 Accuracy 下降。** Full 与 L0–L1 OFF 都为2,627/3,000，即87.5667%；配对差值0.000个百分点，图像聚类95%CI为[-0.299, +0.299]。但10题原对变错、10题原错变对，不能称为逐题无损或证明完全等价。
2. **与 L2–L5 存在非加性联合损害。** 单独关闭 L0–L1 的净损害为0.000个百分点；已经关闭 L2–L5 时，再关闭 L0–L1 的损害为2.400个百分点。两者之差为2.400个百分点，配对95%CI [1.588, 3.225]；五个预声明对比的 Bonferroni 同时95%CI [1.330, 3.493]仍不包含0。这是本批题目上、Accuracy 概率尺度上的交互，不等于证明某个具体 Head 的作用。
3. **关闭 L0–L1 明显改变后续 BOS Attention。** 12个预先固定、未按结果筛选的案例全部出现变化；L2 的平均 BOS Mass 从71.272%降到5.421%，L3–L5分别下降10.419、6.452、2.948个百分点。实际 BOS 缓存 K/V 在两个条件下逐位一致，P−1 的 Query/Hidden 则明显改变。**这些观测不证明 BOS Sink 的下降就是答案退化的中介原因。**

## 协议与复用

- 参考版本：`e495adf4fd93337a3638981bbb66c2bbac81242d`；独立分支：`codex/gpu2-l0l1-bos-mechanism`。
- 成功采集所冻结的代码版本：`b9a85a11360b3135bc7dc88c6c522e45e271537d`。结果提交晚于采集代码版本；源码哈希见 `execution_manifest.json`。
- LLaVA-1.5-7B、FP16/eager、Transformers 4.49.0；模型、环境、原始 Prompt、完整答案评分器与历史实验一致。环境和物理 GPU2 UUID 见 `smoke.json`；本地模型使用固定快照，不下载替换。
- L0–L1 OFF：仅将原 Prefill P−1 和之后每个 Decode Query 在 L0/L1 的 `self_attn.o_proj` 输出置零。其余 Prefill Query、MLP、其他层保留。每个条件独立产生真实 Prefill、KV Cache 和答案前缀，不复用 Full 的后续缓存。
- 复用原有 `supplement_generate` / `JointAttention` / `generate` 与 `gqa_prediction`，只添加独立实验目录及 L0–L5 的定点行观察器，未改动基础引擎。
- Greedy，`max_new_tokens=32`，EOS=2。全部四个条件均无空答案、无截断。
- 原生首 Token、其全词表 LogP、Full logits 哈希复用历史绑定；新增条件只从自己的首步 logits 计算指标，不另跑3,000次 Full 参考 Forward。
- 12个主要机制案例在运行前冻结：每类对原始题号排序，以固定随机种子20261031独立取2题，详见 `protocol.json`。另有3个事后退化示例，不计入12案例统计。

## 完整答案 Accuracy

以下 Accuracy 为原协议下的**完整答案评分**，不是首 Token 命中率。pp表示百分点。区间为按图像 SHA256 整簇重采样的配对95%CI。

| 条件 | 正确/总数 | Accuracy | 相对Full Δ(pp) [95%CI] | 原对→错 | 原错→对 | 首Token Flip | 平均LogP Loss(nat) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full（复用） | 2627/3000 | 87.5667% | 0 | 0 | 0 | 0 | 0 |
| L0–L1 OFF（新增） | 2627/3000 | 87.5667% | 0.000 [-0.299, 0.299] | 10 | 10 | 19 (0.6333%) | -0.002401 |
| L2–L5 OFF（复用） | 2630/3000 | 87.6667% | +0.100 [-0.332, 0.532] | 20 | 23 | 43 (1.4333%) | -0.000130 |
| L0–L5 OFF（复用） | 2558/3000 | 85.2667% | -2.300 [-3.121, -1.492] | 113 | 44 | 138 (4.6000%) | +0.050137 |

LogP Loss定义为 `logP_Full(t_Full) - logP_condition(t_Full)`，其中 `t_Full` 是固定的原生首 Token。使用 FP32 全词表 log_softmax；它是有符号值，不是 Gold Token LogP，不是完整答案损失或准确率。负均值表示这个固定 Token 的平均 LogP 略有提高，不能解释成所有题目改善。完整答案损益计数与首 Token Flip 不是同一个量，后续 Decode 也受干预。

### 非加性联合效应

记 `A=Full`，`F=L0L1_OFF`，`E=L2L5_OFF`，`D=L0L5_OFF`，每个字母表示逐题0/1正确性。

`H01 = A-F`；`H01_given25 = E-D`；`I = (E-D)-(A-F) = F+E-D-A`。

| 对比 | 损害估计(pp) | 普通配对95%CI | 五对比同时95%CI |
|---|---:|---:|---:|
| L0–L1 单独损害 H01 | 0.000 | [-0.299, 0.299] | [-0.396, 0.395] |
| 已关闭L2–L5后的额外损害 H01_given25 | 2.400 | [1.620, 3.198] | [1.377, 3.454] |
| 非加性额外联合损害 I | 2.400 | [1.588, 3.225] | [1.330, 3.493] |

200,000次图像聚类配对 Bootstrap，种子20261031，每次整簇抽样，按题目加权；同图的题目始终一起抽取。交互直接对逐题 `F+E-D-A` 重采样，不相减独立区间。五个预声明对比为 F−A、E−A、D−A、E−D、I；H01只由F−A取相反数，不新增检验。

### 六类结果

每类500题；分组结果是描述性分析，不能将未校正的分组区间当作额外确认性发现。

| 类型 | Full | L0–L1 OFF | L2–L5 OFF | L0–L5 OFF | L0–L1 Δ(pp) |
|---|---:|---:|---:|---:|---:|
| ChooseAttr | 82.6% | 83.0% | 83.0% | 80.0% | +0.4 |
| ChooseCat | 89.6% | 89.6% | 89.8% | 85.4% | 0.0 |
| ChooseRel | 83.6% | 83.6% | 83.4% | 80.6% | 0.0 |
| CompareAttr | 89.2% | 88.6% | 88.6% | 88.6% | -0.6 |
| LogicalObj | 95.8% | 95.6% | 96.6% | 96.0% | -0.2 |
| QueryAttr | 84.6% | 85.0% | 84.6% | 81.0% | +0.4 |

## BOS Sink 与 Query/Hidden

只测**原始 Prefill 的 P−1，负责预测首答案 Token**，不测答案结束后的 EOS。每题每条件仅保存 L0–L5 的逐 Head 注意力行、少量 Query/Hidden 和真实缓存 BOS K/V，不保存32层完整注意力方阵。

Mass分区互斥：BOS为原输入位置0；Image为576个视觉位置；Prompt为其余原始输入文本，包含模板和指令，**不是仅 Question 词元**。下表先对32个 Head取平均，再对12个预选案例取平均，单位为原始 Attention 的百分比。FP16行和误差沿用原有0.002上限，不做事后放宽。

| 层 | Full BOS/Image/Prompt (%) | L0–L1 OFF BOS/Image/Prompt (%) | BOS Δ(pp) |
|---|---|---|---:|
| L0 | 0.243 / 65.774 / 33.983 | 0.243 / 65.774 / 33.983 | 0.000 |
| L1 | 0.833 / 36.935 / 62.232 | 0.557 / 59.552 / 39.891 | -0.276 |
| L2 | 71.272 / 16.214 / 12.513 | 5.421 / 57.336 / 37.243 | -65.851 |
| L3 | 83.681 / 8.721 / 7.599 | 73.261 / 12.340 / 14.399 | -10.419 |
| L4 | 80.151 / 7.279 / 12.570 | 73.699 / 5.725 / 20.576 | -6.452 |
| L5 | 85.681 / 4.401 / 9.916 | 82.733 / 3.198 / 14.070 | -2.948 |

12/12案例的L2–L5平均BOS Mass均下降；并非每个Head都下降，热力图保留局部增强。案例内128个层×Head的平均绝对变化，其中位数为22.431pp；最大单层单Head绝对变化83.682pp。这些是描述性量，不把Head或层当独立样本作显著性检验。

| 层 | P−1层输入Hidden相对L2变化，中位数 | post-RoPE Query相对L2变化，中位数 |
|---|---:|---:|
| L2 | 1.1243 | 0.6973 |
| L3 | 1.4330 | 0.4640 |
| L4 | 1.1630 | 0.4144 |
| L5 | 1.0084 | 0.3612 |

相对变化为 `norm(off-full)/norm(full)`，不是百分比准确率。完整逐案例L2范数、相对范数、余弦相似度和逐位相等标记见 `case_vector_changes.csv`。

两条件在L2–L5的**实际缓存BOS K和V均逐位相同**。干预发生于P−1，因果掩码下不能反向改变早先BOS。与此同时P−1的Query及残差状态改变，说明BOS Mass变化并不要求BOS自身K/V改变；Attention softmax还依赖其他Key和归一化竞争，不能仅凭当前观测将全部变化唯一归因为某个Query方向。

L0 Attention权重逐位不变是预期结果：输出置零发生在权重计算之后；其有效o_proj输出范数确实为0。L1的Attention已经受L0上游干预影响，不能要求其权重也不变。12个预选案例的首Token均未翻转，尽管Sink变化很大；这进一步限制了“Sink减弱必然导致退化”的解释。

## 图表与案例

- [Figure B：12案例BOS曲线 PNG](figures/figure_B_bos_mass_12cases.png) / [矢量PDF](figures/figure_B_bos_mass_12cases.pdf)。L2下降最明显，L3–L5仍保留较强BOS Attention。
- [Figure C：12案例逐Head差异 PNG](figures/figure_C_layer_head_12cases.png) / [矢量PDF](figures/figure_C_layer_head_12cases.pdf)。统一对称色标，差值为OFF−Full。
- Figure A逐案例为Full的L0/L1视觉24×24 Patch热力图，附原图及**实际CLIP中心裁剪**，避免空间错位。显示逐Head平均后、视觉内部归一化的Patch分布，明确使用对数色标，并标注原始Image Mass；不能把它当作对物体区域的独立因果归因。

| 预选类型 | 案例1 Figure A / 独立B+C | 案例2 Figure A / 独立B+C |
|---|---|---|
| ChooseAttr | [05529910 A](cases/ChooseAttr__05529910/figure_A_visual_attention.png) / [BC](cases/ChooseAttr__05529910/figure_BC_bos_attention.png) | [06818059 A](cases/ChooseAttr__06818059/figure_A_visual_attention.png) / [BC](cases/ChooseAttr__06818059/figure_BC_bos_attention.png) |
| ChooseCat | [04784973 A](cases/ChooseCat__04784973/figure_A_visual_attention.png) / [BC](cases/ChooseCat__04784973/figure_BC_bos_attention.png) | [06278071 A](cases/ChooseCat__06278071/figure_A_visual_attention.png) / [BC](cases/ChooseCat__06278071/figure_BC_bos_attention.png) |
| ChooseRel | [04197591 A](cases/ChooseRel__04197591/figure_A_visual_attention.png) / [BC](cases/ChooseRel__04197591/figure_BC_bos_attention.png) | [05799999 A](cases/ChooseRel__05799999/figure_A_visual_attention.png) / [BC](cases/ChooseRel__05799999/figure_BC_bos_attention.png) |
| CompareAttr | [04967002 A](cases/CompareAttr__04967002/figure_A_visual_attention.png) / [BC](cases/CompareAttr__04967002/figure_BC_bos_attention.png) | [06626818 A](cases/CompareAttr__06626818/figure_A_visual_attention.png) / [BC](cases/CompareAttr__06626818/figure_BC_bos_attention.png) |
| LogicalObj | [04863771 A](cases/LogicalObj__04863771/figure_A_visual_attention.png) / [BC](cases/LogicalObj__04863771/figure_BC_bos_attention.png) | [061022270 A](cases/LogicalObj__061022270/figure_A_visual_attention.png) / [BC](cases/LogicalObj__061022270/figure_BC_bos_attention.png) |
| QueryAttr | [04955769 A](cases/QueryAttr__04955769/figure_A_visual_attention.png) / [BC](cases/QueryAttr__04955769/figure_BC_bos_attention.png) | [06273941 A](cases/QueryAttr__06273941/figure_A_visual_attention.png) / [BC](cases/QueryAttr__06273941/figure_BC_bos_attention.png) |

每张PNG都有同名PDF。Figure D按预先冻结的规则，从Full正确而F错误的题目中按类型/题号排序，选最多3个不同类型和图像的案例，**仅为事后说明性案例，不能据此估计退化率或确认机制**：

| 案例 | Full → L0–L1 OFF | 图片与Attention变化 |
|---|---|---|
| ChooseAttr 05895761：沙发颜色 | Yellow → Black | [D1 PNG](figures/figure_D1_posthoc_degradation.png) / [PDF](figures/figure_D1_posthoc_degradation.pdf) |
| ChooseCat 02629218：公交车或火车 | Train → Bus | [D2 PNG](figures/figure_D2_posthoc_degradation.png) / [PDF](figures/figure_D2_posthoc_degradation.pdf) |
| ChooseRel 05888273：自行车相对人物左右 | Right → Left | [D3 PNG](figures/figure_D3_posthoc_degradation.png) / [PDF](figures/figure_D3_posthoc_degradation.pdf) |

## 正确性、计算量和审计

- 31项CPU测试通过；两个真实GPU2 Smoke通过，覆盖首尾任务类别。
- Smoke核验历史Full完整Token与首logits逐位一致、No-op每步有效logits与初始缓存一致、所有32层其他Prefill Query不变、MLP每步正常、独立HF消融生成结果一致、P−1捕获与生成首logits一致、原生恢复、Hook清理及输入不污染。
- 所有15个捕获案例均核验Full首logits精确匹配历史、F首logits精确匹配本轮生成、L0权重不变、L0/L1有效输出为0、BOS真实缓存K/V不变。
- 3,000条原子检查点逐题绑定源码、样本和历史哈希；无重复计数。历史9,000条检查点与5个受保护文件保持原SHA256。
- 成功完成后Hook清理、模型参数版本未变；输出无NaN/Inf，未发生OOM。正式新增生成约973秒；不含加载、Smoke及随后全部15个案例捕获。它不是总端到端计时，也不是模型加速结论。
- 初次启动在正式评分前遇到TASKS字典索引错误；已退出、修复、补回归测试，再冻结成功代码。原始失败日志保留于 `attempts/01_pre_smoke_task_index/` 与 `failures/`，不隐藏失败、不放宽容差。
- 只暴露物理GPU2 UUID，未终止其他任务。GPU工作结束后已释放GPU2，`gpu_release.csv`记录释放时约27MiB/0%。汇总和画图仅用CPU。
- 命令见 `execution_notes.md`；最终离线完整性核验见 `verification.json`。

## 文件索引与解释边界

- `per_question.csv`：12,000行，四条件完整答案、正确性、首Token和LogP指标，明确复用标记。
- `accuracy_summary.csv`、`paired_effects.csv`：总体四条件结果、交互及普通/同时区间；`task_accuracy.csv`、`task_paired_effects.csv`为六类描述性结果。
- `case_layer_mass.csv`、`case_head_bos_changes.csv`、`case_vector_changes.csv`、`case_summary.csv`：逐层原始Mass、逐Head变化、Query/Hidden/KV变化；预选/事后标签分开。
- `cases/`：原始小型NPZ、位置索引、实际CLIP裁剪、案例PNG/PDF；`figures/`为合并B/C和D1–D3。
- `protocol.json`、`reuse_bindings.json`、`execution_manifest.json`、`cpu_validation.json`、`smoke.json`、`completion.json`、`result.json`：冻结配置、环境/数据/源码哈希、测试和完成审计。

**已观察的因果干预效应**：关闭指定Attention输出会改变P−1状态和后续BOS分布；在本批3,000题上，单独关闭净Accuracy不变，联合关闭存在额外损害。

**尚未证实的机制解释**：L0/L1是否通过“维持BOS Sink”来保护Accuracy，仍不能由本实验确定；没有单独操纵/恢复Sink来隔离中介路径。12个机制案例不是额外3,000题的全覆盖机制测量，也不支持推广到其他模型、Prompt或数据集。未训练、未扫描其他窗口、未额外消融。
