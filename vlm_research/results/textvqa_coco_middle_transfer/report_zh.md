# COCO中层Decode视觉回看迁移至TextVQA300

COCO-Middle在TextVQA300得到**43.133%**，相对Decode None +11.733pp，相对Native Full -2.100pp。沿用11/32层的Decode视觉访问，恢复了Native−Decode None净缺口的84.82%（描述性比例）。

本轮完整执行四个冻结新条件，每个300题，共1200条新增Greedy答案；900条A/F/B历史答案直接审计复用。所有2100条用官方TextVQA十人共识软评分重新核验，评分差异为0。没有扩样、训练、Prompt修改、追加层搜索或调整随机种子。

计数边界：1200条是四个新增条件的正式处理轨迹（含12条通过门禁后保留的Smoke处理轨迹）。三题Smoke另有15次原生/No-op/A/F/B验证生成，不进入2100条评分矩阵，也不替换历史基线；本次实际模型生成轨迹共1215条。

## 全300题实测成绩

| 条件 | P−1读图层 | Decode读图层 | Accuracy [普通95%CI] | 相对Full Δpp | 相对Decode None Δpp | 来源 |
|---|---:|---:|---|---:|---:|---|
| Native Full | 32 | 32 | 45.233% [39.700, 50.833] | +0.000 | +13.833 | 审计历史复用 |
| Pminus1 Full / Decode None | 32 | 0 | 31.400% [26.333, 36.601] | -13.833 | +0.000 | 审计历史复用 |
| COCO-Early | 32 | 11 | 32.433% [27.366, 37.667] | -12.800 | +1.033 | 本轮新增 |
| COCO-Middle | 32 | 11 | 43.133% [37.666, 48.667] | -2.100 | +11.733 | 本轮新增 |
| COCO-Late | 32 | 12 | 34.333% [29.099, 39.667] | -10.900 | +2.933 | 本轮新增 |
| Random-11 | 32 | 11 | 34.767% [29.500, 40.067] | -10.467 | +3.367 | 本轮新增 |
| Both None | 0 | 0 | 21.833% [17.300, 26.567] | -23.400 | -9.567 | 审计历史复用 |

Middle/Early/Random均11层；Late为12层，多一层预算，不能把两者总体分差解释为同预算或单位层贡献。只迁移COCO的固定层集合，不复用COCO的Caption输出或CIDEr/SPICE分数；本轮始终使用TextVQA软评分和32 Token上限。

## 五项预先固定的配对比较

| 比较 | ΔAccuracy pp [普通95%CI] | 提升/损害/相同题数 | raw p | Holm p（5项） |
|---|---|---|---:|---:|
| COCO-Middle − Native Full | -2.100 [-4.400, +0.033] | 5/10/285 | 0.074296 | 0.074296 |
| COCO-Middle − Pminus1 Full / Decode None | +11.733 [+8.200, +15.533] | 39/1/260 | 0.000050 | 0.000250 |
| COCO-Middle − COCO-Early | +10.700 [+7.333, +14.333] | 36/1/263 | 0.000050 | 0.000250 |
| COCO-Middle − Random-11 | +8.367 [+5.033, +11.900] | 30/3/267 | 0.000050 | 0.000250 |
| COCO-Middle − COCO-Late | +8.800 [+5.667, +12.167] | 30/2/268 | 0.000050 | 0.000250 |

固定seed=20261105，20,000次Image SHA整簇配对Bootstrap，全部300张图片互不重复。两侧p来自整簇配对sign-flip并用加一修正；五项主要比较统一Holm校正。CI是普通点位95%CI，不是同时区间。跨0不证明无损、等价或非劣效；这里没有开展正式非劣效试验。提升/损害按逐题软分变化计数，不等于二值答对/答错；CSV另保留满分1与低于1的转换。

## OCR冻结分组

| 分组 | n | A Full | F Decode None | Early | Middle | Late | Random-11 | B Both None |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCR-Direct | 230 | 40.783% | 24.261% | 24.522% | 38.043% | 26.565% | 27.522% | 15.391% |
| OCR-Reasoning | 12 | 77.500% | 69.167% | 77.500% | 77.500% | 77.500% | 77.500% | 69.167% |
| Non-OCR Visual | 18 | 38.333% | 38.333% | 38.333% | 38.333% | 38.333% | 38.333% | 36.667% |
| Ambiguous | 40 | 64.250% | 58.000% | 61.750% | 64.250% | 64.250% | 62.000% | 38.000% |
| OCR-Combined | 242 | 42.603% | 26.488% | 27.149% | 40.000% | 29.091% | 30.000% | 18.058% |

OCR-Direct的Middle Accuracy为38.043%；Middle−Decode None为+13.783pp，95%CI [+9.364, +18.405]，Middle−Full为-2.739pp，95%CI [-5.746, +0.043]。这是预先冻结标签下的描述性分组结果，不能把未经校正的小组CI当独立确认性检验。各条件分组Accuracy区间及变化见ocr_group_accuracy.csv；五组全部主要分差均保存在ocr_group_paired_comparisons.csv，未根据表现挑组。

标签逐字节复用Commit `51761dba1b75f0081a7be2df728392483d609c69`：Direct230、Reasoning12、Non-OCR18、Ambiguous40。Combined=242与Direct/Reasoning重叠，不能当独立样本。它们来自上轮输入屏蔽AI复核，并非独立人类真值；本轮没有重新分类。小组样本少、分类误差和与历史数据重复使用均限制外推。

## 层集合与计算图

所有新条件整个Prefill保持原生，包括原始P−1在全部32层直接访问视觉Key。只对真实增量Decode Query→原始576个Image Key做Softmax前−inf Hard Mask：

- Middle：L0、L11–L20（11层）。
- Early：L0–L10（11层）。
- Late：L0、L21–L31（12层）。
- Random-11：[0, 4, 5, 14, 20, 21, 22, 26, 28, 30, 31]；seed=20261105，Python random.Random(seed).sample(range(1,32),10)，另固定L0。

其余Attention连接、Image/R Query、MLP、Residual、RoPE、QKV投影和完整真实KV Cache均保留；每条件独立Cache与前缀。11/32=34.375%仅是允许Decode读图的层比例，不是总计算、显存或延迟比例；其他层仍执行密集Attention与完整KV计算。

## 正确性审计、异常与负面案例

10项CPU测试再次通过，模型所有文件全量SHA与历史冻结版本严格一致。物理GPU2仅加载一份LLaVA，固定UUID `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`，FP16/Eager/Transformers4.49.0，Greedy最多32 Token，logits_to_keep=1。三题Smoke沿用历史冻结题号39419、34726、39145，先全部通过再启动正式矩阵。

真实7B Smoke确认：Native/No-op全部返回有效Logits严格torch.equal；A与Native的完整序列及全部有效Logits严格相等；四个新条件的首位置Logits严格torch.equal Native；全部32层完整Prefill的Attention输出、Attention概率矩阵及层Hidden SHA逐位对应。A/F/B的Token与首Logits SHA复现历史结果。禁止连接的实际Attention概率严格0，非目标Mask保持原值，允许层观测到实际视觉访问；Cache独立、父Cache不变，绝对位置、32层MLP、输入/视觉特征/权重版本、正常及异常Hook清理均通过。

全部1200份检查点再次核对Native A原始首Logits向量SHA、原图SHA、输入哈希、协议/源码SHA、每层实际Mask审计及官方分数。完整首Logits数组不重复归档，仅存精确SHA；Smoke直接在内存比较原始向量。没有放宽Logits容差。

| 条件 | EOS | 截断 | 空答案 | 重复3gram标记 | 平均生成Token（含EOS） | 与Full答案变化 |
|---|---:|---:|---:|---:|---:|---:|
| Native Full | 300 | 0 | 0 | 3 | 4.213 | 0 |
| Pminus1 Full / Decode None | 298 | 2 | 0 | 5 | 3.923 | 135 |
| COCO-Early | 300 | 0 | 0 | 4 | 3.743 | 130 |
| COCO-Middle | 300 | 0 | 0 | 4 | 4.083 | 69 |
| COCO-Late | 298 | 2 | 0 | 7 | 3.920 | 119 |
| Random-11 | 300 | 0 | 0 | 5 | 3.930 | 119 |
| Both None | 297 | 3 | 0 | 5 | 3.940 | 194 |

正常EOS即停止；未EOS而达到32 Token的输出按原协议保留并标记截断，不删题、不延长、不重生成。重复3gram是冻结检测器的标记，不等于事实错误。所有异常仍参加官方评分。

negative_cases.csv完整保留所有新增条件相对A/F及五项主要比较的负差，共137条、涉及47道不同题；同题可跨配对重复。不能把Middle的净提升掩盖的损害题删除。

## 对六个研究问题的回答

1. **Middle得分**：43.133%。
2. **相对Decode None恢复**：+11.733pp，95%CI [+8.200, +15.533]，恢复Full−None净缺口的84.82%。比例是本样本描述性量，分母和逐题正负效应都可能影响它，不是机制解释率。
3. **与Native Full差距**：-2.100pp，95%CI [-4.400, +0.033]。不能把CI跨0解释为已证明无损。
4. **Middle是否优于Early与Random**：高于COCO-Early的点估计10.700pp；Holm校正后支持差异。 对Random：高于Random-11的点估计8.367pp；Holm校正后支持差异。 对Late：高于COCO-Late的点估计8.800pp；Holm校正后支持差异。 Late多一层，保留预算差。
5. **OCR-Direct恢复**：Middle 38.043%；相对None +13.783pp，95%CI [+9.364, +18.405]；相对Full -2.739pp，95%CI [-5.746, +0.043]。仅为本批固定标签下的探索性描述，不证明OCR机制的唯一性或跨题分布泛化。
6. **物理KV算法是否值得实现**：可以把这个固定11层规则列为后续独立工程验证的候选，但本批没有证明非劣效、无损或跨分布质量保持。不能从COCO的成功直接推出TextVQA可用。即使质量接近，也仅支持后续单独验证物理实现；当前功能性Hard Mask没有删除视觉KV，没有测延迟/FLOPs/显存节省。若仍有明确质量缺口，则不足以把当前规则当作质量保持的TextVQA加速方案。

## 图表、运行与复现

![实测Accuracy对比](accuracy_comparison.png)

GPU阶段总耗时14.39分钟（含模型加载与Smoke），正式矩阵阶段12.76分钟；1200份新增检查点中12份来自通过全部门禁的真实Smoke，未重复生成。900条历史基线不计入新增数。资源释放以gpu_release.json中进程退出后的实际快照为准。

预检因GPU占用停止的历史保存在Commit `20d7bde6e005baf1f9437624c9507d4427e54de6`；本次用户授权空闲卡继续后，执行源码冻结到Commit `35621cb2a455c8278dde720f204825c5fce2282a`。原protocol.json和layer_sets.json保持冻结内容，execution_protocol.json记录恢复执行及硬件绑定。precheck的旧状态不是本次最终状态，以completion.json、validation.json及generation_completion.json交叉核验。

CPU复算：在vlm_research目录用原Python环境执行`python -m textvqa_coco_middle_transfer.analysis_final`及`python -m textvqa_coco_middle_transfer.report_final`；运行独立validate_final后校验。不要对已完成目录重跑GPU入口。所有历史文件不覆盖，main不修改，完成提交推送后停止；不追加新策略或物理实现。

首个Tokenizer Token相同不等于首个语义答案词相同，首Token可能仅为空白；per_question.csv另存首个非空白内容子词的位置和ID，分开报告变化。

本轮Random-11只有一个预先固定的随机集合；Middle优于该集合，不代表优于所有随机层组合，也不证明中层是唯一必要的视觉读取位置。

## CPU汇总执行留痕

GPU Smoke及正式生成没有验证失败或OOM。第一次CPU汇总因旧B检查点缺少condition字段而报KeyError；已按冻结历史索引及原检查点SHA绑定B，并额外确认旧writers与query_layout.R完全一致。只修正汇总读取历史Schema的逻辑，没有修改GPU执行源码、层集合、统计规则或任何检查点，没有追加GPU生成。修正后2100条官方重评分及全部主要统计独立复算通过。
