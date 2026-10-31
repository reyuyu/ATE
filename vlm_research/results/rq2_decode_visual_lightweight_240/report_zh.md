# RQ2.4：Decode视觉Attention轻量机制检查

**状态：完成。固定240题、六类各40题、240个Image SHA256簇；只新增240次C生成。**

## 主要结果

C−B完整答案Accuracy变化+0.000pp，按Image SHA256整簇配对Bootstrap200,000次的探索性95%CI [+0.000, +0.000]pp；B正确→C错误0题，B错误→C正确0题。本轮不作确认性非劣效或等价宣称。

**区间解释：** 所有观测到的C−B逐题得分差均为0，因此经验配对Bootstrap退化为[0,0]（CSV已标记degenerate_bootstrap=True）。这不是对未观测样本真实损害边界的精确估计，不能据此证明总体无损。B/C相对Full均纠正4题、损害0题，来自已存在的P−1干预，不是本轮新增Decode限制带来的收益。

| 分组（历史Full非EOS长度） | n | Full | P−1-only B | P−1+Decode C |
|---|---:|---:|---:|---:|
| All | 240 | 85.417% | 87.083% | 87.083% |
| One-token | 206 | 86.408% | 88.350% | 88.350% |
| Multi-token | 34 | 79.412% | 79.412% | 79.412% |

## 内容、后续Token与EOS

B/C首Token在240题上严格一致，首Token全词表Logits逐题精确一致。完整Token序列改变0题，去首尾空白后的解码文本改变0题，官方归一化答案改变0题。
第二个及之后非EOS Token序列改变0题；首次分歧涉及EOS 0题；生成长度改变0题；EOS到达标记改变0题。Token序列变化但解码文本不变0题。首次分歧涉及EOS可能同时伴随新增/减少内容，不能统称为仅EOS变化。

| Full长度分组 | n | 解码文本改变 | 完整Token序列改变 | B对→C错 | B错→C对 |
|---|---:|---:|---:|---:|---:|
| All | 240 | 0 | 0 | 0 | 0 |
| One-token | 206 | 0 | 0 | 0 | 0 |
| Multi-token | 34 | 0 | 0 | 0 | 0 |
| Zero-token | 0 | 0 | 0 | 0 | 0 |

### 全部真实差异案例

没有B/C完整Token序列差异；不存在可列出的真实差异案例。不能据此推广到长答案。

本轮实际审计290个Decode Query、9,280个逐层Attention概率行：30个阻断层的Image概率逐步严格为0，L14/L19的Image Mass为正。观测到的是Greedy Token序列相同，不意味着B/C后续Decode Logits、Hidden States或其他模型计算完全相同。

## 验证、协议及解释边界

样本来自上一轮2,656题图像隔离集合，seed20261101，按任务固定顺序及输入ID/Image SHA256哈希选40个未使用图像簇。选择完全不读取模型得分或答案长度；长度分组仅在选样后按历史Full输出冻结。generation_length_summary.csv保存每任务分布。两组均保留全部样本，不为扩大Decode效果重选多Token题。
Full与B精确复用acc137dc历史结果。模型revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0，Greedy/max32/真实KV Cache。只有当前真实Query的Image Key在30层Softmax前设−inf，L14/L19保留；MLP、BOS、Prompt、自身、位置和其他Prefill Query正常；没有物理Token/KV删除或实际加速。
CPU回归与GPU1真实Smoke通过。No-op所有Prefill及Decode Logits、原生生成和历史Full精确一致；B/C全部Prefill Logits精确相同。其他Prefill Query Attention/Hidden与Full精确相同。每题每一步均直接读取实际Softmax概率，验证30层Image概率严格0、保留两层Image Mass为正、动态绝对Query/576个Image Key/逐层KV长度及非Image Mask不变；不是仅统计Hook调用。原ROW_ATOL=0.002不变。Hook/输入/权重版本、有限输出与EOS均校验，失败不会静默排除。
若变化很少，只能说明当前短答案与实际多Token覆盖下额外Decode限制影响较小；若文本或EOS改变，则后续视觉读取参与了答案延续或结束决定。总体准确率相同不代表逐Token无损。所有结论局限本模型、本固定样本及P−1已受限的前提；不等于所有Prefill Query或长答案都不需要视觉Attention。
图：左为实测Accuracy；右为相对B的去首尾空白解码文本变化率，分组由历史Full非EOS长度确定。原始Token/EOS分歧另列CSV，不混为文本内容改变。图中不放长说明。

## 耗时与输出

正式单GPU任务1.99分钟（含加载），Smoke 0.53分钟；本轮只有240次正式新生成。逐题CSV、差异案例、组别/任务统计、协议、源码/模型/历史绑定、真实Decode概率审计与PNG/PDF均保存。完成后停止，不追加实验。

GPU1本轮模型进程均已退出；资源释放快照见gpu_release.json。
