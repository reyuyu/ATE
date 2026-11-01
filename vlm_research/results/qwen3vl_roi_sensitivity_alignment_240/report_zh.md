# Qwen3-VL 视觉 ROI Attention 与功能敏感性逐层对齐（完整 240 题）

## 完成与有效性

- 同一批六类 GQA 各 40 题，共 240 题；36 层、P−1/真实 Question 两类 Query，得到 17,280 行配对记录。未换题、未重新筛选。
- 固定 Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/eager、官方 Chat Template、Transformers 4.57.6；模型文件逐份 SHA256 同历史一致。
- 本轮仅原生 Full Prefill，无 Mask、无新生成、无消融重跑，DeepStack 全原生。240 次观测前向，加 24 次无 Hook 的 Smoke 对照；不保存完整 Attention 矩阵。
- 9 项采集前 CPU 测试、2 项统计测试通过（含重复图片整簇重采样检查）；240 题真实 Processor 输入和像素重建检查通过。每任务 4 题，共 24 题 GPU Smoke 通过，带只读 Hook 与无 Hook logits 精确一致。
- 240/240 原生首 Token 全词表 BF16 logits SHA256 与同机历史 Full 精确一致；原始图片 SHA256、输入绑定、8,640 条历史 Image OFF 敏感性严格配对。三卡 DeepStack 实际加法、正常/异常 Hook 清理、权重不变均通过。
- Attention 行和最大误差 0.00324464；沿用原 BF16 阈值 0.00390825，未放宽容差；无 NaN/Inf/OOM。

## ROI 与统计口径

- 官方 Cross-modal CSV 120 题；按用户授权，以官方 GQA val scene graph 同图、同对象 ID 补齐另外 120 题。终端语义程序依赖用于定位目标，不按词名猜测。120 个 CSV 题的目标对象集合与 scene graph 坐标全部精确交叉验证一致。
- 共 329 个目标框，3 个框越界 1 像素，均显式裁剪并保留原框、图像/对象 ID 和来源；不存在的 LogicalObj 备选对象不补框。详见 `bbox_provenance.csv`。
- Qwen 的动态网格不是固定 576 Token：16×16 raw patches 经 2×2 spatial merge；验证真实 Processor 打包顺序和官方视觉 rotary 坐标顺序。ROI 为框与 merged 单元正面积相交的并集，边界单元整个计入；细小框的离散覆盖会大于连续像素面积。
- Question 指真实问题子词 Token；先对所有真实 Question Query 与各 Head 的原生 Attention 做算术平均，再计算视觉内部比值。P−1 单独统计；不将模板当作问题。
- ROI Hit Ratio = ROI Mass / Image Mass；ROI Enrichment = Hit Ratio / ROI Token 面积比例。随机基线为每题 128 组等 Token 数均匀无放回随机 patch 集合（允许与真实 ROI 重叠），跨层/Query 共用随机集合；不是空间连续或形状匹配的区域。均匀期望 Enrichment 恒为 1。
- 所有敏感性均来自历史 **P−1 Image OFF**；Question ROI 与之配对不表示测量了 Question 路径的干预必要性。主指标为 Mean Absolute LogP Change，Signed Loss 为辅；不是 Accuracy 或 Gold Answer Loss。
- 图片 SHA256 整簇配对 Bootstrap 2,000 次，保留题目加权估计。普通 95% 区间仅作探索性描述，不以跨层点、Head 或 Query 作独立图像，不报告独立层显著性。

## 核心对齐结果

| Query | 层均值 Enrichment–绝对变化 Spearman [95%CI] | 逐题跨层 Spearman 均值 [95%CI] | L26 同层跨题 Spearman [95%CI] |
|---|---|---|---|
| P-1 | 0.477 [0.187, 0.633] | 0.065 [0.037, 0.094] | 0.153 [0.025, 0.278] |
| Question | 0.224 [0.009, 0.487] | 0.033 [0.009, 0.059] | 0.010 [-0.114, 0.139] |

层均值相关描述“平均曲线是否一起起伏”；逐题相关描述“同一道题的两条曲线是否对齐”；L26 同层相关描述“跨图片 ROI 关注强度是否对应较大 LogP 改变”。这三者不可互相替代，且跨层结构、任务差异与原预测置信度都可能影响关联。

| 任务 | 绝对敏感性峰层 | P−1 Enrichment 峰层 | Question Enrichment 峰层 | P−1 层均值 Spearman | Question 层均值 Spearman |
|---|---:|---:|---:|---:|---:|
| Overall | L26 | L23 | L21 | 0.477 | 0.224 |
| ChooseAttr | L23 | L23 | L21 | 0.379 | 0.356 |
| ChooseCat | L33 | L31 | L21 | 0.375 | 0.178 |
| ChooseRel | L26 | L26 | L21 | 0.029 | -0.291 |
| CompareAttr | L26 | L26 | L20 | 0.492 | 0.311 |
| LogicalObj | L34 | L23 | L21 | -0.143 | 0.022 |
| QueryAttr | L23 | L23 | L21 | 0.465 | 0.316 |

## L26 专项

| Query | Image Mass [95%CI] | ROI Mass [95%CI] | Hit Ratio [95%CI] | Enrichment [95%CI] |
|---|---|---|---|---|
| P-1 | 0.114 [0.108, 0.121] | 0.044 [0.040, 0.049] | 0.374 [0.341, 0.407] | 2.911 [2.505, 3.380] |
| Question | 0.138 [0.134, 0.142] | 0.040 [0.036, 0.044] | 0.292 [0.262, 0.324] | 1.856 [1.672, 2.046] |

L26 历史 Mean Absolute LogP Change = **0.01874348 nat**，Signed Loss = **0.01085126 nat**。
CompareAttr 对 L26 总绝对变化之和贡献为 71.3%；因此不能仅凭总体 L26 峰推广到六种任务。

## 逐题峰值与随机对照

- P-1：逐题 ROI 峰减敏感峰平均 3.23 层 [95%CI 1.82, 4.73]，中位数 1.0；峰层完全相同 27/240，差不超过 2 层 64/240。
- P-1：总体均值曲线的 ROI 峰 L23、敏感峰 L26，差 -3 层 [图片簇 Bootstrap 95%CI -3, -3]。总体曲线峰差与逐题峰差是不同估计量。
- P-1：逐题 Enrichment–绝对变化 Spearman 中位数 0.060；9 条常量曲线相关无定义，保留并明确标记，不填 0；峰值平局预先规定取最早层，常量 ROI 曲线仍列入全部 240 题峰差统计（5 条），应谨慎解释。
- P-1：总体各层随机 patch Enrichment 均值范围 0.998–1.007；真实与随机对照见原始逐题表及曲线。Signed Loss 的层均值 Spearman = 0.387 [0.072, 0.573]。
- Question：逐题 ROI 峰减敏感峰平均 -0.90 层 [95%CI -2.35, 0.59]，中位数 -3.0；峰层完全相同 6/240，差不超过 2 层 33/240。
- Question：总体均值曲线的 ROI 峰 L21、敏感峰 L26，差 -5 层 [图片簇 Bootstrap 95%CI -5, -5]。总体曲线峰差与逐题峰差是不同估计量。
- Question：逐题 Enrichment–绝对变化 Spearman 中位数 0.040；9 条常量曲线相关无定义，保留并明确标记，不填 0；峰值平局预先规定取最早层，常量 ROI 曲线仍列入全部 240 题峰差统计（5 条），应谨慎解释。
- Question：总体各层随机 patch Enrichment 均值范围 0.999–1.003；真实与随机对照见原始逐题表及曲线。Signed Loss 的层均值 Spearman = -0.036 [-0.241, 0.237]。

## 中文研究结论

- P−1 的总体层均值 ROI Enrichment 与绝对敏感性有正向描述性关联（ρ=0.477），但逐题跨层关联很弱（平均 ρ=0.065）。总体曲线的关联不能直接推广为单题功能敏感性的可靠指示器。
- L26 的 P−1 Enrichment 高于均匀面积基线，但同层跨题相关仅 ρ=0.153；Question 的同层关联接近零。没有证据将 L26 的较大 Image OFF 改变直接归因于 ROI 读取。
- 总体 P−1 ROI 富集峰在 L23、Question 富集峰在 L21，不与总体功能敏感峰 L26 重合。ChooseRel/CompareAttr 的 P−1 富集峰与敏感峰均为 L26，但 ChooseRel 全层曲线相关接近零；单个峰重叠不足以说明整条曲线或逐题机制一致。
- 六任务并不一致，LogicalObj 的 P−1 层均值相关为负；总体 L26 敏感峰的主要贡献来自 CompareAttr。应保留分任务和逐题结果，不使用总体均值替代全部任务结论。

## 结论边界

ROI 关注、直接 Image 路径敏感性与预测表现是不同概念。图中的曲线关联及峰值重叠仅为描述性证据；跨层统计未控制所有混杂，不证明某 ROI 路径导致首 Token 改变，更不能推导整层可跳过或答案 Accuracy 不变。
L26 必须同时结合六任务曲线、逐题相关、同层相关和峰差观察，不把总体绝对变化峰自动称为 ROI 功能峰。Image 来源中还包含 ROI 外视觉信息，间接视觉信息也可经其他 Token/残差流传播。

## 文件与运行

- 主图：`overall_roi_alignment`；六任务：`six_task_roi_alignment`；逐题相关/峰差：`sample_correlations_and_peaks`；层均值与 L26 散点：`layer_and_l26_alignment`，均提供 PNG/PDF。
- `per_question.csv`、`layer_summary.csv`、`task_layer_summary.csv`、`layer_curve_correlations.csv`、`per_question_correlations.csv`、`per_question_correlation_summary.csv`、`peak_layers.csv`、`peak_summary.csv`、`l26_correlations.csv`、`l26_layer_summary.csv`。
- `shard_*/rows/*.npz` 只保留 Head/Question 聚合后的 Image Attention 行，不保留完整矩阵；随机种子、输入、框与模型协议均绑定可重算。
- 采集冻结代码 `9f9cc5bc134a84e72f30b5767b82f02cf366d29e`。GPU0/2/3 每卡 80 题；正式采集最长 13.89 秒，Smoke 最长 3.90 秒，均不含模型加载、CPU 数据准备、Bootstrap 与绘图。
- 本轮统计阶段 5.14 秒。GPU0/2/3 已退出并释放模型显存，GPU1 未使用。初次标注不足的 partial 审计保留在提交 `812f3e495`，本次在用户授权后补齐，未将此前 partial 当作完成结果。
