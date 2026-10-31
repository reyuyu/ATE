# 首答案 Query 的层级 Attention Mass 与案例可视化

**状态：complete。** 原始六类 GQA 各500题，3000题全部完成，2409个不同图片 SHA256，96000条逐题逐层记录。只在空闲物理 GPU1 上运行 Native Full；无消融、训练、Decode 或答案 Teacher Forcing。

## 论文图与数据

- [Figure X：总体三曲线 PDF](figure_x_attention_mass.pdf) / [PNG](figure_x_attention_mass.png)
- [Figure Y：正文案例图 PDF](figure_y_visual_cases_paper.pdf) / [PNG](figure_y_visual_cases_paper.png)：7.2英寸宽，适合双栏通栏排版。
- [六任务补充图 PDF](supplement_six_tasks.pdf) / [PNG](supplement_six_tasks.png)
- [视觉内部条件概率案例图 PDF](supplement_cases_conditional.pdf) / [PNG](supplement_cases_conditional.png)
- [逐题逐层 CSV](per_question_layer.csv)、[均值/标准差/区间 CSV](layer_summary.csv)、[案例 Patch CSV](case_patch_attention.csv)
- [Figure 1 严格匹配的层级对照](aligned_layer_sensitivity.csv)、[描述性相关系数](mass_sensitivity_correlations.csv)

## 数据与测量

基线为 Research 1 提交 `e495adf4fd93337a3638981bbb66c2bbac81242d`。Figure 1 Manifest 的 Git blob 与 `221cdde56dde53908aaa953fe26672fd33334f84` 精确一致，没有重新抽样或根据结果筛题。

固定 LLaVA-1.5-7B revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`、FP16、eager、Transformers 4.49.0、eval/inference_mode 和原 Prompt。每题一次原生 Prefill，读取最后有效 Prompt Query P−1 的32个 Head、32层 Attention；不是 EOS 或已输入答案的 Query。

Image、BOS、真实 Question、Prompt Template 为互斥且完备的 Key 分组。通过真实文本偏移定位 Question；答题格式指令属于模板。正文 Prompt = Question + Prompt Template，不包含 Image/BOS。原生 FP16 概率提升至 FP64 后归约，不对行重新归一化。

均值按问题加权，标准差为逐题样本 SD（ddof=1）。95%CI 来自按整张图片 SHA256 聚类的10000次 Bootstrap，种子20261031；是逐点普通区间，不是跨层、来源或任务的同时置信带，不把 Head/层当成独立图片。部分区间比线宽更窄，原值全部保存在 CSV。

## 总体结果

以下为 Attention Mass 百分比；层段行是该段逐层总体均值的平均，不替代逐层统计。

| 层或层段 | Image | Prompt | BOS |
|---|---:|---:|---:|
| L0 | 65.47% | 34.30% | 0.23% |
| L1 | 36.48% | 62.70% | 0.82% |
| L2–L5 | 8.97% | 10.66% | 80.37% |
| L14–L16 | 10.29% | 34.62% | 55.10% |
| L25–L31 | 6.21% | 13.59% | 80.20% |
| L31 | 13.20% | 22.51% | 64.29% |

1. **早层不能概括为“两层都由 Image 主导”。** L0 的 Image 为65.47%，95%CI为65.40%–65.53%；L1 则由 Prompt 主导，62.70%，95%CI为62.62%–62.78%。L0–L1 合并平均 Image 为50.97%、Prompt 为48.50%，这会掩盖两层的差异。
2. **L2–L5 确实 BOS 主导。** 各层 BOS 分别为71.85%、83.84%、80.13%、85.66%，六任务均有这一整体形态。该结果描述分配，不证明 Attention 更新可以安全删除。
3. **中层发生再平衡，以 L14 最明显。** 从 L13 到 L14，Image 从8.34%升到14.24%，Prompt 从37.19%升到41.08%，BOS 从54.46%降到44.68%。随后 L15、L16 的 BOS 又分别升至56.99%、63.62%；不能把三层描述成同一状态。
4. **任务间存在差异。** L14 的 Image 在 QueryAttr 为28.97%，LogicalObj 为7.62%；总体曲线不能替代各任务补充图。深层重新 BOS 主导，但 L31 又出现 Image/Prompt 回升，并非深层完全相同。

## 与 Figure 1 敏感性的关系

按 task、question ID、image ID、layer 严格匹配全部96000条原始结果。原 Figure 1 的总体 mean |ΔLogP| 最大三层依次为 L14、L15、L16，与中层分配变化在深度上对应。下表比较32个层均值，**仅为描述性相关，不作独立样本显著性检验或因果推断**。

| Mass 与 mean \|ΔLogP\| | Pearson | Spearman |
|---|---:|---:|
| Image | −0.125 | +0.171 |
| Prompt | +0.422 | +0.558 |
| BOS | −0.184 | −0.492 |

Image 的关联较弱，线性与等级方向还不一致；BOS 的线性关联较弱，但负向等级关联更明显，不能称为“完全无关”。这些现象不足以从单一 Mass 推导 Attention 必要性。原预测 Token 的 LogP 敏感性也不是完整答案 Accuracy。

## 案例解释与选择边界

在 GPU 推理前，用固定种子/任务/QID 的 SHA256 排序，冻结每类2题、共12题候选池；随后仅依据原图清晰度、可见指代物和问题可展示性选定以下4题，没有查看 Attention、答案对错或消融结果。不是事后从热图挑选“最符合假说”的案例。

| 任务 | QID | 主题 | 复用的 Native Full 答案 |
|---|---|---|---|
| ChooseAttr | `13363937` | 蒸汽颜色 | White |
| ChooseCat | `03268289` | 长颈鹿类别 | Giraffes |
| CompareAttr | `06952284` | 衣服与头盔共同属性 | Color |
| QueryAttr | `04819723` | 窗帘方位 | Right |

正文展示 L0/L3/L15/L31 的逐 Head 平均原始视觉概率。576个 Token 还原24×24，再逆映射真实 CLIP resize/center crop；裁剪外保持原图，不当成零 Attention。每案例四层共享线性色标，图中 M 是原始 Image Mass。正文紧凑版仅改变排版、换行和标签，没有改变案例、概率或色标。

火车与窗帘案例直观展示早层较大的 Image 总量和后层较低总量下的局部热点；长颈鹿与滑板案例也出现背景热点。**这些图没有证明后层自动聚焦答案对象。** 未做 ROI 或中立 Prompt 对照，因此既不把热点认定为任务相关定位，也不认定为已验证的 Visual Sink。

Full 答案字符串复用协议/输入匹配的 Figure 3 原生生成检查点；本轮没有额外自由生成。补充条件概率图除以当前 Image Mass，仅用于观察视觉内部相对分布，不能混作原始 Mass。

## 验证与执行记录

- 7项采集 CPU 测试和3项序列化回归测试通过；2题真实 GPU Smoke 中开启采集前后有效 Full logits 精确一致，最大差异0，全部32层归约与原生输出直接计算一致。
- 3000题当前原生首 Token 与匹配的历史 Full 首 Token 全部一致。历史 BOS Mass 最大绝对差为 `4.47e-8`。
- 最大逐 Head 行和误差 `4.043e-4`；三来源和最大误差 `1.033e-4`，均低于既有 `2e-3` 边界，没有放宽容差。全部结果有限；无 OOM、NaN、Inf、输入或参数变更、残留 Hook。
- 正式3000题前向与逐题保存耗时742.64秒（12分22.64秒，不含模型加载和 Smoke）。结束后 GPU1 仅27 MiB、利用率0%；未终止其他任务。
- 冻结采集代码 Commit：`15134c941a3c27d63cfafac7644305202d64a330`。运行中未改源码；执行配置、测试、环境、原始检查点及产物哈希均保留。
- 首次汇总暴露12个案例的整数层号键被 JSON 字符串化后，排序哈希表示不同。恢复原整数键后12个原始结果哈希完全匹配；2988个无空间图检查点原本即匹配。保留首次错误日志及兼容回归记录，只修复后处理校验表示，没有重算/覆盖数据或更改容差。

**论文结论边界：** 可以表述为“首答案 Query 的 Attention 来源分配具有明显层级结构，L2–L5 BOS 主导，L14 附近再平衡，并存在任务差异”。不应表述为“高 BOS 导致冗余”“低 Image Mass 意味不需要视觉”或“这些层因此可以安全跳过”。看向哪里、更新是否必要、完整生成是否受损是不同问题。
