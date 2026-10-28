# 15133706：Sink 捕捉与增强前后直观对比

问题：**Is the young girl to the left or to the right of the boy?** 正确答案 `right`。完全复用已完成实验缓存，仅CPU绘图，没有模型Forward、GPU补采或新条件。没有修改历史实验文件。

## 三张图

1. [Sink位置与反复热点](sink_capture.png)：真实336×336 Processor裁剪图、两个中性发现提示词与独立第三提示词、原生Full早/中/深层热图。绿色13个非ROI Sink候选；青色男孩ROI；品红女孩ROI；橙色ROI/Sink交集 #364。原生Full热图均为Answer-prediction P−1。
2. [原始 / 局部干预前 / 干预后横向对比](intervention_comparison.png)：三行依次为中层ROI×2、深层L24～31非ROI Sink×0.1、早层L2～4同一Sink×0.1。每行显示相同窗口全部层和全部32Head的平均分布。中列是**干预运行内当前层修改前**，不是独立Full；此前被修改的层已经可能影响它。
3. [注意力转移差分图](attention_change.png)：同一运行的局部After−Before，红色增加、蓝色降低，共用对称线性色标。该图不能与前两张图的对数色标混读。

所有原始/增强热图使用相同颜色范围、相同透明度及原始真实图像，不逐面板拉伸。先对每层每Head在576视觉Patch内归一化，再平均全部Head/窗口层；这是视觉内部相对分布，并非原始Attention Mass。比较图另标注原始Visual Mass。中性缓存只存Head均值，故其图使用保存的均值条件化，再平均提示词/两种Query/32层；不冒充逐Head统计。ROI份额包含橙色交集，非ROI Sink份额不包含交集，两者互斥。

## 捕捉是否准确？

这是冻结的“跨提示词/层持续高Attention位置”定义，没有人工Sink真值，不能计算检测准确率或称为全部无关噪声。冻结候选共14个，其中13个在ROI外，1个为 #364。独立中性提示词保留13/14个候选，集合Jaccard=0.8667。Full P−1的早层L2～4、深层L27～31平均图中，13个非ROI候选全部进入Top58；中层为12/13。这支持**候选确实捕捉到反复热点**，不证明热点没有有用视觉信息。

## 增强后是否把Sink丢掉了？

以下份额为同一次干预运行内局部Before→After，在对应窗口平均，并非对原生Full做固定分布编辑：

| 已有条件 | ROI视觉份额 | 非ROI Sink视觉份额 | 交集#364份额 | 最终Margin |
| --- | --- | --- | --- | --- |
| ROI×2，L14～23 | 16.37%→26.30% | 57.12%→51.06% | 2.38%→4.21% | −0.0234375 |
| 非ROI Sink×0.1，L24～31 | 12.04%→22.88% | 55.13%→14.79% | 1.85%→4.75% | −0.0390625 |
| 非ROI Sink×0.1，L2～4 | 7.72%→28.29% | 80.09%→30.85% | 5.31%→20.10% | +0.1015625 |

**ROI增强没有去掉Sink**，非ROI Sink仍占视觉注意力约一半。直接抑制会降低目标Sink权重，但×0.1不是置零，也没有删除图像像素或Token。早层抑制后#364变得很亮：它属于ROI并被排除在抑制集合外，占新的ROI份额约71.1%。这属于注意力重分布，不能描述成全部Sink已去除。

原生Full Margin=−0.03125，历史Physical K24=+0.0390625。深层目标Sink抑制没有复现Physical改善，早层条件转正；此处只是为解释图像选取已有三种条件，完整31条件及随机背景对照见[原实验报告](../report.md)，没有新增或反复调参。单个事后案例、FP16弱Margin和oracle ROI限制仍然适用，不能推断泛化性能或确定因果机制。

## 复现与核验

工程根目录首次运行（已完成目录拒绝覆盖）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python visualize_sink_capture_case.py
```

[map_statistics.csv](map_statistics.csv)保存所有图的真实份额与原始视觉Mass，[plotted_maps.npz](plotted_maps.npz)保存实际绘图向量，[validation.json](validation.json)保存来源SHA256、统一色标、形状/归一化/局部质量保持检查口径。绘图前后均独立验证既有163文件SHA256未变。新输出的文件哈希见[artifacts_sha256.json](artifacts_sha256.json)。
