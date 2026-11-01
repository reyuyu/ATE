# Qwen3-VL ROI Attention 与功能敏感性对齐：前置检查（partial）

## 状态与停止原因

本轮没有启动 GPU Smoke 或正式模型前向，没有产生 36 层 Attention、ROI 指标、相关系数或 L26 新结论。

用户指定的 Cross-modal Information Flow 官方 CSV 来自 `correct_question` 筛选子集，并不覆盖冻结的全部 240 题。根据“映射不可靠就停止”的要求，不能换题、猜测目标框或把缺失 ROI 当成零 Attention。已完成可复核的数据覆盖审计和部分 CPU Processor 几何检查，暂停正式实验。

## 同一批样本与历史绑定

- 原始 `qwen3vl_image_source_240/sample_manifest.json` 完整复用，六类各 40 题，没有重采样。
- 240/240 图片实际 SHA256、Question/Image ID、问题与答案同原始 GQA 标注绑定通过。
- 240 个历史 layout 的 input_binding 哈希通过；同批历史 8,640 行单层敏感性结果逐题、逐层、图片绑定通过。
- 模型仍为 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；没有加载或修改模型权重。
- 历史 BF16/eager 配置未变；没有新增生成、Mask、训练或层扫描。

## 官方 CSV 覆盖

来源仓库 `FightingFighting/cross-modal-information-flow-in-MLLM` 固定提交 `c9a1111dac4baa175f67716fab473789320187c0`。六份原始 CSV 均保存于 `sources/`，Git blob SHA1 与下载来源逐份精确核验；另保存 SHA256。Question ID 比较仅规范化数字前导零，不采用文本相似匹配。

| 任务 | 冻结题数 | CSV 有该题 | 严格有效 BBox | 缺该题 | 框异常 |
|---|---:|---:|---:|---:|---:|
| ChooseAttr | 40 | 10 | 10 | 30 | 0 |
| ChooseCat | 40 | 28 | 27 | 12 | 1 |
| ChooseRel | 40 | 9 | 9 | 31 | 0 |
| CompareAttr | 40 | 30 | 30 | 10 | 0 |
| LogicalObj | 40 | 38 | 38 | 2 | 0 |
| QueryAttr | 40 | 5 | 5 | 35 | 0 |
| 总计 | 240 | 120 | 119 | 120 | 1 |

异常为 ChooseCat `13708087`：图片 500×375，官方框 `(x=0,y=17,w=501,h=358)`，横向越界 1 像素。本轮不静默裁剪。

LogicalObj 的 31 条记录中，一个备选对象的 ID/框为 `-`，表示该对象不存在。只保留标注明确存在的对象框，不为不存在的对象补框，也不把该缺省误算成损坏标注。上述覆盖通过不等于目标语义已通过全部模型实验验证。

还尝试以原始 GQA 的明确对象 ID 查询六份 CSV 中同图其他题的对象框；这仍不能覆盖全部样本。该尝试仅作覆盖诊断，不把“提到过的对象”自动当成目标 ROI，不将其标记为完成的 ROI 绑定。原始 GQA parquet 包含语义对象 ID，但不含 BBox，不能凭对象 ID 还原缺失框。

## CPU Processor 几何检查

4 项 CPU 单元测试通过。另从直接具备有效框的样本中，按稳定输入哈希固定每类 4 题，共 24 题，仅执行 CPU 几何检查：

- 使用固定模型目录的真实 `Qwen2VLImageProcessorFast` 与官方 Chat Template；24 个实际输入 layout 均与历史精确相同。
- patch=16、spatial merge=2、temporal patch=2；使用实际 `image_grid_thw`，不沿用 LLaVA 的 24×24 网格。
- 将 Processor 输出的 raw patch 流按 merged-row/merged-col/intra-row/intra-col 重建，与官方实际 resize/normalize 的图像张量逐元素精确一致。
- 使用真实动态 merged 网格与 32×32 等效图像单元；原图 BBox 与单元正面积相交得到覆盖索引，同题多框取并集。
- 保存 24 张覆盖图及 `cpu_mapping_overview.png`，检查框、图像方向和空间覆盖对齐。绿色为官方框，粉色为相交单元，白线为 merged 网格；这些不是 Attention 热图。

这批 24 题只验证已有框样本的 CPU 输入与空间几何，不是 GPU Smoke，更不代表剩余 216 题已通过映射检查。没有验证本轮 Full logits、Attention 行和或 DeepStack 运行输出，也没有运行整体统计。

CPU 首次辅助重建调用因将 list 传给缓存参数报 `TypeError: unhashable type: list`；改为官方预处理使用的 tuple 参数后重新运行，24/24 精确检查通过。没有调整数值容差。测试过程记录在 `execution_audit.json`。

## 产物与恢复边界

- `bbox_coverage.csv`：全部 240 题逐题来源覆盖、绑定与错误原因。
- `sample_manifest.json`、`available_csv_boxes.json`、`source_audit.json`：冻结输入及可用官方框。
- `protocol.json`、`validation.json`：明确的 partial/annotation gate failed 状态。
- `cpu_processor_validation.json`、24 张覆盖图及总览：部分 CPU 映射核验。
- `artifact_hashes.json`：本轮文件完整性。

只有补齐可靠目标 BBox、明确边界框处理规则后，才能继续完整的每任务 4 题 GPU Smoke 及 240 题收集。建议申请允许使用 GQA 官方 scene graph 的同图同对象 ID 框补充缺失 CSV，保留逐框来源；未经用户同意，本轮未改用该来源或下载替代标注。

GPU0/2/3 未启动本轮进程，不存在需要释放的模型显存；GPU1 未使用，外部任务未干扰。没有得到足以回答“ROI 与 L26 敏感性是否相关”的新结果。
