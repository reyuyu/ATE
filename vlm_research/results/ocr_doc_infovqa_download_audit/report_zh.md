# OCRBench、DocVQA、InfoVQA 数据下载与完整性审计

本轮只准备数据并做 CPU 验证，没有加载模型权重、初始化 CUDA、运行推理或评分。未删除既有研究文件。

## 汇总

| Dataset | Expected QA | Actual QA | Unique Images | Disk Size (GiB) | Validation |
| --- | ---: | ---: | ---: | ---: | --- |
| OCRBench-v1 | 1000 | 1000 | 930 | 0.137806 | COMPLETE |
| DocVQA-val | 5349 | 5349 | 1285 | 1.565346 | COMPLETE |
| InfoVQA-val | 2801 | 2801 | 500 | 1.925827 | COMPLETE |
| Total | 9150 | 9150 | 2715 | 3.628979 | COMPLETE |

磁盘占用使用 st_blocks × 512 汇总，包含原始 Parquet、原始编码图片和整理标注；保留了所有本轮下载的 Parquet，没有靠删除源数据满足空间约束。Total 图片数为各数据集唯一图片数之和，不是跨数据集 SHA 去重数。

- 开始时 /data 可用：20186722304 bytes（18.800350 GiB）。
- 本报告生成时 /data 可用：16289849344 bytes（15.171104 GiB）。
- 监测到的最低可用空间：15.171371 GiB。
- 本轮数据实际新增占用：3896586240 bytes（3.628979 GiB）。审计工作目录位于 /data1，另行统计。
- 全文件系统可用空间减少：3.629246 GiB；该数值包含其他程序同时写入，不能全部归因于本任务。

## 数据路径与使用约束

`inputs.jsonl` 不包含答案、原始带答案 messages、qa、Gold、lecture 或 solution，供后续模型输入读取。
`answers.json` 以稳定 question_id 映射全部参考答案，仅供离线评分。
`records_with_answers.jsonl` 是满足统一字段要求的完整记录，不可整体传入模型。
`source_records_with_answers.jsonl` 保留所有原始字段和多媒体顺序；只将二进制图片替换为无损导出文件路径及 SHA。
`image_manifest.json` 保存原始字节 SHA、宽高、编码、文件大小和所有源行/媒体索引。

### OCRBench-v1

- 数据绝对路径：`/data/lk/ATE-benchmarks-20261105/datasets/OCRBench-v1`。
- 模型输入：`/data/lk/ATE-benchmarks-20261105/datasets/OCRBench-v1/inputs.jsonl`。
- 参考答案：`/data/lk/ATE-benchmarks-20261105/datasets/OCRBench-v1/answers.json`。
- 来源：`echo840/OCRBench`；Revision：`92a54bd1384387c178d5a07140a2d85e0a3d12e1`；Split：`test`。
- 状态：COMPLETE；QA 1000；原始源行 1000；图片引用 1000；唯一 SHA 930。

Question Type 分布：

```json
{
  "Regular Text Recognition": 50,
  "Irregular Text Recognition": 50,
  "Artistic Text Recognition": 50,
  "Handwriting Recognition": 50,
  "Digit String Recognition": 50,
  "Non-Semantic Text Recognition": 50,
  "Scene Text-centric VQA": 200,
  "Doc-oriented VQA": 200,
  "Key Information Extraction": 200,
  "Handwritten Mathematical Expression Recognition": 100
}
```

原始 Dataset 类别分布：

```json
{
  "IIIT5K": 15,
  "svt": 17,
  "IC13_857": 18,
  "IC15_1811": 8,
  "svtp": 9,
  "ct80": 9,
  "cocotext": 8,
  "ctw": 8,
  "totaltext": 8,
  "WordArt": 50,
  "IAM": 50,
  "ORAND": 50,
  "NonSemanticText": 50,
  "STVQA": 50,
  "textVQA": 50,
  "ESTVQA": 50,
  "ocrVQA": 50,
  "docVQA": 50,
  "ChartQA": 50,
  "ChartQA_Human": 50,
  "infographicVQA": 50,
  "SROIE": 67,
  "FUNSD": 66,
  "POIE": 67,
  "HME100k": 100
}
```

### DocVQA-val

- 数据绝对路径：`/data/lk/ATE-benchmarks-20261105/datasets/DocVQA-val`。
- 模型输入：`/data/lk/ATE-benchmarks-20261105/datasets/DocVQA-val/inputs.jsonl`。
- 参考答案：`/data/lk/ATE-benchmarks-20261105/datasets/DocVQA-val/answers.json`。
- 来源：`vikhyatk/docvqa-val`；Revision：`94ab3ed9c10895c28f77a17847e8202e858628e6`；Split：`validation`。
- 状态：COMPLETE；QA 5349；原始源行 1286；图片引用 5349；唯一 SHA 1285。

Question Type 分布：

```json
{
  "NOT_PROVIDED": 5349
}
```

### InfoVQA-val

- 数据绝对路径：`/data/lk/ATE-benchmarks-20261105/datasets/InfoVQA-val`。
- 模型输入：`/data/lk/ATE-benchmarks-20261105/datasets/InfoVQA-val/inputs.jsonl`。
- 参考答案：`/data/lk/ATE-benchmarks-20261105/datasets/InfoVQA-val/answers.json`。
- 来源：`mm-eval/InfographicVQA`；Revision：`c8e35ef221f0836de9c467e6da181b38d44b552a`；Split：`validation`。
- 状态：COMPLETE；QA 2801；原始源行 2801；图片引用 2801；唯一 SHA 500。

Question Type 分布：

```json
{
  "numeric": 1342,
  "open": 1458,
  "yes_no": 1
}
```

## 完整性与协议

- OCRBench 固定为 echo840/OCRBench 的 test，预期 1000 题与原始 10 个 Question Type，不使用 v2。
- DocVQA 按原始图片行的 qa 列表逐条展开；保留原始图片行与 qa 索引，不把 1286 张图片误当题数。源镜像没有问题 ID 时不伪造官方 ID。
- DocVQA 的 1286 个原始图片记录中有两个记录的图片编码字节完全相同，因此唯一 SHA 图片文件数为 1285；原始 1286 个记录和全部 5349 题均完整保留。DocVQA 源镜像未提供 Question Type，NOT_PROVIDED 仅表示字段缺失，不是新分类。
- InfoVQA 只选择 data/validation-*.parquet；保留原始 id、question_type、messages、全部 answer 和 media。多图片记录使用 images 数组，不截取第一张。单数 image_* 字段只是第一张的兼容索引。
- InfoVQA 的实际 messages 元素使用 role/user + question/hint/options/choices，而不是 content。首次解析因此安全失败、没有标记 COMPLETE；修正 Schema 分支并增加回归测试后，复用已通过 SHA 的原始 Parquet 重做整理。dataset_runs.jsonl 保留首次失败和后续结果，run_results.json 与 data_validation.json 表示最终状态。所有额外原始字段完整保存在 source_records_with_answers.jsonl，未混入模型 inputs.jsonl。
- question_id 由 Dataset、Split、原始行索引及 DocVQA 的 QA 索引构造，原始 ID 单独保留。相同问题、相同原始 ID 或图片均不合并题目。
- 图片仅按完整原始编码字节的 SHA256 去重；不改变编码，不缩放、不裁剪、不重新压缩。源图片路径/行索引的多个关联均保留。
- 全图片使用 PIL load 完整解码两次以上，并重新检查落盘 SHA、格式与宽高。逐题检查非空问题、全部参考答案、唯一 ID 和图片映射。
- 每个源 Parquet 都与固定 Revision 的 HF LFS SHA256 和长度一致；镜像仅为传输通道，不依赖仅同名文件判断等价。原始 HF 元数据及实际镜像地址随 source_manifest.json 保存。
- P50/P90/P95/P99/Max 以唯一原始编码图片为总体，采用线性分位数；宽高为原始编码尺寸。详细数据见 image_resolution_summary.csv。
- 不实现新的评分器；后续 ATE 代码需要薄适配器读取 inputs.jsonl，评分另读 answers.json。数据 COMPLETE 不表示正式评测已完成。

CPU 回归测试 8 项通过，包括 DocVQA 多答案展开、多媒体保留、实际 InfoVQA question 字段 Schema、非法空答案拒绝、已有文件冲突保护、无损导出及重跑一致性、磁盘安全线阻断。见 unit_tests.json。

## Qwen 显存风险审计

已完成全部原图分辨率及像素分布。可选官方 Processor 审计未执行：本机存在 Qwen3-VL 配置，但本轮可用 Python 环境的 transformers 4.56.2 没有 Qwen3-VL 实现。未额外安装依赖、下载模型或加载权重。不把静态分辨率公式冒充实际 image_grid_thw，不声称已计算 ViT Patch 或 1024/1536/2048 Token 阈值计数。后续审计应采用独立 2048 视觉 Token 配置，不修改图片。

## 重现

依赖：huggingface_hub 0.36.2、pyarrow 24.0.0、Pillow 11.3.0。脚本只导入这些 CPU 数据库及标准库，不导入 torch。

```bash
export CUDA_VISIBLE_DEVICES=""
PY=/home/likai/anaconda3/envs/AFAC/bin/python
AUDIT="$(pwd)/vlm_research/results/ocr_doc_infovqa_download_audit"
$PY vlm_research/data_preparation/ocr_doc_infovqa/prepare.py --audit "$AUDIT"
$PY vlm_research/data_preparation/ocr_doc_infovqa/finalize.py --audit "$AUDIT"
$PY -m unittest discover -s vlm_research/data_preparation/ocr_doc_infovqa -p "test_*.py" -v
```

重跑会固定使用已保存元数据中的 Revision，逐文件 SHA 核验后复用；冲突内容不覆盖。下载采用 HF Hub local_dir 断点续传，每文件最多两次任务级尝试、每次最长 600 秒；后台只监视/终止本任务自身的下载子进程。
file_hashes.json 保存数据文件、各图片、原始 Parquet、脚本和报告的 SHA256；该文件不自哈希。大型文件全部留在服务器，不提交图片或 Parquet。

## 停止范围

本任务到数据准备与 GitHub 审计交付为止，不启动 Native、C3-V2、Qwen 或其他 GPU 实验。
