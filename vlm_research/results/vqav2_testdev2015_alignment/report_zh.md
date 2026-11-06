# VQAv2 test-dev2015 数据对齐审计

状态：**COMPLETE**。本轮仅下载、CPU 整理与验证；没有运行模型、加载权重、启动 GPU 推理或调用官方评分服务。

| 项目 | 结果 |
| --- | --- |
| 版本 / Split | VQAv2 v2.0 / test-dev2015 |
| 实际推理问题数 | 107,394 |
| 唯一图片数 | 36,807 |
| 唯一图片 SHA256 数 | 36,807 |
| 缺图 / 重复 Question ID | 0 / 0 |
| 图片全量解码、SHA256、官方 ZIP 大小与 CRC32 | 全部通过 |
| 提供 Gold | 否；gold_available=false；不存在 answers.json |
| 官方评分 | official_server_scoring_required=true；本轮未提交 |
| 官方提交结构 | 离线结构验证通过；实际官方服务接收未测试 |
| 旧 val2014 | 214,354 题安全保留；inputs 与 answers 哈希前后一致 |
| 旧数据混入 | Question ID 和 Image ID 交集均为 0 |
| 新目录多余题目 / 图片 | 0 / 0 |
| 实际生成模型预测 | 0 |

## 服务器路径

- 正式数据：`/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015`
- 推理输入：`/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015/inputs.jsonl`
- 图片：`/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015/images/test2015`
- 图片 SHA 清单：`/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015/image_manifest.json`
- 官方来源原件：`/data1/likai/ATE-vqav2-testdev2015-sources-20261107`（/data1，不在正式推理目录）
- 新正式 Registry：`/data/lk/ATE-benchmarks-20261105/official_benchmark_registry.json`；仓库同源文件为 `vlm_research/benchmarks/official_registry.json`。
- 旧诊断源：`/data/lk/ATE-benchmarks-20261105/datasets/VQAv2`；不会删除旧图像、Checkpoints、历史结果或改写历史冻结协议。

## 协议与官方来源

LLaVA 固定 Commit `c121f0432da27facab705978f83c4ada465e46fd`，保留评测 Shell、提交转换脚本、Evaluation 文档与答案规范化源码及 SHA256。
Google Drive 官方 eval.zip 链接连接超时，因此使用 **VQA 官方 v2 Test 问题包**，没有使用第三方问题镜像。
逐条验证 Test-dev 的 Question ID、Image ID、原始 question 与官方完整 Test2015 一致。
这证明官方 Split 与题目语义对齐，不宣称与未取到的 LLaVA eval.zip 字节完全一致。

问题源：https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
SHA256：`982e2e687a86514b78ea83af356d151976c5e3fb4168a29ca543610574082ad7`。

图片源：https://s3.amazonaws.com/images.cocodataset.org/zips/test2015.zip，固定 ETag `"caf5a3a275bf40dad9a3c8231caa4c76-1582"`。
未下载完整 13,270,587,935 字节 ZIP；用支持 Range 的只读 ZIP 索引确定所需成员，再下载官方单图 URL，逐图核对 ZIP 记录的长度与 CRC32，并保存 SHA256。
保留原始 JPEG 字节，不重新压缩、缩放或裁剪；实际仅保存 36,807 张所需图片。
S3 单图未提供版本 Commit；用官方 ZIP 固定 ETag、成员 CRC32/长度与本地 SHA256 冻结内容，不能把 ETag 误称为整个 ZIP 的 SHA256。

## 隐藏答案与队列

Schema 新增 `answer_type=hidden`，无需答案文件，且拒绝伪造 answer/answers/gold。
ATE 适配器：`load_prepared_benchmark('vqav2_testdev')`；正式入口已切换到 Test-dev。
新队列的 VQAv2 入口仅有 `vqav2_testdev`，旧 `vqav2_val2014_diagnostic` 不在正式队列；其他六项未改变。
独立分支包含新的 Registry、准备器与分析器适配；不会自动升级或重启其他 Worktree 中的旧调度器。
历史队列与检查点原样保留，不能把它们当成新协议的完成结果。

## 提交包装验证

固定版 LLaVA 转换器实际遍历完整 Test2015 的 **447,793** 个 ID，对未推理 ID 填空答案。
本任务推理 Manifest 仍严格 **107,394** 条。导出工具支持该 LLaVA 全量包装及 Test-dev-only 格式。
两种格式都只做离线导出；评测服务当前接受何种包装仍应以提交时的真实要求为准，本轮未调用服务确认。
正式导出强制覆盖全部 Test-dev 问题，拒绝重复/未知 ID 与空原始预测，复用与固定版 LLaVA AST 一致的 EvalAI 规范化器。
CPU 结构测试使用 **2 条显式 synthetic 假数据**，包装为 447,793 条，其中 447,791 条为空占位。
假数据不构成模型预测，不计为完成推理，不可上传；大体积测试包装校验 SHA 后已删除，保留小型 fixture 与审计证据。

```bash
python -m vlm_research.benchmarks.vqav2_submission \
  --dataset /data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015 \
  --predictions /path/to/real_complete_testdev_predictions.jsonl \
  --output /path/to/new_submission.json --mode llava-full-test2015
```

以上仅为未来离线导出用法，本轮没有真实预测或提交。

## 磁盘与安全清理

- 新数据逻辑大小：5,985,421,338 字节（5.5744 GiB）。
- 新数据文件实际分配：6,060,851,200 字节（5.6446 GiB，不含目录块）。
- 含目录块实际占用（对应 du）：6,062,911,488 字节（5.6465 GiB）。
- JPEG 逻辑字节：5,875,375,679。
- /data 下载前可用：16,270,233,600 字节（15.1528 GiB）。
- /data 验收后可用：10,205,446,144 字节（9.5046 GiB）。
- df 实际可用空间减少：5.6483 GiB；共享盘其他进程写入也可能影响此值。
- 本轮图片精确峰值预估后，从自设 10 GiB 预检门槛调整为 **8 GiB** 下载门槛；保留逐块下载保护。
- 原 GPU 实验运行器的 **20 GiB** 门槛未降低；当前磁盘不足以通过它，后续正式 GPU 运行需另行准备空间。
- 清理只涉及本任务创建的精确重复 JSONL 和 synthetic 导出/sidecar，共 110,456,283 字节，完整路径与 SHA 见 cleanup_manifest.json。
- 官方问题原始 ZIP 保留供全量提交 ID 导出使用，不是正式推理队列；没有递归清空共享目录，没有删除旧 val2014、模型、数据集、Git 对象、实验或 Checkpoint。
- 新代码位于独立分支。所有原始图像保留服务器，不加入 Git。

## 复现（CPU-only）

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python \
  vlm_research/data_preparation/vqav2_testdev2015/prepare.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m \
  vlm_research.data_preparation.vqav2_testdev2015.finalize
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  vlm_research.benchmarks.test_vqav2_testdev -v
```

准备器对已有正式文件采用内容一致检查与独占创建，不覆盖冲突；下载支持 Range 续传与有限重试。
本次独立验收运行 12 项 CPU 单元测试全部通过。包含既有评分逻辑的组合回归共 21 项也全部通过，使用同 Commit 旧工作树的只读历史依赖和官方匹配器夹具，未修改它们。
准备器已再次执行成功，完整复用现有图片并保持正式 Manifest 的 SHA256 不变，验证了幂等复跑。
全量独立验收再次读取所有图像并校验官方问题映射，没有静默丢弃异常样本。
服务器其他用户私有目录不可见，未宣称扫描了全服务器所有历史文件。GPU 仅允许读取状态，不启动任何计算任务。
