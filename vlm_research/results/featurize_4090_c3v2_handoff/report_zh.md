# RTX 4090 原始 Physical C3-V2 迁移验收

**状态：BLOCKED**。本轮仅恢复数据、模型、环境并进行固定案例迁移核对；不是正式 Benchmark 成绩。

代码：`e281754ef17b3accc09d6938b5b9f020048b2aec`。模型：`llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
环境：Python 3.11.8；Torch 2.6.0+cu124 / CUDA 12.4；Transformers 4.49.0；FP16、Eager、greedy、batch=1。
GPU：NVIDIA GeForce RTX 4090, GPU-643a4d73-25eb-6b19-1acc-ccf94c744cec, 610.57.04, 24564 MiB, 23715 MiB。模型加载 1 次；峰值 allocated 13.735 GiB，reserved 13.842 GiB。

数据、模型、venv、wheel 缓存均在实例本地 ext4。持续盘只保留代码、报告及小型结果。

- 实际数据：`/home/featurize/ate-local-data-4090/ate-recovery`。
- 实际模型：`/home/featurize/ate-local-models/llava-1.5-7b-hf`。
- Python：`/home/featurize/.venvs/ate4090-frozen/bin/python`。
- 原路径 `/home/featurize/data/ate-recovery` 通过本地盘软链接保留，冻结数据与 Prompt 不变。

数据来自已经验证的 Windows `D:\ATE-dataset-backups-3090`，没有重新下载数据集。六份 portable ZIP 和 OCRBench 迁移包均核对整体 SHA256；七类数据均恢复。六类数据的 14 个 normalized 视图均包含输入和 Gold，合计 28 个文件与 3090 验收 SHA256 完全一致，所有图片映射均存在。OCRBench 保留官方 1,000 题、930 张图及原始清单。模型 14 个文件经过官方固定 revision 内容校验，三份权重 shard 与 3090 SHA256 完全相同。

| 冻结输入视图 | 恢复题数 |
|---|---:|
| MME-paper | 2374 |
| ScienceQA-IMG-paper | 2017 |
| POPE-paper | 8910 |
| TextVQA-OCR | 5000 |
| MMBench-paper | 4377 |
| GQA-testdev-balanced | 12578 |
| OCRBench-v1 | 1000 |

物理 Smoke：**BLOCKED**（原计划五类各 1 个样本，短 decode；复用原冻结物理检查）。检查实际图像输入、L20 前视觉 Hidden 移除、L2–L5 答案 Attention 物理跳算、P−1/Decode 图像访问禁令、紧凑 KV 原位置、RoPE、输出有限性与同语义 dense 短数值参考。未放宽误差门槛。

独立固定案例对比：0/15 完成。物理门禁失败时，后续对比不启动。

**停止原因**：第三个物理样本 POPE:10000001 在 physical 与同语义 dense C3 的数值参考比较中，hidden_relative_rmse=0.010154514573514462，超过冻结阈值 0.01。最大 logits 绝对差 0.015625、argmax 一致、hidden_cosine_distance=0.000030875205993652344；但门禁仍为失败。MME、ScienceQA 两例物理检查通过，TextVQA/MMBench 物理检查及独立 15 案例循环均未执行。没有 OOM 或已报告的 NaN。
这是 4090 上物理实现与同语义 dense 短参考的数值门禁失败，不能据此断言 3090/4090 的答案已经不同。需要定位数值差异的来源；本轮不改变层号、误差阈值、FP16 或 Backend，也不追加 GPU 试跑。

已把两个成功物理 Smoke 的原输出离线对照旧 3090 预测；两者均提前遇到 EOS，短 cap 没有截断答案，不产生额外 GPU 生成：

| 已保存的物理样本 | 3090 | 4090 | token / 文本 / 评分一致 |
|---|---|---|---|
| MME:OCR/0001.jpg:0 | No | No | 是 |
| ScienceQA-IMG:10001 | A | A | 是 |

这两例的 frozen 输入/Gold 文件和图片已校验；独立跨硬件输入 tensor 对比循环尚未开始。不能用这两例替代六类/15 例完整验收。当前结论是数据与模型恢复 PASS、物理迁移门禁 BLOCKED，尚不具备直接开展正式 C3 质量评测的验收结论。

独立对比进度与失败记录保存在 comparison_results.json；完整计划输入绑定和历史来源保存在 comparison_cases.json。输出差异列表（仅适用于已经执行的独立对比）：[]。

首步 logits 逐位哈希仅作跨硬件诊断，未执行的独立对比不记作相同；本轮没有将不同 GPU 的 FP16 中间量逐位相同作为迁移标准。

冻结计划保留 MME 同图双题；本次图片 OCR/0001.jpg 不涉及已知 SHA 差异 landmark/03d5e3bfc958be38.jpg，未替换数据。计划的 TextVQA 为 Paper-OCR 十答案软评分，POPE 保留三策略及 8,910 题论文视图，MMBench 为完整 4 行 Circular 组，GQA 为原历史 C3-V2 入口与答案规范化。未执行的对比与组评分不记为通过。
OCRBench 已恢复全部数据；由于没有此前原始 C3-V2 对照，本轮没有额外生成 OCRBench，也没有启动任何正式全量或回看变体实验。

模型进程已退出；退出后 GPU：`NVIDIA GeForce RTX 4090, GPU-643a4d73-25eb-6b19-1acc-ccf94c744cec, 610.57.04, 1 MiB, 24109 MiB, 0 %`。没有后台模型任务。
Windows 独立归档目录：`D:\ATE-dataset-backups-3090\ATE-results\4090-c3v2-handoff`。GitHub 仅同步本轮轻量证据，具体提交与逐文件 SHA256 验证见 github_sync_receipt.json。

完整原始异常与 traceback 已保存在 comparison_results.json 和 gpu_smoke.log。未修改冻结代码或误差门槛。
