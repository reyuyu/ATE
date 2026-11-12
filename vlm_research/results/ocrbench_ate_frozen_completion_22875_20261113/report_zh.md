# OCRBench v1：原冻结 ATE 协议缺失条件补齐

状态：**PARTIAL**。存在历史检索遗漏造成的重复计算，已纠正、保留并排除重复输出；详见下文。

| 模型 | 方法 | 得分 /1000 | 覆盖 | 接受的新生成 | 历史复用 | 空输出 | 达生成上限 |
|---|---|---:|---:|---:|---:|---:|---:|
| InternVL3-8B | Native | 751 | 1000/1000 | 0 | 1000 | 0 | 0 |
| InternVL3-8B | Physical-D | 32 | 1000/1000 | 0 | 1000 | 0 | 3 |
| InternVL3-8B | Late9 | 563 | 1000/1000 | 997 | 3 | 0 | 0 |
| LLaVA-1.5-7B | SVH-R L11-L24 | 281 | 1000/1000 | 0 | 1000 | 36 | 27 |
| LLaVA-1.5-7B | Physical-C3 | 75 | 1000/1000 | 990 | 10 | 69 | 18 |
| LLaVA-1.5-7B | Native | 301 | 1000/1000 | 0 | 1000 | 0 | 4 |
| Qwen3-VL-8B | Native | 未完成，不报正式分数 | 3/1000 | 3 | 0 | 0 | 0 |
| Qwen3-VL-8B | Physical-Original | 未完成，不报正式分数 | 3/1000 | 3 | 0 | 0 | 0 |

## 输入及协议

1000 道题按原始索引 0000–0999；原始官方 ID 存在重复，未以它覆盖索引 ID。930 张唯一原图均按冻结 SHA256 核验，Question、Gold、原顺序一致。Gold 只供离线评分，不加入模型输入。所有条件独立新建 KV；缺失条件的 Smoke 成功输出计入正式预测；被误重跑的 Native Smoke 随其他重复输出一并排除。

| 模型 | 冻结 Revision | dtype / Transformers | 图像及 Prompt | 生成 |
|---|---|---|---|---|
| LLaVA-1.5-7B | b234b804b114d9e37bb655e11cbbb5f5e971b7a9 | FP16 / 4.49.0 | RGB，CLIP mean 补正方形，336；原 Vicuna-v1、单词/短语指令；576 视觉 Token | Eager，greedy，batch=1，最多128 Token |
| Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | BF16 / 4.57.6 | 冻结默认 RGB Processor，无像素/Token 缩减；原 Qwen Chat Template 和 body 后缀 | Eager，greedy，batch=1，最多128 Token |
| InternVL3-8B | 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 | BF16 / 4.49.0 | **原 ATE 单图块 448×448 /256视觉 Token**，internvl2_5 template，原简短答案指令 | Eager，greedy，batch=1，最多128 Token |

**没有混入官方动态 max_num=24、缩略图新协议结果；本表不是该新协议的官方复现成绩，也不直接证明与论文分数的差距原因。** 所有环境使用 PyTorch 2.6.0+cu124，模型、图片、环境及生成目录位于本地 ext4。

## 复用与物理检查

- InternVL Native751、Physical-D32 的各1000条原始输出全部复用并重评分；Late9 复用0053、0963、0988三条，仅新生成997条。四题 Smoke 检查视觉映射、L28前 Hidden 退出、L19–27选择性视觉 KV、答案跳算、原位置和有限性。
- InternVL Late9 **保留原 `EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER`**。本次结构 Smoke 不消除历史严格数值门禁失败，也不构成新的 Native/Dense 等价性证明。
- LLaVA SVH-R 原1000条直接复用。补查 Windows FastV 归档发现 Native 已完整1000条，输入哈希/Gold/协议逐项完全一致，正式表全部复用该历史301分。Physical-C3 复用另一归档已完成的10条，本次仅接受其余990条。四题 Smoke 的 L20前视觉 Hidden 退出、L2–L5答案 Attention 跳算、原文本位置、禁止答案视觉访问和最终无原始视觉 KV 均独立检查。
- LLaVA 附加 CPU 检查器首版把 compact `query_positions` 当作原 RoPE位置，引发断言；已改查 `visual_audit.original_query_positions`，四题通过。冻结执行器、参数、已有预测均未更改，未重跑样本。
- Qwen 的 ATE/official 两个旧 Manifest 视图对 OCRBench 的1000个请求哈希完全一致，因此不会为相同请求重复生成。Native 与 Physical-Original 使用同一输入，原模型默认预处理保留不变；最长 Smoke 为0738，16307个总Token /16264个图像Token。

## 阻塞及失败

- **qwen BLOCKED**：OutOfMemoryError；请求 `{'question_id': '0738', 'method': 'Native'}`。CUDA out of memory. Tried to allocate 126.13 GiB. GPU 0 has a total capacity of 23.56 GiB of which 4.76 GiB is free. Including non-PyTorch memory, this process has 18.79 GiB memory in use. Of the allocated memory 18.23 GiB is allocated by PyTorch, and 220.29 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

出现 OOM 后停止该模型，不跳题、不补零、不降分辨率、不换 Backend；已成功输出保留，缺失请求仍为缺失。部分正确数只保存在统计文件中，不当作 /1000 正式成绩。Qwen Physical-Original 的历史数值门禁/探索性标记同样保留；本次没有运行 Dense-F，也未宣称新的数值等价性通过。

## 独立评分和证据

每条保存原始文本、输出 Token IDs、Question ID、图像 SHA、输入绑定和来源。官方 OCRBench v1 example.py 的固定评分循环与单独实现的标准化/子串匹配逐题交叉核对，分母固定1000；HME100k 保留大小写并去空格，其他集合转小写。所有已保存得分一致，未删空输出或截断。

`predictions/*.jsonl.gz` 可直接 CPU 复算；`input_manifest.jsonl`、`images_930_SHA256.csv` 和 `inputs/` 记录输入/Gold/图像及处理后张量绑定；`protocols/` 保存冻结来源、源文件 SHA、环境、Smoke 和失败。`execution/` 仅本次隔离入口，冻结核心按指定 Commit 读取，未覆盖原仓库。

执行 `python verify_cpu.py` 可核验文件 SHA、1000题映射、930图像哈希清单和全部已有预测的两套评分。完整逐题 GPU 审计及缓存布局另外压缩归档，模型与原图不会上传 GitHub。

GPU 已释放：`NVIDIA GeForce RTX 3090, GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, 610.57.04, 1 MiB, 24123 MiB`；未关闭服务器。

Windows及持续盘备份的实际传输/SHA回执见 `backup_receipt.json`（提交前完成并核验）。

## 执行失误与纠正

前期历史检索遗漏了 Windows FastV Native 全量归档，误启动并完成 **926条重复 Native 生成**，违反了用户的严禁重复要求。发现后立即暂停并终止首个 LLaVA 进程，恢复/核验历史1000条，后续仅补 Physical-C3 的65条缺失请求，因此 LLaVA 额外加载一次。

重复原始文件全部保留在完整归档 `results/llava/excluded_duplicate_generations/Native/`，未删除、未覆盖历史预测，也未以其替换历史301分或加入正式分母。机器可读的历史检索纠正、排除ID及初始Smoke中已失效的“未重复”库存判断见 `protocols/llava/historical_inventory_correction.json`。本轮不能宣称满足零重复执行要求；已完成条件的输入/评分正确性与该执行失误分开报告。
