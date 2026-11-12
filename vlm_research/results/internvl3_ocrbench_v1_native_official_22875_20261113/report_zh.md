# InternVL3-8B OCRBench v1 官方协议 Native：BLOCKED

本轮仅运行 Native，完整 1,000 题评测未启动。4 题 Smoke 中 3 题成功，第 4 题的官方 25 图块输入在 RTX 3090 上发生原生 Eager Attention OOM，按停止条件终止。**正式 Native 得分：未取得；与论文 880 分的差距：不可计算；尚未达到完整官方协议验收。**

## 资源与固定版本

- GPU：NVIDIA GeForce RTX 3090 24GB，UUID `GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`，驱动 610.57.04。
- 模型：`OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9`。21 文件、15,901,094,763 bytes；4 权重分片下载流 SHA256 与其余 Git blob 哈希通过。官方 HF 429 后从镜像恢复同一 Revision 并校验相同官方哈希。
- Python 3.11.8、PyTorch 2.6.0+cu124、Transformers 4.49.0、torchvision 0.21.0+cu124；BF16/Eager、batch size 1、全部权重位于单 GPU。独立环境仅增加 `timm==1.0.15`，复用原冻结依赖，`pip check` 通过，未修改原 LLaVA 环境。
- 官方参考：VLMEvalKit `3c192143007e4eb3f5c1a36558a4f8c1451980cb`；审计与输入依据：ATE `93ff60f839644fabf81f202d699bbc0638484f42`。服务器原仓库 HEAD `e281754ef17b3accc09d6938b5b9f020048b2aec`，新入口隔离保存，其 SHA256 绑定于 `results/protocol.json`；发布提交是该审计提交的子提交。
- 模型、数据、环境、实际执行目录全部位于 `/dev/vda3` 本地 ext4。工作持续盘只保存轻量结果副本。没有执行 SVH-R、Late9 或其他评测。

## 协议逐项核验

| 项目 | 结果 | 证据 |
|---|---|---|
| 1,000 题/930 原图/Gold/顺序 | PASS | 原 annotation SHA `df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864`；930 图片 SHA256 全部匹配归档 |
| 官方 Prompt | PASS | 执行原 `build_prompt/reorganize_prompt` AST；原题后加官方 single word or phrase 后缀 |
| 动态切块 | PASS 配置及已运行输入 | 原 `dynamic_preprocess/load_image` AST；RGB、448、ImageNet、`max_num=24`；大于1块时加缩略图 |
| 原生 Chat / 视觉映射 | 3 完整通过；25块 Prefill 部分通过 | `internvl2_5`/ChatML、原中文 System、256视觉Token/块；四题的像素、Token IDs、mask 哈希均匹配CPU |
| 生成设置 | PASS 已完成三题 | Greedy、4096上限、原生 EOS `<|im_end|>` 151645；官方传入0.6/.95但不采样；无额外生成参数 |
| 官方评分器 | PASS 已完成三题 | OCRBench v1原循环、VLMEvalKit原函数及独立CPU规范化逻辑三方一致 |
| GPU 4题 Smoke | **FAIL / OOM** | 25块题0743在原生Eager softmax失败 |
| 完整1,000题结果 | **NOT_VALIDATED** | 未启动；不能给正式得分或与880分差距 |

官方模型 config 的 `max_dynamic_patch=12` 没有被误用为评测上限；本轮遵照 wrapper 明确用24，最多加缩略图形成25块。全体输入中有47题为25块。审计包的100文件完整性与原89文件SHA清单重新通过，原始研究证据未改动。

执行器直接编译固定源码的原函数 AST，避免引入整个 VLMEvalKit 的无关依赖。适配仅限 OCRBench 类型/模态查询、返回字节一致的本地原图路径以及透明输入/输出观测。没有重写动态切块、Chat、LLM生成或 Attention；原模型和 ATE 冻结核心均未修改。参考源码及 AST/文件哈希已保存。

## Smoke 与覆盖

| Question ID | 官方 ID | 图块数 | 视觉Token | 总输入Token | 原生输出 | 官方分数 | 峰值分配MiB |
|---|---:|---:|---:|---:|---|---:|---:|
| 0098 | 360 | 1 | 256 | 321 | bakery | 1 | 15285.84 |
| 0759 | 275 | 11 | 2816 | 2894 | 1.00 | 1 | 17771.81 |
| 0053 | 1660 | 16 | 4096 | 4161 | experience | 1 | 20314.36 |
| 0743 | 343 | 25 | 6400 | 6478 | 无；OOM | 空缺 | 未完成 |

三题分别与 Gold `BAKERY`、`1.00`、`EXPERIENCE` 按官方大小写规范及包含匹配得分1；独立复核一致，输出均正常EOS，无空输出或截断。**3题Smoke的得分不能解释为Benchmark成绩。** 四题按1→11→16→25块固定规则选择，选择于GPU运行前，不按答案或得分筛选。

预测覆盖为3/1000；另1题失败、996题未尝试，全量生成未启动。`results/coverage.csv` 保留1000题原始顺序及状态，失败/未运行的分数栏为空。未跳题续跑、未填零、未混用旧751分的单图块预测。

## OOM 原因与结论边界

`0743` 为 `SROIE/image/X51006912976.jpg`，4961×7016，官方网格4×6再加缩略图。其6400个 `<IMG_CONTEXT>` 与 ViT 输出 `[25,256,3584]` 完全对应，图像特征有限，输入哈希正确。错误发生在 Transformers 4.49.0 的 Qwen2 `eager_attention_forward`：

```text
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
CUDA out of memory. Tried to allocate 4.38 GiB ... 1.20 GiB is free.
process has 22.35 GiB memory in use; PyTorch allocated 21.80 GiB.
```

从实际输入长度计算，一个28头FP32注意力张量需 `28 × 6478² × 4 / 2³⁰ = 4.37724 GiB`，与申请量一致。这是本配置原生Eager长Prefill的容量阻塞；没有证据表明由缺图、Gold错配、剪枝算法或错误图像Token数量造成。该计算是内存形状核对，不是性能评测。成功16块题峰值分配20,314.36 MiB，峰值保留22,692 MiB；OOM时的实时数值另存失败记录，不冒充成功峰值。

没有调整切块上限、生成长度、精度、Attention Backend、量化或Offload，也未通过分块Vision/Attention重写来规避问题。模型仅加载一次，失败后停止，不进行额外GPU重试。

要继续同一 BF16/Eager 配置，下一步应迁移到有足够显存的GPU（例如48GB），首先重试同一0743，并保留本次失败证据；通过4题验收后再开始完整1,000题。现有证据不能保证任何特定显存容量必然通过。若希望在3090改用其他Attention实现，需要另行冻结并记录后端协议，不能称本轮已完成。

公开固定 VLMEvalKit 的输入与生成配置已核验；完整GPU执行验收失败。因此当前结论是 **BLOCKED**。即使今后得到完整得分，作者论文880分原始run的精确flags/权重快照来源仍有未公开部分，应与“公开固定源码协议对齐”分开标注。

## 保存与资源状态

- 完整逐题成功输出、Token IDs、输入像素/Token/mask/Prompt/SHA、全部1000输入清单与独立Gold清单、tile数、失败栈及CPU独立评分均在本目录。
- 本地执行：`/home/featurize/ate-experiments/internvl3-ocrbench-native-official-22875-20261113`。
- 模型：`/home/featurize/ate-local-models/InternVL3-8B`；数据：`/home/featurize/ate-local-data/bundles/OCRBench-v1`；环境：`/home/featurize/ate-local-envs/internvl3-torch260-tf449`。
- 模型进程4697已退出；复查无GPU compute进程，显存1MiB使用、24123MiB空闲、利用率0%。服务器未关闭。
- GitHub仅包含轻量证据和必要源码，无模型、图片、密钥或Token。独立CPU复核说明见 `README_REPRODUCE_zh.md`。
