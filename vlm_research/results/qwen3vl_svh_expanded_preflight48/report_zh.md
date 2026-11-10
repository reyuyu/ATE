# Qwen3-VL-SVH 扩大数值验收与六项48题预检

## 结论

**扩大数值验收 FAIL；当前不具备开启双卡六项完整评测的条件。**

计划的59个边界输入执行到第6个时发生数值失败，已严格停止。前5个通过，第6个失败，后53个未执行；六项48题预检均没有启动。没有降低精度、视觉预算、输入分辨率或数值门槛，没有修改 Physical 算法，也没有启动全量评测或推送 GitHub。

既有修复在旧小集上的 PASS 保留；本轮证明它还不能推广为任意真实输入的数值一致性保证。所有历史失败记录保持原样。

## 冻结设置与 CPU 准备

- 起点：`c95e688e3683c3bee68d983dddb3adb0bc40c5f8`。
- 正式执行快照：`9040161acc1102c7727f385430c7e5dc6ee77065`。
- 独立分支：`codex/qwen3vl-svh-expanded-preflight48`。
- 工作树：`/data/lk/ATE-worktrees/qwen3vl-svh-expanded-preflight48`。
- 方案 B：ATE 统一 HF，**非作者官方复现**；方案 A 继续 BLOCKED。
- 模型：Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- 原生 BF16/eager，torch 2.6.0+cu124、Transformers 4.57.6、batch1、last-row LM Head、Greedy、独立真实 Cache。
- 原定视觉面积预算：min_pixels=65,536、max_pixels=2,097,152，未改变。GQA cap32，其他五项 cap128。
- 原修复6项 CPU 测试和新增6项 CPU 测试全部通过；输入冻结阶段未初始化 CUDA。

已经逐源核对35,346条记录的 ID、原始 Gold、问题与选项，并核验8,400个不同路径的图片 SHA256。按官方 Processor 的 resize 规则统计全部输入的视觉 Token 数与总 Token 数；对选中的347个输入真正执行 Processor，确认尺寸、动态 Image Token 连续位置、展开长度及全部输入 Tensor SHA256。

347个绑定由59个数值边界输入和288个预检输入组成；未保存或复制模型与像素大数组。

| 任务 | 全部来源题数 | 本轮冻结输入的关键约束 |
| --- | ---: | --- |
| GQA | 12,578 | 官方 testdev_balanced；原问题+short-answer 后缀，无 Hint/OCR/候选辅助 |
| ScienceQA-IMG | 2,017 | 官方 test 图像题；原 Hint、题干、原顺序 Choices；不输入 lecture/solution/Gold |
| TextVQA | 5,000 | 复用冻结 Paper-OCR 文本、原问题与十个人工答案 |
| POPE | 9,000 | 原始 COCO random/popular/adversarial 各3,000；不沿用旧8,910题版本 |
| MME | 2,374 | 14任务原始问题及 yes/no 后缀；48题清单保留24个完整图片问题对 |
| MMBench | 4,377 | legacy EN dev 20230712、原 Hint/选项顺序；48行保留完整 Circular 组 |

**TextVQA 协议差异明确保留：**旧方案 B 写的是无参考 OCR，本轮按用户最新要求采用冻结 Paper-OCR，属于明确的输入变体。不是旧无 OCR 成绩，也不是作者官方评测复现。Native/SVH 预检输入绑定完全相同，但预检尚未运行。

MMBench 评分设计为固定 VLMEvalKit exact_matching/Circular 诊断，不使用付费 Judge，不声称 MMBench-V1.1。ScienceQA 使用冻结 lmms parser，TextVQA 使用冻结 MMF 十答案软评分，POPE/MME 使用已溯源作者解析规则；评分源码路径和 SHA 均在协议中。

## 边界样本与实际覆盖

六个任务分别按两个维度（实际视觉 Token 数、总输入 Token 数）选择 min、median、P95、P99、max，去重后59题。选择只依赖输入与固定哈希，不依赖模型答案。

| 任务 | 最大视觉 Token 数 | 最大总输入 Token 数 | 实际 GPU 数值覆盖 |
| --- | ---: | ---: | --- |
| GQA | 400 | 442 | 前6个边界输入，5 PASS / 1 FAIL；并未完成该任务全部边界 |
| ScienceQA-IMG | 460 | 561 | 未执行 |
| TextVQA | 2,028 | 2,121 | 未执行 |
| POPE | 400 | 426 | 未执行 |
| MME | 2,048 | 2,081 | 未执行 |
| MMBench | 256 | 573 | 未执行 |

全部边界 ID、选择原因、图像 SHA、Prompt 和 Tensor 绑定见 `length_boundaries.csv`、`numerical_cases.json`、`input_bindings.json`。

## 实际数值结果

冻结门禁保持为：Native/No-op 的全部有效 Logits 必须 `torch.equal`；Dense SVH 与 Physical SVH 每一步全词表 Logits 的 mean_abs<=0.02、max_abs<=0.5，同时要求 Top1、完整 Token 序列和步数一致。

**不要求 Native 与 SVH 算法输出相同。**下面比较的是相同 SVH 功能语义的 Dense/Physical。

| 问题 ID | 视觉 Token | 总输入 Token | 最坏步 mean_abs | 最坏步 max_abs | 门禁 |
| --- | ---: | ---: | ---: | ---: | --- |
| 20705744 | 96 | 132 | 0 | 0 | PASS |
| 201752756 | 260 | 288 | 0.00333454 | 0.125 | PASS |
| 20204868 | 340 | 369 | 0 | 0 | PASS |
| 202003652 | 400 | 425 | 0 | 0 | PASS |
| 202004255 | 400 | 426 | 0 | 0 | PASS |
| **20705853** | **96** | **122** | **0.05127071** | **0.3046875** | **FAIL** |

6个已执行输入的 Native/No-op 全部严格一致；自定义 Native 与 HF `generate` 的有效 Logits/完整 Token/EOS 也全部严格一致。保存的原始全词表 Logits 经独立 CPU/NumPy 复算，再次确认同样的5 PASS、1 FAIL。

### 失败输入

- ID：`20705853`，图片 `n264887`。
- 问题：`What makes up the desk?`。
- 原图 SHA256：`a5ccd2547d9b301ab5b5084804e3ce755721b3f152f4c87cedd097cc15c09238`。
- 对应本轮 GQA 总输入长度最小边界，共122个位置，其中96个视觉位置、26个文本位置。
- Native、No-op、Dense、Physical 最终均生成 `wood`，Token IDs 均为 `[6660, 151645]`，EOS 正常。

| 真实生成步骤 | mean_abs | max_abs | Top1/完整 Token | 结果 |
| --- | ---: | ---: | --- | --- |
| t=0：完整 Prefill 预测首答案 Token | 0.03515845 | 0.25 | 相同 | mean_abs 超0.02，FAIL |
| t=1：消费首 Token 后预测 EOS | 0.05127071 | 0.3046875 | 相同 | mean_abs 超0.02，FAIL |

答案相同并不能使数值门禁变为 PASS；上述误差是原始 Logits 误差，不是答案 Accuracy、NLL 或 LogP Loss。

### 最早分歧与证据边界

失败输入在 Prefill 的 L0–L27 匹配文本位置层输出均逐位相同。**首个已观测分歧是 L28 层输出**：max_abs=0.0625，mean_abs=0.00003396089；随后误差逐层传播，最终有效 Logits 超门槛。

本轮为控制存储，只保留匹配的层输出差值、位置/DeepStack/KV 元数据和全部有效 Logits，没有保留该层所有内部算子激活。因此只能定位到 L28，不能断言内部首先是 Q/K/V、Attention 归约还是 MLP/GEMM；对补零/GEMM对齐在新输入上的普适性仍需后续独立诊断。本轮遵守失败即停，未再运行 GPU 定位或修改修复。

匹配的 mRoPE 位置与 rotary cos/sin、DeepStack 特征/掩码、持久文本 KV 原始位置均通过检查；L22前物理移除视觉 Hidden、L1–L6答案 Attention 物理旁路、MLP/Residual 执行和禁止概率为零的审计通过。它们不抵消最终 Logits 数值失败。

详细证据位于 `numerical/GQA_20705853/`：四条件 JSON、四份完整 Logits NPZ、`logits_comparison.csv`、`layer_comparison.csv`、`trace_metadata.json` 和 `validation.json`。

## 长 Decode、补零开销与单卡最坏输入

**长 Decode 未验证。**本轮要求至少16个自然增量 Decode 步；停止前最长只有2步，不包含任何强制 EOS 后延长、固定长度伪生成或 Teacher Forcing。不能声称长 Decode 逐步一致。

已执行样本覆盖每一步真实 Decode（包括 EOS）的全词表比较。失败输入自身的首个 Decode 已超门槛。

失败输入的补零审计记录：

- 98个后退出线性投影只投影真实文本行，没有 Dense Projection 回退。
- Prefill 14层恢复临时零 Image Key/V 的额外 QK/AV 工作：572,522,496 FLOPs。
- 该样本一个增量 Decode 步的临时零列额外 QK/AV 工作：47,185,920 FLOPs。
- 最大重建 K/V 临时载荷：503,808 bytes，约0.4805 MiB；这是载荷量，不是独立测得的峰值显存增量。
- cuBLASLt 修复工作区为8 MiB。临时零列不持久化为视觉 KV。
- 失败输入 Live KV=3,317,760 bytes，Capacity KV=7,004,160 bytes，持久视觉 KV=0。

该失败输入 Peak Allocated 约16,778.3 MiB；已执行6例的最大 Physical Peak Allocated 为17,190.43 MiB。显存数据包括验收插桩及对照上下文，不是优化方法的独立性能 Benchmark；原始计时仅为执行记录，**不据此宣称任何加速或吞吐收益**。

**最坏输入是否能在3090上运行：尚未验证。**未发生 OOM，但流程在短 GQA 输入就因数值失败停止，未到2,048视觉 Token 或2,121总 Token 的边界，不能外推其显存可运行性。

## 六项48题预检

| 任务 | 冻结题数 | Native 已完成 | SVH 已完成 | Accuracy / 损害 / 纠正 / 空答案 / 截断 |
| --- | ---: | ---: | ---: | --- |
| GQA | 48 | 0 | 0 | 未测量 |
| ScienceQA-IMG | 48 | 0 | 0 | 未测量 |
| TextVQA Paper-OCR | 48 | 0 | 0 | 未测量 |
| POPE 9000来源 | 48 | 0 | 0 | 未测量 |
| MME | 48 | 0 | 0 | 未测量 |
| MMBench legacy | 48 | 0 | 0 | 未测量 |

这是被门禁阻止而不是零分，也不能将6个数值样本当作48题预检得分。`pretest48_status.csv` 对全部六项明确写为 NOT_RUN。

## 资源与交付

- 本轮只暴露 GPU3：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。
- GPU0/1/2 有其他任务，没有占用、抢占或发送信号。
- GPU 阶段从2026-11-10 23:27:30开始，23:28:00记录失败退出；约30秒，包含模型加载和验收。不是全量评测的耗时估计。
- 进程退出后独立核验 GPU3 无 Compute 进程，显存回到5 MiB；见 `gpu_release_audit.json`。进程内部 finally 记录仍可能包含模型引用占用，以进程退出后的独立记录为释放依据。
- 原 Physical 修复源码、历史数值门槛、旧失败目录、main和共享环境均未修改；没有自动推送。

本轮实际命令（历史执行记录，不用于覆盖现有结果）：

```bash
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_preflight48.tests -v
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_numerical_repair.tests -v
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_preflight48.freeze
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -u -m vlm_research.qwen3vl_svh_preflight48.run
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_preflight48_report
```

后续需要先独立定位并修复失败输入 L28 内的数值来源，保留现有门槛，再重新执行完整59例、自然长 Decode 和最坏输入。**当前不得开启48题预检或双卡全量评测。**本轮到此停止。
