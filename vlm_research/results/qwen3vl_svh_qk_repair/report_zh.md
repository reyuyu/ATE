# Qwen3-VL-SVH：两题QK修复与FAST-TRACK重验

## 结论

指定的TextVQA `36053`和POPE `adversarial:205`均已修复：独立新进程中，Dense功能参考与Physical的Prefill及自然生成全部Decode步骤Logits逐位一致，误差为0。此前7个回归样本同样全部通过。**这不等于六任务数值门禁通过。**

恢复原冻结24例后得到13 PASS、1数值FAIL、1原生OOM、9未运行。因此FAST-TRACK仍为BLOCKED；未运行六项各8题预检，也未启动正式质量生成。历史59例仍未完成，方案A仍BLOCKED，本轮仅使用方案B（ATE统一HF，非作者官方复现）。

## 冻结范围与过程

- 起点：`38ce71b5391bbffb5a1f0031f2cc1ab0c4a31b70`，新分支`codex/qwen3vl-svh-qk-numerical-repair`。
- Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16/eager，视觉预算、Processor、Prompt、输入绑定、mRoPE、DeepStack与SVH语义未变。
- 门槛保留：每步有效Logits mean absolute error ≤0.02、max absolute error ≤0.5、Top1和完整生成Token一致；Native/No-op必须`torch.equal`。不要求SVH答案等于Native答案。
- CPU共16项测试通过，含QK数学语义、非突变、正常/异常清理及历史AV/FAST-TRACK回归。
- 捕获与隔离诊断只使用物理GPU3；通过9例独立回归后，才使用重新确认空闲的GPU1/GPU3恢复原24例。
- 历史失败目录、历史实现及main均未修改；未推送GitHub。

## 最早差异与隔离证据

| 样本 | 原始长度/视觉/文本 | 最早算术差异 | 首次层输出差异 | 初始QK最大差 |
|---|---:|---|---|---:|
| TextVQA 36053 | 360 / 320 / 40 | Prefill L26 `QK_text` | L27 | 0.125 |
| POPE adversarial:205 | 152 / 128 / 24 | Prefill L22 `QK_text` | L22 | 0.125 |

从L0起比较真实中间算子，涵盖层输入、Norm、Q/K/V投影、mRoPE、QK、Mask、Softmax、AV、O投影、Residual和MLP。两题首次QK差异之前的匹配文本状态、Q/K数值一致；没有将“首次超过0.02”当作起点。TextVQA的L26算术差异与首次层输出分歧L27也明确区分，不能认定每个最早舍入差异都传递到最终输出。

相同真实文本Q/K的隔离重放：

- TextVQA：Q `[1,32,40,128]`，K转置 `[1,32,128,360]`；临时Query行扩为360。
- POPE：Q `[1,32,24,128]`，K转置 `[1,32,128,152]`；临时Query行扩为152。
- 紧凑BF16乘法严格复现旧Physical；仅把Query空位置填0，保留完全相同的真实Q/K，再取回文本行，即严格复现Dense的合法文本QK值。
- 也用原Dense完整Q乘以同一个Physical K作控制，合法文本结果一致。图像K列在Physical中仍为历史对齐用临时零列并被屏蔽，不是恢复视觉KV。
- FP32仅作隔离诊断，没有用于正式推理。证据支持QK计算形状/布局相关的BF16舍入差异，而非上游Hidden、mRoPE、DeepStack或位置错误；未采集硬件计数器，不指定未经证实的具体CUDA Kernel。

原始逐算子数据位于`diagnosis/*/operator_comparison.csv`，实际Q/K捕获及SHA指针位于`diagnosis/*/qk_capture.json`，隔离重放位于`isolated_QK/`。

## 最小修复及数值结果

新增`vlm_research/qwen3vl_svh_qk_repair/engine.py`：仅Physical Prefill L22–L35的QK乘法临时补零Query行，乘完立即抽取真实文本行，之后的缩放、Mask、Softmax不变。保留此前AV行对齐。投影/Norm/MLP仍仅处理真实紧凑行，没有恢复视觉Hidden、视觉Q投影或持久视觉KV，Decode算法未改。

| 样本 | 修复前最坏逐步Logits Mean Abs | 修复前Max Abs | 修复后Mean/Max | Native/No-op |
|---|---:|---:|---:|---|
| TextVQA 36053 | 0.0464033447 | 0.500000 | 0 / 0 | 精确一致 |
| POPE adversarial:205 | 0.0755626634 | 0.375000 | 0 / 0 | 精确一致 |

修复前两题逐步原始Logits均与旧FAST-TRACK归档逐位重现；修复后Native/Dense基线仍与旧记录一致。

新进程PID `2508477`不同于诊断PID `2504312`。9例独立验证共40.60秒：两个目标、GQA `20705744`、`201752756`、`20204868`、`202003652`、`202004255`、`20705853`、ScienceQA `10391`全部Mean/Max=0。真实自然Decode覆盖1–3步；两目标各1步后EOS，**没有把EOS后伪造步数称为长Decode验证**。

输入哈希、权重地址/版本、Hook清理、逐层位置、mRoPE、DeepStack、Cache逻辑位置、视觉Hidden退出及持久视觉KV=0检查通过。详细见`verification_completion.json`、`independent_verification/*/semantics.json`及`validation.json`。

## 额外计算开销

| 样本 | 新增QK矩阵FLOPs | 最大临时QK输出缓冲 | 视觉持久KV |
|---|---:|---:|---:|
| TextVQA 36053 | 13,212,057,600 | 8,294,400字节 | 0 |
| POPE adversarial:205 | 2,231,369,728 | 1,478,656字节 | 0 |

这是14层Prefill补零行引入的实际执行形状计数，不是硬件TFLOPS，也不是测速。另有临时Q缓冲、原AV补零及历史零列开销，均在每条Physical原始audit中保留；不宣称加速。

## 恢复24例后的阻塞

完整冻结样本、输入绑定、分片和预检名单直接按SHA复用，未重抽或缩图。结果在相邻目录`qwen3vl_svh_fasttrack_qk/`。

1. **ScienceQA `7908`（P99）数值FAIL。** Native/No-op精确一致；Dense/Physical第一个算术差异是Prefill **L1 `q_proj`**，匹配R行最大差0.00390625，早于L22视觉退出。首个层输出差异L1最大0.25。首Token Logits Mean Abs=0.1466039717、Max Abs=0.578125，已越界；后续完整序列分叉，不能把不同前缀下最大均差7.6427解释成固定前缀数值误差。该早层问题不属于本轮两题QK修复范围，未追加修复。
2. **MME `landmark/0adb088e24d88123.jpg:1`（P99）原生OOM。** 输入2062、视觉2028。物理GPU1上的Native视觉编码器Eager Attention在FP32 Softmax尝试申请3.92 GiB时失败，尚未进入语言Decoder或新增QK对齐。错误信息的“GPU 0”是`CUDA_VISIBLE_DEVICES`后的逻辑编号，不是占用物理GPU0。未降低分辨率、改精度或替换后端。
3. 其余9例未运行；未宣称最长输入或长Decode全部通过。未运行各任务8题预检、未生成正式质量分数。

**下一步前提：** 单独定位SQA7908早层投影形状/舍入差异；另提出保持冻结视觉规格和算法的原生视觉编码器显存方案并重新验收。现有证据不足以放行双卡六项全量。

## 资源、保存与复核

- 诊断GPU3：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。
- FAST-TRACK另一张GPU1：`GPU-b08489af-0a8a-eb48-55ba-8346733af417`。
- 结束检查两卡各5 MiB、0%利用率，无本任务计算进程；物理GPU0/2的既有进程未触碰。见`gpu_release_audit.json`。
- Worktree：`/data/lk/ATE-worktrees/qwen3vl-svh-qk-numerical-repair/`。
- Q/K私有捕获：`/data/lk/ATE-resources/experiments/qwen3vl_svh_qk_repair/`，每文件SHA已归档，无模型副本。
- 修复源码冻结Commit：`30535244f2b729a631287ae5460a88c715af9e5c`。
- 9例结果及24例启动冻结Commit：`7de03178f97872d30ea18eb4870bf5e580e1b381`。
- `repair_before_after.csv`、`regression_summary.csv`、相邻目录`numerical_case_status.csv`提供机器可读汇总。

复核汇总命令（只读原始结果、不启动GPU）：

```bash
cd /data/lk/ATE-worktrees/qwen3vl-svh-qk-numerical-repair
/data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_qk_report
```

历史诊断与验收入口有防覆盖及源码哈希校验；若需重跑，必须新建结果目录并绑定对应冻结源码，不能直接覆盖已有记录。
