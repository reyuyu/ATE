# ScienceQA 7908：早层BF16投影行数修复

## 结论

独立新进程验证PASS：7908完整128个生成步及此前13例均通过原门禁。Native/No-op逐步有效Logits严格相同；Dense功能参考与Physical全部逐步Logits误差为0，Token序列一致。7908的SVH达到原128 Token上限而Native自然EOS为6 Token，该截断如实保留，不作为质量成功。

## 首个真实分歧

原冻结输入N=513、Image=460。L0实际输出、L1实际输入、RMSNorm输入和输出完全相同；q_proj真实输入的前512行完全相同。首个真实分歧为L1 q_proj：Dense执行513行，Physical因答案Attention跳算仅执行512行。不是L0误差首次在L1被观测到。

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| L1 q_proj真实行最大绝对误差 | 0.00390625 | 0 |
| L1 q_proj真实行平均绝对误差 | 0.0000949150563 | 0 |
| Prefill全词表Logits平均绝对误差 | 0.146603972 | 0 |
| Prefill全词表Logits最大绝对误差 | 0.578125 | 0 |

相同实际输入矩阵在无模型的新进程中重放，普通BF16 compact GEMM精确复现原Physical差异；固定原始M对应的cuBLASLt描述符、仍只计算compact真实行，可精确复现Dense真实行。L1 K/V同输入实验亦成立。描述符记录是本实验实际选择配置，不冒充已观测到PyTorch原生CUDA Kernel名称。

## 最小修复

仅对Prefill L1-L6的物理Q/K/V/O投影选择原始M的Lt描述符，实际运算仍为N-1行，不补算P-1投影；共享原工作区与计划清理机制。MLP、DeepStack、mRoPE、Mask、KV和历史晚层QK/AV修复不变。没有FP32正式推理、恢复视觉Hidden/KV或放宽门槛。

原门槛：平均绝对误差<=0.02、最大绝对误差<=0.5、Top1及完整Token一致。全部原始失败、逐算子比较和输入捕获SHA保留。

## 证据与范围

- `diagnosis/actual_upstream.csv`：完整实际上游状态及q_proj前后。
- `isolation_completion.json`：相同输入隔离重放及Lt配置。
- `probe_completion.json`：修复后全算子首步检查，无首个分歧。
- `verify_completion.json`、`independent_verification/`：独立新进程14例完整生成、逐步Logits、mRoPE、DeepStack及KV检查。
- GPU1：`GPU-b08489af-0a8a-eb48-55ba-8346733af417`。本轮独立验证进程已退出。

以上使用原视觉Eager，只验证14个可容纳的输入，不代表MME最坏输入或24例门禁通过。原59边界验收仍未完成；正式生成尚未放行。
