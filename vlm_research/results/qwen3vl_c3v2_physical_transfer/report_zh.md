# Qwen3-VL C3-V2物理迁移：数值门槛失败，正式实验停止

本轮状态为 **STOPPED_NUMERICAL_VALIDATION_FAILED**，不是COMPLETE。已实际实现并运行双GPU短输入Smoke；两卡在同一题上均未通过运行前冻结的F/C有效Logits平均绝对误差门槛，因此没有启动Pilot、GQA/VQAv2主质量评测或端到端速度评测。没有放宽容差、改变BF16/Eager、调整层集合或以Dense Mask冒充物理成功。

## 主要结果与停止依据

冻结执行源码：`dae5207d016acea8225f236ce3b81285d3d8bd90`。最终交付Commit另由Git历史及交付消息提供。

首道Smoke为GQA `06905471`，图片`4712`，问题 **Are there any cars or trains?**，Gold为`no`。原始Prefill长122，包含96个真实Image Token及26个非图像Token。两卡均保存了F/P/B/C及两个功能诊断的完整生成和执行审计。

| F与C首步比较 | 实测 | 运行前冻结门槛 | 判断 |
|---|---:|---:|---|
| 有效Logits最大绝对差 | 0.4453125 | ≤0.5 | 通过 |
| 有效Logits平均绝对差 | 0.0834809989 | ≤0.02 | **失败** |
| 首步Top-1 | 一致 | 完全一致 | 本步通过 |
| 完整生成Token IDs | 两者均为`[2152,151645]` | 完全一致 | 本题通过 |
| 答案及终止 | 两者均为`no`，正常EOS | 正常、无异常 | 本题通过 |

GPU2与GPU3的失败数值相同。第一项数值比较失败后，程序没有继续检查B/F与P/W22的Logits，也没有开始剩余两道Smoke。两卡重复的是**同一道题**，不构成两个独立质量样本。不能把这道答对题当成100%正式Accuracy，也不能用Token一致替代未通过的Logits门槛。

Native/No-op逐位Logits一致及手动Native与官方HF生成逐位Logits/Token一致的断言，均位于该题F/P/B/C执行之前，两个Worker已走过这些断言。**原始Native/No-op有效向量没有在失败前单独落盘**，因此这部分证据是冻结代码控制流及失败位置，而不是可离线重算的向量档案。没有宣称两卡各3题Smoke通过。

原始失败见`STOP.json`、`shard*/smoke_failure.json`、`logs/`；机器可读汇总见`gpu_smoke_validation.json`。

## 数据、环境与历史审计

使用已有单份Qwen3-VL-8B-Instruct模型，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、Eager、Transformers 4.57.6、Greedy、batch 1、max_new_tokens 32、logits_to_keep=1。权重及配置逐文件SHA见`model_manifest.json`；官方模型源码SHA为`dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`。没有训练、量化、修改模型权重、重新下载模型或复制大模型。

读取了指定的五份历史报告和相应执行实现，保留原报告及固定Commit CSV副本作为背景。历史L1–L6角色实验、Keep-L0-L21均为功能性结果，不是本轮物理联合路径的质量或速度数据；不同历史原生Full也未混用为本轮N。

在本轮预测前，按历史Manifest的保守SHA并集、输入元数据及固定种子20261105选样，保留整张图片的全部官方问题组。为此仅在独立CPU工具目录准备固定pyarrow 21.0.0读取已有GQA parquet，未修改模型环境依赖或下载数据集。选择规则不读取模型答案、得分或预测。

| 数据集 | 冻结题数 | 图片数 | 历史SHA重叠 | 补充ID审计重叠 | 正式生成数 |
|---|---:|---:|---:|---:|---:|
| GQA | 1012 | 93 | 0 | 0 | 0 |
| VQAv2 | 503 | 117 | 0 | 0 | 0 |

题数略超1000/500是保留最后一张图片完整问题组所致。GQA使用本地官方平衡验证标注，VQAv2使用本地官方验证问题及十答案标注、已缓存COCO图片。历史排除并集覆盖7587个图片SHA。ID核对在停止后补做，没有据此改变冻结样本；ID数据集归属根据Manifest/图片路径保守判别，因此不能证明未登记于这些Manifest的其他实验也没有使用过这些图片。

Prompt采用已有Qwen官方Processor/聊天模板与历史短答案指令；`input_bindings.json`冻结原始Prompt、Token IDs、动态图像位置、问题位置、网格和输入张量SHA。两卡按完整Image SHA分片，没有图片跨Worker。旧数据不填入主样本；也没有复用任何历史预测作为本轮质量结果。

GQA沿用已有提交规范`text.rstrip('.').lower()`后与官方答案比较。VQAv2沿用ATE历史评分：官方标点、数字、冠词归一化同时作用于预测和十个答案，再按leave-one-out十人共识计算软分。没有使用首Token指标替代完整答案Accuracy。本轮正式得分尚未产生。

## 实际物理实现及执行证据

| 条件 | Prefill图像Hidden | R直接读图 | P−1/Decode | 持久KV |
|---|---|---|---|---|
| N | 原生36层 | 原生 | 原生 | 原生完整KV |
| F | 全36层保留 | L0–L21开放，L22–L35禁图 | 全层禁图；L1–L6仅o_proj输出置零 | Dense完整KV |
| P | 进入L22前实际移除 | L0–L21开放 | 全层禁图，文本Attention不跳算 | 各层仅保留文本KV |
| B | 全36层保留 | 与F相同 | 全层禁图；L1–L6真实Attention跳算 | 非跳算层仅文本KV；跳算层无持久KV |
| C | 进入L22前实际移除 | L0–L21开放 | 全层禁图；L1–L6真实Attention跳算 | 非跳算层仅文本KV；跳算层无持久KV |

R使用Qwen图像末位置之后至P−1之前的原始非视觉序列，包括vision_end及相应模板位置，未硬编码576。Image Query与P−1之前文本的Attention均正常执行。

L22入口对Hidden、原始逻辑位置及已经计算的mRoPE cos/sin同步按非图像索引gather；没有重编号文本位置。DeepStack仍在L0、L1、L2后原生注入，早于退出点。后续每层使用原生Qwen块；Norm、MLP、Residual均执行。

L1–L6 Prefill只从Attention计算中移除最后的P−1行；前122−1=121行仍调用原生Q/K/V、Q/K Norm、mRoPE、Eager Attention及o_proj。此层以后所有真实Decode均不读取Attention，因此不需存储仅供这些不存在的未来Attention使用的KV。Decode直接返回零Attention分支，既无Q/K/V、QK/AV和o_proj调用，也不分配该层持久KV。MLP与残差并未跳过。

其他物理层的当前Prefill Attention仍临时使用完整合法KV，但持久Cache只保存未来Decode可以访问的文本KV；按最大输出长度预分配容量并原位追加，区分实际Live与Capacity。Cache保存原始逻辑长度供外层mRoPE定位，不能把压缩后的物理长度当作原始绝对位置。

单题执行审计观察到：

- F/B在L22仍为122行；P/C在L21为122行、L22–L35为26行。
- B/C的L1–L6 Prefill Q/K/V和o_proj均为121行，真实Decode无这些算子；对应KV Live和Capacity均为0。
- 36层每次Forward都有MLP执行及两次残差加法逐位检查；三路DeepStack注入保留；被`-inf`屏蔽连接实际概率为0。
- 原始保留位置序列完整保留，没有将物理压缩索引误作逻辑位置。

`physical_execution_audit.json`及`actual_matrix_shapes.csv`保存实际矩阵形状；这证明代码执行过物理删行及跳算，**不证明数值等价或质量可接受**。

仅为帮助复核该短输入的执行形状，以下列出GPU3的同题观测；GPU2相同。

| 条件 | 语言矩阵GFLOPs | 持久KV Live（bytes） | 持久KV Capacity（bytes） |
|---|---:|---:|---:|
| F | 1717.5103 | 18137088 | 18137088 |
| P | 1195.5774 | 3981312 | 8404992 |
| B | 1716.4205 | 3317760 | 7004160 |
| C | 1194.5442 | 3317760 | 7004160 |

GFLOPs由实际语言线性层GEMM及QK/AV维度按一次乘加2 FLOPs估算，**不含视觉编码器、LM Head、Norm等非矩阵运算，不是完整模型FLOPs或硬件计数器**。这些是在带审计Hook的单题Smoke中观察的形状，不是正式性能评测；原始Smoke中计时字段不可作为E2E加速依据。不同回答长度也不能在未配对测速时混同为算子效率。正式`latency_flops_kv.csv`全部标为NOT_RUN。

## 失败诊断与尚未解决的问题

在最终数值失败前有两次实现错误，均保存了现场、重新冻结源码并重跑CPU测试：首先记录器未兼容视觉Eager函数的关键字参数；其次在紧凑KV Decode路径误用了无法推断其物理长度的HF Mask构造，导致索引越界。最终执行版本修正后完成首题六个非Native条件的生成，但触发预声明数值门槛。没有把早期失败当成通过，也没有在数值失败后再启动GPU任务。

当前能确认的是**执行计算形状改变后，F/C首步有效Logits误差超限**。BF16 GEMM行数、Attention归约顺序、图像行gather及KV压缩都是可能影响数值的环节，但本轮没有逐层Hidden原始张量或完整首步向量档案足以定位首个分歧层。因此不能把误差直接归结为“正常浮点误差”，也不能排除物理实现中的语义或位置问题。没有继续GPU诊断以绕过用户指定的失败停止边界。

另外，两项已保存功能诊断也继承了公共的P−1/Decode禁图，故这里的W22与A16不是以Native为底、只改一个因素的完全隔离诊断。它们的真实权限记录于协议及本报告；本轮不据其作单因素贡献或交互结论。这是未完成项，后续不能直接把这些输出解释为用户要求的纯单因素对照。

P/B/C对比Native还同时改变了答案直接读图权限，Native不能充当“答案禁图固定”的纯2×2空白格。因此即使未来质量实验通过，也只能先解释C−P、C−B等条件贡献，不能把与Native的差异直接命名为两个因素的纯交互。

## 对任务七个问题的回答

1. **Qwen的L22视觉物理退出是否可行？** 首道短输入实际删行并完成生成，但数值验收失败；尚不能宣布得到合格的迁移实现。
2. **L1–L6是否真正跳过答案Attention？** 该短输入审计证明P−1对应行及真实Decode均跳过目标计算，MLP保留；未完成多长度Smoke和后续质量验证。
3. **GQA/VQAv2质量损失是多少？** 尚未知。两套主数据均冻结，但正式生成数为0；历史Accuracy不能填入此表。
4. **退出与跳算贡献、联合交互如何？** 尚未知。仅有一题同答案，且数值门槛失败、部分诊断权限不完全隔离，不能形成质量或交互结论。
5. **实际FLOPs、KV与E2E收益是多少？** 单题实际语言矩阵形状和KV容量如上；完整模型FLOPs、代表性图片性能、E2E加速尚未测得。
6. **与LLaVA的差别是什么？** Qwen有36层、动态图像Token数、三路DeepStack、三轴mRoPE及Q/K Norm，不能直接照搬LLaVA的576位置和Cache布局。历史LLaVA成功不能替代Qwen物理数值验证。
7. **是否已有跨模型物理算法候选？** 有待调试的真实物理原型，但**尚无通过正确性、质量与速度联合验收的跨模型候选**。本轮没有证明无损或加速。

## 文件、复现与释放

CPU测试12/12通过；运行日志见`cpu_tests.txt`。最终数值失败后仅做CPU重算、图片ID重叠补充审计、制图与文件校验，不再加载模型或运行新的GPU样本。三组PNG/PDF分别展示失败门槛、真实Hidden行数及未运行质量/速度状态，作图CSV和脚本均保留。

在源码冻结Commit所在Worktree执行CPU检查：

```bash
cd /data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data/lk/深度学习/.envs/ate-qwen3vl/bin/python \
  -m vlm_research.qwen3vl_c3v2_physical.tests
```

`reproduce.md`提供CPU审计/制图命令和历史GPU运行命令。GPU命令仅用于复核该失败现场，**本次交付不会自动重新执行，原绝对截止时间也没有重置**。原始大模型、图片没有复制进Git；Manifest引用服务器既有文件路径与SHA。

执行设备为物理GPU3 `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`和GPU2 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`，每卡一份模型、UUID绑定，没有使用GPU0/1、抢占或杀其他任务。最终两卡无计算进程，显存均约27MiB；详见`gpu_release_audit.json`。绝对预算为UTC 11:54–15:54，GPU新增任务截止15:24；本轮在12:38:56触发失败后提前结束GPU工作。

main未修改；所有新增实现、冻结输入、失败现场及报告交付到独立分支`codex/qwen3vl-c3v2-physical-transfer`。停止状态保持为失败，不标记实验COMPLETE，不自动搜索层、不训练、不追加GPU实验。
