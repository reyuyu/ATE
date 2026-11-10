# Qwen3-VL-SVH 双3090 FAST-TRACK：数值门禁失败

## 结论

**本轮实际执行后，阶段1为FAIL，未放行阶段2或阶段3。**

- `FAST_TRACK_VALIDATED = false`。
- `FULL_59_NUMERICAL_VALIDATED = false`，原59例完整验收仍未完成。
- 24例已预冻结，实际运行4例：2例通过、2例失败，其余20例因停止规则未运行。
- 六项各8题的96次预检生成未启动；35,346题的70,692次Native/SVH全量生成未启动，不能提供未经测量的六项质量成绩。
- 全部失败记录、未运行清单、逐步Logits及逐算子定位已保留；没有放宽0.02门槛、修改算法或缩减视觉输入。

方案仍为 **B：ATE统一HF协议，非作者官方复现**。方案A继续BLOCKED。

## 输入、代码与冻结

基线：`2e885efa8b237312d3a744cbdfa7814a7757e3e0`。执行源码及清单冻结Commit：`785d97e44137a8a55066e4f2d3cccc6270c493d3`。

独立工作树 `/data/lk/ATE-worktrees/qwen3vl-svh-fast-track-sixbench`，分支 `codex/qwen3vl-svh-fast-track-sixbench`。历史源码和结果只读，main未修改，未推送GitHub。

严格复用上一轮方案B的冻结 `inventory.jsonl.gz`，SHA256 `ede40bc07d85da3f153e00dfcddc7bcaedb6cf831cbff22607daa238940462cd`。

| 任务 | 冻结全量题数 | 输入/评分约束 |
|---|---:|---|
| GQA testdev_balanced | 12,578 | 原题，无OCR/Hint，冻结short-answer后缀和Exact Match |
| ScienceQA-IMG | 2,017 | 真实Hint/Choices，无lecture/solution/Gold输入；Gold索引对应选项字母 |
| TextVQA | 5,000 | 完整Paper-OCR文本，十答案共识软评分 |
| POPE | 9,000 | 原始COCO三策略各3,000，不能与旧8,910混称 |
| MME | 2,374 | 原14任务与同图双题 |
| MMBench | 4,377 | ATE内部Legacy EN Dev 20230712/Circular；不是作者V1.1或API Judge |
| 合计 | **35,346** | Native和SVH各生成一次，共70,692次 |

Qwen revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；原生官方Chat Template、Processor、BF16/eager、原生mRoPE/DeepStack、`logits_to_keep=1`、Greedy与原EOS不变。原方案B冻结生成上限为GQA32、其余128；本轮没有换用旧LLaVA的1024上限。像素预算保持 `min_pixels=65536, max_pixels=2097152`。

每任务按实际总输入长度选择min/median/P99/max，运行前冻结，共24例；GQA min正是 `20705853`。另外预冻结六项各8题的48题清单，保持MME双题和MMBench Circular完整。共72个真实Processor输入已CPU绑定并保存Token/像素Tensor SHA。

全量确定性分片也已冻结：按Image SHA整簇，并连接MME双题与MMBench Circular组，同簇不会跨卡。**这是准备文件，不代表全量已经执行。**

## 双GPU真实结果

| 输入 | 视觉/总Token数 | Native/No-op严格一致 | Dense/Physical最坏逐步mean/max error | 结论 |
|---|---:|---|---:|---|
| GQA 20705853，min | 96 / 122 | 是 | 0 / 0 | PASS |
| ScienceQA-IMG 10391，min | 64 / 106 | 是 | 0 / 0 | PASS |
| TextVQA 36053，min | 320 / 360 | 是 | **0.04640334 / 0.5** | **FAIL** |
| POPE adversarial:205，min | 128 / 152 | 是 | **0.07556266 / 0.375** | **FAIL** |

两道失败题的Dense/Physical均生成 `yes`，完整Token相同，但平均Logits误差超过0.02，因此失败，不按答案相同判通过。

| 失败题 | Prefill mean/max | 首次真实Decode mean/max |
|---|---:|---:|
| TextVQA 36053 | 0.03138009 / 0.25 | 0.04640334 / 0.5 |
| POPE adversarial:205 | 0.07556266 / 0.375 | 0.05147097 / 0.3203125 |

原门禁保持：每步全词表mean absolute error≤0.02、max absolute error≤0.5，且Top1、完整Token和步数一致。表中mean是逐步平均误差的最大值，不是跨输入混合平均。

## 首个分歧算子

发现失败后，两个Worker先写停止标志，再各对本题执行最小Prefill逐算子诊断；未开启任何新的样本或质量评测。比较实际层输入、Norm、投影、mRoPE、QK、Softmax、AV、O投影、残差及MLP，按真实逻辑文本位置对齐。

| 失败题 | 第一个已观察算术差异 | 比较形状 | 最大/平均绝对差 | 首个层输出差异 |
|---|---|---|---:|---|
| TextVQA 36053 | **Prefill L26 QK_text** | `[1,32,40,40]` | 0.125 / 2.44140620e-06 | L27，max0.03125 |
| POPE adversarial:205 | **Prefill L22 QK_text** | `[1,32,24,24]` | 0.125 / 6.78168408e-06 | L22，max0.03125 |

这里 `QK_text` 是实际QK矩阵乘法结果中合法文本Key部分，因果禁止位置不参与差值；不是最终Attention概率。

**这次差异出现于AV之前，说明针对20705853的AV Query零行对齐不足以构成跨任务放行依据。** 当前逐算子证据没有将根因归咎于mRoPE/DeepStack、输入绑定或文本Cache索引；不能仅凭这些日志声称确定了底层CUDA Kernel。下一步有依据的修复方向是独立重放失败QK的实际Q/K及计算形状，验证后再决定是否需要QK几何对齐，本轮没有擅自实施。

证据目录：

- `numerical/TextVQA_36053/first_operator_difference.json`、`operator_comparison.csv`、`logits_comparison.csv`。
- `numerical/POPE_adversarial_205/`下对应文件。
- 每题四条件的实际输出、Token、逐步全词表Logits NPZ和结构审计均保留。
- `worker0_failure.json`、`worker1_failure.json`保存真实异常栈；`STOP_NUMERICAL.json`是共享关闭开关，各题独立记录为最终证据。

## 结构、显存与额外计算

四例中已记录的mRoPE位置、DeepStack、文本Cache位置及物理结构审计通过，禁止视觉连接概率为0，持久视觉KV为0。没有把结构PASS当作数值PASS。KV数值一致性记录保留，不因最终Token相同而忽略。

| 输入 | Physical仪器化Peak Allocated MiB | 本次AV零行对齐额外GFLOPs |
|---|---:|---:|
| GQA 20705853 | 16,778.30 | 1.343226 |
| ScienceQA-IMG 10391 | 16,778.60 | 0.778043 |
| TextVQA 36053 | 17,037.70 | 13.212058 |
| POPE adversarial:205 | 16,801.22 | 2.231370 |

以上来自数值审计路径，不是效率Benchmark，不能据此宣称加速。保留原先补零列/投影几何修复，本表只计新增AV Query对齐部分，完整执行矩阵统计见每题Physical审计。

本轮4例没有OOM；**MME和TextVQA最长输入没有运行，不能宣称它们在单3090上通过显存门禁**。最长清单仍保留，MME最长2081总Token/2048视觉Token、TextVQA最长2121总Token/2028视觉Token。

## CPU复核与GPU释放

新增CPU测试4项、继承修复CPU测试7项通过。另在无CUDA暴露的进程，独立读取已保存Logits，以FP64重算误差，确认两例超限及两例通过；所有执行源码、Scorer及冻结清单SHA不变。

只使用：

- 物理GPU1：`GPU-b08489af-0a8a-eb48-55ba-8346733af417`。
- 物理GPU3：`GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`。

GPU0与GPU2既有进程未终止、未抢占。两个测试Worker均因数值FAIL退出，外部 `nvidia-smi` 已确认GPU1/3均无计算进程、约5MiB基础占用、利用率0%。没有留下自动重试或全量后台任务。

## 阶段状态与后续边界

| 阶段 | 状态 | 实际完成 |
|---|---|---|
| CPU清单/输入/Scorer/分片冻结 | PASS | 全量35,346题清单；72个选中输入Tensor绑定 |
| FAST-TRACK数值验收 | **FAIL** | 4/24已运行，2PASS、2FAIL、20未运行 |
| 六任务各8题预检 | BLOCKED | 0/96次生成 |
| 双卡Native/SVH全量 | BLOCKED | 0/70,692次生成 |
| FULL_59_NUMERICAL_VALIDATED | **未完成** | 未重跑、不改判 |

结果实际保存在 `/data/lk/ATE-worktrees/qwen3vl-svh-fast-track-sixbench/vlm_research/results/qwen3vl_svh_fasttrack_sixbench/`。没有权重或原始图片复制，没有更换协议、放宽误差、跳过失败题或伪造质量成绩。到此停止，等待针对QK分歧的后续修复决策。
