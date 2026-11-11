# Qwen3-VL：P2全量配对评测与GQA官方题型损失归因

## 首页三项结论

1. **P2全量结果（不是150题Validation）：** ScienceQA-IMG Native 94.794%、SVH 94.447%，差-0.347pp；MMBench Native 84.184%、SVH 84.014%，差-0.170pp。下表给出同P2配对区间、损害和纠正。
2. **GQA总体-8.9839pp主要来自官方structural=query：** Query占54.10%题量，却占97.35%原对→错；类内下降16.68pp，加权贡献-9.0237pp，占净损失100.44%。非Query净改善5题，所以贡献超过100%并非计数错误。
3. **不能笼统归因于属性或多跳：** 官方query_argument=name的4931题下降18.94pp，颜色查询578题下降24.91pp；positionQuery的320题仅变化-0.625pp。基于程序的“定位后问属性”没有比直接属性查询更大损害的证据。关系/类别/名称任务受损明显，但本轮不识别具体写入层的因果贡献。

## 一、P2全量主结果

| 任务 | 题/完整组 | Native % | SVH % | SVH−Native pp | 图像簇95%CI pp | 原对→错 | 原错→对 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ScienceQA-IMG | 2017 | 94.794 | 94.447 | -0.347 | [-0.687, -0.049] | 9 | 2 |
| MMBench | 1176 | 84.184 | 84.014 | -0.170 | [-0.516, +0.167] | 3 | 1 |

ScienceQA分数是原冻结严格评分；MMBench主分数要求同一Circular组的所有旋转记录都正确。未把旋转记录级分数当Circular分数。
共12,788条规范预测：新生成11,392条、严格复用1,396条。ScienceQA 2,017题、MMBench 4,377行/1,176完整组。OOM/失败记录0条。
### 样本与独立性

四个预冻结集合分别报告：All（全队列）、PriorValidation（上一轮Validation题号）、UnselectedImages（排除上一轮全部Discovery图片）、UntouchedImages（同时排除Discovery与Validation图片）。当前全量包含Prompt选择用过的Discovery，不是独立外部确认集。PriorValidation中为维持同图同卡重新生成的4题由运行前结构规则决定，不按分数选择。

以下为各集合的严格分数：

| 任务 | 集合 | 题/组 | Native % | SVH % | Δpp | 95%CI |
| --- | --- | --- | --- | --- | --- | --- |
| ScienceQA-IMG | All | 2017 | 94.794 | 94.447 | -0.347 | [-0.687, -0.049] |
| ScienceQA-IMG | PriorValidation | 150 | 98.667 | 98.667 | 0.000 | [+0.000, +0.000] |
| ScienceQA-IMG | UnselectedImages | 1842 | 94.843 | 94.408 | -0.434 | [-0.800, -0.109] |
| ScienceQA-IMG | UntouchedImages | 1681 | 94.468 | 94.051 | -0.416 | [-0.783, -0.115] |
| MMBench | All | 1176 | 84.184 | 84.014 | -0.170 | [-0.516, +0.167] |
| MMBench | PriorValidation | 150 | 85.333 | 85.333 | 0.000 | [+0.000, +0.000] |
| MMBench | UnselectedImages | 1026 | 84.211 | 84.016 | -0.195 | [-0.592, +0.191] |
| MMBench | UntouchedImages | 876 | 84.018 | 83.790 | -0.228 | [-0.697, +0.222] |

### P2高分独立CPU审计

从ScienceQA官方problems.json和MMBench原始TSV按Question ID重建允许字段；逐项核对原Choices顺序、Hint、旋转Gold。Prompt由Question/Hint/Options和固定指令构成，不读取Gold索引、lecture或solution。使用CPU Tokenizer重新解码每条保存的生成Token IDs，并独立评分；没有以标注答案替代模型调用。历史Native/SVH调用分支与模型生成源码经SHA核验。
独立重算旧Validation：SQA Native/SVH均148/150；MMBench均128/150完整组。共702行，701行Token序列相同，701行得分相同；总分相同不等于复制答案。Discovery/Validation图片SHA交集为空。

P2逐字节读取旧protocol.json的`prompts.P2_choices`，而非重写近似指令：
```text
Your entire response must be exactly one uppercase option letter from the listed choices, with no other characters.
```
协议名：`ATE_HF_QWEN_P2_SHORT_ANSWER`。Qwen原生Chat Template、模型revision、BF16/Eager、原视觉预算、Greedy、EOS及max_new_tokens=128均未改；无约束解码，未人为限制为1 Token。正式预测之间只有输出指令相对旧P0发生变化。

### ScienceQA增强解析与旧P0背景

| 评分 | Native % | SVH % | Δpp |
| --- | --- | --- | --- |
| strict | 94.794 | 94.447 | -0.347 |
| conservative_diagnostic | 94.794 | 94.447 | -0.347 |

以下P0只供不同Prompt协议的历史背景；不把P2-SVH对P0-Native当主配对方法差：

| 任务 | 条件 | 题/组 | P0严格 % | P0诊断 % | P0均生成Token |
| --- | --- | --- | --- | --- | --- |
| ScienceQA-IMG | Native | 2017 | 52.405 | 57.164 | 62.371 |
| ScienceQA-IMG | SVH | 2017 | 19.782 | 50.868 | 88.012 |
| MMBench | Native | 1176 | 72.364 | 72.364 | 24.252 |
| MMBench | SVH | 1176 | 59.694 | 59.694 | 40.193 |

### 输出长度、异常与时延

全量每请求的平均/P50/P95生成Token、E2E、TTFT、TPOT、EOS、截断、空答案及解析失败见`p2_output_length_and_latency.csv`；答案字母分布见`p2_letter_distribution.csv`。E2E计时为已有执行器的Vision+语言计算边界，不含图片读取/Processor，不能冒称完整服务端端到端。
生成Token统计包含EOS；常见的2 Token序列是一个选项字母加EOS，不是两个答案字母。正式评分仍用自然生成的完整文本，不从Gold或候选LogP取答案。
固定32 Token性能仅严格复用上一轮完全相同代码、P2输入绑定、模型和视觉预算的独立小型计时探针：每任务2个输入、每输入每方法3次正式重复。原始日志、来源路径与哈希绑定保留，不称为本轮全量固定32测速。它与自然EOS全量每题计时分列；不把回答从几十Token变成一个字母带来的时延下降称为SVH算子加速。

| 任务 | 方法 | 均Token | P50 | P95 | E2E ms | TTFT ms | TPOT ms | EOS % | 截断 | 空答 | 解析失败 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ScienceQA-IMG | Native | 2.000 | 2.000 | 2.000 | 244.906 | 175.471 | 68.877 | 100.000 | 0 | 0 | 0 |
| ScienceQA-IMG | SVH | 2.000 | 2.000 | 2.000 | 260.290 | 187.043 | 70.899 | 100.000 | 0 | 0 | 0 |
| MMBench | Native | 2.000 | 2.000 | 2.000 | 235.410 | 164.588 | 70.243 | 100.000 | 0 | 0 | 0 |
| MMBench | SVH | 2.000 | 2.000 | 2.000 | 252.333 | 177.533 | 72.322 | 100.000 | 0 | 0 | 0 |

历史同P2固定32探针（不是新增全量测速）：

| 任务 | Native E2E ms | SVH E2E ms | Native/SVH倍率 |
| --- | --- | --- | --- |
| ScienceQA-IMG | 2246.451 | 2299.853 | 0.977 |
| MMBench | 2181.837 | 2285.944 | 0.954 |

倍率低于1表示SVH在该固定Token小样本中更慢。不能从自然EOS输出变短或KV减少推导SVH本体加速。

## 二、GQA12578官方题型归因

原始Checkpoint独立重评分：Native 61.63937%，SVH 52.65543%，Δ=-8.98394pp；原对→错1433题，原错→对303题。共有398个图片SHA簇（不是12,578张独立图片），所有本地图像文件SHA均核对，问题/Gold/Prompt及两方法输入绑定完全匹配。官方types覆盖100%，UNKNOWN=0。

| 官方类型 | 题数 | 题量 % | Native % | SVH % | Δpp | 原对→错 | 原错→对 | 净损失题数 | 总体贡献 pp | 净损失贡献 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| choose | 1129 | 8.976 | 85.651 | 85.562 | -0.089 | 15 | 14 | 1 | -0.008 | 0.088 |
| compare | 589 | 4.683 | 67.742 | 68.251 | 0.509 | 8 | 11 | -3 | 0.024 | -0.265 |
| logical | 1803 | 14.335 | 80.699 | 80.699 | 0.000 | 8 | 8 | 0 | 0.000 | -0.000 |
| query | 6805 | 54.102 | 45.393 | 28.714 | -16.679 | 1395 | 260 | 1135 | -9.024 | 100.442 |
| verify | 2252 | 17.904 | 81.838 | 81.972 | 0.133 | 7 | 10 | -3 | 0.024 | -0.265 |

贡献公式为`N_type/N_all * (Acc_SVH_type − Acc_Native_type)`，各互斥完整分组的贡献严格求和回总体差距。Query不只是损害数量最多，也是在五种官方structural类别中类内降幅最大、总体净损失贡献最大。其余类型CI见`gqa_paired_type_bootstrap.csv`，不能将接近0的净变化理解成每道题都不变。

### Query内部：不是所有Query同样脆弱

| 官方字段 | 取值 | 题数 | Native % | SVH % | Δpp | 95%CI pp | 损害 | 纠正 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic | attr | 1766 | 56.908 | 46.546 | -10.362 | [-12.22, -8.50] | 235 | 52 |
| semantic | cat | 1038 | 44.220 | 19.846 | -24.374 | [-28.77, -20.14] | 276 | 23 |
| semantic | global | 108 | 37.037 | 20.370 | -16.667 | [-26.09, -7.48] | 23 | 5 |
| semantic | rel | 3893 | 40.714 | 23.221 | -17.493 | [-20.03, -15.02] | 861 | 180 |
| query_argument | color | 578 | 50.173 | 25.260 | -24.913 | [-29.05, -20.77] | 165 | 21 |
| query_argument | hposition | 320 | 94.688 | 94.062 | -0.625 | [-1.92, +0.61] | 3 | 1 |
| query_argument | material | 234 | 67.949 | 59.829 | -8.120 | [-13.60, -2.65] | 28 | 9 |
| query_argument | name | 4931 | 41.452 | 22.511 | -18.941 | [-21.43, -16.50] | 1137 | 203 |
| query_argument | size | 169 | 57.988 | 55.621 | -2.367 | [-5.70, +0.62] | 7 | 3 |

`semantic=rel`常见程序是`relate → query:name`：它可能要求说出相关物体名称，不等价于只预测left/right等关系词。官方字段里没有统一名为QueryObject的类别，不凭英文问句制造这种“官方标签”。`query_argument`来自真实官方功能程序，单独列示。
名称查询净损失934题；颜色查询净损失144题。位置查询是重要反例：Native 94.69%、SVH 94.06%。activity等低下降子类也保留，但低Native基线可能有地板效应，不能宣称路径无功能。

### 多步定位假说与反证

程序中在query前出现relate/filter定义为“定位”，这是明确标为`HEURISTIC_DIAGNOSTIC`的操作性诊断，不是新增官方题型。属性query中：定位后查询925题下降9.84pp，直接select后查询841题下降10.94pp；差异为+1.10pp、图片簇95%CI[−2.57,+4.79]。因此这轮不支持“先定位后问属性更容易受损”。
关系跳数≥2的451题下降18.40pp，0跳下降16.21pp，1跳下降16.78pp；程序深度≥5仅71题下降21.13pp。这些是探索性关联，受任务组成与难度混杂，不能证明多跳本身导致损失。原始视觉Token数、真实问题Token长度、单/多词答案、Yes/No/open以及计数比较程序标签均另表给出，没有据结果改分组。

### 历史1000/3000题：逐ID绑定后才谈组成

历史三个分支均从固定Commit读取，样本逐Question ID/Image ID/Question/Gold绑定原GQA val_balanced Parquet，相关本地图片SHA逐文件核对。当前testdev与历史1000/3000队列的Question ID+Image SHA严格重合数均为0，图片SHA重合也为0；因此不把旧答案当当前同题基线，也不能直接把两个总体差值作因果分解。
历史1000题有525题官方query（52.5%）；历史恢复L22–L35写入的+9.2pp，分解后query子集恢复+17.52pp，净改善92题，其余structural类别合计净0。这是该历史队列内部的已测因果比较，不能直接当本轮恢复效果。
历史3000题只有500题query（16.67%），且500题全部都是官方`types.detailed=positionQuery`，不是颜色/名称查询。其余为1500 choose、500 compare、500 logical。其历史QueryAttr的0.4pp写入损失与本轮positionQuery约0.63pp的低降幅在描述上相容；不能据旧“六类”称总体Query已低损验证。组成差异有真实元数据证据，但Prompt、评分和干预背景也不同，不能说组成已经解释全部跨队列差距。

### 真实损害、纠正与反例

`gqa_error_cases.csv`按预声明SHA规则，从五个structural类别的损害/纠正单元各取最多4题；另外展示n≥30且净变化最小的3个query细类的哈希案例，并明确这是描述性反例选择。完整12578题答案、官方标签及程序见`gqa_per_question_types.csv`，不只留成功故事。

| ID | 选择单元 | 问题 | Gold | Native | SVH |
| --- | --- | --- | --- | --- | --- |
| 20285576 | query/harm | Which color is the curtain to the right of the picture? | yellow | yellow | white |
| 201056263 | query/rescue | Who is watching the soccer player that is kicking the ball? | spectator | man | spectator |
| 202262056 | choose/harm | Is the gold plate on the left side or on the right? | left | left | right |
| 201952977 | small_net_drop_subtype/positionQuery | Which side are the white houses on? | left | left | left |

Exact Match也会惩罚词形、修饰词和近义表达：例如固定哈希损害案例201663260中Gold/Native为clean，SVH为very clean。本轮没有引入语义Judge或更改规范化，因此并非每一题评分损失都能直接认定为缺乏视觉证据；但颜色yellow→white等实际内容改变的损害也同时存在。官方得分和原文都保留，不能用少数格式或近义案例抵消全部损失。

## 三、有效性、资源和复现

所有质量差使用题目等权、Image SHA整簇配对Bootstrap 20,000次，固定seed=20261111；MMBench按完整Circular组先求分，再以共享Image SHA整簇重采样。子类、程序诊断及多项比较均为探索性，没有把未校正CI当确认性显著性，也不把CI跨0称无损。
历史严格Dense/Physical数值状态仍为**FAIL_RETAINED**；本轮双卡Smoke仅确认Native/No-op严格torch.equal、SVH重复前向一致、冻结结构/位置/MLP/独立Cache/Hook/输入/权重审计正常。本轮质量成功不能补判严格数值等价。方案B为ATE统一HF协议，不是作者官方协议复现；方案A继续BLOCKED。
全量运行与历史记录全部只读继承；只新增独立工作树文件，未改主实验Prompt/Checkpoint/SVH算法/共享环境，未推送GitHub。为了读取25MB历史Parquet，复用已有video-model环境中的PyArrow子进程，未安装或修改Qwen环境；初次CPU缺少PyArrow的日志原样保留。
GPU1/GPU3各自常驻一份模型，按图片分片，同图规范记录Native/SVH均来自同一物理卡。4道混合复用/新增图簇中的Validation题预先按UUID规则补跑8次，旧记录保留，规范结果不重复计数。GPU0/2外部任务不触碰。任务进程已退出，GPU释放审计见`gpu_release_audit.json`。
进程finally内保存的allocator计数仍有约8MiB，原样保留；资源释放的最终依据是进程退出后PID/compute context消失及nvidia-smi回到约5MiB显示基线，不把退出前计数伪写成0。首次报告打包脚本对此误用断言，诊断日志同样保留；没有GPU生成失败或OOM。
所有模型、图片和旧大数组均共享只读，没有复制。本任务压缩检查点和报告使用独立300MiB输出上限、512MiB磁盘最低余量，不改主实验原2GiB保护。

复现命令（从本工作树执行；正式已有结果应先执行只读审计，不重复生成）：
```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m unittest -v test_full test_analysis
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_gqa.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python analyze_full.py
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python finish_report.py
```

推荐下一步仅作提案：以当前官方query细分为重点，预冻结独立验证集比较保持答案禁图时的写入截止深度；本轮没有执行新写入层、P−1回看或任何算法搜索。P2可作为新的短答协议继续验证，不能据它解决GQA语义交接问题。
