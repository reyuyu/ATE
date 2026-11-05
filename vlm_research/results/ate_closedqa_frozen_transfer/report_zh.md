# ATE 固定封闭式问答迁移验证

## 核心结论

Full-QSA在ScienceQA为61.00%，相对Native为-4.60pp，损害65题、纠正42题。四项任务没有一项达到预先冻结的−1pp家族校正非劣效判据。因此，本轮不支持把当前Full-QSA直接认定为封闭式问答的低损害迁移方案；不因负结果调整Writer、跳算、阈值或数据。

- scienceqa_img：Full-QSA自然EOS E2E相对Native 1.214x；相对C3-V2 0.998x；同Native完整序列子集15/20张，速度1.232x。
- pope_random：Full-QSA自然EOS E2E相对Native 1.249x；相对C3-V2 0.998x；同Native完整序列子集18/20张，速度1.248x。
- pope_popular：Full-QSA自然EOS E2E相对Native 1.247x；相对C3-V2 0.999x；同Native完整序列子集18/20张，速度1.246x。
- pope_adversarial：Full-QSA自然EOS E2E相对Native 1.245x；相对C3-V2 0.997x；同Native完整序列子集17/20张，速度1.244x。

## 协议与完整性

ScienceQA-IMG 原冻结500题；POPE COCO Random/Popular/Adversarial各300题。题目、图片字节SHA、原Prompt和官方标注全部绑定。原生LLaVA固定revision、FP16/eager、Transformers 4.49.0，所有条件统一last-row LM Head (`logits_to_keep=1`)、Greedy、最多32 token、自然EOS。
非Native均禁止P−1和全部真实Decode访问原始Image Key。L2–L5图像与P−1之前文本Query正常Attention，Norm/Residual/MLP保留。ScienceQA的Q严格仅为题干，不含选项/Hint/指令；这些文本仍原样出现在输入中，没有lecture、solution或Gold泄露。
条件固定：Fast-P2=All-R/L20/不跳答案Attention；C3-V2=All-R/L20/P−1+Decode跳L2–L5；Full-QSA=Q+完整ASSISTANT Always/L20/保留P−1文本Attention/仅Decode跳L2–L5；历史C4-V2=Conditional-15/L20/P−1+Decode跳算。后者不得混称Full-QSA。
全部7000条条件记录完成；完全相同POPE输入在同卡生成后明确记录哈希复用，不共享请求Cache、不删除任何问题。全部错误/空答案/解析失败/截断见negative_cases.csv。当前主分数来自完整自由生成，不使用候选LogP选择答案。

## 正式生成得分

| 任务 | 条件 | N | Accuracy % | F1 % | Precision % | Recall % | Yes % | 解析失败 | 空答/截断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| scienceqa_img | Native Full | 500 | 65.600 | - | - | - | - | 0 | 0/0 |
| scienceqa_img | Fast-P2 | 500 | 65.800 | - | - | - | - | 0 | 0/0 |
| scienceqa_img | C3-V2 | 500 | 64.200 | - | - | - | - | 7 | 0/0 |
| scienceqa_img | Full-QSA | 500 | 61.000 | - | - | - | - | 7 | 0/0 |
| scienceqa_img | Historical C4-V2 | 500 | 60.400 | - | - | - | - | 6 | 0/0 |
| pope_random | Native Full | 300 | 88.333 | 87.179 | 96.748 | 79.333 | 41.000 | 0 | 0/0 |
| pope_random | Fast-P2 | 300 | 86.000 | 84.091 | 97.368 | 74.000 | 38.000 | 0 | 0/0 |
| pope_random | C3-V2 | 300 | 87.000 | 85.393 | 97.436 | 76.000 | 39.000 | 0 | 0/0 |
| pope_random | Full-QSA | 300 | 86.000 | 84.091 | 97.368 | 74.000 | 38.000 | 0 | 0/0 |
| pope_random | Historical C4-V2 | 300 | 82.667 | 79.365 | 98.039 | 66.667 | 34.000 | 0 | 0/0 |
| pope_popular | Native Full | 300 | 88.667 | 87.500 | 97.541 | 79.333 | 40.667 | 0 | 0/0 |
| pope_popular | Fast-P2 | 300 | 86.000 | 84.091 | 97.368 | 74.000 | 38.000 | 0 | 0/0 |
| pope_popular | C3-V2 | 300 | 87.000 | 85.393 | 97.436 | 76.000 | 39.000 | 0 | 0/0 |
| pope_popular | Full-QSA | 300 | 85.667 | 83.774 | 96.522 | 74.000 | 38.333 | 0 | 0/0 |
| pope_popular | Historical C4-V2 | 300 | 83.000 | 79.681 | 99.010 | 66.667 | 33.667 | 0 | 0/0 |
| pope_adversarial | Native Full | 300 | 83.333 | 82.639 | 86.232 | 79.333 | 46.000 | 0 | 0/0 |
| pope_adversarial | Fast-P2 | 300 | 82.000 | 80.435 | 88.095 | 74.000 | 42.000 | 0 | 0/0 |
| pope_adversarial | C3-V2 | 300 | 82.667 | 81.429 | 87.692 | 76.000 | 43.333 | 0 | 0/0 |
| pope_adversarial | Full-QSA | 300 | 81.667 | 80.144 | 87.402 | 74.000 | 42.333 | 0 | 0/0 |
| pope_adversarial | Historical C4-V2 | 300 | 81.000 | 77.821 | 93.458 | 66.667 | 35.667 | 0 | 0/0 |

## 主要配对比较

图片SHA整簇配对Bootstrap 20,000次、问题等权。Full-QSA对Native的四项Accuracy是预先冻结的主要族；−1pp非劣效使用Bonferroni4单侧下界（quantile=0.0125），仅当下界严格大于−1pp才支持。其余对Native和C3-V2的比较均为探索性，详见paired_comparisons.csv。CI跨0不代表无损。

| 任务 | Full-QSA − Native pp | 普通95%CI | 损害/纠正 | 校正单侧下界 pp | −1pp非劣效 |
|---|---:|---|---|---:|---|
| scienceqa_img | -4.600 | [-8.635, -0.596] | 65/42 | -9.234 | 不支持 |
| pope_random | -2.333 | [-4.333, -0.333] | 9/2 | -4.667 | 不支持 |
| pope_popular | -3.000 | [-5.000, -1.000] | 10/1 | -5.333 | 不支持 |
| pope_adversarial | -1.667 | [-4.000, 1.000] | 10/5 | -4.333 | 不支持 |

### Full-QSA 对 C3-V2

- scienceqa_img: -3.200pp，95%CI [-7.085, 0.795]，损害59题、纠正43题。
- pope_random: -1.000pp，95%CI [-2.667, 0.667]，损害5题、纠正2题。
- pope_popular: -1.333pp，95%CI [-3.333, 0.333]，损害6题、纠正2题。
- pope_adversarial: -1.000pp，95%CI [-2.667, 0.667]，损害6题、纠正3题。

## 同卡实测性能

每任务按输入哈希预先固定20张不同图片。每题五条件3次预热、5次交错重复；每次冷KV并重算Vision；同GPU配对，Processor/I/O不计时，模型与Cache建立、Vision、Prefill及Decode计入E2E。先每题取重复中位数，再跨20题等权均值。完整序列相同子集另外报告，不能用更短输出冒充加速。
| 任务 | 条件 | Prefill ms | E2E ms | Decode ms/token | Speedup | 同序列N / Speedup | FLOPs T | KV Live/Capacity MiB | Peak MiB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|
| scienceqa_img | Native Full | 196.40 | 314.93 | 72.74 | 1.000x | 20 / 1.000x | 9.678 | 348.02/348.02 | 14001.6 |
| scienceqa_img | Fast-P2 | 141.01 | 260.45 | 72.07 | 1.209x | 20 / 1.209x | 6.788 | 60.02/75.03 | 13712.9 |
| scienceqa_img | C3-V2 | 140.91 | 258.93 | 66.97 | 1.216x | 19 / 1.236x | 6.788 | 52.54/65.65 | 13703.4 |
| scienceqa_img | Full-QSA | 140.83 | 259.34 | 66.94 | 1.214x | 15 / 1.232x | 6.788 | 52.54/65.65 | 13703.4 |
| scienceqa_img | Historical C4-V2 | 140.86 | 259.21 | 66.86 | 1.215x | 16 / 1.234x | 6.788 | 52.54/65.65 | 13703.4 |
| pope_random | Native Full | 165.88 | 284.36 | 72.71 | 1.000x | 20 / 1.000x | 8.482 | 304.25/304.25 | 13925.0 |
| pope_random | Fast-P2 | 112.74 | 232.17 | 72.10 | 1.225x | 18 / 1.224x | 5.611 | 16.25/31.25 | 13650.7 |
| pope_random | C3-V2 | 112.76 | 227.40 | 66.92 | 1.250x | 19 / 1.250x | 5.611 | 14.22/27.34 | 13646.8 |
| pope_random | Full-QSA | 112.69 | 227.74 | 66.89 | 1.249x | 18 / 1.248x | 5.611 | 14.22/27.34 | 13646.8 |
| pope_random | Historical C4-V2 | 112.76 | 227.76 | 66.80 | 1.248x | 18 / 1.248x | 5.611 | 14.22/27.34 | 13646.8 |
| pope_popular | Native Full | 163.41 | 281.89 | 72.76 | 1.000x | 20 / 1.000x | 8.482 | 304.27/304.27 | 13925.0 |
| pope_popular | Fast-P2 | 111.15 | 230.37 | 72.06 | 1.224x | 18 / 1.223x | 5.612 | 16.27/31.27 | 13650.5 |
| pope_popular | C3-V2 | 111.16 | 225.80 | 66.91 | 1.248x | 19 / 1.248x | 5.611 | 14.24/27.37 | 13646.6 |
| pope_popular | Full-QSA | 111.09 | 226.13 | 66.93 | 1.247x | 18 / 1.246x | 5.612 | 14.24/27.37 | 13646.6 |
| pope_popular | Historical C4-V2 | 111.19 | 226.16 | 66.87 | 1.246x | 18 / 1.246x | 5.611 | 14.24/27.37 | 13646.6 |
| pope_adversarial | Native Full | 163.88 | 282.19 | 72.70 | 1.000x | 20 / 1.000x | 8.484 | 304.32/304.32 | 13925.1 |
| pope_adversarial | Fast-P2 | 111.49 | 230.79 | 72.06 | 1.223x | 17 / 1.222x | 5.613 | 16.32/31.32 | 13650.8 |
| pope_adversarial | C3-V2 | 111.57 | 226.02 | 66.84 | 1.249x | 18 / 1.248x | 5.613 | 14.28/27.41 | 13646.8 |
| pope_adversarial | Full-QSA | 111.50 | 226.70 | 66.98 | 1.245x | 17 / 1.244x | 5.613 | 14.28/27.41 | 13646.8 |
| pope_adversarial | Historical C4-V2 | 111.56 | 226.51 | 66.82 | 1.246x | 17 / 1.245x | 5.613 | 14.28/27.41 | 13646.8 |

FLOPs是实际执行Linear/Conv/QK/AV张量形状的独立账本（2 FLOPs/MAC），不把Dense Mask连接当成已省计算。估算Algorithmic TFLOP/s见CSV；本轮未采集硬件计数器，硬件TFLOPS/DRAM带宽为N/A。KV为真实Live与分配Capacity，不能混同。

## 评分与结论边界

ScienceQA沿用冻结字母解析器，不因输出格式问题修补得分；学科结果见scienceqa_subject_accuracy.csv。POPE沿用官方首句词法规则，空字符串仍会被其规则解析为yes，空答另外完整标记，不隐藏此局限。三策略共享图片，不将它们视作三个独立图像队列，也不跨任务平均Accuracy/F1。
只验证既定五条件，没有搜索/训练/按结果调参。结果用于这批冻结样本的迁移评估，不能解释为新外部总体上的严格无损；质量与时延须共同判断，不以生成首Token指标替代完整答案。

Full-QSA和C3-V2同时存在Writer集合与P−1跳算方式差异，不能把两者全部差值单独归因于某一种机制。Conditional/QSA的Dense来源Mask本身不减少矩阵形状或KV容量；FLOPs/速度变化必须来自实际物理路径或生成长度，而不是Mask开放Token更少。Fast-P2在ScienceQA的点估计接近Native，也不能仅凭点估计宣布无损或有益。

POPE中Full-QSA的Recall由Native的79.33%下降到74.00%，不是仅凭Yes Ratio即可判断整体改善；三策略共享正例和图片，这一模式不能当成三次独立复现。完整Precision/Recall/F1及其配对差值见CSV。

## 测试、复现与资源

CPU16项测试通过；每任务预定3题GPU Smoke通过：同形状Native/No-op及HF生成有效Logits严格torch.equal；对应全Dense功能参考符合原冻结容差；禁止连接实际概率为0，早期非目标Query、位置/RoPE、完整MLP、Cache隔离及Hook清理均检查。没有放宽数值标准。
MMBench/SEED：Phase4冻结注册中没有对应可验证Manifest，指定资源目录未找到可冻结资产和评分器；本轮仅报告未准备状态，不下载或替换。详见benchmark_readiness.json。
正式复现入口：`CUDA_VISIBLE_DEVICES=<GPU2或GPU3的冻结UUID> python -m vlm_research.closedqa_transfer.benchmark --rank <0或1>`。该入口支持已有检查点恢复，也支持空检查点的Smoke、质量生成和性能采集；原run.py不修改，作为冻结质量源码保留。CPU汇总：`python -m vlm_research.closedqa_transfer.analyze`，进程退出后审计：`python -m vlm_research.closedqa_transfer.analyze --finalize`。
代码、模型、样本、分片和协议SHA见execution_freeze.json；GPU使用及退出记录见worker0/、worker1/和resource_release_audit.json。不使用GPU0/1，不修改main或共享环境。

### 工程异常与恢复

原性能记录包装同时通过显式参数和生成结果传入condition字段，触发Python TypeError；不是模型、Mask或数值验证失败。原STOP与GPU日志完整保留。没有修改正在执行的冻结推理源码，也没有重跑或丢弃已完成的质量记录。独立benchmark.py只修正测量记录的标签组装，另行冻结SHA；恢复时同GPU Native首Token Logits哈希精确匹配。正式计时在新入口重新完成全部预热和重复，不复用失败尝试的时延。
断点复现：质量记录使用原run.py和execution_freeze.json绑定；性能入口使用performance_entry_freeze.json绑定，运行`python -m vlm_research.closedqa_transfer.benchmark --rank <0或1>`。
