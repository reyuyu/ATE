# InternVL3-SVH-R：除L2–L5外全部24层回看

**相同128题TextVQA Image-only，All24_ExceptL2_5 Soft Accuracy=54.7656%（soft score sum=70.1）。** 与历史Native相差-3.7500pp，与此前9层L19–L27相差+4.0625pp。

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。输入、结构、实际KV、评分与完整性通过，原NUMERICAL_EQUIVALENCE_FAILED保留，不从小量Smoke推导全局严格等价。

## 明确定义与冻结协议

只增加用户请求的一个条件：P−1/真实Decode在L0、L1、L6–L27共24层允许访问原始Image Key；L2–L5仍真实不执行答案Attention/QKV/QK/AV/o_proj，Norm、MLP、Residual保持。R→Image从L20开始仍禁止。视觉Hidden保留到L27结束，之后、最终tokenwise Norm之前删除256行，cutoff28；没有Prefill语言Block视觉删行收益。

因此这不等于Native Full：Native没有答案Attention跳算和L20之后R写入限制。不能把任何剩余质量差异只归因于回看层数，或称本条件为完整Native。没有改变历史引擎或算法源码，只传入新KEEP集合，复用此前已通过CPU回归的terminal引擎。

模型OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0、Python3.11.8，batch1 greedy、logits_to_keep1、max_new_tokens32和原生EOS。单图448×448、256视觉Token。GPU固定GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，与前几轮物理窗口相同，运行前确认空闲。

继续使用原128题及原顺序、原生InternVL Chat Template、ATE_SVH_1TILE_SCORER_ALIGNED短答指令。只输入原始图片与问题，无OCR辅助、Gold、Caption或Hint。131个模型输入（128主测试+3 Smoke）的Token IDs、BF16 pixels、Prompt/roles绑定重新CPU核验，准备文件不含Gold。已有模型、图片、环境均本地SSD复用。

新增128次正式质量生成，模型只加载一次、独立请求Cache；1664条旧Main记录经原始SHA、输入/Gold/源码/协议绑定及独立评分核验后复用。父Commit 1407275cbdffca5d4a23cbf8b8d233002e2840dd。Native/R0/R6/R14为用户明确授权复用的历史不同GPU结果，其余窗口为同当前GPU。没有重跑Native正式质量、其他条件、全量Benchmark或效率测试。

这是已多次使用的历史128题，用户在前轮之后追加的探索条件，不是独立Holdout、非劣效确认或全量质量结论。

## 主结果与实际EOS末端缓存

| 条件 | 回看层数 | Soft Accuracy % | EOS末端KV Live=Capacity MiB | 来源 |
|---|---:|---:|---:|---|
| Native | 28 | 58.5156 | 18.3472 | REUSED_OLD_GPU |
| R0 | 0 | 8.5156 | 3.7859 | REUSED_OLD_GPU |
| R14 | 14 | 7.0312 | 10.7474 | REUSED_OLD_GPU |
| Deep8_L19_26 | 8 | 49.6094 | 7.6910 | REUSED_SAME_CURRENT_GPU |
| Late8_L20_27 | 8 | 50.4688 | 7.6764 | REUSED_SAME_CURRENT_GPU |
| Late9_L19_27 | 9 | 50.7031 | 8.1965 | REUSED_SAME_CURRENT_GPU |
| All24_ExceptL2_5 | 24 | 54.7656 | 15.7251 | NEW_CURRENT_GPU |

完整14条件主表见quality_accuracy.csv。Native标注28层读图，仅为历史参考。正常EOS的KV受不同答案长度影响，不是固定32 Token KV@32；新条件未测速，不将缓存、理论Token或旧窗口时间冒充E2E收益。

## 同题配对统计

按Image SHA整簇配对Bootstrap5000次，seed20261112；128题对应128图片簇。普通未校正95%CI；保留多项比较与历史同题探索限制，不把CI跨0解释为等价。

| All24−基准 | 差值 pp | 95%CI pp | 得分提高/下降/不变 |
|---|---:|---|---|
| Native | -3.7500 | [-8.6719, 0.7051] | 3/10/115 |
| R0 | +46.2500 | [37.1855, 55.1562] | 69/5/54 |
| Deep8_L19_26 | +5.1562 | [-0.5469, 11.0176] | 17/10/101 |
| Late8_L20_27 | +4.2969 | [-2.1094, 10.4688] | 18/9/101 |
| Late9_L19_27 | +4.0625 | [-1.7969, 10.0781] | 16/8/104 |

其余13项完整比较、首Token/全文变化、原始Gold、Tokens和所有损害见paired_soft_changes.csv及per_question_changes.csv。24层和8/9层预算不同，质量提升不能描述为同预算优势。相对历史Native的对照同时受已注明换卡限制。

## 答案恢复与异常

相对R0：68题由零分变正分，54题由未满分变满分，29题由元回答变为官方有得分的非元回答。All24元回答0题、unanswerable 0题。恢复依据正式十答案Soft Score与Gold匹配，不以元回答减少代替OCR恢复。

All24正常EOS 128/128，空输出0，截断0，Token P50/P95/最大值=3.00/10.00/20。错误数字/品牌/词语/复述为沿用的重叠规则诊断，仅供辅助，非人工归因或新的评分器。全部1792条保存记录的SHA、输入/Gold/源码/协议、cap/EOS及官方与独立leave-one-out十答案评分逐条核验，未过滤失败题。

## 真实KV、跳算与数值审计

24个KEEP层每层持久保留真实256视觉KV及文本KV；L2–L5没有持久Cache，也不执行答案投影/Attention。所有28层视觉Hidden在Prefill语言Block中真实存在，L27之后显式删行，原始RoPE和逻辑KV位置不重编号，无重复拼接或跨条件共享。真实Tensor shape、dtype、物理长度、Live与底层Storage逐层核验，完整KV布局保留。

CPU官方28层Qwen2 Float64、1 Prefill+8真实Decode，对齐Dense/Physical Logits和Hidden MaxAbs均0，独立Cache、Mask、逻辑位置、Storage及异常清理通过。GPU3题短/中/长Smoke，Native/No-op有效Logits严格torch.equal；原始Image权限、非目标连接、L2–L5真实跳算和28层MLP/Norm均通过，未出现结构错误、NaN/Inf或OOM。

Dense/Physical Smoke：完整Tokens一致3/3、标准化答案一致3/3；对齐前缀Logits MaxAbs=0.00000000，各对齐步MeanAbs均值=0.00000000。全部逐层差异、真实概率、算子输入行数、KV原始位置与物理退出事件有完整原始审计。历史数值豁免保留，不从这3题的数值观察宣称全局等价，不进行新的内核或层搜索。

## 存储、同步与结束

SSD执行目录：/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090。持久化原件：/home/featurize/work/ATE-internvl3-svh-r-all24-textvqa-3090/vlm_research/results/internvl3_svh_r_all24_textvqa_3090/。完整SQLite、原始物理审计和CPU/GPU日志在server_only；模型、图片和准备张量继续本地SSD，不传到Windows。

GPU前冻结96个Python文件SHA保持不变，所有历史父源码逐字节保留。后处理/归档/发布代码另列analysis_source_manifest.json。独立分支codex/internvl3-svh-r-all24-textvqa-3090，不改main、不强推、不建PR或覆盖此前结果。artifact_manifest.json与SHA256SUMS列出上传和服务器原件；固定Commit HTTPS逐文件回读及SSH分支头一致性回执在server_only/git_publish_receipt.json。

128次新增正式生成已完成，GPU已释放，无遗留计算进程，服务器保持运行；未自动扩大样本、继续其他配置或运行性能测试。
