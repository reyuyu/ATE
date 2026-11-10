# Qwen3-VL / SVH 连续研究报告

| Benchmark | Native | SVH | Native/SVH成功 | OOM N/S | 配对有效/计划 | ΔSVH [95%CI] | 完整指标 | 数值/结构 |
|---|---:|---:|---|---|---|---|---|---|
| GQA | 61.3785 | 52.8854 | 10432/10432 | 0/0 | 10432/12578 | -8.4931 [PENDING, PENDING] | False | RELAXED / 结构PASS |
| ScienceQA-IMG | PENDING | PENDING | 0/0 | 0/0 | 0/2017 | PENDING [PENDING, PENDING] | False | RELAXED / 结构PASS |
| TextVQA | PENDING | PENDING | 0/0 | 0/0 | 0/5000 | PENDING [PENDING, PENDING] | False | RELAXED / 结构PASS |
| POPE | PENDING | PENDING | 0/0 | 0/0 | 0/9000 | PENDING [PENDING, PENDING] | False | RELAXED / 结构PASS |
| MME | PENDING | PENDING | 0/0 | 0/0 | 0/2374 | PENDING [PENDING, PENDING] | False | RELAXED / 结构PASS |
| MMBench | PENDING | PENDING | 0/0 | 0/0 | 0/4377 | PENDING [PENDING, PENDING] | False | RELAXED / 结构PASS |

| SVH-R候选 | 层集合/K | 六项得分 | 类型/KV/速度 | 阶段 |
|---|---|---|---|---|
| 尚未开始 | 冻结SVH后才执行 | PENDING | 不宣称物理节省或加速 | 等待第一项终态 |

当前检查点：{'COMPLETE': 20864, 'PENDING': 49827, 'RUNNING': 1}。第一项状态：PARTIAL。
表中配对分数只使用共同成功且MME双题/MMBench Circular组完整的集合。覆盖不足时不是正式全量指标，缺失样本没有按错题补零。
方案B，非Qwen作者官方复现；方案A仍BLOCKED。原0.02严格数值FAIL保留，质量结果不能证明严格数值等价或实际加速。
周期数值检查：0/92次严格FAIL；这是抽检分母，不是全部质量题目的失败率。
本轮原视觉/语言Eager、BF16、原像素预算；OOM跳过后保留全部计划分母。Bootstrap按图像簇及必要完整组，不把随机层集合当作独立图片。
MME为14任务Acc+Acc+总分；MMBench为冻结ATE Circular离线版本，不是作者V1.1。POPE主视图9000，历史8910辅助视图另行校验。
原始数据：/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090/vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench；逐条完整输入绑定和预测在分卡SQLite（zlib无损JSON+SHA），不上传公共仓库。
