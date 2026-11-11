# Qwen3-VL / SVH 连续研究报告

| Benchmark | Native | SVH | Native/SVH成功 | OOM N/S | 配对有效/计划 | ΔSVH [95%CI] | 完整指标 | 数值/结构 |
|---|---:|---:|---|---|---|---|---|---|
| GQA | 61.6394 | 52.6554 | 12578/12578 | 0/0 | 12578/12578 | -8.9839 [-10.0423, -7.9674] | True | RELAXED / 结构PASS |
| ScienceQA-IMG | 52.4046 | 19.7819 | 2017/2017 | 0/0 | 2017/2017 | -32.6227 [-34.9428, -30.3588] | True | RELAXED / 结构PASS |
| TextVQA | 80.1862 | 58.0981 | 4995/4995 | 5/5 | 4995/5000 | -22.0881 [-23.4117, -20.7833] | False | RELAXED / 结构PASS |
| POPE | 89.1222 | 89.4222 | 9000/9000 | 0/0 | 9000/9000 | 0.3000 [0.0444, 0.5778] | True | RELAXED / 结构PASS |
| MME | 2373.5613 | 2405.0463 | 1968/1976 | 406/398 | 1960/2374 | 31.4850 [-0.9816, 67.0512] | False | RELAXED / 结构PASS |
| MMBench | 72.3639 | 59.6939 | 4377/4377 | 0/0 | 4377/4377 | -12.6701 [-14.7669, -10.6383] | True | RELAXED / 结构PASS |

| SVH-R候选 | 层集合/K | 六项得分 | 类型/KV/速度 | 阶段 |
|---|---|---|---|---|
| 尚未开始 | 冻结SVH后才执行 | PENDING | 不宣称物理节省或加速 | 等待第一项终态 |

当前检查点：{'COMPLETE': 69878, 'SKIPPED_OOM': 814}。第一项状态：TERMINAL。
表中配对分数只使用共同成功且MME双题/MMBench Circular组完整的集合。覆盖不足时不是正式全量指标，缺失样本没有按错题补零。
方案B，非Qwen作者官方复现；方案A仍BLOCKED。原0.02严格数值FAIL保留，质量结果不能证明严格数值等价或实际加速。
周期数值检查：7/287次严格FAIL；这是抽检分母，不是全部质量题目的失败率。
本轮原视觉/语言Eager、BF16、原像素预算；OOM跳过后保留全部计划分母。Bootstrap按图像簇及必要完整组，不把随机层集合当作独立图片。
MME为14任务Acc+Acc+总分；MMBench为冻结ATE Circular离线版本，不是作者V1.1。POPE主视图9000，历史8910辅助视图另行校验。
原始数据：/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090/vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench；逐条完整输入绑定和预测在分卡SQLite（zlib无损JSON+SHA），不上传公共仓库。
