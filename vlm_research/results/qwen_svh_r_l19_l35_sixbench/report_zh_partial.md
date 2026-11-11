# qwen-SVH-R L19-L35 六项固定配置评测

## 协议与边界

- 方案B：ATE统一HF；不是作者官方复现。方案A继续BLOCKED。
- P−1与全部真实Decode统一在0-based L19–L35共17层回看。R写入L0–L21，主路径视觉Hidden在L22前退出，L1–L6答案Attention跳算不变。
- ScienceQA/MMBench采用冻结P2短答案提示；TextVQA保留Paper-OCR；POPE9000；MMBench为ATE内部legacy Dev Circular，不称V1.1。
- 沿用选层时的功能性质量参考，包含真实因果前缀sidecar。未验证本配置物理加速，不宣称17层持久KV或E2E收益。
- 历史严格数值FAIL_RETAINED未被改判。按用户近期指示不重复Smoke；CPU绑定通过不等于GPU数值门禁通过。
- 固定候选来自重复探索的Discovery，扩展队列不是完全独立外部确认，CI跨0不等于无损。

## 分数与覆盖

| 任务 | Native完整/计划 | SVH完整/计划 | qwen-SVH-R完整/计划 | Native | SVH | qwen-SVH-R |
|---|---:|---:|---:|---:|---:|---:|
| GQA | 12578/12578 | 12578/12578 | 12578/12578 | 61.6394 | 52.6554 | 61.1385 |
| ScienceQA-IMG | 2017/2017 | 2017/2017 | 2017/2017 | 94.7942 | 94.4472 | 94.5959 |
| TextVQA | 4995/5000 | 4995/5000 | 118/5000 | BLOCKED/PARTIAL | BLOCKED/PARTIAL | BLOCKED/PARTIAL |
| POPE | 9000/9000 | 9000/9000 | 864/9000 | 89.1222 | 89.4222 | BLOCKED/PARTIAL |
| MME | 1968/2374 | 1976/2374 | 0/2374 | BLOCKED/PARTIAL | BLOCKED/PARTIAL | BLOCKED/PARTIAL |
| MMBench | 4377/4377 | 4377/4377 | 4377/4377 | 84.1837 | 84.0136 | 84.2687 |

MME单位为14任务Acc+Acc+总分，其余为百分比，MMBench按完整Circular组。
全量分数仅在全部必需题/组完成时给出。OOM不计错误、不更换分母；部分结果仅可见available/common subset字段，不冒充正式完整分数。

## 同题配对变化

| 任务 | 对照 | 共同题数 | 差值 | 图片簇95%CI | 损害/纠正 | 范围 |
|---|---|---:|---:|---|---:|---|
| GQA | Native | 12578 | -0.5009 | NOT_AVAILABLE | 202/139 | FULL |
| GQA | SVH | 12578 | +8.4831 | NOT_AVAILABLE | 253/1320 | FULL |
| ScienceQA-IMG | Native | 2017 | -0.1983 | NOT_AVAILABLE | 6/2 | FULL |
| ScienceQA-IMG | SVH | 2017 | +0.1487 | NOT_AVAILABLE | 1/4 | FULL |
| TextVQA | Native | 118 | -2.6271 | NOT_AVAILABLE | 6/3 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| TextVQA | SVH | 118 | +17.3729 | NOT_AVAILABLE | 6/28 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| POPE | Native | 864 | +0.2315 | NOT_AVAILABLE | 4/6 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| POPE | SVH | 864 | -0.4630 | NOT_AVAILABLE | 6/2 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| MME | Native | 0 | N/A | NOT_AVAILABLE | 0/0 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| MME | SVH | 0 | N/A | NOT_AVAILABLE | 0/0 | PAIRED_COMPLETE_COMMON_SUBSET_ONLY |
| MMBench | Native | 4377 | +0.0850 | NOT_AVAILABLE | 4/6 | FULL |
| MMBench | SVH | 4377 | +0.2551 | NOT_AVAILABLE | 2/5 | FULL |

最终统计按Image SHA整簇、问题等权、20,000次配对Bootstrap，固定种子20261111；MME保留双题、MMBench保留完整Circular组。
TextVQA损害/纠正表示软分下降/上升；MME子任务与POPE三策略另存CSV。不同任务不平均。

## 完成状态

- PARTIAL_SNAPSHOT_RUNNING。
- 原始数据、模型和历史结果只读复用；新逐题检查点写入本结果目录。
- 本轮没有新层搜索、性能Benchmark、缩图、精度/后端切换。
- 全部原始失败保留；运行日志、协议与源码SHA可复核。

## 路径

/data/lk/ATE-worktrees/qwen-svh-r-l19-l35-sixbench/vlm_research/results/qwen_svh_r_l19_l35_sixbench

冻结配置：QWEN_SVH_R_FIXED.md。复现：qwen_svh_r_full.py --freeze / --launch；summarize_qwen_svh_r.py。
