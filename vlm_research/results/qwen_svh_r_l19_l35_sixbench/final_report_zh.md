# qwen-SVH-R L19–L35：六项固定配置质量结果

**配置已固定：P−1及全部真实Decode统一在0-based L19–L35回看，共17层。**

采用方案B（ATE统一HF），不是作者官方复现。ScienceQA/MMBench为冻结P2；其余Prompt与视觉预算不变。
R写入L0–L21、主路径L22前视觉退出、L1–L6答案Attention跳算均保持不变。

## 分数

| Benchmark | Native | SVH | qwen-SVH-R | 覆盖 |
|---|---:|---:|---:|---|
| GQA | 61.6394 | 52.6554 | 61.1385 | 完整 |
| ScienceQA-IMG | 94.7942 | 94.4472 | 94.5959 | 完整 |
| TextVQA | 80.1862（共同子集） | 58.0981（共同子集） | 78.9129（共同子集） | 不完整；三方共同有效4995题 |
| POPE | 89.1222 | 89.4222 | 89.3222 | 完整 |
| MME | BLOCKED：无完整总分 | BLOCKED：无完整总分 | BLOCKED：无完整总分 | 不完整；三方共同有效1898题 |
| MMBench | 84.1837 | 84.0136 | 84.2687 | 完整 |

表中POPE为Accuracy，MMBench为完整Circular准确率；TextVQA为十答案软分。MME应为14任务Acc+Acc+总分，不用普通准确率代替。

### POPE F1

| 策略 | Native | SVH | qwen-SVH-R |
|---|---:|---:|---:|
| ALL | 88.6387 | 89.0323 | 88.9017 |
| random | 90.9936 | 91.4773 | 91.3167 |
| popular | 88.4028 | 88.8276 | 88.6969 |
| adversarial | 86.6281 | 86.9096 | 86.8065 |

## 逐题配对变化

| 任务 | 对照 | R减对照 | 图片簇95%CI | 损害/纠正 | 同题数 |
|---|---|---:|---|---:|---:|
| GQA | Native | -0.5009 | [-0.8659, -0.1465] | 202/139 | 12578 |
| GQA | SVH | +8.4831 | [7.4904, 9.5006] | 253/1320 | 12578 |
| ScienceQA-IMG | Native | -0.1983 | [-0.4873, 0.0503] | 6/2 | 2017 |
| ScienceQA-IMG | SVH | +0.1487 | [-0.0500, 0.3941] | 1/4 | 2017 |
| TextVQA | Native | -1.2733 | [-1.8097, -0.7467] | 210/133 | 4995 |
| TextVQA | SVH | +20.8148 | [19.5565, 22.0753] | 161/1314 | 4995 |
| POPE | Native | +0.2000 | [-0.0222, 0.4444] | 15/33 | 9000 |
| POPE | SVH | -0.1000 | [-0.3111, 0.1111] | 25/16 | 9000 |
| MME | Native | N/A | N/A | 3/9 | 1898 |
| MME | SVH | N/A | N/A | 4/2 | 1898 |
| MMBench | Native | +0.0850 | [-0.2586, 0.4288] | 4/6 | 4377 |
| MMBench | SVH | +0.2551 | [0.0000, 0.5932] | 2/5 | 4377 |

可定义的统计量采用20,000次Image SHA整簇配对Bootstrap；保持MME双题、MMBench完整Circular。MME因缺少完整14任务覆盖，总分及总分CI均不可计算，不插补。TextVQA损害/纠正为软分下降/上升。
普通探索性区间，不据CI跨0宣称无损、等效或非劣效；本队列包含选层时重复使用的Discovery图片。

## 主要结论

- TextVQA共同4995题相对SVH恢复+20.8148pp，但相对Native仍为-1.2733pp，95%CI [-1.8097, -0.7467]。不能宣称无损。
- GQA、ScienceQA、MMBench和POPE均得到完整质量分数；TextVQA仅有4995题共同有效子集，MME没有可比较的完整14任务总分。
- MME中有51条Native已有有效预测、本轮参考路径却OOM的记录，不能把所有缺失都解释为历史基线限制；具体算子线索见oom_attribution.csv和原始Trace。

## 完整性与局限

- 所有35,346个固定输入均已核验并处理，含292条严格复用；实际成功、OOM与分组覆盖逐项保存在accuracy_summary.csv。
- 原Eager Native在TextVQA已有5条OOM、MME已有406条OOM；历史失败未删除或改判。新配置失败同样保留。
- 未缩图、换精度/后端、改评分器或丢弃失败题。缺失不会记错后混入分数，也不会用可用子集冒充完整测试。
- 选层时与本轮使用同一功能质量参考，含因果视觉前缀sidecar；参考保留30个active层视觉KV，不能据17层回看声称17层持久KV或真实加速。
- 历史严格数值FAIL_RETAINED继续保留；没有新增数值PASS，没有运行性能Benchmark。
- GQA的Query/Non-Query见gqa_query_nonquery.csv；POPE各项指标及F1区间、MME子任务、全部逐题答案与负例另存CSV。
- 回看改善属于答案直接访问视觉的收益，不能称原SVH已完成无损信息交接。单一固定配置结果不证明层集合全局最优。

## 资源与复现

- 本任务GPU1/GPU3的两个Worker已退出；GPU0/GPU2及其他进程未干扰。
- 双卡阶段墙钟7315.24秒，含模型加载和生成，不含CPU准备/统计；不是方法E2E延迟或加速比。
- 冻结代码Commit：706a36edc8d90ae98d1bc07c9d7849524c9f2993；分支codex/qwen-svh-r-l19-l35-sixbench。
- CPU入口：summarize_qwen_svh_r.py；gqa_svh_r_subgroups.py；finish_qwen_svh_r.py。
- 全部结果：/data/lk/ATE-worktrees/qwen-svh-r-l19-l35-sixbench/vlm_research/results/qwen_svh_r_l19_l35_sixbench

到此停止，不自动搜索新层集合或追加实验。
