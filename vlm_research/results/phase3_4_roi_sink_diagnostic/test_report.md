# Phase 3.4 测试与验收记录

全部真实GPU阶段完成，详细机器可读证据见correctness_validation.json、observation_validation.json、各完成门和逐样本检查点。

| 检查 | 真实结果 |
|---|---|
| GPU之前CPU回归 | 83项通过，logs/cpu_tests.txt |
| 最终CPU回归 | 85项通过，logs/final_cpu_tests.txt；旧72项+新增13项 |
| 环境依赖 | pip check通过；沿用原环境/权重 |
| ROI映射 | 240/240真实Processor pixel_values逐元素一致，CLIP embedding行优先顺序验证 |
| 样本隔离 | 主48左右各24，独立历史案例5，主统计不混入案例 |
| 观察 | 53图，所选行[32,32,2,576]，Attention开启/关闭预测和Question hidden精确相同 |
| 7B Smoke | 3图、96次Attention审计Forward、3072层矩阵，目标连接严格0 |
| 非目标及因果 | Mask clone不改非目标，只有预定层窗生效；A-only Question状态精确Full |
| 行和 | Smoke最大误差0.000383377；观察最大0.000414848 |
| 历史基线 | 53图原Prompt/候选映射、原生Full分数与Phase3.2精确相同 |
| 正式与Smoke | 同样本Margin/logP/Question drift精确一致 |
| Hook清理 | 正常/异常CPU测试及每图GPU原生后检均通过，无后续Forward污染 |
| 数量控制 | 主48全部三seed控制精确同Patch数量；空间匹配不足单独披露 |
| 不可用组合 | 仅案例0216546的12个ROI随机组合；无主样本缺失 |
| 精度诊断 | 五案例55条件，同FP16 hidden的CPU FP32输出投影，0分类变化；不是全FP32复验 |
| 有限性 | 真实GPU无NaN/Inf/OOM，所有保存数组有限 |

首次observe启动在加载前遇到资源API返回值错误，失败日志和修复说明已保留；后续所有阶段通过。阶段完成门已绑定冻结实现/数据，最终artifact_audit.json另核对文件计数、SHA256、检查点、随机控制和实际序列长度。未修改第三方源码或旧实验结果，也未终止其他GPU任务。
