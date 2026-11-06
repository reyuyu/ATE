# 完整GQA val_balanced：C3-V2 vs L11–L24回看

状态：COMPLETE。共132,062题、10,234个Image ID、10,234个图片SHA簇，264,124条新生成。

| 条件 | Accuracy | 95%图片簇Bootstrap CI |
|---|---:|---:|
| C3-V2 | 65.2512% | [64.8842, 65.6123]% |
| C3-V2 + L11–L24回看 | 68.3338% | [67.9759, 68.6951]% |

回看相对C3-V2：+3.0826pp，配对95%CI [+2.9010, +3.2677]pp。
原对→错 3,464题；原错→对 7,535题。

## 固定算法
两条件均维持L0–L19原生R视觉写入、L2–L5仅P−1和Decode的物理Attention跳算；MLP/Residual/Norm不变。
C3-V2在L20前移除视觉Hidden，答案阶段无Image KV。
回看条件仅在L11–L24让P−1及真实Decode读图，保留这14层Image KV，视觉Hidden在L25前移除。
L20–L24的普通R依然不能新增读图，不是完整Native Prefill，也不是最近TextVQA的Native-Prefill Decode-only条件。

## 评分与边界
Greedy、max_new_tokens=32、自然EOS、H1末行LM Head、独立语言KV Cache。
沿用固定LLaVA GQA输出转换及官方短答案严格匹配；不使用候选LogP代理Accuracy。
20,000次按Image SHA整簇配对Bootstrap、问题等权；类型分组为描述性结果。
这是完整验证集val_balanced，**不是隐藏test/testdev**。图片复用本地MMEB版本，未证明与Stanford原图分辨率完全相同，不能直接当作官方排行榜复现。
没有复用历史3000题分数，也没有运行Native Full质量矩阵或性能测试。本轮仅回答两个固定C3条件的全验证集Accuracy。

详细结果见accuracy_summary.csv、paired_comparisons.csv、task_accuracy.csv及per_question.csv；负面案例全部保留。
源码、模型、输入、数据协议哈希见execution_freeze.json；CPU/GPU验证及资源释放记录保存在同目录。
