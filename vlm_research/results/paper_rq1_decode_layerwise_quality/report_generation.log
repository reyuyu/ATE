# 中文研究总结

Full: CIDEr 1.235110; SPICE 0.239601; TextVQA 46.900%.

COCO CIDEr损害最大：L0, L11, L18, L2, L24。SPICE损害最大：L0, L14, L20, L30, L21。TextVQA损害最大：L9, L27, L18, L31, L29。

三端点96项校正非劣效候选：无；仅探索，不认定安全。

COCO L0-off有176/200条达到64-Token上限；不能从早层Prefill/单步低敏感性推出持续Decode关闭安全。
L14–L16没有复现为COCO CIDEr损害峰。历史单步敏感性与完整生成质量没有稳定统一对应，详细离散层Pearson/Spearman及协议差异见报告。
本轮400样本×33条件=13,200次真实完整生成；Prefill和首Token原生，仅后续Decode单层Attention输出置零。所有MLP保持，三卡同样本配对。
单步原生Token LogP不等于自由生成质量；历史相关性见report.md。文本变化/平均损失小不证明语义安全，单层不推导多层，Hook置零不代表加速。
已保存全部结果、负向尾部、普通与96比较校正区间、六组PNG/PDF及真实图像案例；下一步候选交由研究决策，本轮不自动扩展。
