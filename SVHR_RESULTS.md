# Qwen3-VL-SVH-R 阶段结果

2026-11-12：Discovery及Validation完成，Held-out因MME视觉编码器OOM停止。
**总体状态PARTIAL，不是最终完整实验。**

- [中文阶段报告](vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/progress_report_zh.md)
- [完整Validation汇总](vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/Validation_Unified_summary.csv)
- [逐题记录、测速与SHA256](vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/completed_snapshot/)
- [冻结协议](vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/protocol.json)

TextVQA Validation同题有效498题：Native 81.51%，原SVH 62.97%，Wide30 80.32%。
另有2题历史OOM保留。宽回看提升质量，但小规模自然EOS测速比Native更慢；
不宣称无损、非劣效、作者官方复现或总体真实加速。
