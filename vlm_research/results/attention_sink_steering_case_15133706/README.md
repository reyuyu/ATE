# 15133706：Attention Sink去偏与ROI增强（真实GPU完成）

[报告](report.md) · [31条件完整CSV](per_condition_results.csv) · [冻结协议](protocol.json) · [数值复核](artifact_validation.json) · [全部31热图](heatmap_gallery.md)

中层B逆权重去偏变差；中层ROI增强微小改善但仍错；两个深层窗口Sink抑制均未复现Physical K24转正。只有L2～L4对同一非ROI Sink的P−1抑制转正（factor.5/.1 Margin +.0078125/+.1015625），三背景对照未转正。仅单例FP16探索，不能推断泛化或无损性。

![目标与随机对照](target_vs_matched_controls.png)

![早中深空间分布](offline_spatial_comparison_clean.png)

`gpu_attempts/20261028_230938/`保存实际31条件、完整原始logits、逐Head局部前/后行、396审计、GPU smoke和环境。历史实验不变；本轮完成后停止。
