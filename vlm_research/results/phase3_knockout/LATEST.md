# Phase 3 最新运行

2026-10-27 23:10，按用户“开始继续phase3”手动启动并完成真实 7B GPU 实验。自动轮询仍关闭，cron 未恢复。

- 完成现有全部 9 条 QA × 8 个 cutoff；32 层 Hook/Mask 检查通过，K=32 logits 与基线最大绝对差 0。
- 44 项回归测试与 pip check 通过，无 CUDA OOM、NaN/Inf。
- 完整实验墙钟时间 42.201 秒；校验峰值 allocated 14344.958 MiB，正式 Forward 峰值 allocated 13649.285 MiB。
- 存在非单调变化与平均值掩盖关键词变化的现象；没有确定统一融合深度，不宣称可安全删除视觉 Token。

[完整分析报告](phase3_report.md) · [原始运行报告](manual_gpu_20261027_2310/phase3_report.md) · [资源与跨样本汇总](manual_gpu_20261027_2310/comparison_summary.json) · [准备阶段历史](phase3_preparation_report.md)

实验停止在 Phase 3。`/home/zsy/.local/state/ate-phase3-watch/status.json` 的 disabled 仅表示 watchdog 关闭，不表示本次手动实验未完成。
