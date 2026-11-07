# RTX 3090 Physical C3-V2 结果证据

本轮五项正式评测 COMPLETE：22,678/22,678 条；真实 GPU 物理正确性 Smoke PASS。
HF Native 既有 15 条 Smoke 的独立 CPU 评分审计 PASS，原评分全部正确；这不是正式 Native 基线。

- 正式结果：[report_zh.md](report_zh.md)、[score_summary.json](score_summary.json)、[run_metadata.json](run_metadata.json)。
- 物理执行门禁：[physical_smoke_summary.json](physical_smoke_summary.json)。完整跟踪的服务器路径、SHA、大小保存在摘要中。
- 独立审计：[审计报告](../featurize_new3090_scoring_audit/report_zh.md)及逐题 CSV/JSON。
- 既有 Native Smoke：[原报告](../featurize_new3090_gpu_smoke/report_zh.md)、15 条原始输出与 Token 的 smoke_results.json。
- c3_runner.py / finalize.py 是本轮入口与核对脚本快照，依赖服务器已恢复的数据、REC/scripts/ate_data.py 和冻结 Manifest；运行配置见元数据，不是新的通用安装入口。

实验代码固定在 e281754ef17b3accc09d6938b5b9f020048b2aec。本证据分支仅补充结果。
MME 保留已知单图版本差异；无完整同协议 HF Native 对照，不计算配对差值。

完整逐题预测 predictions.sqlite、predictions.jsonl.gz、映射 Manifest 和原始物理跟踪均保留在服务器
/home/featurize/work/ATE/vlm_research/results/featurize_new3090_c3v2_evaluation/，未纳入 Git。
evidence_manifest.json 记录此次白名单文件以及服务器保留文件的大小、SHA。

用户于 2026-11-08 明确要求：后续每次实验结束后同步轻量结果至 GitHub。
同步范围为简洁报告、配置、小型评分证据及必要代码；完整预测、数据、模型和大日志留在服务器。
未完成的实验必须保持真实 PARTIAL / BLOCKED 状态；后续用户的明确要求优先。
