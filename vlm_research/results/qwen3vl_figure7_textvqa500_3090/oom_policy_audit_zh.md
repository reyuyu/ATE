# Qwen Figure7 OOM 策略来源审计

2026-11-12；仅只读核查 GitHub 和已完成的本轮容量 Smoke。

- Commit 066c037f88931aeb73937e34b663b61de3f24ba5，qwen3vl_svhr_l11_l24_budget_search/progress_report_zh.md：历史 Validation 500题中498有效。36759、36747保留 BLOCKED，六条件合计12条。原始Validation_blocked_records.jsonl.gz已独立解压验证，不补零、不换题。
- Commit a9030774cf88205dd5b84b158ff0f7257e4ef3e8，qwen_svh_r_l19_l35_sixbench/report_zh_partial.md：TextVQA Native/SVH各4995/5000；完整分数仍 BLOCKED/PARTIAL，available/common subset另报。qwen_svh_r_full.py169及232–245行确认捕获 OOM、保留失败并清理自身缓存，不更换后台、精度或图片。
- 上述500题不是本轮Figure7的同一500题。独立核对ID：36759在本轮；36747不在本轮。全量另五例的完整ID清单未在上述轻量交付公开，不臆测其他重叠。
- 本轮36759已原图、BF16、Vision/LLM全Eager真实OOM；4800视觉Token，原始输入4833Token。失败在官方Vision eager QK矩阵分配，非Decode权限Mask。
- 用户明确确认沿用历史OOM策略：保持原500 Manifest、原图、全Eager。所有条件共用完整成功交集评分，标明N/500，保留BLOCKED，不补零、不换题、不将子集称为完整500结果。
- 历史SVH算法和Paper-OCR成绩不用于本轮；本轮严格沿用Figure7的Image-only、Native Prefill/P−1、仅真实Decode视觉访问权限。

来源：
https://github.com/reyuyu/ATE/blob/066c037f88931aeb73937e34b663b61de3f24ba5/vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/progress_report_zh.md
https://github.com/reyuyu/ATE/blob/a9030774cf88205dd5b84b158ff0f7257e4ef3e8/vlm_research/results/qwen_svh_r_l19_l35_sixbench/report_zh_partial.md
