# SVH / SVH-R 跨模型物理优化：GitHub 交付

本分支在实验完成后，根据用户后续“推送github”的明确授权发布。原实验报告、源码、CSV、协议和验证文件保持原始字节及SHA256。原报告中的“未推送GitHub”、validation.json中的no_git_writes、原交付回执中的git_pushed=false描述实验结束时的状态；本次上传是后续独立操作，不改变实验事实。

- [正式中文报告](final_report_zh.md)
- [两轮正式效率均值/中位数](optimized_efficiency_32tokens.csv)
- [同Session同题配对比较](paired_comparisons.csv)
- [1200条正式逐请求数据及Token IDs](efficiency_per_request.csv)
- [真实逐层KV布局](kv_physical_audit.csv)
- [数值与完整输出回归](numerical_equivalence_report.md)
- [验证结果](validation.json)
- [上传/服务器保留文件清单](GITHUB_ARTIFACT_MANIFEST.json)

本轮完成1200个正式请求；全部使用GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6、原GQA15+TextVQA Paper-OCR15、固定32输出Token及31步真实Decode。Native/原版/优化版在各自Session内交错执行；全部慢样本和负结果保留。没有重跑质量Benchmark。失败的去补零D候选没有进入正式效率表，InternVL无稳定E2E收益的版本不建议替换默认原版。

## 代码与依赖

opt_b/、opt_c/、opt_d/、opt_e/保留完整独立适配包，code/保留实际Runner、CPU验证、统计、Profiler解析及全部版本。source_diffs/供直接审阅修改；execution_runner_versions.json绑定真实执行版本。初始build脚本是过程记录，最终独立包和SHA为权威，不要仅执行某个早期build脚本覆盖最终代码。

原始冻结源码清单含6390个文件，可发布源码归档source_archives/frozen_sources.tar.gz包含6386个文件。4个未参与本轮运行、且含凭据样式常量的外部测试/API评测脚本仅保留服务器原件，路径与原SHA见source_archives/excluded_unused_source_files.json；实际运行依赖未排除。归档不含模型权重、图片、数据集、密钥或系统缓存；保留原绝对路径对应的目录结构及逐文件SHA256。它们可以解压到新的普通目录，不会写入原始/home路径。

    cd vlm_research/results/svh_cross_model_physical_optimization
    python source_archives/verify_and_extract_sources.py --verify-only
    python source_archives/verify_and_extract_sources.py --extract-to /new/local/ate-source-restore

SHA校验原始字节，不以重新格式化代码或JSON代替原文件。runtime_source_snapshots/保存实际HF/模型代码，frozen_runtime_resources/保存必要小型资源；model_manifest.json记录模型Revision与可信完整回执，不含权重。官方已有许可证位于third_party_notices/。

原始复现上下文及运行命令见[reproduction_readme.md](reproduction_readme.md)、[run_commands.txt](run_commands.txt)、protocol.json及protocol_refinement_addendum.json。脚本保留冻结服务器的路径与GPU UUID断言；换实例复现时，应在新独立工作区恢复模型/图片/环境及归档源码，审查并统一更新路径与协议，不要覆盖已有输出，也不要把不同GPU的时间拼接。

[原18条件已发布证据](https://github.com/reyuyu/ATE/tree/108cbdc1e2f782b6b59bd6c87a44cd28868330e8/vlm_research/results/rtx3090_unified_efficiency_18conditions)是历史来源，最终冻结源码与SHA优先于旧分支默认文件。

## GitHub范围与服务器完整原件

GitHub包含轻量完整证据、所有逐请求/逐层CSV、数值与实际Audit Mode结果、环境记录、代码与冻结源码归档。新实验完整SQLite检查点、原始requests.jsonl、连续GPU/CPU/进程遥测及Profiler trace仅保存在服务器：

/home/featurize/work/ATE/vlm_research/results/svh_cross_model_physical_optimization/

GITHUB_ARTIFACT_MANIFEST.json逐文件区分GITHUB_FILE、GITHUB_SOURCE_ARCHIVE、SERVER_ONLY；不能声称SERVER_ONLY文件存在于GitHub。frozen_reference_evidence/内的历史逐题记录用于核对冻结原版输出，和本轮仅保存在服务器的新原始Checkpoint分开标记。

原SHA256SUMS和evidence_manifest.json对应服务器完整8604文件，故其中部分路径不会以原始文件形式出现在GitHub。检查本次GitHub发布内容请使用：

    sha256sum -c GITHUB_SHA256SUMS

GITHUB_SHA256SUMS不自包含自身，以避免循环哈希。归档内容通过独立解压流逐文件验SHA。GitHub Commit与远端核验回执另外保存在服务器结果目录旁的svh_cross_model_physical_optimization_github_sync_receipt.json，避免Commit自哈希循环。

本分支不修改main，不创建PR；本次同步不运行任何GPU任务。
