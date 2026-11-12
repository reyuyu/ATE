# Section 5.5 GitHub 核心证据同步

本目录由用户在实验完成后明确授权上传 GitHub。冻结 `protocol.json` 的 `no_push: true`、原始报告中“未 Push”的表述描述实验执行阶段及当时状态，保留原文与原 SHA；此次后续授权只增加结果归档，不改变算法、预测、计时或统计，也不启动 GPU。

- 发布分支：`codex/section55-svh-ablation-length-3090-20261113`；未修改 main，未创建 PR。
- 实验执行与正确性：PASS；性能状态：`COMPLETE_WITH_PERFORMANCE_WARNINGS`。
- 质量：新增 32781 条自然 EOS 预测，复用已验证 Full SVH 10927 条，共 43708 条配对记录。
- 效率：两个独立 Session，各 360 条正式请求，共 720 条；所有 16/32/64/128 Token 均实际生成。
- 24 项校准检查中 12 项超过 2%；最大绝对 E2E 漂移 4.473%，TPOT 漂移 4.936%。未筛选最快 Session 或删除慢请求。

## 同步范围

本次同步原核心证据包的 65 个文件：中文报告、消融与 POPE 子集表、720 条逐请求效率数据、两个 Session 独立统计与配对区间、校准与警告、环境及 Smoke 证据、协议、源码与 SHA、矢量 PDF/PNG 和绘图源码。本说明为额外添加的发布说明。原 65 个文件逐个保持 SHA256 一致。

完整质量逐题预测、SQLite、连续 GPU/CPU 遥测及原始运行日志仍保存在服务器持久盘：

`/home/featurize/work/ATE/vlm_research/results/svh_ablation_efficiency_3090/`

`artifact_manifest.json` 覆盖服务器完整 115 个文件；它包含本次未放入 Git 的大文件条目，因此不能将该清单误读为 GitHub 上所有文件均已同步。模型权重、数据集图片和缓存不包含在本次提交中。

## 冻结来源与复现

计算时 Git HEAD：`e281754ef17b3accc09d6938b5b9f020048b2aec`；原效率参考：`5ee2a0dbe78df336229b55fed14a4a6d0d880bed`。当前发布分支基于远端 main 的 `2bda69c738c85537594858707ce7f6e1035dd59a`，仅增加结果目录；复现计算必须使用上述冻结执行源码及 SHA，不应直接假定当前 main 的模型/算法代码与实验时一致。

模型：`llava-hf/llava-1.5-7b-hf`，Revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。完整路径、环境版本与复现命令见 `README_reproduce_zh.md`、`source_manifest.json` 和 `report.md`。

原始轻量证据包 SHA256：`c23f9a530e4424cff81a2aff7a2f626d54d9d10a927f7cdb7d41da6e3f02806a`。
