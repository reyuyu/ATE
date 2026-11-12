# interVL-SVH-R-new：OCRBench v1 全1000题

状态：**COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER**。实际完成1000/1000题、930张唯一原图，得分 **717/1000（71.7%）**。继承上一轮Late14的100条原始预测，仅新增缺失900条；没有重跑Native、Physical-D、Late9或其他条件。

## 命名与配置

正式配置名按用户指定保留为`interVL-SVH-R-new`。模型为InternVL3-8B，Late14=L14–L27（0-based，共14层）允许P−1与真实Decode回看原始视觉KV；R仍在L0–L19读图，L2–L5答案Attention/QKV物理跳算，视觉Hidden在L27后/Final Norm前退出，选择性持久KV与独立Cache不变。命名不改变算法，也不代表全局数值等价门禁已经通过。

[运行前命名与协议提交](https://github.com/reyuyu/ATE/commit/4b5b1f7899a8a60c17a1d48280a1ad25f76d60c9)；配置文件为`vlm_research/configs/interVL-SVH-R-new.json`，SHA256=80d0a2e80c56d1ad9f8c3d7b22ef5d55f68cc2e83d12b47d4e4e23b716b279a5。

## 全集正式分数（原ATE单图块协议）

| 方法 | 覆盖 | 得分 /1000 | Accuracy | 空输出 | 截断 |
|---|---:|---:|---:|---:|---:|
| Native | 1000 | 751 | 75.1% | 0 | 0 |
| Physical_D | 1000 | 32 | 3.2% | 0 | 3 |
| Late9 | 1000 | 563 | 56.3% | 0 | 0 |
| interVL-SVH-R-new | 1000 | 717 | 71.7% | 0 | 0 |

对照均为同一模型Revision、1000题、930张原图、单448图块/256视觉Token、Prompt、生成上限与评分器；只复用已有逐题结果，不使用官方动态24图块新协议，不把不同协议公开论文分数当作直接对照。

## 配对差异

| 对照 | 新方法减对照（分） | 差值pp | 错→对 | 对→错 | 图片SHA簇Bootstrap 95% CI（pp） |
|---|---:|---:|---:|---:|---|
| Native | -34 | -3.4 | 11 | 45 | [-4.90, -2.00] |
| Late9 | +154 | +15.4 | 165 | 11 | [13.10, 17.79] |

930个图片SHA簇、10000次配对Bootstrap、seed=20261113，未做多重比较校正。Late14是依据此前100题探索选择的配置，且该100题包含在本次1000题中；因此全量覆盖并不等于独立留出验证，也不能由未显著差异推断统计等价或非劣效。逐题改善与恶化均保留。

## 输入、评分与Smoke核验

原1000题Question ID、顺序、Question、Gold、Image ID及类别全部与原始OCRBench注释一致。930张图片沿用此前可信SHA清单，不重复全盘哈希；生成时逐题核对实际图片、Prompt、input_ids和BF16 pixel_values绑定。继承100条的原文件SHA及旧协议都逐项复核，未重新生成。Gold与模型输入独立保存，Gold未拼入Prompt。

4道新题0001、0500、0700、0900的真实GPU结构Smoke通过，并计入新增900条。允许回看层视觉Attention概率>0，禁止层及L20后R的视觉Attention=0；物理Attention跳算、Norm/MLP、RoPE原始位置、独立Cache、持久视觉KV层集合以及Hidden退出均通过。无OOM、NaN或结构异常。此前Late14首题2步Dense短参考一致的记录仅作为局部证据，本轮不重复生成该参考；原严格阈值与历史失败、EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER继续保留，不能宣称全局数值等价。

所有1000条原文和Token IDs保存，CPU重新解码一致；包括三个历史对照在内的4000条预测，冻结OCRBench v1官方评分循环与独立包含匹配实现逐项一致。普通类别小写化、首尾空白处理及换行转空格；HME100k区分大小写并去除空格/换行。错误、空输出、截断均未删除或用0分伪造补齐。

## 环境与边界

模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645、原Chat Template及短答后缀。实际模型仅加载一次；数据、模型、环境和运行目录均在本地盘。冻结59个来源文件SHA在运行前后核验，核心未修改。

该终端实现的全部28层Prefill仍处理视觉Hidden，非保留层用瞬时零scratch保持原Attention归约宽度，持久原始视觉KV则物理选择性保留。未进行KV/FLOPs/TPOT/延迟实验，不把回看层减少直接当作真实效率改善。

实际GPU及释放后状态：NVIDIA GeForce RTX 3090, GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, 1 MiB, 24123 MiB。无残留计算进程，不关闭服务器，不启动其他实验。

独立分支同步配置、全集轻量预测、输入哈希、评分、对照统计及外围执行源码；完整原始记录、逐题结构与日志单独归档，真实Windows和持续盘SHA256备份回执见backup_receipt.json。

Windows完整原始结果备份已完成，路径：D:\ATE-dataset-backups-3090\ATE-results\internvl-svh-r-new-ocrbench1000-22875-20261113\full_results.tar.gz；SHA256：69802fa222713aaa7f42d2d650bcfee51d28821ac4d76e3bce84fca45a9e7118。持续盘副本同SHA。
