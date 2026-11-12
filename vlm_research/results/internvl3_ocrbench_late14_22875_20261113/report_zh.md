# InternVL3-8B Late14：同一OCRBench100追加验证

状态：**COMPLETE_EXPLORATORY_QUALITY**。Late14明确为0-based L14–L27，共14层回看。本轮仅新增该条件100次生成，得分 **77/100**；其他六组600条逐题输出全部复用。

| 方法 | 回看层（0-based） | 得分 /100 | 空输出 | 截断 |
|---|---|---:|---:|---:|
| Native | 原生完整 | 79 | 0 | 0 |
| Physical_D | 无回看 | 5 | 0 | 0 |
| Late9 | L19–L27 | 57 | 0 | 0 |
| Late14_L14_27 | L14–L27 | 77 | 0 | 0 |
| Late16_L12_27 | L12–L27 | 76 | 0 | 0 |
| Late20_L8_27 | L8–L27 | 77 | 0 | 0 |
| All24_ExceptL2_5 | L0–L1及L6–L27 | 77 | 0 | 0 |

本批样本上，Late14以14个回看层达到与Late20、All24相同的77分，逐题得分向量也一致，但分别有10条、8条回答文本不同。它相对Late9恢复21题、恶化1题，相对Native仍少2题。Late14与Late20/All24的配对Bootstrap区间为[0,0]只是本样本差分全为0的结果，不能推断总体方法等价。可把Late14作为后续独立验证候选，但本轮不再扩展，也没有验证效率收益。

## 配对比较

| 对照 | Late14减对照（pp） | 错→对 | 对→错 | 图片簇Bootstrap 95% CI（pp） |
|---|---:|---:|---:|---|
| Native | -2 | 0 | 2 | [-5.05, 0.00] |
| Late9 | +20 | 21 | 1 | [11.88, 28.28] |
| Late16_L12_27 | +1 | 1 | 0 | [0.00, 3.06] |
| Late20_L8_27 | +0 | 0 | 0 | [0.00, 0.00] |
| All24_ExceptL2_5 | +0 | 0 | 0 | [0.00, 0.00] |

配对Bootstrap按98个图片SHA簇重采样10000次，seed=20261113，未做多重比较校正。Late14是看到此前100题结果后的追加窗口，因此本次仍是同批样本上的探索，不能当作独立验证、全量OCRBench成绩或统计非劣效。差异不显著也不等于方法等价。

## 输入与实现不变

完整复用[上一轮提交](https://github.com/reyuyu/ATE/commit/d47fc0d20d26df01d42b89754f50df3d9ea5dd62)的100题ID、顺序、图片SHA、Gold、Prompt、input_ids、BF16 pixel_values、选择seed和10类比例；未重新挑选样本。只把P−1与Decode允许视觉回看的层集合设为L14–L27。R读图边界仍为20、答案Attention跳算仍为L2–L5、cutoff仍为28。原始终端Physical核心59个来源文件SHA256逐项核验，未修改冻结源码。

模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645；ATE_SVH_1TILE_SCORER_ALIGNED，单448×448图块/256视觉Token，原Chat Template与短答后缀不变。模型只加载一次，模型/数据/环境/运行文件均在本地盘。

## Smoke、数值与评分

4题Smoke为0000、0560、0743、0926，全部通过，并计入100题。真实视觉输入、允许层视觉Attention概率>0、禁止层及L20后R视觉Attention=0、L2–L5答案Attention/QKV跳算、Norm/MLP、RoPE原始位置、独立KV和K集合选择性持久视觉KV均通过，视觉Hidden在L27后/Final Norm前删除。

首题0000只生成2Token（包含EOS），另做1次同前缀Dense参考；两步logits与共享Hidden逐元素一致。原阈值未改；该极短参考不能消除历史全局数值等价失败。本轮保留EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，不声称全局等价。OOM、NaN及执行异常为0。

700条新旧预测均在CPU使用冻结OCRBench v1官方评分循环与独立包含匹配核算；普通数据大小写不敏感，HME100k区分大小写并去除空格/换行，全部一致。新增100条Token IDs离线解码与保存原文一致；没有丢弃错误、空输出或截断。旧600条不重复上传完整内容，其来源文件哈希见protocol.json、配对分数见per_question_scores.csv，新增逐题原文见predictions_Late14.jsonl.gz。

终端cutoff28意味着全部28个语言层Prefill仍处理视觉Hidden；非保留层使用瞬时零scratch保持Attention归约宽度，仅对持久视觉KV作物理选择性保留。本轮只验证质量，不测试也不声称FLOPs/KV/延迟收益。

真实GPU及释放后状态：NVIDIA GeForce RTX 3090, GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, 1 MiB, 24123 MiB；无遗留计算进程。不关闭服务器，不增加窗口、样本或Benchmark。

完整原始记录、逐题审计和日志会保存在独立归档，Windows和持续盘的真实SHA256核验见backup_receipt.json；轻量结果在独立Git分支，main和旧结果不修改。

Windows完整原始结果备份已完成，路径：D:\ATE-dataset-backups-3090\ATE-results\internvl3-ocrbench-late14-22875-20261113\full_results.tar.gz；SHA256：a9434eeeb67651123ab1afbc1bde7d4df4f817daf8218689f261a8db2a77bb83。服务器持续盘副本同SHA，见backup_receipt.json。GitHub远端核验回执保存在同一Windows备份目录。
