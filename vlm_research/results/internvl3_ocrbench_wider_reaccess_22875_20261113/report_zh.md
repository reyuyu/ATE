# InternVL3-8B 扩大视觉回看范围：OCRBench100 固定小样本

状态：**COMPLETE_EXPLORATORY_QUALITY**。只运行预先固定的三组各100题；没有重跑Native、Physical-D、Late9，也没有扩大到1000题或测量效率。

本轮固定100题中，扩大回看范围将Late9的57分恢复到76–77分：Late16=76，Late20=77，All24=77，Native=79。Late20相对Late9净增20题（21题改善、1题退步），相对Native仍少2题；All24没有额外增加得分。因此L8–L27是值得后续独立验证的较小回看集合，本轮不自动扩大评测，也不宣称两方法等价或有效率收益。

## 同一批100题结果

| 方法 | 回看层（0-based） | 得分 /100 | 相对Late9（pp） | 空输出 | 截断 |
|---|---|---:|---:|---:|---:|
| Native | 原生完整 | 79 | +22 | 0 | 0 |
| Physical_D | 无回看 | 5 | -52 | 0 | 0 |
| Late9 | L19–L27 | 57 | +0 | 0 | 0 |
| Late16_L12_27 | L12–L27 | 76 | +19 | 0 | 0 |
| Late20_L8_27 | L8–L27 | 77 | +20 | 0 | 0 |
| All24_ExceptL2_5 | L0–L1及L6–L27 | 77 | +20 | 0 | 0 |

这里的分数是100题探索性结果，不能乘十当作OCRBench1000正式分数。既有全量结果为Native751、Physical-D32、Late9563；本轮未重复生成它们。

## 配对比较

| 新方法 | 对照 | 差值pp | 错→对 | 对→错 | 按图片SHA簇Bootstrap 95% CI（pp） |
|---|---|---:|---:|---:|---|
| Late16_L12_27 | Late9 | +19 | 21 | 2 | [10.68, 27.72] |
| Late16_L12_27 | Native | -3 | 0 | 3 | [-7.00, 0.00] |
| Late20_L8_27 | Late9 | +20 | 21 | 1 | [11.88, 28.28] |
| Late20_L8_27 | Native | -2 | 0 | 2 | [-5.05, 0.00] |
| All24_ExceptL2_5 | Late9 | +20 | 21 | 1 | [11.88, 28.28] |
| All24_ExceptL2_5 | Native | -2 | 0 | 2 | [-5.05, 0.00] |

固定seed=20261113，10000次图片簇配对Bootstrap，共98个图像SHA簇。区间未经多重比较校正；样本小，同批三组用于探索，不用于证明非劣效或挑出配置后宣称独立验证。逐题改善与恶化均保存在changed_score_cases.csv。

## GitHub历史依据与设计

- [All24历史TextVQA实验](https://github.com/reyuyu/ATE/commit/5219e469d8b65f90f84669b29126ad573b6bfae9)：相同TextVQA128上All24=54.765625%，Late9=50.703125%。这是Image-only TextVQA，不是OCRBench。
- [终端窗口实验](https://github.com/reyuyu/ATE/commit/1407275cbdffca5d4a23cbf8b8d233002e2840dd)：保留到L27的晚层信息有帮助；单纯扩展较早中层并不保证恢复。
- 本轮预先冻结Late16、Late20、All24三组；共同终点L27、cutoff28、R读图边界20与答案Attention跳算L2–L5保持不变，只改变P−1及Decode允许读取原始视觉KV的层集合。
- GPU前检查了GitHub历史分支、服务器实验目录与Windows已有备份目录。找到的宽窗口结果是TextVQA，未发现这三组OCRBench预测；三组基线各1000条则已存在并直接抽取。

## 输入与评分

按官方10个类别各取10%，由SHA256(seed|题号|图片SHA)排序确定：六类文字识别各5题，Scene/Doc/Key Information各20题，数学10题。保留原始顺序；选题不依赖Gold或预测。完整100题ID、Prompt、Token IDs、图片SHA和选取规则见protocol.json与inputs_light.jsonl.gz。

100题输入绑定逐项核对冻结图片、Prompt、Gold、input_ids、BF16 pixel_values；Gold独立保存。初次CPU准备遇到单答案scalar与历史规范化singleton-list的序列化表示差异后停止，改为比较相同答案列表，未改变任何答案或评分规则，未因此重复GPU生成。

使用原OCRBench v1官方评分循环并独立实现答案包含匹配：普通数据转小写、去首尾空白并换行转空格；HME100k区分大小写且移除空格/换行。600条预测全部重新核算且两种评分实现一致；新增300条Token IDs在CPU解码与保存原文一致。

## Physical与数值检查

每组4题结构Smoke（0000、0560、0743、0926），共12次生成计入各组100题。检查了真实视觉Prefill、允许层读图概率>0、禁止层和L20之后R读图概率=0、L2–L5答案Attention/QKV物理跳算、Norm/MLP正常、RoPE原始位置、独立KV、持久视觉KV仅在K集合保留，以及L27后/Final Norm前物理移除256行Hidden。全部通过；OOM/NaN/异常为0。

每组首题0000仅生成2Token（包含EOS），另做同前缀Dense参考，共3次：这2步全部logits及共享Hidden逐元素一致，最大差值/相对RMS均0。未降低旧阈值；**该短参考仅覆盖这一题，不能消除历史严格门禁失败，也不能证明全局数值等价**。本轮继续明确标注历史EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。

直接导入历史冻结终端Physical执行器，核验全部59个来源文件SHA256，未修改核心。该实现有物理持久KV裁剪，但使用瞬时零scratch保持原Attention归约宽度；cutoff28意味着所有28层Prefill仍处理视觉Hidden。因此本轮不能据此声称延迟、FLOPs或KV收益，未运行任何效率测试。

## 环境与限制

真实GPU：NVIDIA GeForce RTX 3090, GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, 1 MiB, 24123 MiB。模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645。模型仅加载一次。

继承ATE_SVH_1TILE_SCORER_ALIGNED协议：RGB单448×448图块/256视觉Token，原Chat Template及短答后缀不变；未混入官方动态max_num24图块协议。该预处理仍可能限制OCR质量，不能把恢复不足全部归因于回看层。

执行结束后GPU无遗留计算进程；不关闭服务器。轻量报告、配置、逐题结果、统计、来源哈希与执行器外围代码在独立Git分支；原始记录已归档到服务器持续盘及Windows D:/ATE-dataset-backups-3090/ATE-results/internvl3-ocrbench-wider-reaccess-22875-20261113/full_results.tar.gz，双端SHA256核验通过，见backup_receipt.json。GitHub提交与远端核验回执另保存在同一Windows备份目录。

## 分类得分（正确题数）

| 类别 | 题数 | Native | Late9 | Late16 | Late20 | All24 |
|---|---:|---:|---:|---:|---:|---:|
| Artistic Text Recognition | 5 | 5 | 5 | 5 | 5 | 5 |
| Digit String Recognition | 5 | 5 | 4 | 5 | 5 | 5 |
| Doc-oriented VQA | 20 | 12 | 11 | 10 | 11 | 11 |
| Handwriting Recognition | 5 | 4 | 4 | 4 | 4 | 4 |
| Handwritten Mathematical Expression Recognition | 10 | 9 | 5 | 9 | 9 | 9 |
| Irregular Text Recognition | 5 | 5 | 5 | 5 | 5 | 5 |
| Key Information Extraction | 20 | 12 | 6 | 12 | 12 | 12 |
| Non-Semantic Text Recognition | 5 | 4 | 3 | 4 | 4 | 4 |
| Regular Text Recognition | 5 | 5 | 5 | 5 | 5 | 5 |
| Scene Text-centric VQA | 20 | 18 | 9 | 17 | 17 | 17 |
