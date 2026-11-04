# 跨模型完整答案Accuracy敏感性

两个模型按顺序、各自双卡完成，同一图片全部条件同卡配对。Native Full均为原生，不是All-Writer。
仅关闭一个语言层原始P−1的整个o_proj更新；后续Decode原生。图中阴影为各模型32/36项Bonferroni校正95%CI。

- LLAVA最大点估计损害L15：-0.8333 pp；校正后可检测损害层[26]。
- QWEN最大点估计损害L24：-1.4000 pp；校正后可检测损害层[24, 27, 34]。

原始Prompt、FP16/BF16、视觉布局及模型版本不同，不能把峰位差异单独归因于架构。
首Token与最终ACC峰位及相关性详见各模型report_zh.md和first_token_vs_accuracy.csv。
本轮不证明独立视觉读取模块、不证明单层可安全省略，也未测量物理跳层或加速。到此停止，不运行InternVL或联合条件。
