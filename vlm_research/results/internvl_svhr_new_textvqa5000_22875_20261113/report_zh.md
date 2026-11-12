# interVL-SVH-R-new：TextVQA5000

状态：COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER。5000/5000题、3166张唯一图片，VQA Soft Accuracy **64.512%**。仅新生成Late14，Native与Physical-D复用历史逐题结果；Late9引用已发布全集汇总。

## 协议与结果

TextVQA v0.5.1 validation，严格继承ATE冻结Image-only输入（无OCR文字）、单448×448图块/256视觉Token、原问题及短答指令、InternVL Chat Template、BF16/Eager、Greedy、max_new_tokens=32、EOS151645。使用10答案leave-one-out VQA软评分，不是二值Exact-match，也不是LLaVA Paper-OCR或动态24图块新协议。

| 方法 | 覆盖 | VQA Soft Accuracy | 空输出 | 截断 |
|---|---:|---:|---:|---:|
| Native | 5000 | 66.342% | 0 | 8 |
| Physical-D | 5000 | 10.598% | 0 | 1 |
| Late9 | 5000 | 56.004% | 历史汇总未载入 | 历史汇总未载入 |
| interVL-SVH-R-new | 5000 | 64.512% | 0 | 5 |

Late14相对已发布Late9：+8.508 pp。Late9为同冻结协议历史全集56.004%的汇总引用，本轮没有下载其40MB全Benchmark逐题表，也不把4370条changed-only案例误当全集；因此不提供Late9配对改善数量或置信区间。

| 对照 | Late14−对照 pp | 软分提高 | 软分降低 | 相同 | 图片SHA簇Bootstrap95% CI pp |
|---|---:|---:|---:|---:|---|
| Native | -1.830 | 227 | 331 | 4442 | [-2.625, -1.030] |
| Physical-D | +53.914 | 2981 | 149 | 1870 | [52.417, 55.421] |

Bootstrap以3166个图片SHA簇配对采样，10000次、seed20261113，未做多重比较校正。按连续软分统计提高/降低，不能把它们称为二值错→对；探索性结果不构成非劣效或等价证明。

## 物理与输入验收

Late14正式名interVL-SVH-R-new：P−1/Decode仅L14–L27回看原始视觉KV，R在L0–L19读取图像，L2–L5答案Attention物理跳算，视觉Hidden于L27后/Final Norm前退出，选择性持久视觉KV与独立Cache保持冻结实现。仅外层执行及评分代码新增，59个冻结源码SHA在运行前后不变。
4题Smoke ['34602', '35836', '37102', '39601']通过结构检查且计入5000题：图像实际输入、允许/禁止视觉Attention、原始位置/RoPE、退出、KV层集合及有限性检查通过。原全局严格数值比较失败和EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER保留，不宣称全局数值等价。

原5000题ID、Question、Gold10答案、Image ID和顺序核对原始注释；3166图片沿用可信SHA记录，实际生成逐题核对image、prompt、input_ids和BF16 pixels哈希。Gold独立保存，不加入模型输入。全部5000条Token IDs CPU解码与原文一致，15000条新旧预测经官方评分及独立规范化/软分计算一致。错误、空输出和截断全部保留。

## 环境与归档

OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9，RTX3090，PyTorch2.6.0+cu124、Transformers4.49.0、batch1。模型仅加载一次，数据/模型/环境/运行文件在本地盘；进度每250题备份至持续盘，不重跑对照，不做效率实验或其他Benchmark。
GPU释放后：NVIDIA GeForce RTX 3090, GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, 1 MiB, 24123 MiB。未关闭服务器。
原始预测、配置、源码与日志分别归档；Windows与持续盘真实SHA回执见backup_receipt.json。独立分支发布轻量全集预测、输入哈希、分数、统计及核验脚本。

历史证据：[Native/Physical-D全集及冻结Manifest](https://github.com/reyuyu/ATE/commit/3297559f21d3de55a6630a5491c9ecf66301c4b7)、[Late9全集汇总及冻结物理源码](https://github.com/reyuyu/ATE/commit/c60a9919112a3cf5bec8c494a533cba56a97f29b)、[interVL-SVH-R-new命名与前项OCRBench全量结果](https://github.com/reyuyu/ATE/commit/20043c577c6e08588becfe7470344b830b357640)。

Windows完整原始结果备份已完成，路径：D:\ATE-dataset-backups-3090\ATE-results\internvl-svh-r-new-textvqa5000-22875-20261113\full_results.tar.gz；SHA256：4b6cbff02d13604179c00d13a1b562c820dc2f083551262587c5d704ebf03b60。持续盘副本同SHA。
