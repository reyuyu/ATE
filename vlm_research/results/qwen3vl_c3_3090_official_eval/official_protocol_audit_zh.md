# Qwen3-VL 原始 C3：本轮官方协议审计

本轮比较的是固定 Qwen3-VL 权重在 BF16/Eager/Greedy 下的 Native、Dense-F 与原始 Physical C3。模型作者公开的 vLLM、采样和长输出配置与本轮授权配置不同，因此本轮不会宣称复现 Qwen Model Zoo 成绩。模型原生 Processor、Chat Template、RGB 处理及默认像素范围保持原样，不降低图片分辨率。

| 任务 | 数据与评分 | 本轮输入及差异 | 生成上限 |
|---|---|---|---:|
| 历史 GQA1000 | val_balanced 固定1000题、918个图像SHA簇；完整答案 exact match | 1000份输入张量哈希逐一与历史绑定完全相同；仅重映射图片物理路径 | 32 |
| GQA 正式 | testdev_balanced 12,578题、398张图；提交转换后 exact match | 原问题及固定 short-answer 指令，Qwen原生模板 | 32 |
| ScienceQA-IMG | test 中有图的2017题；正确选项 Accuracy | 保留原 Hint、原问题及全部选项；明确为IMG子集，不冒充4241题整体成绩 | 1024 |
| TextVQA | validation5000题；官方十答案 leave-one-out VQA Soft Accuracy | 明确保留Paper-OCR，含Reference OCR token；不冒充image-only输入 | 128 |
| POPE | 官方9000题（三策略各3000）；ATE论文8910题另存；Accuracy/P/R/F1/Yes ratio | 官方原问题与ATE短答案指令视图分开生成；完全相同请求可共享本轮新预测 | 128 |
| MME Classic | 2374题、1187同图双题组、14任务；逐任务Acc+Acc+，再求Perception/Cognition/Total | 官方保留Please answer yes or no；ATE论文短答案替代指令另存；已知单图SHA差异保留 | 128 |
| MMBench-EN | Dev20230712，4377个Circular行、1176完整组；作者exact_matching匹配与全组全对聚合 | Hint、选项及循环行全部保留；不调用付费LLM，不随机猜字母；需要官方裁判的未解析项另标AWAITING_OFFICIAL_SCORING | 1024 |
| OCRBench v1 | 全部1000题，930图片；官方包含匹配及HME100k大小写/去空格特例，总分0–1000 | 发现旧ATE user_text使用字面反斜杠n，而官方脚本使用真实换行；保留ATE与OFFICIAL两套输入及独立结果 | 128 |

官方规则来源均冻结到 `official_sources/manifest.json` 的真实 Commit 和 SHA256：

- [GQA 作者评分定义](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)：按完整答案匹配，balanced数据计分。
- [ScienceQA 作者](https://github.com/lupantech/ScienceQA)：`tools/evaluate_acc.py` 定义原测试4241题及has_image分组。这里仅生成已授权的2017张图题，对应IMG指标。
- [MMF 作者 TextVQA evaluator](https://github.com/facebookresearch/mmf/blob/72f898ec35af86567423c7a019dd1dc5175d1a5a/mmf/utils/m4c_evaluators.py)：十答案平均留一法，不是简单二值准确率。
- [POPE 作者 evaluate.py](https://github.com/RUCAIBox/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/evaluate.py)：只保留首句、去逗号，词中有No/not/no则no，否则yes；空输出也按原规则处理并单独计数。
- [MME 作者 Eval Tool](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/dd2950902889cd614d4edf606827240166d29381/tools/eval_tool.zip)：直接读取官方ZIP中的calculation.py，保留首四字符yes/no解析，未额外strip或改规则。
- [MMBench 作者](https://github.com/open-compass/MMBench/blob/8eaac9ae42e22eaf42863b218455acd51268ba1c/README.md)及[作者Circular实现](https://github.com/open-compass/VLMEvalKit/blob/3c192143007e4eb3f5c1a36558a4f8c1451980cb/vlmeval/dataset/utils/multiple_choice.py)：本轮只使用明确的exact_matching，不启动LLM裁判。
- [OCRBench 作者脚本](https://github.com/Yuliang-Liu/MultimodalOCR/blob/d1901a9500c7232040b5b5806df77a15312a3509/OCRBench/scripts/LLaVA1_5.py)：原始包含匹配、HME100k特例及128输出上限。
- [Qwen3-VL 作者复现说明](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md)：作者公开模型复现使用采样vLLM；本轮遵守用户指定的BF16/Eager Greedy，明确区分两者。

这些来源不会覆盖旧ATE输入、预测或评分器。只对冻结清单所列请求生成；不同协议的结果不合并成一个官方成绩。代码、模型和每条输入绑定均进入断点记录。
