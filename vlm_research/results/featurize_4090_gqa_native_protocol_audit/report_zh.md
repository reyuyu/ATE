# GQA Native 60.63% 与官方 62.0%：协议差异审计

**审计状态：COMPLETE；差距归因：LIKELY / 尚未完全确定。** 最有实证支持的是 Vicuna System Prompt 与 square-pad 预处理的联合协议偏差。数据映射、计分实现、漏题及32-token截断未发现问题；不能把全部1.37 pp归因于Prompt。

历史12,578条Native预测经独立逐题逻辑、官方转换脚本及官方GQA evaluator复算：**7,626 / 12,578 = 60.6296709%**。相对官方四舍五入62.0%的差为1.3703291 pp。历史预测、Gold和冻结Scorer均未修改。

[官方Model Zoo](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/MODEL_ZOO.md)明确列出1.5-7B为62.0%，13B为63.3%。[2023原始GQA脚本](https://github.com/haotian-liu/LLaVA/blob/ce1aa08d129bd7df931a7067bcb79cf3cb3a1af5/scripts/v1_5/eval/gqa.sh)默认13B，因此只取其数据与评测流程，未误用13B权重或成绩。官方发布包只含13B预测，未用它冒充7B对照。

下表的“官方公开流程”来自原始发布脚本、7B模型配置与对话/预处理源码。62.0%的7B成绩本身已核实；产生该成绩的完整运行命令、依赖锁定与逐题预测未公开在所查发布包中，因此不能把公开通用脚本当作该次7B运行的逐项日志。HF_conversion_reference.py也仅作转换机制参考，不代表已证实本模型转换时使用了该版本脚本。

| 项目 | 官方公开流程 / 来源 | 历史ATE Native | 结论 |
|---|---|---|---|
| 公开参考成绩 | LLaVA-1.5 7B full_ft-1e:62.0%;13B:63.3% | Native 7626/12578=60.6296709% | VERIFIED |
| 数据集与计分分母 | GQA testdev_balanced,12578,每题等权exact-match | 同12578题；所有qid/question/imageId/Gold/split匹配 | VERIFIED |
| Prompt | vicuna_v1 System + USER:<image> + question/short suffix + ASSISTANT: | USER:<image> + 完全相同question/short suffix + ASSISTANT:，没有System | VERIFIED |
| Tokenizer | 原生SentencePiece，BOS1/EOS2 | 同SentencePiece文件SHA；HF fast；所有12578官方Prompt token IDs与slow官方算法一致 | VERIFIED |
| 预处理 | RGB→按CLIP mean颜色补方形→CLIP336 bicubic/normalize | RGB→短边336 resize→center crop336；不先pad | VERIFIED |
| 图片版本 | Stanford GQA images.zip对应398张 | 398张完整；旧官方ZIP CRC/size全匹配、80张Stanford直接byte匹配；当前像素哈希全部匹配历史Native | VERIFIED_WITH_LIMIT |
| 生成与停止 | greedy/beam1/FP16/EOS2；2023release cap128 | greedy/beam1/FP16/EOS2/Eager，cap32 | VERIFIED |
| Scorer | 输出strip后，text.rstrip('.').lower()；与官方answer完全相等；balanced等权 | 同转换、同分母及exact-match；官方完整脚本离线复算一致 | VERIFIED |
| 模型/权重/Backend | 原生LlavaLlamaForCausalLM；Vicuna7B32层4096；CLIP-L/14-336；原始词表32000 | HF LlavaForConditionalGeneration；同架构/视觉层-2/投影结构；词表扩至32064；Transformers4.49/Eager | UNRESOLVED |

**Prompt的确切差别（VERIFIED）**：历史HF Prompt从 `USER: <image>` 开始；官方vicuna_v1在其前面加入标准System段，后面的图像换行、原问题、单词/短语回答指令和 ` ASSISTANT:` 均相同。示例qid201030233从616个展开位置变为646，增加30个文本token。完整字符串与Token IDs见cpu_audit_summary.json。原生-200图像标记与HF32000表示实现不同，二者均注入576个图像patch，不是缺图或多图。官方slow与HF fast对12578个官方Prompt的Token IDs完全一致。
[官方对话模板](https://github.com/haotian-liu/LLaVA/blob/ce1aa08d129bd7df931a7067bcb79cf3cb3a1af5/llava/conversation.py)与[预处理实现](https://github.com/haotian-liu/LLaVA/blob/ce1aa08d129bd7df931a7067bcb79cf3cb3a1af5/llava/mm_utils.py)已固定到原始release commit；实际官方prepared GQA题干从eval.zip只按HTTP Range读取所需文本成员，不下载整包或图片。12578个qid、题干/回答后缀及图片ID逐项完全一致。

**数据、图片与Scorer（VERIFIED，保留来源限制）**：raw官方testdev_balanced题库SHA256为 `14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318`；当前/历史Question、Gold、Image ID、split均相同。398张图片无缺失，尺寸已检查；当前处理后FP16像素哈希全部匹配历史Native。旧官方ZIP的398个size/CRC检查、80张Stanford直接byte匹配记录与本轮迁移整体ZIP校验共同支持图片版本可信。本轮没有重新对398张做全量SHA或重新下载，未把80张的直接验证写成398张。
实际执行[官方GQA转换](https://github.com/haotian-liu/LLaVA/blob/ce1aa08d129bd7df931a7067bcb79cf3cb3a1af5/scripts/convert_gqa_for_eval.py)和[LLaVA指定GQA evaluator](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187)：源码SHA与历史冻结evaluator相同，逐题及分母均一致。没有缺预测、重复qid、空输出、32-token截断或保存得分错误；历史最长输出仅6token（含EOS），增加cap不会改变这些已EOS结束的答案。

**冻结200题的GPU控制实验**：按structural类型比例分配（compare9/logical29/query108/verify36/choose18），仅按预先固定seed与qid哈希排序选题，未查看正确性来挑题。3个Native Smoke输入tensor、token和文本均匹配历史；同一模型一次加载，随后完成A/B/C各200题，共600次控制生成。没有额外完整基线、C3或参数搜索。

| 条件 | 正确数/200 | Accuracy | 相对同题历史Native | 改变答案 | 改善 / 恶化 |
|---|---:|---:|---:|---:|---:|
| 历史Native（复用） | 121/200 | 60.5% | — | — | — |
| A | 120/200 | 60.0% | -0.5 pp | 10 | 2 / 3 |
| B | 121/200 | 60.5% | +0.0 pp | 24 | 8 / 8 |
| C | 127/200 | 63.5% | +3.0 pp | 23 | 10 / 4 |

A=仅换官方Vicuna-v1 Prompt；B=仅改官方square-pad；C=两者同时对齐。其余保持同一HF权重、FP16/Eager、greedy、EOS2和cap32；600条输出均有限、非空且未截断。此表是固定小样本诊断，不能作为正式GQA成绩，也不能直接外推完整1.37 pp。

图像cluster数=156；4000次cluster bootstrap的配对差值探索区间：A [-3.35, +1.97] pp；B [-3.96, +3.96] pp；C [-0.52, +6.73] pp。小样本区间用于显示不确定性，不作为因果分解或恢复62.0的证明。
389/398张原图的crop/pad像素不同，补方形后的当前CLIP核心输出与官方CLIP预处理逐像素一致。A单独-0.5 pp、B单独0 pp，C联合+3.0 pp，表明不能把效应简单相加或认定单独System Prompt必定提高成绩。

**逐题复核**：全部200题的三条件原输出/解析/分数见paired_predictions.csv；57条改变答案的条件记录见changed_answer_cases.csv。对四例原图做了人工复核（不改Gold）：

| qid | Gold | 历史 → A / B / C | 复核结论 |
|---|---|---|---|
| 20886973 | blue | Gray → Gray / Blue / Blue | 右下角蓝鼠标处于中心裁剪边缘；B/C改善支持边缘图像证据变化 |
| 201896132 | blue | White → White / Blue / Blue | 底部蛋糕有浅色主体和蓝装饰，目标/颜色有歧义；保留官方Gold，不宣传为确定视觉纠错 |
| 20482509 | racket | Racket → Tennis racket / Racket / Tennis racket | 网球拍可见；更具体的tennis racket仍按官方exact-match计0，属于表述损失，不是Scorer错误 |
| 201676091 | bottom | Bottom → Bottom / Top / Top | 图中多个手柄，指代有歧义；B/C按官方Gold恶化，pad并非保证提高每题 |

**模型与残余差异（UNRESOLVED）**：HF模型与原生模型的7B架构、SentencePiece及CLIP336/视觉层-2设定对应。原生词表32000、HF扩为32064；历史没有输出新增token。没有下载另一套大型权重，也没有独立逐tensor确认原生权重或原生Backend的数值等价；官方2023代码依赖与本次Transformers4.49/Eager不同。3个Native Smoke一致不能排除其余题的跨GPU浮点影响。因此本轮确认“协议不完全可比”，联合小样本正向证据使其成为最可信的部分解释，但完整差距的占比仍未确定。

**最少下一步**：优先取得官方7B同split逐题预测与精确运行配置，先CPU比对具体错误集合；若拿不到，下一轮需原生LLaVA backend与HF在同一固定小样本上对齐权重与逐token输出，才能隔离转换/Backend残余差异。要量化完整1.37pp仍需更大、预先授权的配对证据，本轮不启动。

本轮环境：NVIDIA GeForce RTX 4090, GPU-643a4d73-25eb-6b19-1acc-ccf94c744cec, 610.57.04, 23715 MiB；代码 `e281754ef17b3accc09d6938b5b9f020048b2aec`；HF revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；Torch2.6.0+cu124 / Transformers4.49.0。模型加载1次，allocated峰值13.635GiB。全部GPU工作已结束：`NVIDIA GeForce RTX 4090, 1 MiB, 24109 MiB, 0 %`。原数据/模型保留本地盘，C3 Physical数值门禁未改动。本轮没有完整Benchmark重跑或任何V²Drop/其他研究任务。

结果保存在独立目录featurize_4090_gqa_native_protocol_audit；GitHub只发布报告、协议清单、必要源码、固定样本/统计及小型逐题CSV。12,578条完整CPU审计表、600条完整原始GPU预测和完整样本绑定留在本地与Windows独立备份；同步提交见github_sync_receipt.json。
