# ScienceQA 离线评分审计：保留冻结解析和整数 Gold

本轮新入口的初版 scorers.py 存在两项后处理差异：将解析字母直接与整数 Gold 比较，以及采用了比历史 ATE 更宽松的通用选项匹配。问题均在 ScienceQA 生成开始前发现；题目、选项、Gold、图像、Prompt 和生成协议没有问题。

初版文件及每条预测里的初始分数均原样保留。v2 仅修正字母→整数索引，作为通用解析的明确对照保留。正式主计分使用 scorers_v3.py：从 ATE 冻结 Commit e281754ef17b3accc09d6938b5b9f020048b2aec 的原 score.py 按 AST 提取 ScienceQA 分支，逐字保留其答案提取语义，再按官方 ScienceQA 的整数选项索引评分。

冻结解析接受单个大写选项、`A. `样式前缀、或唯一的 `The answer is X.` 匹配；例如单独 `A.` 在冻结规则下不能解析，不能为了提高分数改成宽松匹配。不能解析的答案按错误保留，不随机猜测、不调用付费裁判。

scienceqa_offline_score_corrections.csv 保留初始分数与主计分修正；scienceqa_parser_alternate_per_question.csv 和 summary.csv 另列通用匹配对照及差异。独立 CPU 逻辑另行逐题核对冻结解析、整数 Gold 和汇总；不重跑 GPU，不改历史冻结评分器或正在运行的推理绑定文件。

MMBench 三个选项解析函数及 verbose 正则与历史冻结源码 AST 完全一致。TextVQA 规范化类 AST 完全一致；十答案评分类唯一差异是去除 tqdm 进度条包装，移除该包装后 AST 完全一致。官方 LLaVA 评分文件 SHA256 8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774，与历史冻结 vendor 文件相同。详细证据见 scoring_reference/。

本说明替代早期只描述字母索引桥接的 v2 说明，旧说明仍保存供审计。所有修改均仅为离线后处理，不改变模型、输入、算法、生成 Token 或执行时间。
