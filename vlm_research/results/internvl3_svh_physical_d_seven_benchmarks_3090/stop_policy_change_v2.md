# stop_policy_v2 变更说明

父版本：6c294aae67ddb2a5cb3706ded76dcf3edb26c859。
本轮授权只补齐 GQA / TextVQA / OCRBench。原四项全量结果不重新推理或计分。

唯一研究执行变更是停止策略。单题错误、无关解释、重复、原始上限截断、解析失败均标记 OUTPUT_ANOMALY 并纳入评分，不再据此停止整个任务。模型、Revision、BF16/Eager、单图块256视觉Token、Prompt、Gold、图片、原生EOS、Greedy、独立Cache、生成上限、Scorer和Physical-D均不变。原protocol.json字节、暂停记录与异常文本保留。

持续硬停止：NaN/Inf、OOM、突破生成上限、KV/RoPE/位置/Attention结构错误、输入或执行源码SHA不符、Native/No-op/官方Chat不一致、检查点损坏、同题两方法输入不一致。

系统性异常监测在新增GPU推理前冻结：每方法100题的非重叠窗口，连续两个窗口空输出≥10%、重复≥10%、其他行为格式异常≥25%或截断≥50%即暂停；连续10题空输出或重复也暂停。不会使用Gold/Accuracy决定停止，不自动调阈值。该规则允许单题异常并保存各窗口的审计记录。

原检查点仅在完整SHA256、SQLite完整性、模型/源码/输入/Prompt/Gold/协议核验匹配后复制并恢复。父版本已存在预测JSON逐字节保留；V1_REUSED与V2_GENERATED单独记录SHA，不覆盖任何旧预测。只生成第一个未完成Question ID起的缺失方法记录。

原严格BF16数值验收继续是 FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER；结构Smoke通过不能改写此状态。结果是ATE单图块同卡配对研究，不能宣称作者完整排行榜配方。

TextVQA仍为32 Token，长书名截断按冻结Scorer照常计分。未来64/128 Token版本必须作为新生成协议完整运行，不能混入本轮。
