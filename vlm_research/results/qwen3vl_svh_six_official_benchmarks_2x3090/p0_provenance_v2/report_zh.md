# Qwen3-VL-SVH 六项评测 P0 协议溯源复核

日期：2026-11-10（Asia/Shanghai）。审计起点：`f238d375ffda785d3062069ce6e4fd0c729cd9a6`。

**本轮完成的是静态协议审计，不是评测完成。GPU 请求、新预测、模型加载均为 0；未安装 vLLM，未修改 Physical 算法、main、共享环境及旧 P0/P1 记录。**

## 1. 结论

更可行的是 **方案 B：ATE 统一 HF 配对协议**，但它明确不是作者官方复现，也还不能立即启动 Physical 正式评测：历史 `NUMERICAL_EQUIVALENCE_FAILED` 必须先解决并按原门槛重新验证。本轮没有解决或改判它。

**方案 A 尚未闭环。** 作者 README 给出了 vLLM、两个框架名称和全局采样参考，却没有这六项逐任务的框架 SHA、实际命令、视觉预算及 Judge 配方。补读作者技术报告 `arXiv:2511.21631v1` 后，找到 **MMBench-V1.1** 及其正文 Prompt 的直接证据；不能再说 MMBench 完全没有作者证据，但这些证据仍不足以复原完整执行配方。其余五项在本轮归档材料中没有找到可靠的作者逐任务配方，保持 **UNRESOLVED**，不据此断言作者从未测过。

本报告的 `[Q01]`、`[V08]` 等为证据 ID。`evidence_index.md` 和 `parameter_evidence.csv` 给出每项的绝对源码路径、固定 Commit/论文版本、起止行、完整文件 SHA256 和片段 SHA256。行号以本地归档文件为准；论文使用已归档 HTML 的文本抽取行及 Appendix 小节，不虚构 PDF 页码。

## 2. 证据分级与固定版本

| 类型 | 本轮可支持的结论 | 不能替代的证据 |
|---|---|---|
| MODEL_AUTHOR_DIRECT | 作者 README 复现综述、论文 Appendix A/B | 不能自动确定每个任务的运行参数 |
| MODEL_AUTHOR_README_EXAMPLE/DEPLOYMENT | 部署、像素预算和 HF 示例 | 不是作者正式评测的逐任务配方 |
| MODEL_REVISION_ASSET | 固定模型的生成配置、Tokenizer、Processor 默认值 | 可能被框架/命令行覆盖 |
| FRAMEWORK_CANDIDATE | 固定候选源码的实际默认执行路径 | 不证明作者当时使用这个 Commit |
| DATASET_AUTHOR/REFERENCE_SCORER | 数据集 Gold、评分或 Circular 定义 | 不证明 Qwen 作者选用了该解析器/Judge |
| ATE_PROPOSED | 本报告方案 B 的显式内部设计 | 绝不能标为作者官方设置 |

固定证据版本：

- Qwen 作者仓库：`96588727e44c78b25ba03ea03b8e12f7e64fd0da`。
- 作者技术报告：`arXiv:2511.21631v1`，Appendix A 与 B.2。
- VLMEvalKit 候选：`051aa811ec27c494919bda944654de884fd70c41`。
- lmms-eval 候选：`f9b577e76bac2fb5a17f6e2d5f9ad9b69156c292`。
- 模型：`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- HF 环境历史元数据：torch `2.6.0+cu124`、Transformers `4.57.6`；未安装 vLLM。本轮仅复用旧环境记录，没有导入模型或初始化 CUDA。

候选框架是本次审计锁定的快照，不是作者论文当时版本的证明，不能反向填成作者使用的 Commit。

## 3. 实际生效路径：不能只读 PromptMixin

### 3.1 作者全局设置

作者 README 的 Evaluation Reproduction 明示：vLLM；VLMEvalKit/lmms-eval；Instruct 为采样而非 greedy，seed=3407、T=0.7、top_p=0.8、top_k=20、repetition_penalty=1、presence_penalty=1.5、out_seq_length=32768。[Q01]

这些是**全局参考**，不能证明每项最终 max_new_tokens 都是 32768，也不能证明 seed 被实际传到了引擎/请求。README 中 `vllm>=0.11.0` 是部署支持下界而非精确复现版本。[Q04]

### 3.2 VLMEvalKit 候选的真实路径

`config.py:1474–1485` 注册 Qwen3-VL-8B-Instruct 时设置 `use_vllm=True`、`use_custom_prompt=False`、temperature=0.7、max_new_tokens=16384、top_p=0.8、top_k=20、repetition=1、presence=1.5。[V01]

`inference.py:183–187` 因而走 **dataset.build_prompt**，不是 Qwen 的 PromptMixin。[V02] 后者即使写有 yes/no 或 concise 模板，也不能拿来描述此注册项的有效输入。

- vLLM 构造器实际写入 **seed=0**，并按全部可见设备数设置 TP；与 README 的 seed=3407 不一致。[V04]
- SamplingParams 未显式传请求 seed，cap 来自 16384 注册值，stop_token_ids=None；实际 EOS 还依赖未固定的 vLLM 版本及模型资产，不能补成已证实的逐任务 EOS 配方。[V06]
- 注册项没有固定模型 revision，也没有给这六项显式 min/max_pixels。官方 Processor 默认值、qwen-vl-utils 的预缩放与可能的消息级覆盖均需追踪，不能仅凭模型 preprocessor_config 推断最终预算。[V04,V05,V06,M02]
- 关闭 vLLM 的候选 HF 路径使用 `flash_attention_2`，且 generate_kwargs 没有 presence_penalty；不是作者 vLLM 路线的同义替代，更不是当前 ATE eager 物理端口。[V03,V04]

### 3.3 lmms-eval 候选的真实路径

已归档 `models/simple/qwen3_vl.py` 是 **HF BF16 simple adapter**，不能称为作者 vLLM 评测。类默认 cap128 会被任务 YAML 覆盖；T=0 时转为 greedy。PIL 图像消息显式使用 min_pixels=262144、max_pixels=2097152；并且默认有系统消息。[L01,L02,L04]

任务配置选择按**精确模型名**查表。`qwen_vl` 的专属模板不能直接套给注册名 `qwen3_vl`；无该键时走 default。[L05] 这是 ScienceQA、GQA 和 MME Prompt 追踪的重要区别。

HF simple adapter 显式使用 tokenizer.eos_token_id，而固定 generation_config 的 EOS 是 `[151645,151643]`；两者不能无核查地混用。[L02,M01,M03] `until` 字符串还会在生成后裁剪，不能把裁剪后的答案误当成模型原生 EOS 停止。[L03]

## 4. 六项协议核查

完整机器可读矩阵：`six_task_protocol_matrix.csv`；逐参数展开：`parameter_matrix.csv`。

| 任务 | 作者实际配方证据 | 候选源码可确定的路径 | 仍缺什么 |
|---|---|---|---|
| GQA | 未找到逐任务作者配方，UNRESOLVED | VLMEval `GQA_TestDev_Balanced`，question+single-word/phrase 后缀；分派 accuracy。lmms `lmms-lab-encoder/GQA:testdev_balanced_instructions/testdev`，default 同类后缀，cap16、greedy。[V08,L10,L05] | 作者所用任务版本、框架 SHA、实际 Prompt/预算/cap/seed/scorer |
| ScienceQA-IMG | 未找到逐任务作者配方，UNRESOLVED | VLMEval MCQ 为 Hint/Question/Options/select；`ScienceQA_TEST` 名称本身不能证明过滤 IMG。lmms 明确 `ScienceQA-IMG/test`，default 为 Context+题干+原顺序选项+字母要求，cap16；Gold 整数转字母。[V09,V10,L11,L16] | 作者 IMG 过滤、Hint/格式、自由文本解析器、实际运行参数 |
| TextVQA | 未找到逐任务作者 OCR/输入配方，UNRESOLVED | VLMEval `TextVQA_VAL` 的 builder 未插入 OCR，只附 single-word 后缀；评分分派 vqa_score。lmms utils 会 capitalize 题干，并在 ocr=True 时插入 OCR，但本轮未取得被 include 的完整默认模板，不能确定该候选有效 OCR/cap。[V08,L12,L17] | 作者是否使用外部/reference OCR、图像预算、完整任务模板及 scorer 版本 |
| POPE | 未找到逐任务作者配方，UNRESOLVED | VLMEval `POPE` 保留数据中的 question，不因 PromptMixin 自动添加 yes/no；lmms COCO 三策略 default 短答后缀，Random cap128、greedy。[V12,L13] | 作者使用哪一 POPE split/题集、parser/Judge、生成配置 |
| MME | 未找到逐任务作者配方，UNRESOLVED | VLMEval `MME` 保留数据内 question；lmms 会移除原 yes/no 后缀再添加 default 短答后缀，cap16、greedy。两个输入不能视为相同。[V12,L14,L19] | 作者实际 Prompt、评分解析器、版本、预算及 cap |
| MMBench | **已确认论文使用 MMBench-V1.1，且 B.2 给出 Question/Options/select 模板**。[P01,P02] | VLMEval 区分 V1.0/V1.1；MCQ 默认 exact_matching、可接 Judge。lmms EN 模板 cap1024、until="ASSISTANT:"，Judge 默认可由 MODEL_VERSION 覆盖。[V09,V10,V11,L15,L18] | 作者实际 split/数据哈希、Hint、Judge 型号/版本/API/失败策略、框架 SHA 和运行命令 |

### 4.1 OCR 与视觉预算不能互相代替

模型具有读字能力，不等于给模型附带 OCR 字符串。旧 LLaVA Paper-OCR 的 reference token 不能自动继承给 Qwen；TextVQA 图像-only 与 reference-OCR 必须分别命名。

固定模型 Processor 默认 size 对应面积范围 65536–16777216，patch16/merge2；作者 README 另有 256–1280 visual-token 的预算演示。这两项都不是已证实的六项实际评测预算。[M02,Q02] 更不能硬编码 LLaVA 的 576/336/crop 或 Vicuna System Prompt。

### 4.2 Scorer 与 Judge 是协议的一部分

- **ScienceQA**：作者 evaluate_acc.py 比较零基整数答案，原 test 4241 题，has_image 子集 2017；它没有定义所有自由文本输出通用的官方解析器。字母/整数桥必须明确，不得把历史错误评分直接继承。[S01,L16]
- **TextVQA**：复用 MMF 的标准化、十答案 leave-one-out 共识软分，不以简单字符串完全匹配或单次 min(count/3) 替代。[S02]
- **POPE**：作者 parser 保留首句，识别 No/not/no，否则判 yes；空输出也可能被该规则判 yes，须单独报异常，不能偷偷更换 parser。[S03]
- **MME**：作者 parser 使用前四字符 yes/no；14 子任务按原图片两题成对计算 Accuracy+Accuracy+。Gold 的 Yes/No 转小写是作者现有规则，不是更改标注。[S04,S05]
- **MMBench**：Circular 要同一道原题的所有选项轮换均正确，不是 4377 行准确率均值。作者规则先抽取、失败后 ChatGPT 匹配；具体 Judge 是复现依赖。[S06,V14] VLMEval 候选允许 Judge 不可用后退到 exact_matching，这种自动回退在正式协议中必须禁止或单独命名，不能仍报原 Judge 协议成绩。[V11]
- **Judge 默认值冲突**：VLMEval MCQ fallback 为 exact_matching，但 CLI 可改；lmms EN utils 默认 `gpt-4o-2024-11-20`，环境变量可覆盖。都不能据此推断 Qwen 作者的实际 Judge。[V11,V15,L18]

本轮没有调用任何 Judge API。部分候选评分依赖源码和 TextVQA include 模板因 GitHub API 403 未取到，错误保存在 fetch manifests；它们保持未闭合，未使用其他 Commit 的同名代码冒充。

## 5. 本地 Dataset ID 与 Gold 映射审计

`check_datasets.py` 仅做 CPU 元数据检查，不解码图片、不导入 torch。输出 `dataset_id_gold_audit.json`，含原始标注绝对路径和 SHA256。

| 数据 | 核对范围 | 结果 |
|---|---|---|
| GQA | 12578 道 testdev_balanced 的 ID、imageId、原 question、answer | 全量一致 |
| ScienceQA-IMG | 官方 test 的 has_image 子集；PID、Gold index、选项顺序、Hint | 2017 题集合相同，绑定一致；不输入 lecture/solution |
| TextVQA | TextVQA_0.5.1 validation 的 question_id、image_id、原题和十答案 | 5000 题集合相同，绑定一致 |
| POPE | 本地官方 COCO JSONL 三策略各 3000；与旧 8910 题逐项比较 | 旧 Random 缺 90 题，另外两策略各 3000；已存在 8910 条的题/图/Gold 全部一致 |
| MME | 原 question 与 lowercased Gold；14 任务与配对 | 2374 条，1187 对，每对两题，绑定一致 |
| MMBench | 原 `mmbench_dev_20230712.tsv` 的 index、question、选项、Gold | 4377 行一致；1176 Circular 组，1132 图片 SHA；**不能认证为作者 V1.1** |

**POPE 的 90 条不是随机少几题。** 它们全部为 no，涉及 84 张图片；旧 Random 为 1500 yes/1410 no，原始 Random 为 1500/1500。因此分母和标签比例均变化，不能混报 8910 与 9000 的 Accuracy/F1。缺失 ID、原问题和图片路径已逐条保存到 `pope_9000_vs_8910_missing.csv`；缺失原因仍 UNRESOLVED，不凭差集猜测历史过滤动机。

POPE 必须以 `(strategy, 原始 question_id)` 对齐。旧 paper 的 `author_question_id` 对 Random/Popular 加了全局偏移，不能直接与各 JSON 内 1–3000 匹配。

本轮检查引用图片文件存在性，**没有重新计算全部图片的当前 SHA256**；图像字节绑定复用历史清单及其有限 spot checks。后续生成前仍需全量当前图片与输入张量绑定。MME 的历史个别图片字节版本例外也不因此自动成为官方版本。

原题/图片路径数不总等于独立图像字节数：清单中 ScienceQA-IMG 为 1799 个 SHA 簇，MME 为 1177 个 SHA 簇；MME 仍须保留全部 1187 个原始题对，不能用 SHA 去重删题。

## 6. 方案 A：严格作者官方协议

### 已具备

1. 固定 Qwen 权重/Processor/Tokenizer 缓存和 SHA 清单。
2. 作者全局 vLLM/采样参考；MMBench-V1.1 与正文 Prompt 的直接证据。
3. 两框架固定候选源码、真实 Prompt 分派链、任务 YAML 和数据集作者评分参考。
4. 六项本地数据库存及 Gold 映射；POPE9000 完整原始标注已在本地，不需下载图片补齐。

### 必须补齐的证据

1. 作者实际逐任务 launch/config、框架 Commit、vLLM/Transformers/qwen-vl-utils 及依赖 lock；不能用候选最新版回填。
2. 六任务具体 Dataset ID/split/revision/字节哈希、Prompt override、系统消息、OCR 政策和视觉预算。
3. 实际 per-task cap、EOS/stop、seed 向 engine/request 的传递方式、采样方式及 Judge 配置。
4. MMBench V1.1 的确切 split/Gold/参考 Judge。现有 20230712 旧 dev 不能替代。

### 同 Runtime Physical 端口的前提

现有 `qwen3vl_c3v2_physical/engine.py` 是 HF eager/自定义 CompactKV 代码，不是 vLLM port。[H03] 严格路线需要未来在**作者固定 runtime**上实现、验证相同写入边界、P−1/Decode 禁图、DeepStack/mRoPE、物理 Hidden 退出、层1–6答案 Attention 跳算及 KV 生命周期，并让 Native 与 SVH 都走同一运行时/精度/输入/采样边界。只把 Native 放 vLLM、SVH 放 HF 不构成官方同条件对照。

vLLM 的 PagedAttention、调度/TP/Cache 与 HF 不同，安装软件本身不等于完成端口。未来须先验证 Native/No-op 严格一致，再按冻结 Dense/Physical 门槛检查；本轮不实施这些工程，不安装 vLLM。

**A 状态：BLOCKED_AUTHOR_PROTOCOL + BLOCKED_SVH_OFFICIAL_RUNTIME_PORT；没有官方六项分数。**

## 7. 方案 B：ATE 统一 HF 配对协议（非作者官方复现）

完整设计见 `plan_B_ate_hf_design.json`；仅设计，不是已授权待运行队列。

### 共享运行与输入规则

- Native/SVH 使用同一固定 Qwen revision、BF16、HF torch2.6.0+cu124/Transformers4.57.6、batch1/eager、last-row LM Head、独立真实 KV。
- 使用模型自身官方 Chat Template；显式 user 单图在前、任务正文在后、不追加 System Prompt。跨模型可共享任务正文，但各用自己的官方 Chat Template/视觉处理；不要求不同架构有相同视觉 Token 数。
- **显式 ATE 预算**：Qwen Processor image size 面积 min65536、max2097152，patch16/merge2，RGB，保比例、仅 resize 一次。这个上限不是作者官方任务预算，是为内部配对提出的固定有限预算，不宣称已在 3090 验证可行，也不与旧 full-resolution 输入混用。
- 不提供外部 OCR/reference token；模型原生视觉读字能力保留。TextVQA 与历史 Paper-OCR 不直接配对。
- 全量题、不做 few-shot/重采样；greedy、num_beams1、repetition1，不启用 presence penalty；记录 seed3407。EOS 显式 `[151645,151643]`、pad151643、无额外字符串裁剪。greedy 与作者 stochastic 有意不同，命名必须保留 ATE-HF。

### 六任务设计矩阵

| 任务/规模 | 任务正文 | cap | 评分 |
|---|---|---:|---|
| GQA testdev_balanced /12578 | 原问题+single-word/phrase 后缀 |32| ATE 固定 strip/lower Exact Match，明确不是未核验的作者特定 scorer |
| ScienceQA-IMG /2017 | 可用 Hint + Question + 原顺序 Options + select-correct 请求 |128| 固定 lmms [L16] 自由文本 parser；字母转零基原 Gold；非法输出记错 |
| TextVQA val0.5.1 /5000 | 原问题+single-word/phrase 后缀；不附 OCR |128| MMF 十答案共识软分 [S02] |
| POPE COCO /9000 | 原问题+固定 yes/no 请求 |128| 作者 parser [S03]；三策略独立五指标，空输出另审计 |
| MME classic /2374 | 保留原问题及其原始 yes/no 请求 |128| 作者 [S04,S05] 的14项 Acc+Acc+ |
| MMBench 20230712 EN dev /4377 | Hint + Question + 原顺序 Options + select 请求 |128| 固定 VLMEval exact_matching + Circular，显式命名为无外部 Judge 的诊断变体 |

合计 **35346** 条/条件，是补齐 POPE90 题后的内部数据规模；不是作者六项实验的已证实总数。所有 caps、改写及分数解析都是方案 B 的事前选择，不冒充作者配置。最终若改用 MMBench V1.1+外部 Judge，应作为另一个版本重新冻结，不能覆盖这份设计。

### 实施前仍需的工作

1. 固定缺失源码依赖和完整 Python/Pillow/Processor依赖 lock，编写精确任务输入适配与 parser CPU 回归；禁用 MMBench Judge 自动回退。
2. 生成本方案自己的 Prompt/Input/Pixel/Gold/Scorer 哈希，不沿用旧 LLaVA Prompt 或 Qwen Paper-OCR 哈希；对图片逐字节复核，POPE补齐90行，保持 MME 对与 Circular 组完整。
3. 按本方案输入预算做 Native 视觉显存预检。旧 TextVQA full-resolution eager OOM 保留，不因设计一个 cap 就宣称修复；新预算仍 OOM 时阻塞，不静默缩图、跳题或换后端。[H02]
4. **在另行授权的工程任务中先修复 Physical 数值门禁。** 原 mean≤0.02/max≤0.5、Top1/序列要求不变；中位输入仍未通过、最长题未完成的事实继续保留。[H01] 这份静态设计不产生 GPU PASS。
5. 全部通过后才运行同机 Native/SVH 配对，按 Image SHA 簇统计损害/纠正及配对 CI；既有分数只有输入、运行和 scorer 全绑定相同才复用。本设计基本改变了 OCR/预算/部分 Prompt，因此旧结果通常不能直接当本轮基线。

**B 状态：DESIGN_ONLY_NON_AUTHOR_OFFICIAL；工程量较 A 小，但目前仍不是 READY_FOR_PHYSICAL_EVAL。**

## 8. 可复用代码与不可直接复用项

- **模型、图片、标注清单**：本地共享资源只读；本轮无需模型/图片下载。
- **HF 输入绑定**：`7e18895a...:vlm_research/results/qwen3vl_c3_3090_official_eval/scripts/benchmark_inputs.py`，已有图像 SHA、官方 Chat Template、动态 Image Token/grid、输入张量哈希；需显式接入新正文/预算，不能照搬其旧 body。
- **HF生成和 Cache 审计**：`vlm_research/qwen3vl_c3v2_physical/engine.py`；保留原算法只读。本轮仅定位复用点，不以复用代码表示数值合格。
- **Gold/Scorer桥**：旧 `scripts/scorers_v2.py`、`scorers_v3.py`、`scoring_reference/` 可作为已记录的 ScienceQA 整数/字母错误及修复参考；两版解析策略不同，不自动混为一种官方 parser。
- **本轮新增工具**：`check_datasets.py`、`build_matrix.py` 和 `test_static_provenance.py`，只做 CPU 静态审计，不含模型运行入口。
- **不能直接复用**：旧 LLaVA Vicuna/crop/336/576 协议、POPE8910完整分数、TextVQA Paper-OCR+full-resolution 的输入/时延、未绑定的 MMBench Judge分数、失败的 Physical 等价 PASS、vLLM Native 对 HF SVH 的所谓官方配对。

上述历史 HF 文件的只读快照及 SHA 见 `reusable_code_manifest.json`；没有修改原文件。

## 9. 交付与边界

本目录新增中文报告、六任务矩阵、逐参数矩阵、源码/Commit/行号/SHA证据目录、两方案设计、本地 ID/Gold 审计和 POPE90行差集。旧 audit 根目录的 P0/P1 protocol/validation/report/failure 文件保持不变。

本轮没有运行 Accuracy、显存、速度或 GPU Smoke，没有新质量结果。无须释放本任务 GPU，因为没有建立 GPU 上下文或模型进程。只保存本地提交，不推送 GitHub、不创建 PR。
