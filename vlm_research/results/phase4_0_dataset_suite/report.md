# Phase 4.0：ATE 跨数据集评测套件

日期：2026-10-29，Asia/Shanghai。**六个评测单元的冻结子集、图片与 Native Full 链路均已准备；没有运行全量 Benchmark 或任何干预。**

分支：`exp/phase4_0-benchmark-datasets`。独立 Worktree：`/data/lk/ATE-worktrees/phase40-benchmark-datasets`。
远端 main 基线为 `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`，未修改 main 或其他 Worktree。
模型、原始 Processor/Chat Template、FP16/eager、Transformers4.49.0、576 个视觉 Token 保持不变。

## 1. 六项准备状态

| 评测单元 | 完整标注规模 | 本轮冻结题数 | 冻结图片数 | 实际状态 |
|---|---:|---:|---:|---|
| GQA ChooseAttr | 1000题 / 937图 | 240 | 240 | 图片、真实候选、完整 LogP 和生成链路就绪 |
| GQA QueryAttr | 1000题 / 934图 | 240 | 240 | 开放生成就绪；不构造错误候选 |
| GQA LogicalObj | 991题 / 931图 | 240 | 240 | yes/no 各120；候选与生成链路就绪 |
| POPE COCO | 每策略3000题 / 500图 | 300×3=900 | 三策略共用50图 | 每图6题完整保留；各策略 yes/no 各150 |
| ScienceQA-IMG Test | Test 4241题，其中2017题带图 | 500 | 500个原始PID/文件，475个不同图片SHA | 原始选择题、Gold Index、图片与字母答案评分就绪 |
| TextVQA v0.5.1 Val | 5000题 / 3166图 | 500 | 309 | 按完整图片问题组；保留每题10个人工答案 |

合计2620条评测记录、1528个不同图片 SHA256；不是2620张独立图片。
POPE 分策略保存，故共有8个 Manifest。`ready_native_full` 表示数据及链路就绪，**不表示2620题已经评分**。
独立注册的官方 GQA Balanced Test-Dev 状态为 `blocked_assets_and_official_scorer`，冻结0题，不冒充上述三个作者子集。

## 2. 数据来源与获取

- 三个 GQA CSV 来自 `FightingFighting/cross-modal-information-flow-in-MLLM@c9a1111dac4baa175f67716fab473789320187c0`，原行数分别1000、1000、991；完整 CSV SHA、原始字段、每行 SHA 均保存。
- GQA 图片复用 `/data/lk/ATE-resources/data/image-tasks/MMEB/GQA/`。资源记录对应 MMEB-V2 revision `e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1`，已存在归档 SHA `feeaf4a8319cba587b982ec976fba225d13e134a3217e1cad5c251e64299346a`；本轮重新校验使用图片，不重复下载归档。
- POPE 原始 JSONL 来自 `AoiDragon/POPE@08d957b917e5a378a2f99d35b6293c536a66298b` 的 `output/coco/`。严格匹配 `COCO_val2014_*.jpg`，通过 COCO 官方 S3 HTTPS 对象端点取得50张原图，没有使用 val2017 或缩略图。
- ScienceQA 原始 `problems.json`、`pid_splits.json` 固定在 `lupantech/ScienceQA@2cbf8318e07b9ece895bb2ae605e71e38d623264`。仅选择官方 Test 中的带图题，保留学科、题型、年级和候选等元数据。
- ScienceQA 图片来自用户指定 HF 数据集的国内镜像，固定 revision `f18b0a70359ebfb41f658fd564208d0355b013f4`。只下载122386007字节 Test parquet，LFS SHA 为 `235e92d1f30155266df76bc9f28fc6e4fcb6bec2c6a8c7d67f9086ea6b392a84`。发布者构建脚本按官方 JSON 顺序遍历；4241条 Test 的12个内容字段、图片存在标记及官方 Split 顺序全部一致，才按原PID提取图片。
- TextVQA 使用官方 `TextVQA_0.5.1_val.json` 和官方 `train_val_images.zip`。ZIP 总大小7072297970字节，本轮只读取92821887字节的必要范围，取得309张原始图片，使用标准库 ZIP/CRC 验证。保留 ETag、S3 Version ID 和各范围哈希；**没有计算或声称完整 ZIP SHA256**。

全部新增共享资源位于 `/data/lk/ATE-resources/benchmarks/phase4_0/`，当前约431MiB，包含标注、下载的Test parquet、选中原图和隔离的 pyarrow 工具；不含已有模型/GQA归档。原始图片、权重和工具环境没有提交 Git。

## 3. 冻结与图片审计

随机种子固定 `20261029`，排序后随机化，不读取模型预测。抽样锁先于 GPU 写入。
GQA 每任务内部图片不重复，优先不与旧 ChooseRel240 重叠，三个任务实际旧图重叠均为0；LogicalObj预设120no+120yes。
POPE三策略使用同一50张图和完整6题组；TextVQA按完整图片问题组累计到目标，本次恰为500题，未拆组。

所有选中图片存在性、完整解码、尺寸（源标注提供时）和 SHA 均通过，缺失/损坏为0。
GQA三个完整来源池的图片也全部通过。POPE、ScienceQA、TextVQA只恢复本轮选中图片：其全集清单内仍分别有每策略2700、1517、4500条未选题缺少本地图片，这是明确保留的未下载范围，不能声称完整Benchmark图片已就绪。

ChooseAttr与QueryAttr共享8张图，ChooseAttr与LogicalObj共享10张，QueryAttr与LogicalObj共享9张，三者共同交集1张。
**权威重叠口径为 `canonical_image_overlap_audit.json`：按数据集命名空间及图片SHA核对。**
原 `overlap_audit.json` 仅保留最初的字面ID字符串交集；例如 GQA图片ID与ScienceQA PID同为2993，不代表同一图片。该文件没有覆盖删除，新增回归验证此情况。
未来 Bootstrap 使用 `metadata.cluster_id`，按真实图片SHA聚类，不能默认每题独立，也不能将多Head/层当作样本。

## 4. 答案与评分边界

- ChooseAttr：保留作者真实选项及原顺序。75类Gold；最多的 white30、blue23、black19。240题中60题至少一个候选包含多个Token，完整序列Teacher-forcing评分，不只取第一个Token。
- QueryAttr：虽然标签恰好为 left110/right130，但 CSV 没有候选，仍按开放答案处理，候选与 Margin 字段不启用。
- LogicalObj与POPE：候选为真实 yes/no，提供候选诊断及实际生成评分。POPE正式生成指标保留 Accuracy、Precision、Recall、F1、Yes Ratio。解析沿用上游词法规则，不偷偷改进解析器；零分母额外标记并按0扩展。
- ScienceQA：保留原始 Choice 列表、位置和Gold Index。候选数量2/3/4/5分别162/140/189/9题；Gold字母A/B/C/D/E分别187/178/93/41/1。冻结Prompt要求输出选项字母，因此诊断LogP对应完整的**字母答案编码**，不是把候选文本首Token当成整段答案；原始候选全文另存。图片存在不等于视觉必需，没有根据模型表现剔除可纯文本回答的题。
- TextVQA：保留每题全部10个答案；原样复用固定 LLaVA `m4c_evaluator.py` 的归一化与leave-one-annotator-out软准确率，源码 SHA `8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774`。不简化成单标签或二选一。本轮没有把OCR Token注入输入。

候选诊断字段以 `diagnostic_` 开头；Margin为 Gold完整LogP减最强错误候选完整LogP，平局算错，保留所有候选长度/映射/LogP，不加EOS。
实际生成评分字段与诊断字段隔离。三个作者GQA子集只提供明确命名的机制用生成字面匹配，**不是官方GQA评价器结果**。
这些CSV本身经过作者预筛选，不代表无偏GQA总体，零旧图重叠也不能消除此来源偏差。

## 5. 输入协议与可比性

完整Prompt、模板哈希、原生Chat Template、图像预处理配置、Tokenizer/Processor文件SHA、动态Question/P−1、候选所有答案位置均已冻结。
GQA/POPE/TextVQA采用问题加短答案指令；ScienceQA采用真实Hint（如有）、问题和原选项加字母答案指令，不输入Lecture、Solution或答案。
不是把ChooseRel左右二选一Prompt强套到所有任务，也没有为不同干预改变Prompt。

Native Full采用确定性Greedy，`max_new_tokens=32`、EOS=2、beam=1，生成使用模型原生Cache；没有新增干预的Decode/Cache实现。
18个Smoke全部遇到EOS，无长度截断。未来正式结果必须继续记录截断及失败，不因回答错误更换样本。
本套件是冻结子集、固定HF Processor和明确的**image-only TextVQA**协议；评分器兼容不意味着可直接对比含额外OCR提示、不同图像预处理或完整测试集的论文榜单分数。
参考论文和官方LLaVA评测材料已下载并记录文件SHA，不将DyVTE的结论当作本轮观察。

## 6. CPU、GPU 与独立审计

- GPU前CPU：新增28项，全工程136项全部通过，无跳过；18个真实图片Processor输入与冻结Token映射一致。
- 固定执行代码 Commit：`24523aef248f5bf1943d8e0d51d8febb187a105b`。每单元2例，ChooseAttr另加首个多Token候选案例，ScienceQA另加首个最多选项案例，共18例，全部按输入特征提前冻结。
- 只使用空闲物理GPU0 RTX3090；没有终止其他任务。18例Native Full真实生成及候选评分全部完成，原始全词表Logits、生成Token和评分逐例保存。
- 无NaN、Inf、OOM；输入、Parameter Version、Hook足迹不变；模型参数与Buffer前后字节哈希完全相同。峰值allocated约14137.42MiB。GPU0已退出，验收时仅27MiB显示占用。
- 发布审计独立用NumPy Float64从原始Logits重算每个候选完整LogP、核对Greedy Token及答案评分；最大LogP差 `2.6645352591003757e-14` nats。仅此重算使用预设1e-10浮点求和界，不是放宽模型FP16正确性容差。
- 发布复核新增命名空间和下载范围保护，最终CPU141项全部通过、无跳过；`pip check` 无损坏依赖。原136项验收、原Smoke及原始抽样数据不覆盖。最终下载代码补齐自动获取发布者构建脚本与跨重试归档身份保护，不改变已执行评分算法或样本。

Smoke只证明链路可执行；不从这18例估计总体能力，更没有Full/B4/C2改进结论。

## 7. 阻塞、恢复与下一步前提

官方 GQA 下载和评价站点本机连接超时，未取得匹配的Balanced Test-Dev标注、完整图片映射与官方评价包；仅登记预检，不下载超大包，不拿ChooseRel替代。
其他六项冻结子集已无图片资源阻塞。此前POPE两张图超时及ScienceQA24题文本匹配歧义保留在下载审计，之后按原冻结ID恢复；没有丢弃、替换或补造问题。

**可直接使用的数据和链路：** 六项任务的Native Full；其中ChooseAttr、LogicalObj、POPE及ScienceQA支持候选诊断，QueryAttr/TextVQA采用开放生成，TextVQA必须软评分。
**尚不能宣称直接通过的干预：** B4/C2需要移植既有定义，先验证真正Question范围与选项/Hint/模板分离、Prefill P−1与后续答案Token语义、多Token候选因果性、No-op精确一致和Hook清理。单Token任务也不能跳过新输入映射验收。
**Physical Exit：** 所有新任务均需另行验证Prefill与Teacher-forcing/Decode、位置映射、退出后KV Cache和等价性；开放生成任务尤其不能复制旧单Token实验假设。没有授权就不运行这些实验。

## 8. 文件与复用

本报告绝对路径：`/data/lk/ATE-worktrees/phase40-benchmark-datasets/vlm_research/results/phase4_0_dataset_suite/report.md`。
`dataset_inventory.json`、`sampling_lock.json`、`manifests/`、`full_inventory/`、`resource_image_audit.json`、`resource_provenance.json`、`evaluation_protocols.json`、`canonical_image_overlap_audit.json`为冻结资料。
`tests/validation.json`为GPU前验收，`tests/pre_push_validation.json`为最终141项复核，`smoke/validation.json`为真实GPU状态，`publication_validation.json`为独立重算。
代码用法见 `../../benchmarks/README.md`；其他GPU Worktree通过只读 `load_manifest` 使用共享资源，数据变化会拒绝加载。使用独立输出目录，不回写本目录。
完成后只推送本实验分支；不合并main，不自动开始后续研究。
