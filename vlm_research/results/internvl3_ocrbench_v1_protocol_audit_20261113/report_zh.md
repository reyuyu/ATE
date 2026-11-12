Native **751/1000 可独立复算，但与作者公布880并非同一已验证评测配置**。已确认的主要错配是单图块256视觉Token与官方OCRBench动态切块，以及自定义Prompt后缀；动态分辨率是最有证据的主要性能差异候选，129分差距的定量归因仍为UNVERIFIED。**Late9仅3/1000源于997题原图SHA无法匹配，不是第3题生成崩溃、OOM或Token上限。当前原图已全部恢复并通过CPU核验，本轮没有补跑。**

本轮状态：CPU审计完成；{'PASS': 10, 'FAIL': 4, 'UNVERIFIED': 2}。FAIL指相对公开参考的协议错配或历史完整性缺失，不代表751计分失真。当前新增GPU生成=0、模型权重加载=0；完成4题CPU预处理Smoke，并读取历史3题严格Native/No-op/official GPU一致性证据。未启动1000题正式评测，未修改冻结源码、历史结果或GitHub。

| 核查项 | 状态 | 现有记录/本轮证据 | 官方参考/限制 | 证据文件 |
|---|---|---|---|---|
| 原始标注与题目映射 | PASS | 1000题逐项核对Question ID、原ID、问题、Gold、分类、数据集、相对图片路径；零差异。 | OCRBench v1原始JSON；SHA df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864 | cpu/dataset_validation.json |
| 原始图片SHA | PASS | 930张唯一图片逐一SHA256核验，覆盖1000题全部匹配历史原图Manifest；未换图、未重编码。 | 原始Google Drive ZIP历史回执SHA c82acb33a2f00cdb499ea13a3cee7f3d1584269af339aed3fcb4e381d5a9f975 | cpu/image_integrity.csv; historical/OCRBench_bundle_SOURCE.json |
| 动态切块及缩略图 | FAIL | 固定max_num=1、448x448、256视觉Token；与官方动态配置不符。 | VLMEvalKit OCRBench max_num=24，RGB动态448切块，多块额外thumbnail，最多25块/6400视觉Token。按同一原图947题会多块。 | cpu/official_dynamic_tile_plan.csv; cpu/cpu_smoke.json; official/VLMEvalKit_baseline_pinned/vlmeval/vlm/internvl/internvl_chat.py:316 |
| RGB、插值、归一化 | PASS | 4题CPU Smoke：历史像素哈希复现；同样max_num=1时官方预处理Tensor完全一致。 | RGB、448、BICUBIC及ImageNet mean/std；动态切块差异另列。 | cpu/cpu_smoke.json |
| 问题Prompt后缀 | FAIL | Give only the requested text or answer. Do not explain.；1000题均不同于官方问题后缀。 | Answer the question using a single word or phrase.；OCRBench公开默认不启用MPO改写。 | cpu/official_path_and_late9_closure.json; official/VLMEvalKit_baseline_pinned/vlmeval/vlm/internvl/internvl_chat.py:242 |
| 原生Chat Template及Tokenizer | PASS | internvl2_5/ChatML、原生中文system；1000题完整Prompt和input_ids复现，无Vicuna混用。 | 冻结模型官方conversation.py；EOS=<|im_end|>、151645。 | cpu/cpu_smoke.json; model_reference/conversation.py |
| 生成上限 | FAIL | max_new_tokens=128；Native 1000题均EOS，最长55 Token，截断0。 | VLMEvalKit公开默认max_new_tokens=4096；作者880原始run的完整参数未公开核验。该差异没有截断本批Native答案。 | cpu/scoring_validation.json; historical/native_stopv2/OCRBench/output_monitor_v2.json |
| Greedy、EOS和数值配置 | PASS | BF16/Eager、Batch1、Greedy、use_cache、logits_to_keep=1；独立Cache、原生EOS。 | 公开wrapper使用BF16与Greedy；精确作者backend/软件版本另列UNVERIFIED。 | historical/native_stopv2/protocol.json; historical/native_stopv2/environment_v2.json; cpu/cpu_smoke.json |
| Native前向及源码绑定 | PASS | 9个冻结执行源码SHA全部匹配；Native不安装Physical干预；2000条历史记录使用同一个协议及源码Manifest SHA。 | 历史Native/No-op/official logits严格相等Smoke 3题：0418、0007、0686；本轮只读取该GPU证据。 | cpu/source_integrity.json; historical/native_stopv2/OCRBench/smoke_validation.json |
| 答案解码及规范化 | PASS | 2000条Token IDs重解码与历史答案一致；raw_decode与计分文本零评分差。 | 普通答案小写、去外侧空白、换行替空格；HME100k大小写敏感且去空格；accepted-answer子串，非exact match/ANLS。 | cpu/cpu_smoke.json; cpu/rescored_predictions.csv |
| 官方0–1000评分 | PASS | Native独立重算751；原始官方循环、VLMEvalKit utility及OCRBench实际class evaluator全部一致；10类计分保留。 | 1000题每题至多1分，原始总分751/1000。VLMEvalKit附加Final Score Norm=75.1不能冒充原始分。 | cpu/scoring_validation.json; cpu/official_path_and_late9_closure.json; cpu/category_scores.csv |
| Late9历史完整性 | FAIL | 仅3题，2分；997题BLOCKED_IMAGE_SHA_UNAVAILABLE，不能报全量成绩。 | 只接受冻结SHA匹配原图；缺图不允许替换或补零。 | historical/late9/blocked_requests.csv; historical/late9/data_recovery_attempts.json; historical/late9_source/worker.py:164; cpu/official_path_and_late9_closure.json |
| Late9当前输入恢复 | PASS | 原997条阻塞题所需原图现全部匹配；CPU数据阻塞已解除。本轮新增Late9生成0。 | 原有单图块协议可在将来独立恢复；不等于已补完或通过新GPU生成。 | cpu/dataset_validation.json; late9_resume_readiness.json |
| 作者880发布值 | PASS | 原论文Table3与官方项目页均列InternVL3-8B OCRBench=880。 | https://arxiv.org/html/2504.10479v1#S3.T3 | references.json |
| 作者880原始运行快照 | UNVERIFIED | 尚无可验证的880逐题预测、原始运行Commit、完整flags、软件/backend及图片打包SHA。 | 当前/历史固定VLMEvalKit默认配置是可核对的公开参考，不能冒充880原始run的完整快照。 | references.json; source_manifest.json |
| 129分差距的定量归因 | UNVERIFIED | 单图块/Prompt错配已确认；动态切块是最有证据的主要候选，未做消融，不能给出各项贡献或保证修复后880。 | 880-751=129原始分；128上限未造成Native截断；计分错误和缺题已排除。 | cpu/scoring_validation.json; cpu/official_dynamic_tile_plan.csv |

**751与880的差异证据**

- 作者论文Table3和[官方项目页](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)均报告880，故不是将88.0误读成880。751与880相差129个原始分。
- [固定VLMEvalKit InternVL wrapper](https://github.com/open-compass/VLMEvalKit/blob/3c192143007e4eb3f5c1a36558a4f8c1451980cb/vlmeval/vlm/internvl/internvl_chat.py)将OCRBench分配到max_num=24；load_image使用RGB动态448切块并在多块时加thumbnail，最多25块、6400视觉Token。模型config自身max_dynamic_patch=12不能覆盖wrapper的OCRBench=24设置。当前全部固定1块、256视觉Token。按原始尺寸运行官方网格选择器，947/1000题需多块；实际4题CPU预处理覆盖1、11、16、25块，单图块分支与历史像素Tensor严格相同。
- 历史后缀为 `Give only the requested text or answer. Do not explain.`；官方为 `Answer the question using a single word or phrase.`。本轮实际调用官方build_prompt CPU路径，1000题均确认该差异。ChatML/internvl2_5模板和system本身可正确复现，不是误用了LLaVA/Vicuna。
- 128与公开默认4096上限不同，但Native1000/1000正常EOS、零截断、最长55 Token，不能把本批129分损失归为Token不足。未因追分提高上限。
- 官方原始评分循环、VLMEvalKit utility及OCRBench实际class evaluator均得到751；2000条历史Token解码与存档一致，raw_decode与计分text不引起任何评分变化。Gold、原图SHA、题数与分数单位错误已排除为已观测问题。
- Native错误249题中，Doc-oriented VQA错90、KIE错78，合计168/249约67.5%，与单块压缩细文本能力受限相容；这只是支持证据，不能代替分辨率消融或证明129分全部来自该因素。作者880原始预测/运行完整参数/图片打包SHA没有可核验快照，精确作者运行仍UNVERIFIED。

Native十类分数（CPU独立重算）：

| 类别 | 原始分 |
|---|---:|
| Regular Text Recognition | 49/50 |
| Irregular Text Recognition | 45/50 |
| Artistic Text Recognition | 47/50 |
| Handwriting Recognition | 45/50 |
| Digit String Recognition | 48/50 |
| Non-Semantic Text Recognition | 47/50 |
| Scene Text-centric VQA | 164/200 |
| Doc-oriented VQA | 110/200 |
| Key Information Extraction | 122/200 |
| Handwritten Mathematical Expression Recognition | 74/100 |
| 总分 | 751/1000 |

**Late9为什么只有3题**

冻结分支Commit为`c60a9919112a3cf5bec8c494a533cba56a97f29b`。worker.py第164–167行在expected image_sha256不在local_image_map时写BLOCKED_IMAGE_SHA_UNAVAILABLE并continue，不生成。997条blocked_requests与历史map无法匹配的题集完全相等，与3条已完成记录的并集恰好1000、交集为空。

当时官方Google Drive两次下载超时；备用echo840/OCRBench快照有不同图片字节。恢复审计记载另一次镜像包检查1000题、930唯一图片匹配0，已拒绝替换；本地已有资源仅匹配0053、0963、0988。这3题Native=2/3、Physical-D=0/3、Late9=2/3；9条保存答案均正常EOS，截断/空输出均0。这是数据SHA可用性阻塞，不是全量结果，也不得按2/3外推1000题成绩。

当前`/home/featurize/ate-local-data/bundles/OCRBench-v1/OCRBench_Images/`中930唯一原图均匹配，覆盖原997条缺失题；数据阻塞已解除，生成恢复仍NOT_RUN。原有三题和全部失败/下载诊断均保留。

**最小修复与下一步边界**

最小修复方案（本轮仅提出，未启动生成）

1. 保留751/1000原始结果并标记ATE_SVH_1TILE_SCORER_ALIGNED；不改分、不改图、不覆盖协议与Checkpoint。它适用于原单图块同条件内部比较，不是作者880的同协议复现。
2. 若仅补齐旧Late9：使用cpu/image_integrity.csv重绑定本地SSD路径，保持旧single-tile、原Prompt、BF16/Eager、128上限和冻结算法。997题所需原图已恢复；先在新卡对0053/0963/0988及少量缺失题进行绑定、原生No-op、物理Cache/位置Smoke，再严格恢复997题；本轮未执行该步骤，不保证跨GPU逐Token一致。
3. 若要对照作者公开配置：另建Native-only版本协议，固定官方VLMEvalKit Commit及模型Revision，使用OCRBench配置max_num=24、448动态块和thumbnail、官方问题后缀、原生chat及Greedy/正常EOS/公开默认max_new_tokens=4096。不是根据成绩选参数。先测1块/多块/最大块数资源Smoke；3090显存不足就记录BLOCKED，不偷偷缩图、降块数或换backend。未获后续正式评测指令前不跑1000题。
4. 固定256视觉Token的现有SVH/Late9不能直接套入多图块；任何将来动态多块联合对照都需独立输入/角色/KV正确性门槛，不能混入本轮或复用旧三题冒充同协议。
5. 评分器原样保留。可直接复用本轮官方原始循环与VLMEvalKit实际class evaluator双重CPU核验，不替换成exact match、ANLS、TextVQA软分或追分规范化。
6. 要解释129分各项贡献，还需要预先冻结、单独授权的分辨率/Prompt控制实验。当前证据只能确定协议不一致，不能承诺修复后880或把全部129分归因于切块。


**复现与追溯**

- Native完整结果Commit：`3297559f21d3de55a6630a5491c9ecf66301c4b7`；原始PARTIAL Commit：`6c294aae67ddb2a5cb3706ded76dcf3edb26c859`；单图块推理协议SHA：`f68ed284c5874532b997100bc829b9e434ded282c2d12256c7a23dc0ab2a7ada`；来源Manifest SHA：`9fb0eb9b68cf5406e1881bfe44f6c7f7d31d053bf65ed30657c7ac1ad30db816`。本轮核对9个执行源码SHA全部相符。
- 官方固定VLMEvalKit：`3c192143007e4eb3f5c1a36558a4f8c1451980cb`；本轮读取当前`c247c4d61d92be5643802aa9be5f2bc34d913d2a`，相关wrapper/utils/scorer与前者字节一致。它是可审计公开默认参考，不冒称2025年880原始run的完整快照。
- OCRBench原始注释Commit：`6ddb0a570419ce7ccb6913cf1c5ed84930320e7e`；本轮官方快照`d1901a9500c7232040b5b5806df77a15312a3509`的1000题JSON与历史SHA一致；未使用FullTest/v2。官方图片ZIP SHA来自已存原始传输回执，当前各图片逐一与冻结Manifest核对；作者880内部使用的数据包装字节不能由该回执单独证明。
- 模型固定Revision：`853e3a797a661694b1b8ece0cb72dc2b23e3dac9`，复用现有VERIFIED模型回执，不重复读取15.89GB权重。旧751运行GPU为GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e，本轮当前卡为GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6；本轮仅CPU，不生成可混淆的新成绩。
- CPU脚本：code/audit_cpu.py及code/audit_final_checks.py；使用`/home/featurize/ate-local-envs/internvl3-torch260-tf449/bin/python -B`。前者强制CUDA_VISIBLE_DEVICES为空，cpu/cpu_smoke.json确认cuda_initialized=false。GPU必要一致性证据仅引用历史Smoke，未假称本轮重跑。
- 服务器持久化审计目录：`/home/featurize/work/ATE/vlm_research/results/internvl3_ocrbench_v1_protocol_audit_20261113`；执行SSD目录：`/home/featurize/ate-audits/internvl3-ocrbench-v1-20261113`。所有主要证据在此目录单独保存，并由evidence_manifest.json及SHA256SUMS绑定；未推送GitHub。

来源链接与对应证据见references.json。当前GPU空闲和保护文件核对见validation.json。
