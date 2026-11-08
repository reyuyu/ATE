# RTX3090 Physical C3-V2 + L11–L24 七项正式评测

**状态：COMPLETE**。36,256/36,256 条：本轮新生成23,678；GQA12,578条严格校验后复用历史完整B2结果。
冻结模型：llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9；计算源码Commit：e281754ef17b3accc09d6938b5b9f020048b2aec。
环境：{'python': '3.11.8', 'torch': '2.6.0+cu124', 'transformers': '4.49.0', 'cuda_runtime': '12.4'}；FP16/Eager，Greedy，Batch1；本轮模型加载1次。

| Benchmark | 题数 | 正式指标 |
|---|---:|---:|
| GQA | 12578 | 59.922086% |
| MME | 2374 | 1806.602541 |
| ScienceQA-IMG | 2017 | 68.468022% |
| POPE | 8910 | 87.317621% |
| TextVQA | 5000 | 54.790000% |
| MMBench | 4377 | 63.010204% |
| OCRBench-v1 | 1000 | 281.000000/1000 |

MME 分任务（Accuracy+ 为同图两题均正确的比例）：

| Task | Accuracy % | Accuracy+ % | 任务分 |
|---|---:|---:|---:|
| existence | 96.666667 | 93.333333 | 190.000000 |
| count | 80.000000 | 63.333333 | 143.333333 |
| position | 73.333333 | 50.000000 | 123.333333 |
| color | 88.333333 | 76.666667 | 165.000000 |
| posters | 84.693878 | 70.748299 | 155.442177 |
| celebrity | 74.411765 | 54.117647 | 128.529412 |
| scene | 86.000000 | 73.500000 | 159.500000 |
| landmark | 89.250000 | 78.500000 | 167.750000 |
| artwork | 73.000000 | 50.000000 | 123.000000 |
| OCR | 77.500000 | 55.000000 | 132.500000 |
| commonsense_reasoning | 70.000000 | 45.714286 | 115.714286 |
| numerical_calculation | 37.500000 | 15.000000 | 52.500000 |
| text_translation | 60.000000 | 25.000000 | 85.000000 |
| code_reasoning | 50.000000 | 15.000000 | 65.000000 |

MME 分类总分：{"Perception": 1488.3882553021208, "Cognition": 318.2142857142857, "Total": 1806.6025410164066}

POPE 论文视图三策略：

| Strategy | 题数 | Accuracy % |
|---|---:|---:|
| random | 2910 | 89.450172 |
| popular | 3000 | 88.033333 |
| adversarial | 3000 | 84.533333 |

OCRBench v1 十子类（原始整数分，不换成 VQA 软分）：

| Category | 得分 | 满分 |
|---|---:|---:|
| Regular Text Recognition | 42 | 50 |
| Irregular Text Recognition | 39 | 50 |
| Artistic Text Recognition | 42 | 50 |
| Handwriting Recognition | 25 | 50 |
| Digit String Recognition | 2 | 50 |
| Non-Semantic Text Recognition | 4 | 50 |
| Scene Text-centric VQA | 105 | 200 |
| Doc-oriented VQA | 20 | 200 |
| Key Information Extraction | 2 | 200 |
| Handwritten Mathematical Expression Recognition | 0 | 100 |

与前次同协议原始 C3 的正式配对结果（MME 单位为 points，其余为百分点）：

| Benchmark | 前次 C3 | Re-access | 差值 |
|---|---:|---:|---:|
| MME | 1827.751801 | 1806.602541 | -21.149260 |
| ScienceQA-IMG | 68.963808 | 68.468022 | -0.495786 |
| POPE | 87.205387 | 87.317621 | +0.112233 |
| TextVQA | 50.400000 | 54.790000 | +4.390000 |
| MMBench | 63.010204 | 63.010204 | +0.000000 |

历史 GQA：Native 60.629671%；C3 57.274606%；Re-access 59.922086%；Re-access−Native -0.707585 pp，Re-access−C3 +2.647480 pp。本轮未重跑这些对照。

Physical正确性：七类真实图像Smoke通过。R仅L0–19写入；P−1/Decode仅L11–24读原始Image Key；L25前物理移除视觉Hidden；L2–5跳算答案Attention，Norm/MLP/Residual继续；SelectiveKV仅指定层保存图像，原始RoPE和位置索引精确。
同B2三token Dense仅作数值参考，无正式Dense实验。冻结atol0.10/rtol0.003、hidden RMSE0.01/cosine1e-4未放宽。详见physical_smoke_summary.json。
GQA保持原正式HF crop/USER模板与cap32；Paper五项逐字段复用前次Manifest、pad预处理和Gold；MME/POPE/TextVQA cap128，ScienceQA/MMBench cap1024。TextVQA是Paper-OCR，POPE为8910题（2910/3000/3000）。MMBench1176完整Circular组。
OCRBench v1官方1000题、930图；固定官方Commit6ddb0a570419ce7ccb6913cf1c5ed84930320e7e，原始标注与GoogleDrive图像ZIP保留；十子类官方子串评分，HME100k大小写敏感去空格；总分0–1000。Prompt=Vicuna_v1+单词/短语后缀，pad/CLIP336，cap128，未注入外部OCR。
官方原始id有143个跨dataset重复；dataset_name+id无重复。ATE使用官方行号0000–0999作为唯一键，原始id/Gold未改写。
MME保留landmark/03d5e3bfc958be38.jpg已知单图SHA差异，两题仍保留；本轮与前次C3使用同一恢复图片。
独立评分核验：36256条冻结评分重算通过；TextVQA5000条额外独立留一计算；OCR1000条运行原官方AST评分循环；MME14任务/1187对和MMBench1176组独立聚合一致。逐题得分不以Smoke正确率替代正式成绩。
可比性：五项与前次原始C3全部Manifest及输入张量哈希一致，可报告配对差值；GQA历史Native/C3/B2同输入绑定可比；无完整Paper Native或OCR对照。详见paired_comparability.json。
GQA历史Python3.11.15与本机3.11.8、驱动/GPU实例不同；Torch/Transformers/模型/算法/输入张量逐项一致；当前已存真实Physical Smoke的首步logits SHA及Greedy Token也与历史同题完全一致。保留历史来源，不宣称本轮重新生成GQA正式全量。
峰值/释放：{'before': 'NVIDIA GeForce RTX 3090, GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e, 610.57.04, 24576 MiB, 266 MiB, 23859 MiB', 'peak_allocated_mib': 14665.2099609375, 'peak_reserved_mib': 15366.0}；{'allocated_mib': 32.0, 'reserved_mib': 32.0, 'process_exit_releases_context': True}。最终资源检查另见shutdown_verification.json。
原始预测及完整映射Manifest均压缩保存，包含Gold、原始输出、解析答案、逐题得分、配对/题组和源码/协议绑定；不必依赖SQLite重评分。Windows和GitHub实际归档状态由最终receipt确认；本脚本不提前宣布安全关机。

人工逻辑复核补记：17/17 条原始输出、Gold、选项解析与得分一致，涵盖 MME 完整双题及 MMBench 完整四轮组。TextVQA34619 的十答案留一软分为0.3；不是二值准确率。ScienceQA10015 已额外核对官方problems.json和原图：金属选项指牛仔裤的纽扣/拉链，未发现映射错误。原Native15条审计仍为PASS15/15，原文件保存在前次C3 Windows归档。

质量输出标记：OCRBench 空输出36条、达到冻结128Token上限27条，HME子类为0/100；TextVQA空输出27条、截断5条；GQA历史空输出2条。所有记录均保留并进入完整分母，未为提高分数删题、放宽规则或重跑。未发现新增Scorer、Gold映射或Prompt问题，未发生OOM或非有限有效Logits；模型质量的局限不等同于执行失败。

OCRBench标注附加核对：1,000条完整原始记录及dataset+image+question组合均无重复；跨数据集重复原始id不作为全局主键。
