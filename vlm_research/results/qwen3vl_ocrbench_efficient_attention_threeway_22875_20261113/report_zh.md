# Qwen3-VL OCRBench1000：补充冻结 SVH-R 与三组对照

状态：**COMPLETE_WITH_NUMERICAL_WARNING**。只新增 SVH-R；已完成的 Native/Physical-Original 原始预测和归档保持不变，未重复生成。

| 方法 | 得分 /1000 | 覆盖 | 空输出 | 截断 | 自然EOS | 峰值分配 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Native | 868 | 1000/1000 | 0 | 0 | 1000 | 21.109 |
| Physical-Original | 38 | 1000/1000 | 0 | 88 | 912 | 20.003 |
| SVH-R | 835 | 1000/1000 | 0 | 5 | 995 | 21.085 |

Native/Original 来自本轮内存高效 Attention 已完成实验；三组共用同一原1000题、930张图、Question/Gold/顺序、四项输入Tensor哈希、全分辨率Processor、Prompt、BF16与新Efficient后端。不是旧Eager结果，也不是混入其他分辨率的新协议。

| 配对比较 | 分差 | Accuracy差 pp | 错→对 | 对→错 |
|---|---:|---:|---:|---:|
| Physical-Original - Native | -830 | -83.0 | 3 | 833 |
| SVH-R - Native | -33 | -3.3 | 17 | 50 |
| SVH-R - Physical-Original | +797 | +79.7 | 798 | 1 |

## 冻结算法与真实物理检查

Qwen SVH-R 使用0-based **L19–L35**回看，沿用原写入L0–L21和L22前视觉Hidden退出、L1–L6答案Attention跳算。它不是InternVL Late14或LLaVA L11–L24配置。主路径L22后只有文本Hidden；晚层视觉KV从实际L22输入的因果图像前缀构造：L22–L34共13个完整块，L35只计算Q/K/V与RoPE并截获KV，随后释放临时视觉Hidden。无第二次Vision/DeepStack，不把辅助分支开销省略成零。

持久原图KV仅存在L19–L35共17层；其余13个非跳算层只存文本KV，6个跳算层无持久KV。KV Live和Storage Capacity分开审计，每题缓存独立。所有1000题实际Q/K长度、Key访问集合、视觉退出、KV合并位置与文本Token顺序检查通过。不是Dense Mask假物理压缩。

原 Physical.forward、SelectedKV、Original.context 源码原样继承。唯一新适配：将冻结构造器原本仅拦截Eager的KV捕获钩子桥接至本轮SDPA调度，之后调用与Native/Original相同的Efficient-only算子。原权重、层集合、Mask逻辑值和图像预算未改变。GitHub原始physical.py/engine.py与服务器快照逐字节哈希核验相同。

## Smoke与数值限制

0308、0601、0738三题完整自然EOS Smoke保留并直接计入1000题，另对0308做一次同前缀冻结Functional-R诊断，不计入正式分数。SVH-R Gate为 **PASS_STRUCTURAL_AND_GREEDY_WITH_NUMERICAL_WARNING**，原严格阈值max_abs≤0.5、mean_abs≤0.02；本次严格数值门禁为 **FAIL**，短题全部Greedy Token一致。逐步误差见numerical_comparison.json。

此前Original对Dense-F的严格数值FAIL保留不变；其38分严重退化不能因本次回看结果而隐藏或改写。结构通过/Greedy短样本一致不证明全局数值等价；没有全量Functional-R或Eager对照，不能将所有质量变化归因于某一单因素。

0738保留4961×7016原图，grid=[1,304,214]，16264个LM视觉Token，总长16307；没有降分辨率、跳题或失败补零。三组所用实际Attention为aten::_scaled_dot_product_efficient_attention / CUTLASS，Math/Flash/Cudnn后端禁用。Smoke保存真实CUDA算子证据。

## 协议与可追溯性

- 模型：Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，复用本地已校验权重。
- RTX3090 24GB；PyTorch2.6.0+cu124、Transformers4.57.6；BF16、Batch1、Greedy、自然EOS、上限128、logits_to_keep=1；原单线程与TF32关闭。
- 冻结SVH-R源码Commit：f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760。本次隔离适配与执行源码SHA见protocol.json和SOURCE_SHA256.json；工作区HEAD e281754ef17b3accc09d6938b5b9f020048b2aec没有修改。
- 1000题官方OCRBench v1评分与独立注释逻辑逐题完全一致，Token IDs重新解码与保存文本一致；全部错误/空输出/截断原样保留。
- 逐题轻量预测、Gold、输入哈希、类别分数和配对表都保存；完整Attention/KV审计、日志与Gate Logits保存在本地独立目录及压缩归档。
- 输入和Gold分离，Gold未传入模型。主请求每题独立KV；三次Smoke直接复用，仅997条后续正式生成，无Native/Original重跑。

## 保存与资源

模型、图片、环境及实际运行输出在本地ext4盘；持续盘仅保存轻量证据和压缩checkpoint/归档。新SVH-R目录与此前Base目录独立，旧完整预测和archive的SHA不变。

GPU实验进程已退出，GPU释放证据见gpu_release.json。Windows实传与逐文件SHA校验见windows_backup_receipt.json；GitHub提交由后续独立分支发布回执确认，报告生成时不虚构Commit。未修改main，未关闭服务器，未启动其他Benchmark或效率实验。

检查覆盖说明：Gate检查全部文本Query和固定的视觉Query；正式逐题使用固定的文本/视觉Query子集验证可见Key集合，并核对实际Q/K长度与完整持久KV，不声称逐题比较全量Attention矩阵或全局数值等价。Gate保存的全部主路径绝对位置索引已在CPU复核。耗时包含验收审计，未作为标准效率或TPOT实验成绩。
