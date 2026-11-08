# V²Drop 快速复现

**REPRODUCTION_STATUS：PASS**

固定官方Commit `2a9431150f8ace3a1d9abcdc96a7d46acca597df`；模型 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，RTX4090 / FP16 / Eager / Transformers4.49 / batch1。
服务器Git HEAD：`e281754ef17b3accc09d6938b5b9f020048b2aec`；实际执行的冻结ATE源码以protocol.json中的逐文件SHA256为准，发布分支继承既有证据Commit `71a5762cd8ccb8cc99efe9b6b0267d8abef70c46`。Python 3.11.8 / Torch 2.6.0+cu124 / CUDA 12.4。所有200题当前图像起点恰好为35；动态起点适配另经起点5/35的CPU检查，不声称这些真实输入存在起点错误。
官方L2差分、top-k、L3/L17/L22、0.5/0.5/0和至少1 Token均保持不变；从固定原始源码AST直接提取选择代码，仅把4处起点35替换成真实图像起点。采用HF层前钩子收缩Hidden、Mask、原始位置和共享RoPE，使用原生DynamicCache的逐层不同长度，保留较早层KV；没有修改任何ATE冻结核心。
官方环境为Torch2.1.2/Transformers4.37.2，README要求修补旧RoPE长度。当前HF4.49原生位置驱动RoPE保持原位置；不下载第二套模型、不安装新环境。
Smoke：PASS。4张真实图片，禁用剪枝时与Native Token及Logits完全一致；官方未改动固定35选择参考与动态起点选择完全一致；576→288→144→1、文本保留、RoPE、因果Mask和真实Decode/KV检查通过。CPU另测起点5/35的文本保护。

| Benchmark | 条件 | 题数 | Accuracy | 相对Native | 空输出 | 截断 |
|---|---|---:|---:|---:|---:|---:|
| GQA | Native | 100 | 59.0000% | +0.0000 pp | 0 | 0 |
| GQA | V2Drop | 100 | 61.0000% | +2.0000 pp | 0 | 0 |
| TextVQA | Native | 100 | 54.4000% | +0.0000 pp | 0 | 0 |
| TextVQA | V2Drop | 100 | 53.4000% | -1.0000 pp | 0 | 0 |

TextVQA为Paper-OCR十答案VQA软分，不是普通二值准确率。每组同一图片、Prompt、Gold、输入张量哈希与Scorer；GQA100 Native与此前同4090/同官方协议完整结果逐Token一致。仅探索性样本，未复现官方完整成绩，也未证明统计非劣效。

效率样本：同一200题中的GQA前15+TextVQA前15（按固定SHA排名）；每方法先32/128各预热一次。每输入/长度计时一次，方法顺序与长度顺序均交错；下表为30输入算术均值。固定32/128输出包含1次Prefill+31/127次Decode，忽略EOS，和正常Accuracy生成分开。
Prefill列包含Vision+Projector，Full=Prefill+Decode；统一Torch实际算子计数（矩阵乘/卷积，MAC=2），并保存每层真实Q/K/V形状和持久KV布局。另按真实逐层投影Token数、Q×K及模型维度独立复算全部Language FLOPs，按CLIP与Projector维度复算Vision，逐项精确一致。计数独立运行128步，其前32步与单独32步计时的全部Token一致；计数开销未混入计时。元素操作/Norm/Softmax/排序/索引不计入FLOPs，运行时间全部计入。


## 固定32输出Token

| 方法 | Full TFLOPs | Prefill TFLOPs | Decode TFLOPs | KV Live MiB | 视觉KV% | TTFT ms | TPOT ms | E2E ms | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 9.7697 | 9.3489 | 0.4208 | 351.50 | 100.0000 | 108.671 | 42.441 | 1424.338 | 13980.9 |
| V2Drop | 4.7557 | 4.3409 | 0.4148 | 164.91 | 35.2105 | 77.534 | 42.860 | 1406.208 | 13684.0 |
| C3-V2 | 6.8586 | 6.4640 | 0.3946 | 55.56 | 0.0000 | 89.031 | 38.871 | 1294.027 | 13690.2 |
| C3-V2+L11-L24 | 8.0646 | 7.6659 | 0.3987 | 181.56 | 43.7500 | 98.191 | 38.922 | 1304.775 | 13826.3 |

## 固定128输出Token

| 方法 | Full TFLOPs | Prefill TFLOPs | Decode TFLOPs | KV Live MiB | 视觉KV% | TTFT ms | TPOT ms | E2E ms | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native | 11.0762 | 9.3489 | 1.7273 | 399.50 | 100.0000 | 109.562 | 42.407 | 5495.277 | 13980.9 |
| V2Drop | 6.0433 | 4.3409 | 1.7024 | 212.91 | 35.2105 | 79.258 | 42.845 | 5520.592 | 13730.8 |
| C3-V2 | 8.0834 | 6.4640 | 1.6194 | 97.56 | 0.0000 | 89.045 | 38.839 | 5021.552 | 13717.5 |
| C3-V2+L11-L24 | 9.3021 | 7.6659 | 1.6362 | 223.56 | 43.7500 | 103.138 | 38.978 | 5053.379 | 13860.1 |

本轮128输出Token时，V²Drop相对Native的Full FLOPs减少45.44%，Prefill FLOPs减少53.57%，Decode FLOPs仅减少1.44%。TTFT均值下降27.66%，但E2E均值增加0.46%；32输出Token时E2E均值下降1.27%。本轮未观察到明显完整请求加速；这些是一次固定样本测量的描述统计。选择及Hidden gather的GPU时间均值为32输出0.288 ms、128输出0.292 ms。

V²Drop逐层持久视觉KV为L0–2:576，L3–16:288，L17–21:144，L22–31:1；合计6490，Native18432，实际Residency约35.21%。最后只剩1个视觉Hidden不表示历史层KV只剩1个。C3的视觉KV为0；Reaccess只在L11–24保留，共14层、43.75%。
Token选择额外时间在efficiency_summary.csv的selection_hidden_gather_GPU_ms；其余位置/Mask收缩包含在TTFT/E2E中，未单独隔离。TTFT、E2E为墙钟时间；主表TPOT=(E2E−TTFT)/(输出Token数−1)，CUDA逐步TPOT另列保存。Peak列为PyTorch峰值allocated显存（含模型），reserved另存CSV。GPU输入已预处理，计时不含图片读取、Tokenizer及H2D；包含真实Vision、投影、Prefill、Decode、冷请求控制器与KV构建。
ATE效率两组直接调用冻结MiddleWriter，未改层集合或算法；四方法同一输入、H1最后位置LM Head、同循环、同计数器，适合本轮同硬件效率对照。此次没有新增ATE Accuracy实验，不能将200题V²Drop分数与ATE全量成绩直接配对。
GPU进程已结束：`NVIDIA GeForce RTX 4090, 1 MiB, 24109 MiB, 0 %`。模型加载次数：1；未启动其他研究任务、未关闭服务器。

完整输入Manifest、预测、逐层布局与算子计数压缩保留本地并备份Windows；GitHub仅发布轻量源码、来源与许可证、Smoke、200题双条件预测、配置、统计及原始计时。windows_backup_receipt.json随结果发布；github_sync_receipt.json与completion_receipt.json在归档同步阶段生成并保存于服务器和Windows目录，实际Commit链接由最终汇报提供。

[固定官方源码](https://github.com/xuyang-liu16/V2Drop/blob/2a9431150f8ace3a1d9abcdc96a7d46acca597df/llava/model/language_model/V2Drop.py)。
