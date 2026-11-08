# GQA 官方对齐协议三组正式评测

**状态：COMPLETE。** 三组各12,578题，共37,734条正式生成；全部逐题预测保留，独立CPU复核及官方评分脚本通过。

| 方法 | 正确数 / 12,578 | GQA Accuracy | 相对 Native |
|---|---:|---:|---:|
| Native | 7788 | 61.9176340% | — |
| C3-V2 | 7244 | 57.5926220% | -4.3250119 pp |
| C3-V2+L11-L24 | 7453 | 59.2542535% | -2.6633805 pp |

| 配对差值 | 改善 / 恶化 | Accuracy差值 | 按Image SHA簇Bootstrap 95% CI |
|---|---:|---:|---:|
| C3-V2 − Native | 588 / 1132 | -4.3250119 pp | [-5.1200, -3.5493] pp |
| C3-V2+L11-L24 − Native | 562 / 897 | -2.6633805 pp | [-3.3420, -1.9966] pp |
| C3-V2+L11-L24 − C3-V2 | 545 / 336 | +1.6616314 pp | [+1.0181, +2.2897] pp |
| Native − Historical Native | 662 / 500 | +1.2879631 pp | [+0.6800, +1.8933] pp |

图像SHA簇398个，有放回抽取398簇并按实际题数重算Accuracy差值；10,000次，seed20261108，percentile区间。此为固定testdev集合上的配对不确定性估计，不用于证明模型与官方原生后端等价。
新协议Native比历史60.6296709%变化 **+1.2879631 pp**；比官方四舍五入公开62.0%变化 **-0.0823660 pp**。官方值只作参考，未据此调参。历史与新协议的Prompt、预处理及硬件运行记录不同；三种新方法内部输入完全一致。新旧Native差值不能单独分解System、pad和跨硬件浮点残余贡献，本轮没有重跑旧协议全量基线。

## 协议与可追溯性

模型 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；代码基线 `e281754ef17b3accc09d6938b5b9f020048b2aec`，冻结Physical源码哈希与历史GQA执行冻结逐项相同。复用4090本地盘模型、数据和既有Torch2.6.0+cu124 / Transformers4.49.0环境；FP16/Eager/batch1/greedy/EOS2/max_new_tokens32。
完整Vicuna-v1 System + USER + 图像 + 官方原题及single-word-or-phrase指令；RGB→CLIP mean色补方形→CLIP336。按官方prepared jsonl原始顺序完成Native、C3、Reaccess；旧清单使用另一顺序，历史对照全部以qid对齐。每条新预测绑定同一个样本输入张量哈希、Prompt哈希及protocol哈希。
Protocol SHA256 `1a7db2cacd807105fe6ccd1bf9c91652032050c242c2a49ddd5d7f490dcdeec4`；输入Manifest SHA256 `0ddd9d4112e7fe7ce9cd77b8ce48b603581c0abf6a3feefca4de4446587caeb8`；官方Gold SHA256 `14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318`。详见protocol.json、execution_config.json、prompt_examples.json和本地完整input_manifest.jsonl.gz。

## Physical验收

3道固定GQA题×2种Physical条件，共6次严格Dense数值比较全部PASS；没有使用PASS_WITH_NUMERICAL_WARNING豁免。L20/L25前视觉Hidden退出、R只在L0–L19读取Image Key、回看只在L11–L24、L2–L5物理跳算、Norm/MLP/Residual、RoPE和紧凑KV位置均通过。Dense仅作短序列检查，正式37,734条均使用各自冻结实现。
六次比较最大Hidden Relative RMSE为0.004527092，原阈值仍为0.01。2道旧协议题×2种C3的跨硬件输入及完整Token输出均匹配历史。上一轮POPE约0.0101545的严格FAIL保持原样，本轮没有重分类或修改阈值。
正式进程三组共享一次模型加载。此前新入口误将回看结果从仅含Native/C3的历史表读取，导致0条正式生成时退出；已修正为读取独立历史回看文件，并在CPU阶段验证来源。失败代码、日志和配置保存在attempt1_entry_lookup_failure；总加载次数记录为2，未隐瞒。

## 独立评分审计与资源

全部37,734条记录检查qid唯一性、数量、官方Question/Gold/Image映射、输入绑定、token解码、EOS/截断和原始输出。另以逐字符移除尾句号再lower的独立逻辑复算，并实际调用未修改的官方转换和GQA evaluator；0差异。此前同4090的200条官方对齐C组与本次Native的Token和分数全部一致。

| 条件 | 空输出 | 截断 | 最长生成Token数（含EOS） |
|---|---:|---:|---:|
| Native | 0 | 0 | 7 |
| C3-V2 | 0 | 0 | 8 |
| C3-V2+L11-L24 | 0 | 0 | 8 |

GPU已释放：`NVIDIA GeForce RTX 4090, 1 MiB, 24109 MiB, 0 %`。无遗留推理进程；未启动其他Benchmark、层搜索或V²Drop。逐题SQLite同步事务持久化，断点只跳过绑定相同的已完成样本。
完整预测压缩文件、Manifest、逐题评分表及必要配置留在独立本地目录；Windows备份和GitHub远端逐文件SHA校验结果见本地windows_backup_receipt.json及github_sync_receipt.json。GitHub仅同步轻量源码、正式分数、统计、哈希和报告，完整预测不上传。

[官方7B Model Zoo参考](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/MODEL_ZOO.md)；[此前协议审计](https://github.com/reyuyu/ATE/commit/25989d8c0e00cf53a980c726be94d84feaabbc14)。
