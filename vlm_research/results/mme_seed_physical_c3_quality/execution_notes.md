# 执行记录与协议边界

本轮只进行质量评估。Native与Physical C3使用同题、同图片Tensor和模型配置，没有运行E2E性能基准、Kernel搜索、训练或新的层集合搜索。

## 冻结与来源

- 独立分支：`codex/mme-seed-physical-c3-quality`。
- 正式推理检查点绑定执行Commit：`c4bf0ce4f81acc9f6ec623c5507857c1ddb7efe6`。
- GPU执行核心最后变更Commit：`fea02fcaa6cbc1170aadf844d9cd42a16a97b379`；前者增加完整Qwen CPU输入审计，未改动GPU核心。
- `source_manifest.json`绑定GPU执行文件。统计与报告脚本明确不属于该GPU核心哈希集合；交付Commit保留这些脚本及最终结果。
- 历史Qwen Physical-Original执行器逐字节复用，未换成Dense F、Repaired、Zero-Key或cuBLASLt版本。其历史严格数值等价验收失败限制保留。

## Qwen 2,048视觉Token协议修订

原Processor最大分辨率在24GiB RTX3090上不可运行。用户随后明确批准统一2,048个语言侧视觉Token上限；只改变官方ImageProcessor的`size.longest_edge=2048*32*32`，LLaVA输入完全不变。旧预检记录与新协议均保存，没有覆盖历史研究结果。

实际安装的Processor使用`patch_size=16`、`merge_size=2`。全部MME及SEED图片均实际执行修改前、修改后Processor，记录语言侧视觉位置数、grid、像素Tensor SHA及缩放尺寸。修改后没有图片超过2,048，未截断Token、删除大图或按输出筛题。

两张卡分别在首个Native最坏输入Smoke中OOM，发生于Qwen视觉编码器的Eager FP32 Softmax，尚未进入语言Decoder或Physical C3。一个输入含8,160个ViT Patch，另一输入含8,192个ViT Patch；这与合并后2,040/2,048个语言侧视觉Token相符。两卡均只有一份模型。完整Tensor Shape、峰值显存和异常堆栈分别保留。

因此Qwen状态为`STOPPED_OOM`，没有启动任何正式Native/C3问题预测，也没有新增Dense/Physical数值等价通过结论。没有降低到1,024 Token、切换Attention Backend、调整精度或进行Kernel修复。Qwen进程退出后，空闲设备用于已通过门禁的LLaVA完整质量评估。

## LLaVA与评分

LLaVA保持冻结FP16/Eager协议。两卡分别通过真实Native/HF/No-op有效Logits逐位一致、物理视觉Hidden退出、目标Mask、真实Attention跳算、RoPE、独立KV及Hook清理检查。SEED四候选Cache分叉评分与四次独立冷请求的有效Logits严格相同。

MME保存完整自由生成与官方Yes/No解析结果，最终调用原始官方工具交叉复算全部14项分数。官方工具依赖的scikit-learn、SciPy、joblib及threadpoolctl安装在隔离CPU目录，未修改模型推理环境；固定版本写入`official_scoring_dependencies.json`。

SEED对完整候选答案逐Token计算条件LogP，以官方平均负LogP排序选择选项，最终报告ACC。官方重复Question ID保留全部原始记录。候选首Token由P−1预测，后续Token经真实增量Decode，因此没有绕过C3答案阶段干预。没有以32 Token上限截断候选。LLaVA候选标签跨度与官方适配器逐题核对无差异。

MME和SEED分别进行20,000次Image SHA整簇配对Bootstrap。MME在每次重采样中重算Accuracy、Accuracy+与官方任务/总分；其差值单位为Score Points。SEED差值单位为Accuracy百分点。所有区间均为探索性95%CI。

## 交付状态

只有实际完成全量、通过检查点与官方评分复核的模型/Benchmark矩阵才报告完整得分。Qwen未测得分不填零、不引用官网Native代替，也不推测质量差。整体任务因Qwen资源门禁停止而标为`PARTIAL`。实际覆盖、GPU释放与完整产物哈希以最终`completion.json`、`validation.json`及`gpu_release_audit.json`为准。

Smoke原始Logits数组保存在服务器资源目录；GitHub只包含路径与SHA索引，不包含这些数组本身。
