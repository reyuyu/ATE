# CVPR 2026 Information-Horizon 协议对齐：部分进度

**状态：PARTIAL；已按 2026-11-07 用户授权配置空闲卡自动接力。实时状态见 `relay/status.json`，服务为 `ate-information-horizon-relay.service`。**

已完成作者 Native 阶段 2,683/22,678 条；全部为 TextVQA/OCR。阶段 B 为 0/68,034。
这些完成项是确定性执行前缀，不是随机抽取的评测子集。不得将其均分当成 5,000 题正式 TextVQA 分数。
SQLite 检查点全部通过记录 SHA、输入绑定、协议绑定及独立重评分检查；无重复、无覆盖。

## 当前主表

| Benchmark | Published Reference | 作者 Native | HF Paper Native | Physical C3 | L11–L24 |
|---|---:|---|---|---|---|
| TextVQA | 58.3% (OCR) | PARTIAL 2683/5000 | NOT_RUN | NOT_RUN | NOT_RUN |
| MME | 1862 /2800 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| ScienceQA-IMG | 69.5% | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| MMBench | 64.6% | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| POPE | 85.9% (metric requires qualification) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

未完成比较不填 0、不计算伪 CI。阶段 A 完成后才允许阶段 B。

## 已完成的验证与复现边界

- 作者源码固定为 `75909b2936a13d7214e83514f0bc0cb9cc91139e`，其工作区未修改。
- 作者原始 `PrunedLLM.forward` 在 `pruned_config=None` 时使用未初始化的 K。本实验显式绑定作者既有的未剪枝父类 forward；未将剪枝预算 576 冒充 Native。
- 四卡各通过 15 个预冻结 Smoke：Full/No-op 与无实际剪枝分支的全部有效 Logits 严格 `torch.equal`，32 层完整 Prefill/Decode，无剪枝及残留 Hook。
- 分类为 `AUTHOR_CODE_REPRODUCTION / native-parent dispatch repair`，不是逐依赖完全复现。实际 Python3.11、Torch2.5.1、FlashAttention2.8.3，与作者 Python3.10、Torch2.1.2、FlashAttention2.5.9 不同；Transformers4.37.2 相同。
- 权重映射检查了 719 个张量/派生缓冲：686 个共享可学习参数张量全部精确相同；HF 词表额外 64 行不视作原生 LM 的对应参数。
- 32 个旧 Checkpoint RoPE `inv_freq` 与 FP32 公式不同，最大绝对差约 0.00122315；它们是非持久缓冲，不是学习参数。作者当前构造器重新生成公式。仍不能据此假定两后端运行数值相同。
- 阶段 B 的物理执行器真实 GPU 门禁尚未执行，不能标记已通过。
- 另有 15 个冻结输入的 CPU 跨实现对照通过：实际作者 Smoke 绑定与 HF 原生 Processor 的 Token IDs、视觉位置和图像 Tensor SHA 精确一致。这不是两后端 Logits 等价证明。

## 数据与评分

冻结实际输入共 22,678 条，7,774 个图片字节簇。
- 取得的作者格式镜像文件中 POPE Random 为 2,910、Popular 3,000、Adversarial 3,000，共 8,910，不补齐成 9,000。Random 若朴素顺序 zip 原标签，会出现 1,417 处标签不匹配；因此按图片和问题绑定，保留作者词法解析规则。对齐文件和原始顺序均保留；不能把这个子集的分数直接称作论文 9,000 题 POPE 分数。
- TextVQA 保留作者 OCR 内容和顺序，官方十答案共识软分；旧 Image-only 48.128% 不作为 OCR 版本基线。
- MME 保留 14 任务及图片双题，报告必须区分 Perception、Cognition、Total；本轮未完成该项，未产生正式分数。
- ScienceQA 保留 Hint/选项顺序及 single-pred Prompt，不输入 Gold 对话。MMBench 保留 4,377 个旋转记录和 Circular 分组；离线官方匹配器为 OFFLINE_REPRODUCTION，不冒充服务端裁判。
- 评测输入来自注明来源和 SHA 的镜像 eval.zip；未能独立取得原 Google Drive 文件核对字节。此限制完整保留。
- 常规任务生成上限 128；ScienceQA/MMBench 为 1,024；不沿用旧 32 Token 上限。

| Benchmark | 旧 ATE Native（历史，不重算） | 新 Paper Native |
|---|---:|---|
| TextVQA | 48.128% (Image-only) | 未完成 |
| MME | 1771.16 /2800 | 未完成 |
| ScienceQA-IMG | 66.04% | 未完成 |
| MMBench | 62.84% (Circular) | 未完成 |
| POPE | 85.91% (Accuracy) | 未完成 |

## 中断、清理与恢复

- 正式生成在 2026-11-06 22:02（Asia/Shanghai）触及磁盘安全线而停止；异常与 GPU 释放日志保留，检查点没有损坏。
- 本轮安全删除约 8.9128 GiB 的可再下载原始压缩包；MME/SEED 解压文件先逐项校验大小与 CRC32。原 GQA 图片和所有历史结果保留。
- `testdev_balanced` 的 12,578 题、398 张图片及本轮输入图片已重新验证 SHA256。没有为节省空间删掉评测题。
- 本轮暂存于内存盘的环境/权重/结果已全部恢复到 `/data`。共享模型、共享环境、main 和历史结果未改动。
- 2026-11-07 接力启用时四卡均由其他用户训练占用。本任务仅每 10 秒检测，连续两次确认无计算进程、显存占用不超过 1,024 MiB 且利用率不超过 5% 才启动对应冻结分片，不抢占或终止外部进程。
- 接力代码不纳入模型计算路径，冻结推理源码及 A/B 协议哈希未变；新增 8 项接力 CPU 测试与原有 6 项 CPU 测试通过。当前未因此新增答案，仍为 2,683 条；实时运行数量以检查点和接力状态为准。
- A 全部完成并审计后才进入 B；B 必须先通过四个分片的真实 GPU Smoke。失败、哈希变化或磁盘不足会停止，不自动重试或放宽门槛。服务退出不等于所有正式指标已完成审阅。
- `workers/A/rank*/results.sqlite` 是原始本地恢复点。Git 同时保存全量已完成记录 `per_question.jsonl.gz` 和摘要 CSV，不需要重跑这 2,683 条。

## 下一步（尚未执行）

GPU 空闲后继续阶段 A 剩余 19,995 条；其完成后，在统一 HF/eager 后端验证并执行 B 的 68,034 条。跨后端 Native 差异和同阶段 C3 算法损失必须分开。
最终才运行完整官方聚合与 20,000 次图片簇配对 Bootstrap；本轮没有新增测速、搜索、GQA GPU 任务、SEED 或 MM-Vet。
