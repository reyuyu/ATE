# Qwen3-VL-SVH 六项作者完整协议审计
**状态：BLOCKED_AUTHOR_PROTOCOL / BLOCKED_SVH_OFFICIAL_RUNTIME_PORT。正式生成 0 条，未加载模型、未启动 CUDA。**
## 结论
没有把旧 HF/Eager/Greedy 队列恢复成所谓官方评测。P0 缺少作者逐任务、固定版本的完整配方；P1 缺少同一官方 vLLM Runtime 的已验证 Physical-Original 端口。按本次停止规则保存阻塞证据，不用旧成绩填补。
## 已验证与缺失
- 作者 Qwen3-VL 源码固定为 `96588727e44c78b25ba03ea03b8e12f7e64fd0da`。README 明确为 vLLM，并提及部分任务定制 Prompt 尚需文档；没有给出这六项逐任务框架 Commit、Runtime/Backend 版本、有效输出上限和 OCR 政策的完整锁定。
- VLMEvalKit 候选 `051aa811ec27c494919bda944654de884fd70c41` 注册项：`use_vllm=True`、`use_custom_prompt=False`、`max_new_tokens=16384`、temperature=0.7。README 全局参考上限32768不能直接覆盖任务，框架注册上限16384也不能当作六项已核验配方。不能把当前禁用的 Prompt Mixin 当作实际生效 Prompt。
- 该 vLLM SamplingParams 调用未显式传逐请求 seed；需查清 seed=3407 的引擎/请求作用域。lmms-eval 的当前 simple Qwen 入口是 HF，不是同一个官方 vLLM 运行路径。
- 现有 Physical-Original 直接使用 Transformers 的模型、Eager Attention、DynamicCache 和自定义 CompactKV。没有通过验证的 vLLM/TP2 端口，不能拿 Native vLLM 对比 Physical HF。当前模型环境没有安装vLLM；未修改共享依赖或下载大型环境。
- 固定模型 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 共16个资产逐文件重算SHA256，均与历史绑定一致；没有加载权重。EOS配置为[151645,151643]。
- 历史 `NUMERICAL_EQUIVALENCE_FAILED` 保留。最后修复仅固定短题成功，中位输入仍失败、最长题未完成；本轮没有重新判PASS。若未来同Runtime质量探索获准，其数值限制必须继续明确标注。
## 数据盘点（不是作者推理协议通过）
|任务|要求题数|本机历史清单题数|引用图片缺失|当前结论|
|---|---:|---:|---:|---|
|GQA|12578|12578|0|作者配方未验证|
|ScienceQA-IMG|2017|2017|0|作者配方未验证|
|TextVQA|5000|5000|0|作者配方未验证|
|POPE|9000|8910|0|作者配方未验证|
|MME|2374|2374|0|作者配方未验证|
|MMBench|4377|4377|0|作者配方未验证|

POPE官方三策略标注另有3000×3条，已保存绝对路径和SHA；旧8910题子集不用于本轮9000题。其他任务清单已盘点文件存在性并做小范围图片SHA抽核，**未完成作者输入冻结和全集图片SHA复核**。未沿用LLaVA Prompt、生成上限或Paper-OCR作为Qwen官方输入。MMBench的Judge要求尚未锁定，不能以规则解析替代后宣称官方分数。
## 硬门槛与未运行项
P0未通过，故未冻结单卡分片或TP2模式；P1官方Native/Physical Smoke、P2最短/中位/P95/P99/最长真实输入显存门槛、P3每任务48题输出验收、P4全集均未运行。CSV空表和空分数表示未测，不表示0分。不能给出Native/SVH得分、退化幅度或Bootstrap区间。
历史POPE约71 Token长输出、TextVQA 15552视觉Token/115.33GiB Eager申请、ScienceQA字母/整数Gold错误和NFS父路径阻塞均作为反例保存。ScienceQA零基整数Gold到原始选项字母的CPU回归通过；这不等于运行了作者评分器。因为本轮没有GPU请求，不能说已解决长输出或OOM，也不能判定两卡官方高效Attention一定不够显存。
## GPU与存储
原四卡主机由hostname、PCI Bus ID和UUID重核。候选GPU2 `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`、GPU3 `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185`；实际占用0张，GPU0/1其他任务未干扰。前序SparseVLM+SVH 35256条已完成独立复算和本地提交，之后才进入本轮。
工作树和输出均在/data的本地ext4，父目录挂载及剩余空间记录于gpu_binding.json。当前磁盘仅约数GiB空余，未清除历史数据、未复制模型，也未自动安装大型Runtime。没有NFS检查点队列和后台GPU轮询。
## 交付与后续条件
已提供六任务协议矩阵、原始官方源码快照/SHA、历史失败证据、模型/数据清单、CPU门禁测试、Runtime兼容诊断和空结果表。审计图只展示门槛状态，不是质量曲线。
要进入正式运行，须先取得可证明的六项作者版本化配方（Prompt、像素/OCR、采样/EOS/上限、Scorer/Judge），再实现并独立验证同Runtime的Physical-Original端口；随后才可解锁最坏输入和48题预检。不自动更换为HF/Eager、不缩分辨率、不恢复旧采样队列。
main和历史源码未改。仅本独立分支提交/推送本次真实阻塞证据；完整分数和性能图留空。
## 复现

```bash
cd /data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_official_audit.audit sources
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_official_audit.audit freeze
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.qwen3vl_svh_official_audit.audit report
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_official_audit.audit release
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python -m vlm_research.qwen3vl_svh_official_audit.audit checksums
```
