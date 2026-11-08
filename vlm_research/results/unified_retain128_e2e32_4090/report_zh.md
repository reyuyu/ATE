# ATE Retain-128 统一 E2E@32 效率复现

完成 150/180 个正式请求：Native 与 4 个外部方法各30题。FastV 缺少可核验的官方平均128工作点，未运行，不填数值。只运行效率实验，OCRBench/质量/训练请求均为0。

| Method | Actual Avg Visual Tokens | KV@32 MiB | E2E@32 ms | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|
| Native Full | 576.000 | 351.500 | 1879.056 | 120.513 | 56.727 |
| FastV-128 | — | — | — | — | — |
| SparseVLM-128 | 145.406 | 136.203 | 1997.044 | 172.807 | 58.846 |
| V²Drop-128 | 131.000 | 129.000 | 1884.914 | 105.921 | 57.387 |
| ApET-128 | 128.000 | 127.500 | 1888.067 | 108.217 | 57.415 |
| DUET-VLM-128 (C+all) | 128.000 | 127.500 | 1907.798 | 108.420 | 58.044 |

配置与论文对应关系：

- SparseVLM：官方 retained_tokens=128，top-k [303,110,36]，保留原始回收聚类，实测层布局 576×3、312×4、117×9、39×16，均值145.40625；不能称实际平均128。
- V²Drop：论文 Appendix A 的保留比 [0.28,0.25,0]，进入L3/L17/L22执行；原始 floor/min1 不变，576→161→40→1，均值131。旧“192”复现是演示参数576→288→144→1，并非论文192工作点。
- ApET：官方已发布192→64、L16、FPS/basis10组合，在同一32层骨干的均值128；此组合出现在LLaVA-NeXT脚本，LLaVA1.5脚本公开288→96及96→32。按论文128预算迁移已发布参数，未变更算法；缺少证明它就是LLaVA1.5论文质量表128行的独立配置记录，不能宣称完全对应。
- DUET-VLM：论文附录Table12 的170 dominant（含CLS）+35 contextual、width4，205→102→0，均值128。固定官方推理源码默认所有图后文本参与排序，故明确为C+all；C末Token路径仅为注释，没有改写成C，不使用训练权重。
- FastV：官方固定Commit提供K/R机制和静态KV分支，但没有可唯一确认的平均128 preset。不能把深层保留128当作层平均128，也没有反解K/R凑预算；本项BLOCKED_EVIDENCE。

同机改善（相对本轮Native，按30题算术均值）：

- SparseVLM-128：KV减少61.25%，E2E减少-6.28%。
- V²Drop-128：KV减少63.30%，E2E减少-0.31%。
- ApET-128：KV减少63.73%，E2E减少-0.48%。
- DUET-VLM-128 (C+all)：KV减少63.73%，E2E减少-1.53%。

测量及核验：统一FP16/Eager/Batch1/Greedy、use_cache=True、logits_to_keep=1，同一冻结GQA15+TextVQA15输入Tensor SHA256；含Vision/Projector/Prefill/31次Decode与必要清理，排除预处理/H2D/写盘。KV以真实终端K/V形状、dtype、有效长度独立重算，未使用总显存或理论预算替代。每题32输出Token、忽略EOS。
全程GPU UUID：`GPU-724fa9c3-d33a-df3f-e349-533cb4634a53`。输入轮换方法顺序，统一预热；每5题Native探针检查，共6次全部PASS，最大相对漂移0.999%，未自动重设基线。Native与原已验收32Token结果逐题完全一致；Native/Bridge、V2Drop禁用No-op的Token/首Logit完全一致，真实Hidden/Mask/RoPE/KV与31步Decode Smoke通过。

与此前约192档比较：见 comparison_with_previous192.csv，保留真实旧均值。虽然GPU UUID和输入相同，两个预算属于不同运行时段，不能把跨时段差值当作严格的配对预算因果效果；同时给出以各轮Native归一化的E2E变化。旧V²Drop参数本就不对应论文192，不能做论文192→128质量效率联合结论。

用途限制：可作为同一冻结HF模型、Eager、静态真实KV、32输出Token条件下的效率证据；必须显示真实视觉均值和DUET C+all标签。它不是六方法均严格平均128的齐全主表，也不能绑定到未验证的公开质量分数。

官方来源：
- FastV：https://github.com/pkunlp-icler/FastV/tree/d1659729b5bf1be225e99ee15783deeea80f63b1
- SparseVLM：https://github.com/Gumpest/SparseVLMs/tree/734d9003879689713a132b83297dfad060a5c9af
- V²Drop：https://arxiv.org/html/2509.01552v2 （Appendix A）；源码Commit 2a9431150f8ace3a1d9abcdc96a7d46acca597df。
- ApET：https://arxiv.org/html/2602.19870v1 ；https://github.com/Maqkccx/ApET/tree/70cc01b215f4d74c5661c699919fe3aeda48bed0
- DUET：https://arxiv.org/html/2602.18846v1 （Appendix C/Table12）；https://github.com/AMD-AGI/DUET-VLM/tree/cd69e489f39a2a51e3b278a470dc8dc32c00e0fb

持久化结果根目录：`/home/featurize/work/ATE/vlm_research/results/unified_retain128_e2e32_4090`。独立适配器：`/home/featurize/work/ATE/vlm_research/external_baselines/retain128_compatibility/`。所有主要文件绝对路径见absolute_file_paths.txt。检查点与完整Smoke轨迹留工作盘，GitHub发布的实际文件范围与未上传项由github_publication_manifest.json记录。本任务不向Windows传输备份。GPU已释放，无计算进程，不关闭服务器实例。

## GitHub 发布

分支：`codex/unified-retain128-e2e32-4090`。报告链接：https://github.com/reyuyu/ATE/blob/codex/unified-retain128-e2e32-4090/vlm_research/results/unified_retain128_e2e32_4090/report_zh.md 。完整Commit及远端读取校验记录保存在工作盘 `github_upload_receipt.json`；该回执在提交后生成，不属于自身所指提交。未上传大文件及其SHA256见 `github_publication_manifest.json`。

首次Smoke因检查器漏列BridgeNative标签而停止，正式请求为0；修复仅作用于新检查器，失败原件保留在 `preformal_failures/attempt001/`，摘要随提交发布。
