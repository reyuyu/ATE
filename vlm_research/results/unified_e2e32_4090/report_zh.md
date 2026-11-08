# ATE 八方法统一 KV Cache@32 / E2E@32 效率复现

状态：PASS；正式请求 240/240；每种通过审计的方法 30 题（GQA15+TextVQA15）。本次未运行任何质量 Benchmark、训练或 OCRBench。

| Method | KV Cache@32 MiB ↓ | E2E@32 ms ↓ | TTFT ms | TPOT ms/token |
|---|---:|---:|---:|---:|
| Native Full | 351.500 | 1950.498 | 119.901 | 59.052 |
| FastV | 177.500 | 1973.426 | 106.661 | 60.218 |
| SparseVLM | 166.750 | 2026.046 | 145.425 | 60.665 |
| V²Drop | 164.906 | 1952.974 | 107.955 | 59.517 |
| ApET | 159.500 | 1945.451 | 115.032 | 59.046 |
| DUET-VLM (C+all) | 159.375 | 1970.826 | 111.849 | 59.967 |
| llava-SVH | 55.562 | 1766.342 | 107.966 | 53.496 |
| llava-SVH-R | 181.562 | 1780.817 | 111.462 | 53.850 |

以上为同一批 30 题算术均值；中位数及 GQA/TextVQA 分组均值见 efficiency_summary.csv。KV 是真实持久缓存的 live bytes，不包含权重、峰值显存或临时张量。32 个输出只写入 31 个生成 Token 到缓存。

| Method | KV 减少 | E2E 减少 | TPOT 减少 |
|---|---:|---:|---:|
| Native Full | 0.00% | 0.00% | 0.00% |
| FastV | 49.50% | -1.18% | -1.98% |
| SparseVLM | 52.56% | -3.87% | -2.73% |
| V²Drop | 53.08% | -0.13% | -0.79% |
| ApET | 54.62% | 0.26% | 0.01% |
| DUET-VLM (C+all) | 54.66% | -1.04% | -1.55% |
| llava-SVH | 84.19% | 9.44% | 9.41% |
| llava-SVH-R | 48.35% | 8.70% | 8.81% |

本轮 Native TPOT 为 59.052 ms/token，仍接近59。FastV E2E 比 Native 慢 1.18%，SVH/SVH-R 分别比 Native 快 9.44% / 8.70%。外部方法之间约百分之几以内的小差异不宜过度解读；本次是单轮同题配对计时，不声明统计显著性。

## 可比性与漂移

- GPU UUID 始终为 `GPU-724fa9c3-d33a-df3f-e349-533cb4634a53`；Driver 610.57.04。正式阶段一个进程加载一次模型，统一外层 Runtime、相同输入 SHA、32-token 生命周期与同步边界；原始逐请求记录带 UUID。
- FP16 / Eager / Batch1 / Greedy / use_cache=True / logits_to_keep=1；Torch 2.6.0+cu124、Transformers 4.49.0，实际导入路径和源码 SHA 见 environment.json。SparseVLM 沿用官方手写 Eager FP16 softmax，并未换成融合 SDPA/Flash。
- 所有方法均计入 Vision、Projector、Prefill、31 次 Decode、必要的选择/聚类/剪枝与缓存管理。图片读取、Tokenizer、预处理、输入 H2D、模型加载、最终 KV 审计与日志 I/O 在计时之外。TextVQA 复用原30题清单中的冻结OCR上下文视图；本次不调用外部OCR，也不是重新选择500题Image-only清单。
- 每方法两个相同 Smoke/预热输入；按冻结题序交错方法并轮换顺序。Native/No-op Token 与首 Logits 完全一致；独立紧凑位置兼容循环与 stock Native 32-token/首 Logits 完全一致。实际 Hidden、Mask、RoPE、QKV、每层缓存及至少 8 个真实 Decode 步均通过 Smoke。
- Native 漂移检查 6/6，状态：PASS；每 5 题使用同两题探针，对比本次三轮暖机基线，5% 为暂停门槛。探针共 18 请求，均不计正式结果。
- Native E2E_ms 相对暖机基线变化范围 -2.73% 至 -0.02%。
- Native TPOT_ms 相对暖机基线变化范围 -3.00% 至 +0.10%。
- GPU 监控范围：{"clocks.current.sm": {"min": 210.0, "max": 2670.0, "mean": 2527.608695652174}, "clocks.current.memory": {"min": 405.0, "max": 10251.0, "mean": 10108.304347826086}, "power.draw": {"min": 12.44, "max": 163.98, "mean": 127.63894927536232}, "power.limit": {"min": 450.0, "max": 450.0, "mean": 450.0}, "temperature.gpu": {"min": 34.0, "max": 65.0, "mean": 52.06159420289855}, "utilization.gpu": {"min": 0.0, "max": 82.0, "mean": 30.32608695652174}}；其他计算进程：[]。

## 官方工作点与限制

- Native、FastV、V²Drop、SVH、SVH-R 直接复用此前冻结实现；各方法独立控制器的必要开销保留。三种新方法使用同一组冻结 HF 权重与模块，兼容层单独保存，官方源文件未改。
- FastV：固定 d1659729b5bf1be225e99ee15783deeea80f63b1，K3/R2⁄3 静态 KV，L0–L2 保留 576，进入 L3 后 192。属于当前已验证静态 KV 工作点，不能直接绑定公开论文其他 K/缓存分支的质量分数。
- V²Drop：固定官方 2a9431150f8ace3a1d9abcdc96a7d46acca5597df，保持已有 L3/L17/L22 的 576→288→144→1 配置。与论文 192-token 质量工作点不同。
- SparseVLM：固定 ICML v1.5 734d9003879689713a132b83297dfad060a5c9af；main 已转为 SparseVLM+，本次未使用。名义192预算 [300,200,110] 加入官方回收聚类后，实际视觉 KV 分布为 576×3层、309×4层、204×9层、113×16层，平均206.5；没有人为更改为恒定192。官方旧类 SDPA/flash_attn 接线和废弃接口由独立 HF 桥接绕过，官方手写 Attention 函数、评分与剪枝/合并 AST 原样执行。不能把本次效率无条件配到论文其他版本或精确192-token质量数值。
- ApET：固定 70cc01b215f4d74c5661c699919fe3aeda48bed0；官方288个视觉表示进入LLM、L16起96个，平均192；FPS basis=10，官方重建误差排序、基底替换、合并和位置重编号原样执行。匹配公开192预算的超参数类型；本次未验证论文质量分数。Vision桥接读取相同hidden_states[-2]，不保留或stack原版encode_images完全丢弃的辅助CLIP Attention输出；此项是移除未使用的导出，不改变算法输入或排序/合并。
- DUET-VLM：固定 cd69e489f39a2a51e3b278a470dc8dc32c00e0fb；dominant300（含CLS）+contextual7，cluster_width4，307→153→0，进入L16/L24剪枝，层平均191.75。当前官方函数使用全部图后文本 Token 排名，对应 C+all；论文主表 C 使用最后文本 Token，二者不可冒充同一质量工作点。本次没有使用训练后的 DUET 权重。
- llava-SVH：视觉 Hidden 在进入L20前移除，答案阶段不持久保留原始视觉KV；SVH-R 在进入L25前移除Hidden，只于L11–L24持久保留视觉KV。两者L2–L5物理跳过答案位置Attention，普通Prefill其他位置仍执行，不宣称整层Prefill全部跳算。冻结源码未改。
- Native/FastV/V²Drop 使用原始位置或其子集；ApET/DUET/SparseVLM 按官方实现压缩后重编号。真实逐层 KV 长度与视觉表示数量见 kv_layout_per_layer.csv。
- CSV 导出时修正了原始运行器对 SparseVLM 位置规则的通用字符串注释：应为 compact。原始记录、实际位置、数值 RoPE 检查及计时未作改动。

结论：本次通过完整审计的行可以作为同机、同输入、固定32-token条件下的效率比较。论文主表须注明上述具体版本/预算/缓存分支及DUET C+all变体；不能直接与不匹配的公开质量分数拼成八方法联合质量–效率表。

官方参考：[SparseVLM ICML](https://github.com/Gumpest/SparseVLMs/tree/v1.5)、[ApET](https://github.com/Maqkccx/ApET)、[DUET](https://github.com/AMD-AGI/DUET-VLM)、[DUET论文配置与变体](https://arxiv.org/html/2602.18846v1)。

## 路径

- 结果：`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_4090`
- 完整官方源码与兼容层：`/home/featurize/work/ATE/vlm_research/external_baselines`
- 模型：`/home/featurize/ate-local-models/llava-1.5-7b-hf`；Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
- protocol.json、environment.json、smoke_summary.json、input_verification.json、efficiency_per_input.csv、efficiency_summary.csv、paired_comparison.csv、kv_layout_per_layer.csv、formal_requests.jsonl.gz、checkpoint.sqlite、drift_summary.json、gpu_telemetry_summary.json、validation.json、evidence_manifest.json。

本次没有修改冻结算法、GPU设置或远端GitHub。完成后释放GPU；释放记录另见gpu_release.json。
