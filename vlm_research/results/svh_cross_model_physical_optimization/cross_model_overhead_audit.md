# 跨模型共享开销静态审计（GPU前冻结）

正式来源为 rtx3090_unified_efficiency_18conditions 的 source_manifest、cross_source_manifest 和 supplement；所有源文件实际SHA已核对并另存 frozen_source_manifest.json。原结果不变。各模型层号遵循各自冻结定义，不强行统一。

| 路径 | 实际核心文件/函数（vlm_research/前缀） | Prefill/Decode、KV与退出 | 已确认问题/分类 | 计划 |
|---|---|---|---|---|
| LLaVA SVH | c3v2_middle_reaccess/engine.py:MiddleWriter,SelectiveKV；physical_handoff_fastpath:InplaceKV | L20退出；L2–L5仅答案跳算；Mask请求内预计算；KV预分配+原位写，Decode已Compact | 中层Mask在keep空时仍初始化，但无每Token isin/同步；没有数值Dense补零；NECESSARY_ALGORITHM_COST | 不为统一代码而改写。保持冻结实现，除非后续证据支持修改 |
| LLaVA SVH-R | 同上；cutoff25、keep L11–L24 | 仅14层图KV，原位置，已Compact | 真正的selected attention是必要成本；无已知逐步重建KV | 同上，不扩大正式复测 |
| Qwen SVH | svh_sqa7908_repair→svh_qk_repair→svh_l28_repair→svh_numerical_repair→c3v2_numerical_repair2 | L22退出；L1–L6跳算；真实文本预分配KV | GPUtolist、热路径标量断言；临时补零列/行及固定Lt；SHARED_OVERHEAD、NUMERICAL_ALIGNMENT_OVERHEAD | B移元数据/调试检查，保留Lt与补零；C直接gather；D独立去补零 |
| Qwen SVH-R | svhr_search/physical.py:Physical,SelectedKV | L22主退出；L19–L35图KV；L22–L35辅助前缀；Decode原位append | 初始化N次tolist；每步36 isin、72显式bool；动态Mask/positions；MODEL_SPECIFIC_OVERHEAD、SHARED_OVERHEAD；晚层图KV必要 | B预计算同值Mask/索引；C直接写最终KV。保留必要辅助分支，不改层选择 |
| InternVL SVH | internvl3_svh_physical_d_pilot/engine.py:Control,CompactCache | L20退出；L2–L5答案跳算；Decode每层cat；位置不重编号 | 逐步cat分配/复制、重复全Mask再gather；SHARED_OVERHEAD | B重用全零合法Decode Mask；C预分配 |
| InternVL SVH-R | internvl3_svh_r_terminal_windows_textvqa/engine.py | Late9 L19–L27，末端28退出；四层跳算；只有selected层图KV | 逐步cat；zero_image_scratch补零；无用keys构造；SHARED、NUMERICAL_ALIGNMENT_OVERHEAD | B去无消费者映射、缓存Mask；C缓存buffer；D独立紧凑Attention |
| FastV+SVH | fastv_svh/engine.py:Fusion.rank_l2/forward | 576→192@L3→0@L20；TextOnlyKV预分配；L2–L5跳算 | 逐层SHA/选择详情GPU回传、标量断言；额外rank Q/QK与部分K；PRUNING_INTEGRATION_OVERHEAD | B仅移调试同步。保留独立ranking，避免改变官方排名与residual |
| SparseVLM+SVH | sparsevlm_svh/engine.py:Fusion.score_only/recycle | 576→309→204→113→0@L20；真聚类/RNG/锚点；TextOnlyKV预分配 | 调试synchronize/回传、geometry断言；三次score-only；PRUNING_INTEGRATION_OVERHEAD | B门控调试检查，保留输入种子摘要和选择/聚类语义。不能随意删必要RNG/raters操作 |
| V²Drop+SVH | results/v2drop_prefix_svh_accuracy_3090/fusion_engine.py:PrefixKV.update/PrefixSVH | L3/L17前缀；L20退出；无L22正式独立剪枝；预分配 | 28层Prefill重复local_text.tolist、相同阶段Mask/Rotary多次gather；PRUNING_INTEGRATION_OVERHEAD | B阶段边界缓存CPU布局和Mask/Rotary；选择器不变 |
| ApET+SVH | 原正式code/apet_fusion.py:ApETFusion | Vision288→96@L16→0@L20，既有TextOnlyKV；无独立额外ranking | Decode每层相同logical位置重复tensor/RoPE，Prefill重复text_index；PRUNING_INTEGRATION_OVERHEAD | B按原logical位置组复用RoPE/positions。官方FPS/基底替换/排序保持 |

共享问题是控制/元数据/缓存管理，不能仅凭同名API认为同成本。Qwen isin 的612 Kernel及SVH临时Dense恢复已有真实Profiler；融合的D2H与rank成本见前置两份诊断。InternVL cat实际源码明确，但收益需本轮受控测量；LLaVA原SVH/R已经Compact，不声称存在没有证据的Dense补零。

所有方法专用初始化/Mask/Buffer仍在计时内，禁止提前移出E2E边界。Audit Mode保留原检查；正式性能移出的仅为可证明不参与输出的诊断。B/C先要求32步有效Logits与输出严格相同；D单独使用预冻结BF16数值门槛，并在输出改变时进行质量回归，不能直接拼接历史Accuracy。无收益和失败阶段原样保存，不强制整合。暂未启动GPU。
