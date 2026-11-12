# 独立物理推理优化修改说明

所有 A 条件均调用原18条件冻结 Runtime，使用原始模型、Processor、Prompt、位置定义、层选择、精度与 Eager 后端。原目录只读；opt_b/opt_c/opt_d 为独立命名空间，未修改 main、GitHub、模型或历史成绩。实际包文件与最终 SHA 为复现依据，code/build_*.py 为实现过程脚本，不能仅凭一份初始构建脚本覆盖最终包。

## 阶段与数学边界

A 为冻结原版。B 仅缓存静态索引、Mask、位置组，并把审计专用检查与 GPU 回传移到 Audit Mode，原排序、聚类、FPS、基底替换、QKV、Softmax、数值补零、Residual、MLP 不变。C 在 B 上优化真实 KV 存储与复制。D 单独尝试去不可访问的数值补零，不以答案相同替代数值门槛。E 为通过正确性与收益筛选的阶段别名，不是另一个算法。

B/C 要求原30输入全部32步有效 Logits torch.equal 与完整 Token 一致；三个长度代表输入保存层输出及0/1/8/31步缓存快照。D 的 BF16 门槛在 GPU 前冻结于 protocol.json，失败不放宽。全部方法专用初始化仍在 TTFT/E2E 内。

## Qwen

B 保存已知 CPU Text Positions，避免每层回传相同索引。位置断言与线性/列审计进入 Audit Mode。SVH 保留原 Lt descriptor、Softmax/QK/AV 对齐补零。SVH-R 预生成同值 Prefill Mask，Decode 使用两组请求级零 Mask 的有效切片；SelectedKV 初始化只建立一次 Text 集合，原 mRoPE/Cache Positions 不变。

C 把文本 Gather、晚层文本/视觉 KV 合并直接写入最终容量缓冲，避免中间副本。每层允许的 Key、KV 数值与31步增长不变。真实 Late Visual KV Construction 保留，不能删除必要算法开销。

D 仅针对 SVH 去补零；SVH-R 不适用这一候选。失败记录与首分歧层保留，失败候选不进入正式计时。

## InternVL3

B 缓存全长零、紧凑零和禁图三类 Decode Mask，取有效长度视图，删除无消费者 Key Positions 重建。Prefill Writing、物理退出、原始 RoPE 与答案跳算定义不变。

C 以真实 Buffer/used/capacity 替代逐 Token cat，Prefill 首次分配、Decode 原位写入。分别审计有效视图 Live、底层 Storage 与 Capacity，无隐藏完整 Dense KV。

D 仅对 SVH-R 数值补零分支独立尝试紧凑 Attention；不能自动与历史全量 Accuracy 拼成同一个质量—效率工作点。

## LLaVA 与四种融合

LLaVA SVH/SVH-R 已预计算 Mask、预分配 KV 并使用紧凑 Attention，两条路径不改写、不自动重测。新增 Native 仅用于发生变化的四个融合条件的同机对照。

FastV+SVH B 门控审计专用布局 SHA、选择详情、断言与同步；保留官方额外 Q/QK Ranking 和 K 复用。部分结构索引、score_min/max 回传仍存在，不声称清除全部同步或排名成本。

SparseVLM+SVH B 门控审计专用结构断言及计时 Event 同步；保留 Token Recycling、Clustering、输入绑定 RNG 与 Text/RoPE 锚点。为确定性与选择账本仍需的 SHA/位置回传保留，不用近似选择器替代官方算法。

V²Drop+SVH B 缓存0/3/17/20阶段的 CPU Text 索引及阶段内重复 Mask/Rotary gather；冻结前缀剪枝与 L20退出不变，不新增 L22剪枝，不冒充论文独立192档。

ApET+SVH B 按原逻辑位置组复用 Decode Position/RoPE，Prefill Text 索引仅在0/16/20边界更新。FPS、基底替换、误差排序和预算不变。

融合路径既有 TextOnlyKV 已预分配，无需再造 C/D；对应阶段 NOT_APPLICABLE。

## 审计、计时与限制

CPU 测试在 GPU 前完成，生产 Mask/KV 函数与独立参考比较。GPU 包括 Native/No-op、原输出复现、原30输入完整32步回归、真实 Audit Mode 与逐层 KV。

Audit/Profiler 请求不进入正式计时表。Qwen Session1 已通过全部30输入数值筛选，但实际 Audit Mode 因协调器调度在 Session1 后执行；顺序差异保留，Session1 在审计通过前为临时证据，不为选择更快结果重跑。

计时沿用冻结边界：GPU 输入准备完毕，起始同步后，计入控制器/Mask/Buffer初始化、Vision/注入、Prefill、31步Decode及Context恢复，最终同步后结束。有限性汇总、Token回传、KV元数据审计、磁盘写入和最终Python对象释放在该边界外；原版与优化版一致。专用初始化没有移到边界外。

两独立 Session 分别报告30题均值、中位数、同题配对改善；所有漂移警告和慢请求保留，不重设基线、不选择最快 Session。Profiler 只用于算子/Kernel/同步审计，扰动墙钟时间不当作真实效率。

## 独立E补充修正版

Profiler发现Qwen SVH仍在每个Decode Attention层创建相同dead-column索引，新增opt_e将同一步索引与Mask缓存复用，原补零QK/AV不变。InternVL SVH原C被有效长度检查拒绝，原失败包保留；opt_e中的C2修正Decode遗漏的原位追加。两者均先做生产函数CPU回归，按protocol_refinement_addendum.json在GPU前冻结。仅在数值/缓存/Audit Mode通过且相对B有额外收益时，新增原30题两独立Session；不是调层、调精度或调算法参数。
