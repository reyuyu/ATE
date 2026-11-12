# ATE RTX3090 八方法统一效率实验：连续完成两轮并评估复现性

你现在接续 ATE 项目的 RTX3090 八方法效率复现实验。

**用户明确授权：在当前同一张 RTX3090 上执行两个独立完整 Session，每轮240次正式请求，共480次。**

本次重点不再是使用极严格的预热波动门禁阻止实验，而是完整测量、保留所有性能波动，并分析两次独立运行的结果是否一致。

请直接执行，不要只输出实验计划，不需要再次请求授权。

## 一、已有实验与本次调整

GitHub：https://github.com/reyuyu/ATE

冻结八方法参考 Commit：

`5ee2a0dbe78df336229b55fed14a4a6d0d880bed`

当前 RTX3090 UUID：

`GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27`

已有失败实验：

- `vlm_research/results/unified_e2e32_eight_methods_3090/`：31/240，PARTIAL。
- `vlm_research/results/unified_e2e32_eight_methods_3090_session2/`：0/240，稳态准入失败。

以前的性能门禁曾因稳健CV 1.079%、窗口变化1.057%、单请求偏离3.035%等轻微超限阻止实验。

**本轮决定：性能漂移从硬性停止条件改为完整记录与警告。**

具体要求：

1. 预热后无论是否满足原1% CV、1%窗口变化、3%单请求偏离，都允许进入正式效率测量。
2. 正式阶段Native漂移超过2%时记录 WARN，不因为单纯测速波动停止。
3. 即使发生更大的性能波动，也要保留原始结果并继续测量，以便研究其影响。
4. 不得自动删除、重跑或替换慢请求。
5. 若发生输入不一致、算法正确性失败、OOM、非有限Logits、GPU设备切换、其他计算任务冲突或硬件安全问题，则必须停止受影响的实验。

本轮是**放宽性能波动门禁，不放宽正确性和硬件安全门禁**。

## 二、严格保持原八方法

固定八种方法：

1. Native Full
2. FastV K3-192
3. SparseVLM 名义192，含官方Recycling
4. V²Drop 历史576→288→144→1
5. ApET 288→96
6. DUET-VLM C+all
7. llava-SVH
8. llava-SVH-R

保持此前4090正式实验完全相同的算法配置，不引入Retain-128。

模型：

`llava-hf/llava-1.5-7b-hf`

Revision：

`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`

运行条件：

- FP16
- Eager Attention
- Batch Size 1
- Greedy Decoding
- PyTorch 2.6.0+cu124
- Transformers 4.49.0
- use_cache=True
- logits_to_keep=1
- 固定32输出Token
- ignore_eos=True
- 1次Prefill + 31次真实Decode

不修改任何冻结算法、剪枝策略、Attention跳算或KV生命周期。

## 三、复用已有3090修复成果

复用此前已经通过CPU测试和GPU Smoke的FastV验收修订。

FastV与历史4090的输出差异若仅发生在EOS之后，不再因此触发硬失败，但必须保留完整Token差异记录。

Native仍严格要求32 Token与历史输出一致。

其他方法的跨GPU历史输出差异必须记录；若没有独立正确性失败，不能只因历史Token不完全相等就判定物理实现失效。

全部方法必须通过输入SHA、算法结构、真实KV布局、No-op及必要的数值正确性检查。

不得重新开发已经完成的外部方法。

## 四、固定相同30题

复用此前4090统一八方法正式实验的：

- GQA：15题
- TextVQA：15题

每轮使用完全相同的30题、Prompt、图像Tensor及输入SHA256。

同一道题的八方法交错执行，并按原4090冻结规则轮换方法顺序。

**两个Session使用相同的预先冻结顺序**，便于直接比较两次完整执行的重复性。

不能重新抽样或改变输入。

## 五、固定预热协议，不再无限追求稳态

每个Session开始前：

1. 检查GPU空闲及UUID。
2. 加载冻结模型。
3. 执行固定16次Native完整32 Token预热。
4. 其余七种方法各执行2次完整32 Token预热。
5. 完成必要的算法Smoke和Native No-op检查。
6. 使用相同的固定Native探针建立本轮独立参考基线。

两个Session必须使用相同的预热次数、输入和执行规则。

预热期间仍记录E2E、TTFT、TPOT和设备状态。

但不再使用1% CV、1%窗口变化或3%单请求偏离作为开始正式实验的硬门槛。

如果预热出现明显快慢状态切换，记录 WARN，继续正式测试。

不得为了等待最快状态而无限预热。

## 六、正式实验：两个完整Session

### Session 1

执行八方法 × 30题：

**240次正式请求。**

按冻结输入顺序逐题交错运行全部八种方法。

每5题测量固定Native双探针，计划6次校准检查。

记录：

- 校准相对基线变化
- 是否超过2%
- 是否出现中途慢化
- GPU/CPU性能状态
- 各方法所在请求顺序

任何单纯性能波动都记录为WARN，不提前停止正式实验。

全部240次完成后生成该Session独立结果。

### Session 2

Session 1完成且未触发正确性或安全硬失败后，自动启动Session 2。

重新建立独立模型进程、重新加载模型，使用同样的固定预热和测试协议。

同样执行240次正式请求。

不要将两个Session的结果提前合并，也不要沿用Session 1的Native基线。

如果Session 1因正确性或安全失败而未完成，保留PARTIAL证据并停止，不自动用新的Session覆盖失败记录。

## 七、统一测量指标

每次正式请求保存：

- Persistent KV Cache@32（MiB）
- TTFT（ms）
- Decode TPOT（ms/token）
- CUDA Event Decode TPOT
- E2E@32（ms）
- 每次Decode的单步耗时（尽可能保留）
- 输入与输出SHA256
- 实际KV逐层布局
- 请求开始和结束时间
- GPU UUID及状态
- 当前Session及方法执行顺序

计时定义严格与此前4090统一实验一致。

TPOT：

`(E2E - TTFT) / 31`

KV必须通过实际持久K/V Tensor形状、dtype和有效长度独立重算。

## 八、重点比较两次实验到底相差多少

分别计算Session 1和Session 2各方法的30题平均值、中位数、标准差及P90。

输出表一：

| Method | S1 KV | S1 TTFT | S1 TPOT | S1 E2E |
|---|---:|---:|---:|---:|
| Native Full | | | | |
| FastV | | | | |
| SparseVLM | | | | |
| V²Drop | | | | |
| ApET | | | | |
| DUET-VLM | | | | |
| llava-SVH | | | | |
| llava-SVH-R | | | | |

输出表二：Session 2相同格式。

输出表三：跨Session变化：

| Method | TTFT变化% | TPOT变化% | E2E变化% | 相对Native Speedup变化 |
|---|---:|---:|---:|---:|
| Native Full | | | | |
| FastV | | | | |
| SparseVLM | | | | |
| V²Drop | | | | |
| ApET | | | | |
| DUET-VLM | | | | |
| llava-SVH | | | | |
| llava-SVH-R | | | | |

定义：

`跨Session变化 = (Session2均值 / Session1均值 - 1) × 100%`

每个Session内单独计算：

`E2E Speedup = Native E2E均值 / Method E2E均值`

`TPOT Speedup = Native TPOT均值 / Method TPOT均值`

并分别计算KV、TTFT、TPOT、E2E改善百分比。

重点分析：

1. Native的绝对E2E和TPOT在两次Session之间变化多大？
2. 八种方法是否出现近似共同的性能变化？
3. 各方法相对同轮Native的加速率是否更加稳定？
4. SVH和SVH-R是否两轮都表现出Decode TPOT改善？
5. FastV、V²Drop、ApET等外部方法的TTFT优势是否重复出现？
6. 小于1%–2%的差异是否存在排名变化？
7. 哪些异常请求造成明显跨轮变化？

还要按照同一道题比较Session 1与Session 2的耗时差异，保存配对统计及探索性95% Bootstrap区间。

对有明显漂移的Session如实标注，不因240次完成就宣称稳定性PASS。

## 九、结果解释

明确区分以下两项：

**执行完整性：** 是否完成240/240条请求，是否通过算法正确性与输入检查。

**测速稳定性：** Native是否存在明显漂移，两轮结果是否一致。

允许出现：

`COMPLETE_WITH_PERFORMANCE_WARNINGS`

这表示全部正式请求成功，但仍有性能波动，不等于高精度测速稳定性已经通过。

不允许删除异常请求后将结果改写为稳定PASS。

如果两轮绝对延迟差异很大，但相对同轮Native的加速比比较一致，应在报告中同时展示这一现象及其局限。

不能为了让SVH显得更快而挑选Session。

## 十、保存及权限

新建独立目录：

`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_repeatability/`

下设：

- `session1/`
- `session2/`
- `comparison/`

完整保存：

- Protocol及源码SHA
- 两轮环境记录
- 预热记录
- GPU/CPU遥测
- Smoke与No-op审计
- 全部480次原始正式请求
- 两轮独立效率汇总
- Native漂移记录
- 跨Session逐题对照
- 统计分析代码
- 中文报告
- Checkpoint及失败日志
- SHA256校验清单

不覆盖任何已有实验目录。

不运行自然EOS、不跑七项完整质量Benchmark、不重新训练、不调参、不修改main。

不要求Windows备份。

暂不授权GitHub Push或创建PR。

完成后释放GPU，不关闭服务器。

## 十一、最终验收和报告

最后用中文直接回答：

1. 两轮是否均完成240/240？
2. Native两轮E2E和TPOT相差多少？
3. 每种方法的TTFT、TPOT、E2E跨轮变化多少？
4. 哪些方法相对Native加速比最稳定？
5. 两轮SVH/SVH-R的Decode TPOT改善是否一致？
6. 两轮中有没有异常慢请求？
7. 这批数据是否适合用于论文主表，还是仅能作为波动分析证据？
8. GPU是否已经释放？

**本次实验的目标是实际测量两轮重复性，而不是强迫所有请求满足1%的绝对延迟波动。**

现在开始执行两轮完整3090效率实验。只在算法正确性、输入一致性或硬件安全出现问题时停止，不因单纯性能统计阈值超限而停止。