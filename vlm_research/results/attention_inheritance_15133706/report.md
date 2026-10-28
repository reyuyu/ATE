> **后续完成通知（2026-10-28）**：六条件新运行已完成，见 [completed_20261028_2245/report.md](completed_20261028_2245/report.md)。下文和根部CSV/JSON/validation仍是首次尝试的历史未完成记录，未被改写或并入新运行。

# 15133706 单案例 Attention 继承：实验未完成

**状态：已实现并通过101项CPU回归；真实GPU仅完成L23、λ=0。非零λ的答案结果尚不存在。** 不能判断继承是否提高right Margin或使其转正。所有CSV空值均为未完成，不是零值、负面结果或推算结果。

问题：**Is the young girl to the left or to the right of the boy?** 标准答案right；候选顺序left/right保持历史不变。仅此一张图，不训练、不扩展样本。

## 已实际获得的分数

| 条件 | Margin (nats) | 来源 |
| --- | --- | --- |
| Full | -0.03125000 | 本轮真实7B，精确匹配历史Full |
| Full L23→L24～L31，λ=0 | -0.03125000 | 本轮两候选真实前向，整个P−1词表logits与Full逐元素相同 |
| Physical K24 | +0.03906250 | Phase3.2历史数据；本轮没有重跑Physical |
| Full L23，λ=0.5/1 | 未完成 | 第一次λ=.5在L25验证停止，没有输出候选分数 |
| Full L24，λ=0/0.5/1 | 未完成 | GPU资源阻塞，未完成条件 |

![只展示实测与明确标注的历史分数](margin_comparison.png)

Full的right/left logP为−18.5991971377/−18.5679471377，原始FP16 logits为8.2578125/8.2890625。正确候选的全词表概率约8.3651e−9；二候选相对概率约0.49219。Margin接近零，候选正确不等于开放生成正确。

## 干预实现与定位

使用既有load_model、Processor、问题/候选定位及完整候选评分。Prompt长度603；P−1=602；输入候选后长度604；视觉位置5～580，共576。left/right各一Token，其预测Query均为602，**不使用已经输入的答案位置603**。

固定Full L23或L24的每个Head视觉行，分别归一化成p_ref；新鲜Full参考与Phase3.4缓存逐元素相同。L24～L31每层保持Head对应关系，令当前层视觉行a、m=sum(a)，仅在Softmax后、乘当前层V之前写入：

`new_visual = m * ((1−λ) * a/m + λ * p_ref)`

其余Query、P−1非视觉权重与原始Mask不改，不做全行重新归一化。λ=0跳过写入。数值修改发生在现有FP32 Softmax转成FP16之后，替换值也保持FP16；模型权重、V、RoPE、位置、候选及Prompt不变。后层局部编辑前的权重可能因前层干预自然变化，不能宣称其等于Full。

通过模型实例私有forward/globals包装eager函数，无全局softmax猴子补丁，不改第三方源码；上下文finally恢复。头h跨深度直接对应，不假定它们具有相同语义功能。无KV Cache/Decode、无物理删除、无加速声明。

## 验证结果与首次停止

7项新增CPU测试覆盖λ0全logits/Attention精确、混合公式、因果早期位置、P@V计算、FP16量化、异常清理、实例隔离及非法输入；全工程共101项回归通过，见`cpu_regression.txt`。

真实GPU初次执行：Full分数和输入映射与历史完全一致；Full L23/L24参考与缓存完全一致；L23 λ0两个候选的P−1全部logits、Question最终hidden精确等于Full，16个层级审计全部通过；行和最大误差0.000296473503，视觉Mass改变量0，非目标连接逐元素不变。

首个非零条件（L23 λ=.5）在L25停止：某Head视觉Mass约0.00209993，FP16条件分布L1误差0.00429884，超初始固定0.002阈值。行和误差0.00024229，最大Mass绝对误差7.81e−6，非目标连接精确。该条件**没有完整Margin**。完整失败、初始配置和已完成分数保存在`initial_attempt_fp16_validation/`。

调查表明576个FP16子正常数的绝对舍入，经极小Mass条件化会放大；固定L1阈值未覆盖此情形。干预公式与计算完全不改，新增逐元素等于公式FP16舍入值的bitwise检查，并以dtype/质量推导的界验证：`u=eps/2; d=tiny*eps; E=u*sum(abs(x))+576*d/2; L1<=2E/sum(round(x))+2e−6`。这是舍入误差和三角不等式的上界，不从答案或实测误差拟合。原0.002阈值保留为显式告警；原行和0.002、Mass atol2e−5+rtol0.001不变。**修正验证器只通过CPU，尚未经真实7B非零λ验证，不能称整个实现已通过GPU验收。**

## GPU资源阻塞和真实开销

第一次运行时GPU0约17GB空闲，Full及λ0成功。修正验证后重跑前，其他用户进程1536339新增约10104MiB；结合原进程7524MiB，GPU0空闲仅约5971.6MiB，低于既有16000MiB门槛。GPU1也被占用。资源检查在模型加载前停止，没有CUDA OOM、没有终止其他任务，没有启用自动轮询。见`resource_blocked_attempt/`。

已完成λ0两候选诊断耗时**1.010206秒**、峰值allocated **13640.625MiB**、reserved **13734.000MiB**。计时含本地审计开销且GPU共享，非稳定性能比较。首次环境/加载记录保存在初次尝试目录；当前仍为Python3.11.18、PyTorch2.6.0+cu124、Transformers4.49.0、CUDA12.4及原模型revision。安装库SHA256复核未改变。

## 观察与机制解释

实际仅证明λ0实现路径保持Full；**尚不能回答非零继承是否改善答案**。Full与历史Physical的符号差异本就很小。即便后续某条件转正，单个已挑选案例、FP16弱Margin和一个Query行的干预，也不能证明Sink有害、解释Physical全部效应或代表一般能力提升。Physical改变多位置的信息流和计算，不能与本干预等同。

## 恢复方式

GPU资源足够后，手动运行；保留已有尝试，使用新的独立目录：

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_inheritance_pilot.py --output-dir results/attention_inheritance_15133706/completed_retry
```

只会重跑此一个案例的Full与6条件；仍先检查资源并拒绝覆盖已有配置目录。若再次失败，保留failure.json及逐条件已完成结果，不更改候选/强度/层以追求转正。此摘要可纯CPU重建：`CUDA_VISIBLE_DEVICES='' python summarize_attention_inheritance.py`。本次到此停止，需GPU可用后才能完成用户要求的真实对比。
