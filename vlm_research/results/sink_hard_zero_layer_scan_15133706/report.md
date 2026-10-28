# 15133706：不同层硬置零Sink实验

## 实际状态

**cpu_passed_gpu_pending：真实GPU完成 0/144 条件。**


历史Full Margin=−0.03125、历史Physical K24=+0.0390625，仅作旧对照。未完成时不得将它们写成新基线或推断任何层的预测收益。

## 冻结设计

仅Girl/Boy案例15133706，原问题/候选left/right，gold right；FP16/eager/Transformers4.49.0、use_cache=False、batch1。动态定位真实P−1和576视觉位置，只改该Query、全部32Head，Question隐藏状态应保持因果不变。

固定32个单层L0～31及四窗口L2～4、L8～15、L14～23、L24～31。每个scope含冻结13非ROI Sink置零，以及三组Phase3.4等数量/空间bin匹配背景置零，共144条件。不扫描强度、不按结果选层/Head/窗口；单层和多层累计窗口分开比较。ROI/Sink交集#364不被抑制，不能称为所有Sink均已清除。

在Softmax转原生FP16后、乘当前V之前：`r=a_visual; r[blocked]=0; new_visual=sum(a_visual)*r/sum(r)`。保留当前Head视觉总质量、非视觉权重、其他Query与因果Mask；零质量保持零，正质量却无剩余支持则停止，无任意fallback。不同于旧Phase3.4整行Mask后Softmax重分配，也不同于×0.1软抑制或L22−L2离线作图。被选Patch必须严格为零。

复用未修改的实例局部eager包装、FP16公式逐元素舍入及dtype推导界。行和atol0.002；视觉Mass atol2e−5+rtol0.001；原conditional L1阈值0.002保留告警，不据结果放宽。先CPU测试，再7B无干预Full/全层no-op、L14硬置零Sink/随机对照Smoke，最后正式评分。每条件两完整候选全词表LogP、Margin、原始logits、Question hidden、区域份额、局部Before/After行、审计和原子完成指纹保存；绝不保存完整Attention矩阵。

## 结果和限制

**尚无本轮真实Full或干预分数，不能回答不同层扣除是否改善预测。没有绘制虚构Margin曲线。** GPU不足时不加载7B，不终止其他任务；没有自动轮询。

单个事后选出的弱Margin FP16案例；高Attention不等于有害噪声。转正只是二候选排序变化，须同时报告正确词全词表LogP和三背景对照；无泛化、因果机制已证明或推理加速声明。Hook正常/异常清理，历史代码/权重/结果保持原样。

## 执行/恢复

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
# 已冻结，勿再次prepare覆盖：
# CUDA_VISIBLE_DEVICES='' python sink_hard_zero_case.py --stage prepare
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python sink_hard_zero_case.py --stage gpu
# 只有该attempt存在有效完成checkpoint且代码Commit/环境相同时：
# python sink_hard_zero_case.py --stage gpu --resume-attempt gpu_attempts/<原目录>
```

固定代码Commit必须先于真实7B运行；运行中的源码不修改。资源门槛16,000MiB空闲。每次gpu默认新建尝试目录，保留失败/资源记录；`--resume-attempt`严格核验冻结协议、代码Commit、环境及逐条件文件SHA，拒绝不完整checkpoint覆盖或重复计数。若原尝试没有分数，直接新attempt即可。结果汇总仅在全部144条件及清理验证通过后画图；partial保留真实分数但不伪装完整。

协议见[protocol.json](protocol.json)，状态见[status.json](status.json)，CPU记录见[cpu_tests.txt](cpu_tests.txt)。只推送独立实验分支，不合并main。
