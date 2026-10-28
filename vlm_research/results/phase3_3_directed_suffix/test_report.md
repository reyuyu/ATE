# Phase 3.3 正确性测试记录

## CPU 回归（正式 GPU 执行前完成）

执行既有六组回归和新增 `test_directed_suffix`，合计 **72 项全部通过**；完整输出见 `logs/cpu_tests.txt`。小模型同样使用 Transformers 4.49.0 的真实 LLaVA/LLaMA eager 实现，共32个小型Decoder层。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest test_attention_probe test_attention_positions test_knockout \
  test_phase3_watch test_physical_exit test_fixed_k_exit test_directed_suffix -v
```

新增10项测试覆盖：真实掩码行列方向；仅目标连接变为负无穷、其余掩码逐元素不变；仅K之后层修改且先Clone；实际Attention严格为零；所有K32精确原生Full；A-only不改变更早Question hidden；QA Question hidden等于Q-only；Global精确复用Phase3实现；答案输入P不泄漏到预测Query P−1；正常及异常清理Hook；不修改输入/参数；非法映射和损坏/错配/重复检查点拒绝。

“非目标连接不修改”指加性Mask，不能要求softmax后所有非目标Attention概率不变；归一化和层间传播会改变概率与hidden。

## 真实7B Smoke（正式队列之前通过）

固定样本为冻结Manifest中各标签前3条，加用户指定的3个历史问题，去重得到8张，不根据新实验结果选择。所有四条件×三个K×两个候选都返回真实32层Attention，共192次审计Forward、6144个层矩阵，检查形状、有限性、归一化、因果Mask、padding和目标区域严格为零。矩阵未保存到硬盘。

另外验证Global对当次重新计算的Physical，以及对Phase3.2历史分数的差异；沿用Phase3.1的全部容差，不修改阈值。K32有效文本hidden/logits精确Full，A-only与QA的Question因果不变量精确成立，全部干预结束后再次原生Forward与Full精确一致。Smoke耗时见 `smoke_complete.json`。

## 正式全样本审计与汇总

完整完成门和样本数量见 `run_complete.json`，独立离线审计见 `correctness_validation.json`。每个样本均核验冻结图片哈希、原Prompt/候选/Question和Answer映射、历史Full精确分数、全部K32、Question因果不变量及Global对历史Physical容差。逐层Hook记录确认32层均运行、仅后缀生效、Mask已Clone、原始位置不变和完整序列长度未改变。正式阶段不额外返回完整Attention；Smoke与正式的logP/Margin/Drift在相同样本上要求完全相同。

`implementation_audit.json`记录编译、pip check及本阶段源码SHA256。没有修改旧阶段Python/结果、安装包源码、环境或权重；GPU资源检查不会终止他人进程。数值近似等价不保证接近零Margin时分类严格一致，此类差异独立保存在 `global_vs_historical_physical.csv`。
