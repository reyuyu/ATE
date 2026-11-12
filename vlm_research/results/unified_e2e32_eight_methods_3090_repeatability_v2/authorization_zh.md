# 当前对话授权：RTX3090 repeatability_v2

授权来源：用户本轮直接消息《ATE RTX3090：修订历史Token验收，完成两轮八方法效率实验》。

授权两个新的独立Session，每轮240个正式请求，共480；旧151条记录保持PARTIAL且不得混入。

Native跨GPU历史Token验收改为首个EOS之前及EOS本身一致，EOS位置必须一致；仅EOS后差异记录并继续。Native/No-op在同一GPU上的完整32Token与数值执行一致性仍严格。FastV原首EOS门禁及其他方法物理、数值正确性门禁保留。

每次仍实际执行固定32Token、1次Prefill及31次真实Decode；不修改Greedy、模型、Attention、KV或计时口径。

本轮在启动前冻结每Session40次Native完整32Token预热，其他七方法各2次；执行既有八方法Smoke、旧失败FastV与Native额外Smoke、独立Native20次参考基线、原冻结顺序240正式请求、每5题双探针各3次共36次校准。两轮完全相同，不临时增加预热。

性能波动、CV及漂移仅WARN，全部保留；输入、算法/物理/No-op正确性、OOM、GPU设备/冲突或硬件安全硬失败时停止。Session1完整且正确性通过后自动执行独立新进程Session2，不授权第三轮或自动重试。

使用新目录unified_e2e32_eight_methods_3090_repeatability_v2，不修改冻结算法、main或历史结果，不跑其他Benchmark、不训练、不Push/PR、不做Windows备份。完成后释放GPU并保持服务器运行。
