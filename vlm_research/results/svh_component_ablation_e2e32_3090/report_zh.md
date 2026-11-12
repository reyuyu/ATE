# SVH Component Ablation：RTX3090 E2E@32 效率补测

状态：**COMPLETE_WITH_PERFORMANCE_WARNINGS**；执行与正确性：PASS。两个独立Session各120条，合计240条正式请求，全部保留。GPU UUID `GPU-4612d0cb-a47a-674d-b647-3f4065f1627f`。

四个配置与原SQA/POPE消融使用完全相同的物理实现和模型Revision。Accuracy直接引用已验证旧结果，不重新评测。**效率输入为原冻结GQA15+TextVQA Paper-OCR15，并非SQA/POPE全集的自然EOS延迟。** 固定输出32Token，忽略EOS，1次Prefill与31次真实Decode。FP16/Eager/Batch1/Greedy、torch2.6.0+cu124、transformers4.49.0。

## 四配置综合表

下表时间为两个Session均值的平均；±为两个Session均值的样本标准差。完整逐题离散度、逐Session均值/中位数/标准差及同轮配对区间另存CSV。

| Configuration | SQA % | POPE % | KV@32 MiB | TTFT ms | TPOT ms/token | E2E@32 ms | E2E speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native Full | 69.51 | 86.85 | 351.5000 ± 0.0000 | 237.72 ± 2.91 | 64.558 ± 0.383 | 2239.02 ± 14.79 | 1.0000× |
| SVH w/o Selective Transfer | 68.67 | 87.22 | 55.5625 ± 0.0000 | 239.97 ± 0.79 | 59.342 ± 0.587 | 2079.57 ± 18.99 | 1.0767× |
| SVH w/o Attention Bypass | 69.36 | 86.40 | 63.5000 ± 0.0000 | 182.50 ± 1.64 | 64.100 ± 0.315 | 2169.61 ± 11.40 | 1.0320× |
| SVH Full | 68.96 | 87.21 | 55.5625 ± 0.0000 | 184.81 ± 1.21 | 59.165 ± 0.495 | 2018.92 ± 16.54 | 1.1090× |


## 逐Session实测

| Session | Condition | KV MiB | TTFT ms | TPOT ms/token | CUDA Event TPOT | E2E ms | E2E reduction % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Native Full | 351.5000 | 235.660 | 64.287 | 64.281 | 2228.560 | +0.000 |
| 1 | SVH w/o Selective Transfer | 55.5625 | 239.407 | 58.927 | 58.919 | 2066.142 | +7.288 |
| 1 | SVH w/o Attention Bypass | 63.5000 | 181.337 | 63.878 | 63.870 | 2161.545 | +3.007 |
| 1 | SVH Full | 55.5625 | 183.956 | 58.815 | 58.808 | 2007.222 | +9.932 |
| 2 | Native Full | 351.5000 | 239.780 | 64.829 | 64.822 | 2249.477 | +0.000 |
| 2 | SVH w/o Selective Transfer | 55.5625 | 240.529 | 59.757 | 59.748 | 2093.000 | +6.956 |
| 2 | SVH w/o Attention Bypass | 63.5000 | 183.658 | 64.323 | 64.315 | 2177.665 | +3.192 |
| 2 | SVH Full | 55.5625 | 185.667 | 59.515 | 59.506 | 2030.618 | +9.729 |


## 相对同轮Native的变化

降低比例先在各Session计算，再取两轮平均；负值表示比Native更高或更慢。

| Condition | KV reduction % | TTFT reduction % | TPOT reduction % | E2E reduction % |
| --- | --- | --- | --- | --- |
| Native Full | +0.000 | +0.000 | +0.000 | +0.000 |
| SVH w/o Selective Transfer | +84.193 | -0.951 | +8.081 | +7.122 |
| SVH w/o Attention Bypass | +81.935 | +23.229 | +0.709 | +3.100 |
| SVH Full | +84.193 | +22.254 | +8.355 | +9.831 |


## 执行路径与物理KV

- 直接调用冻结`unified_e2e32_4090/code/e2e32_runtime.py`原`generate`函数对象，未复制或修改其计时循环、Greedy、模型或Attention。仅在函数外选择已验证消融类；新增KV审计在原计时结束后运行，投影/位置/RoPE观察器仅用于Smoke，正式运行无观察钩子。控制器和掩码初始化仍位于原计时范围内。
- w/o Selective Transfer：实际复用质量评测NoSelectiveTransfer类，视觉隐藏状态通过全部32层Prefill，Relays全层读取图像；答案不直接访问图像，L2–5物理Attention Bypass保持，答案持久KV仍为文本。这一消融不应被误写成保留全部图像KV。
- w/o Attention Bypass：实际复用NoAttentionBypass类，L20视觉隐藏提前退出与图像KV移除不变；L2–5恢复正常Attention及真实文本KV，32层均有文本KV。
- 两探针×四配置的32步logits与对应质量路径逐位相同。数值影子验证仅使用实际Greedy输出作为固定前缀，逐步argmax也必须一致，不使用Gold。Native/No-op及Native/Full-SVH原运行器的全部32步logits严格一致；真实Q/K/V调用、隐藏行数和RoPE检查通过。
- KV@32由实际K/V形状、有效长度、FP16元素大小及独立存储指针统计；生成第32个输出时缓存中有31个Decode输入。逐题逐层核验独立字节公式，不能用总GPU显存代替KV。

## 稳定性与配对统计

每轮独立进程加载模型；固定40次Native完整32Token预热，加四条件各2次预热，双探针各10次建立独立中位数基线。每题交错四条件，顺序按题号循环轮换，两轮调度完全相同。每5题双探针各3次校准，性能漂移只WARN，无重置基线、重跑或删除请求。5Hz GPU、1Hz CPU和其他GPU进程遥测保留。

24项校准中0项超过2%；最大绝对E2E漂移1.987%，最大绝对TPOT漂移1.950%。两轮同题同配置32Token输出完全一致的数量为120/120；所有比较在首EOS之前及EOS位置一致。

| Session | Window | Probe | E2E drift % | TPOT drift % | Flag |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | GQA:202121628 | +1.629 | +1.631 | within2% |
| 1 | 1 | TextVQA:38571 | +0.987 | +1.376 | within2% |
| 1 | 2 | GQA:202121628 | -0.202 | -0.473 | within2% |
| 1 | 2 | TextVQA:38571 | -0.490 | -0.347 | within2% |
| 1 | 3 | GQA:202121628 | -0.497 | -0.170 | within2% |
| 1 | 3 | TextVQA:38571 | -0.131 | -0.105 | within2% |
| 1 | 4 | GQA:202121628 | +1.987 | +1.950 | within2% |
| 1 | 4 | TextVQA:38571 | +0.639 | +0.368 | within2% |
| 1 | 5 | GQA:202121628 | +0.244 | +0.492 | within2% |
| 1 | 5 | TextVQA:38571 | -0.208 | -0.082 | within2% |
| 1 | 6 | GQA:202121628 | +0.864 | +0.759 | within2% |
| 1 | 6 | TextVQA:38571 | +0.787 | +0.768 | within2% |
| 2 | 1 | GQA:202121628 | +0.543 | +0.707 | within2% |
| 2 | 1 | TextVQA:38571 | +0.053 | +0.374 | within2% |
| 2 | 2 | GQA:202121628 | -1.343 | -1.231 | within2% |
| 2 | 2 | TextVQA:38571 | -1.658 | -1.684 | within2% |
| 2 | 3 | GQA:202121628 | -1.243 | -1.127 | within2% |
| 2 | 3 | TextVQA:38571 | -1.385 | -1.451 | within2% |
| 2 | 4 | GQA:202121628 | +0.733 | +1.296 | within2% |
| 2 | 4 | TextVQA:38571 | -1.309 | -1.217 | within2% |
| 2 | 5 | GQA:202121628 | -0.545 | -0.405 | within2% |
| 2 | 5 | TextVQA:38571 | +0.956 | +1.009 | within2% |
| 2 | 6 | GQA:202121628 | +0.516 | +0.786 | within2% |
| 2 | 6 | TextVQA:38571 | +0.367 | -0.228 | within2% |


| Session | Condition | E2E faster /30 | Paired Native−condition E2E ms | Exploratory95% CI ms |
| --- | --- | --- | --- | --- |
| 1 | SVH w/o Selective Transfer | 30 | 162.417 | [152.869, 171.966] |
| 1 | SVH w/o Attention Bypass | 28 | 67.014 | [51.897, 82.131] |
| 1 | SVH Full | 30 | 221.338 | [213.643, 229.032] |
| 2 | SVH w/o Selective Transfer | 30 | 156.477 | [143.538, 169.416] |
| 2 | SVH w/o Attention Bypass | 28 | 71.812 | [56.424, 87.200] |
| 2 | SVH Full | 30 | 218.859 | [211.058, 226.661] |


区间为30题配对差的探索性t区间，含系统状态相关性，未作多重检验校正。两轮绝对时延变化、逐请求异常和所有慢请求均保留；即使全部执行完成，性能WARN也不能称稳定PASS。

## 结果与复现

`summary.csv`为四条件综合表；`session_summary.csv`、`paired_statistics.csv`、`per_request.csv`、`kv_per_layer.csv`、`calibration.json`及两轮SQLite保存完整证据。`protocol.json`和`source_manifest.json`为测量前冻结记录，`artifact_manifest.json`为完成后清单。原SQA/POPE结果与此前长度实验均未覆盖。

仅在新目录、核验相同冻结源码与输入并适配实例UUID后运行；已存在的协议与Session不允许覆盖。

```bash
source /home/featurize/work/ATE/vlm_research/results/featurize_22484-20261113-r1_resource_restore/activate.sh
cd /home/featurize/work/ATE/vlm_research/results/svh_component_ablation_e2e32_3090
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -B code/c55_prepare.py
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/c55_run.py --session 1
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -u -B code/c55_run.py --session 2
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -B code/c55_analyze.py
```

## 相对Full SVH的组件影响

正值表示关闭该组件后更慢或KV更大；这里按同题、同Session与Full SVH配对。

| Session | Condition | Metric | Condition−Full mean | Exploratory95% CI | Increase vs Full % |
| --- | --- | --- | --- | --- | --- |
| 1 | SVH w/o Selective Transfer | KV_MiB | +0.0000 | [0.0000, 0.0000] | +0.000 |
| 1 | SVH w/o Selective Transfer | TTFT_ms | +55.4509 | [52.9595, 57.9424] | +30.144 |
| 1 | SVH w/o Selective Transfer | TPOT_ms | +0.1119 | [-0.1452, 0.3690] | +0.190 |
| 1 | SVH w/o Selective Transfer | E2E_ms | +58.9204 | [50.0838, 67.7570] | +2.935 |
| 1 | SVH w/o Attention Bypass | KV_MiB | +7.9375 | [6.4581, 9.4169] | +14.286 |
| 1 | SVH w/o Attention Bypass | TTFT_ms | -2.6195 | [-4.7250, -0.5140] | -1.424 |
| 1 | SVH w/o Attention Bypass | TPOT_ms | +5.0627 | [4.6322, 5.4931] | +8.608 |
| 1 | SVH w/o Attention Bypass | E2E_ms | +154.3232 | [140.6757, 167.9707] | +7.688 |
| 2 | SVH w/o Selective Transfer | KV_MiB | +0.0000 | [0.0000, 0.0000] | +0.000 |
| 2 | SVH w/o Selective Transfer | TTFT_ms | +54.8613 | [52.5763, 57.1464] | +29.548 |
| 2 | SVH w/o Selective Transfer | TPOT_ms | +0.2426 | [-0.1700, 0.6552] | +0.408 |
| 2 | SVH w/o Selective Transfer | E2E_ms | +62.3821 | [49.2421, 75.5221] | +3.072 |
| 2 | SVH w/o Attention Bypass | KV_MiB | +7.9375 | [6.4581, 9.4169] | +14.286 |
| 2 | SVH w/o Attention Bypass | TTFT_ms | -2.0091 | [-5.1581, 1.1399] | -1.082 |
| 2 | SVH w/o Attention Bypass | TPOT_ms | +4.8083 | [4.4055, 5.2111] | +8.079 |
| 2 | SVH w/o Attention Bypass | E2E_ms | +147.0476 | [134.0036, 160.0916] | +7.242 |


## 性能警告分类

| Session | Type | Category | Count |
| --- | --- | --- | --- |
| 1 | within_request_event_decode_change_over5pct | formal | 1 |
| 1 | within_request_event_decode_change_over5pct | smoke | 14 |
| 1 | within_request_event_decode_change_over5pct | warmup | 2 |
| 2 | within_request_event_decode_change_over5pct | calibration | 1 |
| 2 | within_request_event_decode_change_over5pct | formal | 2 |
| 2 | within_request_event_decode_change_over5pct | smoke | 14 |
| 2 | within_request_event_decode_change_over5pct | warmup | 2 |


请求内早/晚Decode差异超过5%是诊断标记，不能单凭它判定GPU降频；Smoke含数值捕获与观察钩子，其计时不用于正式统计。Native校准、正式请求内标记和预热/Smoke标记分别列出，不能混为同一种稳定性结论。
SVH w/o Selective Transfer相对Full SVH，两轮同轮比值的平均变化：KV +0.000%，TTFT +29.846%，TPOT +0.299%，E2E +3.004%。是否能推广到其他输入和GPU状态，仍受固定30题、两个Session与上述性能警告限制。
SVH w/o Attention Bypass相对Full SVH，两轮同轮比值的平均变化：KV +14.286%，TTFT -1.253%，TPOT +8.343%，E2E +7.465%。是否能推广到其他输入和GPU状态，仍受固定30题、两个Session与上述性能警告限制。

## GPU释放与归档

两个测量进程均已退出。最终GPU显存占用1 MiB；其他计算进程记录见`final_resource_status.json`。进程退出前分配量原样保留，未伪装为零。服务器保持运行。完整原始SQLite、遥测、校准和失败记录保留在本目录；轻量交付不包含模型权重或图片。
