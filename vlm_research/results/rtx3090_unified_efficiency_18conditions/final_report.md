# RTX3090统一效率实测

本报告只汇总本次实际记录。所有正式请求绑定 GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6。表中数值以 **Session1 / Session2** 展示；不混用旧GPU、不选最快Session、不将两轮简单混合。质量Benchmark请求数为0。

实际正式请求：1080/1080（目标18条件×30题×2轮）；性能告警：6项。总体状态：COMPLETE_WITH_PERFORMANCE_WARNINGS。

## Table A：LLaVA-1.5-7B

|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|TTFT减少%|TPOT减少%|KV减少%|状态|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|Native Full|351.50 / 351.50|240.75 / 241.44|66.85 / 66.47|2313.21 / 2301.95|0.00 / 0.00|0.00 / 0.00|0.00 / 0.00|COMPLETE|
|SVH|55.56 / 55.56|184.25 / 182.67|61.50 / 61.03|2090.88 / 2074.52|23.55 / 24.41|8.00 / 8.18|84.65 / 84.65|COMPLETE|
|SVH-R|181.56 / 181.56|210.03 / 208.94|61.56 / 61.11|2118.40 / 2103.37|12.81 / 13.53|7.91 / 8.06|48.58 / 48.58|COMPLETE|
|FastV|177.50 / 177.50|156.79 / 155.50|67.96 / 67.51|2263.63 / 2248.39|35.20 / 35.90|-1.66 / -1.57|49.82 / 49.82|COMPLETE|
|SparseVLM|166.75 / 166.75|175.00 / 175.71|70.30 / 69.88|2354.27 / 2342.07|27.18 / 27.07|-5.16 / -5.14|52.90 / 52.90|COMPLETE|
|V²Drop|164.91 / 164.91|147.19 / 146.88|67.52 / 67.06|2240.42 / 2225.61|38.87 / 39.18|-1.00 / -0.89|53.43 / 53.43|COMPLETE|
|ApET|159.50 / 159.50|161.29 / 160.38|68.53 / 67.93|2285.76 / 2266.28|32.89 / 33.45|-2.51 / -2.21|54.97 / 54.97|COMPLETE|
|DUET-VLM (C+all)|159.38 / 159.38|162.55 / 162.76|69.52 / 69.26|2317.79 / 2309.92|32.43 / 32.50|-4.00 / -4.21|55.01 / 55.01|COMPLETE|
|FastV+SVH|55.56 / 55.56|171.20 / 170.05|61.29 / 60.95|2071.34 / 2059.54|28.90 / 29.59|8.31 / 8.30|84.65 / 84.65|COMPLETE|
|SparseVLM+SVH|55.56 / 55.56|189.96 / 190.06|61.60 / 61.25|2099.46 / 2088.77|20.87 / 21.03|7.86 / 7.85|84.65 / 84.65|COMPLETE|
|V²Drop+SVH|55.56 / 55.56|177.11 / 175.87|61.73 / 61.23|2090.70 / 2074.12|26.18 / 27.00|7.66 / 7.87|84.65 / 84.65|COMPLETE|
|ApET+SVH|55.56 / 55.56|164.28 / 164.18|76.35 / 76.41|2531.14 / 2532.95|31.51 / 31.74|-14.21 / -14.96|84.65 / 84.65|COMPLETE|

## Table B：同Backbone配对比较

|Backbone|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|E2E减少%|
|---|---|---:|---:|---:|---:|---:|
|LLaVA|Native Full|351.50 / 351.50|240.75 / 241.44|66.85 / 66.47|2313.21 / 2301.95|0.00 / 0.00|
|LLaVA|SVH|55.56 / 55.56|184.25 / 182.67|61.50 / 61.03|2090.88 / 2074.52|9.61 / 9.88|
|LLaVA|SVH-R|181.56 / 181.56|210.03 / 208.94|61.56 / 61.11|2118.40 / 2103.37|8.42 / 8.63|
|Qwen|Native Full|86.25 / 86.25|383.67 / 383.40|93.53 / 90.10|3283.11 / 3176.36|0.00 / 0.00|
|Qwen|SVH|10.48 / 10.48|361.30 / 357.28|92.65 / 88.82|3233.52 / 3110.83|1.50 / 2.04|
|Qwen|SVH-R|45.27 / 45.27|467.18 / 459.92|107.67 / 102.82|3804.98 / 3647.23|-15.87 / -14.84|
|InternVL|Native Full|20.97 / 20.97|144.62 / 148.43|61.22 / 62.32|2042.52 / 2080.22|0.00 / 0.00|
|InternVL|SVH|5.97 / 5.97|129.69 / 130.26|56.02 / 56.58|1866.30 / 1884.23|8.65 / 9.43|
|InternVL|SVH-R|10.47 / 10.47|147.80 / 148.23|57.88 / 58.27|1942.14 / 1954.59|4.94 / 6.00|

## Table C：Pruning与其SVH组合

所有收益严格以同题、同Session原始方法为分母。负数代表组合更慢/更大。

|原始方法 → 组合|Session|配对题数|KV减少%|TTFT减少%|TPOT减少%|E2E减少%|
|---|---|---:|---:|---:|---:|---:|
|Native Full → SVH|session1|30|84.65|23.55|8.00|9.61|
|Native Full → SVH|session2|30|84.65|24.41|8.18|9.88|
|FastV → FastV+SVH|session1|30|69.93|-10.44|9.81|8.49|
|FastV → FastV+SVH|session2|30|69.93|-10.40|9.72|8.40|
|SparseVLM → SparseVLM+SVH|session1|30|68.01|-8.71|12.35|10.80|
|SparseVLM → SparseVLM+SVH|session2|30|68.01|-8.49|12.34|10.81|
|V²Drop → V²Drop+SVH|session1|30|67.65|-20.86|8.57|6.67|
|V²Drop → V²Drop+SVH|session2|30|67.65|-20.18|8.68|6.80|
|ApET → ApET+SVH|session1|30|66.56|-2.19|-11.44|-10.77|
|ApET → ApET+SVH|session2|30|66.56|-2.73|-12.50|-11.78|

### Table C：原始/组合逐行绝对时间

|配对组|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|
|---|---|---:|---:|---:|---:|
|Native Full|Native Full|351.50 / 351.50|240.75 / 241.44|66.85 / 66.47|2313.21 / 2301.95|
|Native Full|SVH|55.56 / 55.56|184.25 / 182.67|61.50 / 61.03|2090.88 / 2074.52|
|FastV|FastV|177.50 / 177.50|156.79 / 155.50|67.96 / 67.51|2263.63 / 2248.39|
|FastV|FastV+SVH|55.56 / 55.56|171.20 / 170.05|61.29 / 60.95|2071.34 / 2059.54|
|SparseVLM|SparseVLM|166.75 / 166.75|175.00 / 175.71|70.30 / 69.88|2354.27 / 2342.07|
|SparseVLM|SparseVLM+SVH|55.56 / 55.56|189.96 / 190.06|61.60 / 61.25|2099.46 / 2088.77|
|V²Drop|V²Drop|164.91 / 164.91|147.19 / 146.88|67.52 / 67.06|2240.42 / 2225.61|
|V²Drop|V²Drop+SVH|55.56 / 55.56|177.11 / 175.87|61.73 / 61.23|2090.70 / 2074.12|
|ApET|ApET|159.50 / 159.50|161.29 / 160.38|68.53 / 67.93|2285.76 / 2266.28|
|ApET|ApET+SVH|55.56 / 55.56|164.28 / 164.18|76.35 / 76.41|2531.14 / 2532.95|

## 协议与限制

- LLaVA：FP16/Eager、Torch2.6.0+cu124、Transformers4.49.0；Qwen：BF16/Eager/4.57.6；InternVL：BF16/Eager/4.49.0，单448图块/256视觉Token。同组输入、后端、精度一致，跨Backbone绝对时间不可排名。
- 固定原30题GQA15+TextVQA Paper-OCR15。LLaVA三类输入Tensor哈希与历史完全一致；其他模型使用同一图片、问题和OCR内容及各自原生Chat。Gold从不传入模型。每次冷请求独立Cache，精确32输出，记录首次EOS但不提前停止。
- E2E从GPU输入准备完毕开始，包含Vision/Projector/Prefill/31次真实Decode及控制器建立与清理；不含文件读取、预处理、Tokenizer、H2D、最终KV审计、输出写盘。TPOT=(E2E−TTFT)/31，逐步CUDA Event另存。KV为真实有效Tensor Live，allocated及峰值显存单列。
- FastV为K3静态KV的192深层保留版本；SparseVLM实际层均值206.5；V²Drop独立版576→288→144→1，融合版仅L3/L17前缀后L20退出；二者不是相同完整剪枝计划。ApET独立版288→96层均值192；DUET为C+all而非论文C。
- 新ApET+SVH保留官方FPS/基底替换/误差排序，288→96，L20退出与零视觉持久KV同时生效。通过物理/Dense诊断数值门槛；没有重测质量，不能直接绑定尚未找到的旧ApET+SVH质量成绩。
- Qwen SVH保留历史BF16数值对齐实现，含临时零行/零Key及固定cuBLASLt描述符桥接；临时零不是持久视觉KV，相关计算开销全部计入。SVH-R额外构建晚层视觉KV的因果前缀支路，完整计入TTFT/E2E，不能称为免费KV。InternVL SVH历史数值Waiver明确保留，SVH-R保留Late9_L19_27与终端退出28实现及临时零Key。
- 融合冻结源码中内置选择验证、CPU元数据提取和同步均保持原路径，其成本计入实测。这是所列具体实现的效率，不能当成无控制开销的理想算子效率。正式请求没有新增逐层调试Hook。
- 原始协议、首轮SparseVLM路径错误及修订记录完整保留。无算法调参、无GPU功耗/时钟设置修改、无质量评测、无训练、无GitHub写入。

## 完成状态与失败

```json
{
  "LLaVA_session1": {
    "status": "COMPLETE",
    "formal_completed": 360,
    "formal_expected": 360
  },
  "LLaVA_session2": {
    "status": "COMPLETE",
    "formal_completed": 360,
    "formal_expected": 360
  },
  "Qwen_session1": {
    "status": "COMPLETE_WITH_PERFORMANCE_WARNINGS",
    "formal_completed": 90,
    "formal_expected": 90
  },
  "Qwen_session2": {
    "status": "COMPLETE",
    "formal_completed": 90,
    "formal_expected": 90
  },
  "InternVL_session1": {
    "status": "COMPLETE",
    "formal_completed": 90,
    "formal_expected": 90
  },
  "InternVL_session2": {
    "status": "COMPLETE_WITH_PERFORMANCE_WARNINGS",
    "formal_completed": 90,
    "formal_expected": 90
  },
  "LLaVA_smoke": {
    "status": "PASS",
    "usable": [
      "Native",
      "SVH",
      "SVH-R",
      "FastV",
      "SparseVLM",
      "V2Drop",
      "ApET",
      "DUET",
      "FastV+SVH",
      "SparseVLM+SVH",
      "V2Drop+SVH",
      "ApET+SVH"
    ]
  },
  "Qwen_smoke": {
    "status": "COMPLETE",
    "formal_completed": 0,
    "usable": [
      "Native",
      "SVH",
      "SVH-R"
    ]
  },
  "InternVL_smoke": {
    "status": "COMPLETE",
    "formal_completed": 0,
    "usable": [
      "Native",
      "SVH",
      "SVH-R"
    ]
  }
}
```

性能告警详见performance_warnings.json；每轮完整均值、中位数、标准差、P90见results_summary.csv。原始32Token、逐步时间、逐层真实KV、SQLite及SHA证据保留。复现命令见reproduce.sh。

## 实测结论

- LLaVA SVH的KV减少84.65%，同题配对E2E减少9.61% / 9.88%；SVH-R分别为48.58%和8.42% / 8.63%。
- FastV/SparseVLM/V²Drop与SVH组合后，本轮配对E2E均改善；ApET+SVH虽减少KV，却比独立ApET慢10.77% / 11.78%，负结果原样保留。未据此修改适配器或重测。
- Qwen SVH的E2E减少仅1.50% / 2.04%，在已观察漂移下不应宣称稳定小幅加速；SVH-R则慢15.87% / 14.84%。因果前缀视觉KV构建及冻结控制器开销全部计入。
- InternVL SVH的E2E减少8.65% / 9.43%，SVH-R减少4.94% / 6.00%，但保留历史数值Waiver及本轮性能告警。
- 全18条件的540对跨Session请求，输入及完整32Token序列均一致。所有正式样本完成且无OOM/非有限值/结构故障；SQLite与JSONL逐字节一致。
- Qwen Session1末尾探针TPOT慢约16%；InternVL Session2多次探针快约13%—14%。LLaVA未触发5%告警。温度最高60°C，监控未发现其他GPU计算进程；不能据此确定漂移原因。
- 这些数据可作为已列明实现/配置的同机效率证据；论文使用时须保留分Session结果、性能告警和算法配置差异，不与未核实的历史质量配置绑定，不做跨Backbone绝对延迟排名。

GPU释放核验：{"at_ns": 1794505665823730359, "GPU_UUID": "GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6", "all_compute_apps": "", "released": true, "server_closed": false, "nvidia_smi": "NVIDIA GeForce RTX 3090, GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6, 1 MiB, 0 %, 35, 23.32 W"}

存在性能漂移告警，结果按Session分别展示；不能将其表述为没有漂移的稳定加速。原始慢样本全部保留，未择优重试或重置基线。告警不能仅凭时间差强行归因于GPU频率或CPU调度。
