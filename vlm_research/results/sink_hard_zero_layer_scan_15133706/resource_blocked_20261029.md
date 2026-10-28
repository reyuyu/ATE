# 2026-10-29 01:28（Asia/Shanghai）资源阻塞记录

执行代码已先冻结于 `fcf043d91e044c5318fb26d2ad939d3d3af3da3d`，118项全工程CPU及24项专项/相关回归通过；GPU阶段再次通过24项CPU检查。

GPU0实际空闲 7742.3 MiB，门槛16000MiB。nvidia-smi同期GPU0空闲8134MiB、GPU1空闲2274MiB，均在运行其他计算任务；CUDA报告与nvidia-smi的可用量口径/采样时刻不同，以实际门禁日志为准。没有终止任务、量化、offload或尝试7B加载，没有GPU Smoke及新候选分数。当前0/144条件完成，不能评价不同层硬置零对预测的影响。

原始资源JSON及24测试记录在 [gpu_attempts/20261029_012846_240504](gpu_attempts/20261029_012846_240504/)。该文件是此次尝试的历史记录，不代表以后GPU状态。

GPU释放后在本实验分支、原Conda环境执行：

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python sink_hard_zero_case.py --stage gpu
```

没有任何已完成预测需要拼接，默认新attempt即可。新结果将执行同一冻结协议，先7B Smoke再144条件。若在后续有分数的尝试上断点恢复，必须使用其binding.json记录的代码Commit/环境，不可将当前blocked尝试当成成功。
