# Middle16 质量评测

Frozen execution source: `470da109f16c6c6f014d1fa745d88eac2ab4e5e2`.

工作树：`/data/lk/ATE-worktrees/llava-svh-r-middle16-3090-sevenbench`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.svh_middle16.tests
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.svh_middle16.run driver
CUDA_VISIBLE_DEVICES='' /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.svh_middle16.analyze
```

不要重新运行已存在冻结协议的 prepare；新运行器会验证源码、Manifest、分片和既有Checkpoint再续跑。

当前恢复了35,256条原始记录，其中MME整项受指定图像字节不匹配阻塞；OCRBench官方原始图片包尚缺，未写入替代样本。可运行的五项共有32,882条。此状态不能标记七项COMPLETE。

`sample_manifest.jsonl.gz`保留全部已恢复数据，不删除MME的两条受影响题目；`partition.json`只分配验收通过的完整Benchmark。两卡按Image SHA整簇分片，MMBench Circular组并入同簇。

Native/No-op和旧SVH-R只用于短Smoke；正式生成只运行新Middle16。没有复用旧GQA输出作为新算法结果。

历史全量逐题控制输出在原Featurize机器，当前本机只有Git提交中的汇总。未恢复严格绑定的逐题记录前，不计算伪配对区间或损害/纠正数。
