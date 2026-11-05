# Middle10质量恢复：CPU准备记录

**状态：PARTIAL / CPU_READY_GPU_PENDING。没有新增GPU实验结果。**

起点：`a3727ba0808b183bc8bede4396a86e82855c7d5a`。
独立分支：`codex/c3v2-middle10-quality-recovery`。

已实现固定E1–E5物理路径及对应Dense功能参考；未修改历史引擎。
19项CPU测试通过，包含12项历史回归和7项恢复条件测试。
已核验同一500题、309个Image SHA簇、模型权重和2,000条历史质量断点。
原48题性能集及图片整簇分片保持不变。

E1恢复L20 R读图；E2增加指定7层Prefill P−1读图；E3联合二者；
E4仅恢复Prefill P−1的L2–L5文本Attention，Decode继续跳算且不增加持久KV；
E5保持Middle9回看权限，只将视觉Hidden退出后移至L21。
所有新条件的物理KV及CPU Dense等价测试通过，不代表真实GPU数值验证已通过。

附件要求本轮重新确认设备授权，不能继承历史授权。
预检时物理GPU0–3均为空闲RTX3090（各27MiB、0%利用率）；已询问是否授权GPU0和GPU1。
尚未加载GPU模型、运行GPU Smoke、启动2,500条新增生成或性能测试。
收到授权后须重新检查实时占用，再冻结GPU/源码协议并通过双卡真实Smoke。

CPU复现：

```bash
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.middle10_recovery.prepare
```

不把CPU通过、历史基线或待执行条件记为本轮完整实验结果。
