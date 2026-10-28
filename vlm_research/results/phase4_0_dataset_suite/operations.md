# 操作审计

## 仓库与环境

开始与发布前分别通过 `git ls-remote --heads origin` 核对远端。
main均为 `e89c8acce6e674ecfbd2c8acad57faf71bf4d547`。
已读取Phase3.7B结果提交 `fc0d802f1d1ee50e54b150bb2ffb398402966175`、Phase3.8提交 `2415b69264d20c5da1c47c18bdb0b5adcafa69a7` 和Phase3.8b C2结果 `1a451e14260f04aeafbd27515cad60ed7d5e7dd0` 的报告；没有修改对应Worktree。

```bash
git fetch origin main
git worktree add -b exp/phase4_0-benchmark-datasets /data/lk/ATE-worktrees/phase40-benchmark-datasets FETCH_HEAD
```

实际Python：`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`。
命令中针对失效代理取消代理变量，没有修改全局配置或打印认证信息。
磁盘预检 `/data` 约393GB可用，设置下载50GiB安全余量；四GPU检查空闲，没有停止其他人的进程。

## 数据准备

在新Worktree的 `vlm_research/` 执行：

```bash
python -m benchmarks.acquire --root /data/lk/ATE-resources/benchmarks/phase4_0/sources
python -m benchmarks.fetch_resources annotations
python -m benchmarks.fetch_resources pope
python -m benchmarks.fetch_resources scienceqa
python -m benchmarks.fetch_resources textvqa
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.prepare_datasets
```

最初GitHub raw连接超时，取消本任务自己的慢下载/全盘文件搜索，改用固定Commit的GitHub Contents API取得同一文件；没有停止任何GPU任务。HF官方连接不可用，ScienceQA改用同一公开仓库固定revision的国内镜像。
可选 `pyarrow==19.0.1` 仅安装在共享资源的 `tools/`，ScienceQA提取时通过 `PYTHONPATH=/data/lk/ATE-resources/benchmarks/phase4_0/tools` 使用，未改变固定VLM环境。
原始下载失败及重试见 `resource_provenance.json`；额外参考论文、LLaVA许可与ScienceQA发布者脚本也保存于外部sources。最终acquire入口已包含必要发布者脚本，干净资源目录不再依赖手工取脚本。
ScienceQA初次仅文本字段匹配不能唯一确定24题，暂停这些题的图片提取；随后阅读发布者脚本并验证全部4241题顺序/字段，再按原PID恢复，不使用任意同文本图片。
TextVQA仅读取官方ZIP所需范围，所有309张图完成，无全ZIP落盘。归档身份由原始下载记录固定，后续重试拒绝ETag/版本变化。

## 验证顺序

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.acceptance
git commit -m 'Freeze Phase 4.0 cross-dataset suite and CPU acceptance before native smoke'
CUDA_VISIBLE_DEVICES=GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.smoke
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.publication --cpu-review
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m benchmarks.publication --cpu-review --review-name pre_push
```

GPU前136项通过；冻结代码 `24523aef248f5bf1943d8e0d51d8febb187a105b`；GPU18例完成且无失败尝试。
第一次发布复核139项通过；之后补齐下载重建入口、归档身份及范围读取测试，最终141项通过。所有旧测试日志均保留。
发布层新增权威跨数据集图片重叠审计，避免混淆不同命名空间的同名数字ID；没有修改原Manifest、抽样、Logits或评分。
最后执行 `pip check`、原始Logits独立重算、制品哈希封存、Git差异范围检查和远端分支核验。

上述为命令与实际状态摘要；GPU逐例JSON/NPZ是真实运行输出，不把本页文字当作原始终端全量转录。
