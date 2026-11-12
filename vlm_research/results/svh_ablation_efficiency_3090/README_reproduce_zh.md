# ATE Section 5.5 复现说明

本交付中的数字来自真实运行。完整逐题预测、SQLite、原始日志、遥测与校准记录保存在服务器：

`/home/featurize/work/ATE/vlm_research/results/svh_ablation_efficiency_3090/`

Git HEAD：`e281754ef17b3accc09d6938b5b9f020048b2aec`。效率运行器参考提交：`5ee2a0dbe78df336229b55fed14a4a6d0d880bed`。模型：`llava-hf/llava-1.5-7b-hf`，Revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。

`code/` 包含独立入口、两个组件消融子类、长度泛化副本、分析与绘图源码。冻结仓库源码没有被修改。`source_manifest.json` 记录运行前冻结的计算源码，`artifact_manifest.json` 记录完成后的实验文件 SHA256。不要把旧结果目录作为新实验输出目录，也不要直接覆盖已完成的 Checkpoint。

## 资源与环境

- 模型：`/home/featurize/ate-local-models/llava-1.5-7b-hf`
- 数据：`/home/featurize/ate-local-data`
- Python：`/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python`
- Python 3.11.8、PyTorch 2.6.0+cu124、Transformers 4.49.0，全部锁定版本见 `requirements.reference.txt`。
- FP16、Eager、Batch 1、Greedy；OMP/MKL/OpenBLAS 1，torch interop 6。

`s55_common.py` 有实例 UUID、恢复回执和输出目录门禁。跨实例复现应先恢复并核验同一模型及冻结输入，在新的独立结果目录保存实例映射修订证据，不能把新 GPU UUID 冒充为原 UUID。硬编码原路径是本次实验的可追溯记录，不是通用安装器。

## 执行顺序

以下为原运行入口的顺序示例；仅应在上述门禁已适配并核验、全新输出目录中运行。

```bash
source /home/featurize/work/ATE/vlm_research/results/featurize_22484-20261113-r1_resource_restore/activate.sh
cd /home/featurize/work/ATE/vlm_research/results/svh_ablation_efficiency_3090
python -u -B code/s55_prepare.py
python -u -B code/s55_ablation_run.py
python -u -B code/s55_analyze.py --quality-only
python -u -B code/s55_length_run.py --session 1
python -u -B code/s55_length_run.py --session 2
python -u -B code/s55_analyze.py --length-only
python -u -B code/s55_report.py
```

质量评测使用自然 EOS 和原解析规则；效率请求实际生成 16/32/64/128 Token，忽略 EOS。两者不可互换。Full SVH 质量历史预测的复用审计见协议与报告；若复现时不存在已核验的历史文件，应在新任务中补齐对应预测，不能拿公开分数代替。

## 只重绘图，不运行 GPU

使用 Matplotlib 3.8.1：

```bash
python code/s55_plot.py --csv generation_length/latency_two_sessions.csv --output generation_length/latency_vs_tokens.pdf
```

三条曲线使用两个独立 Session 均值的平均；误差线表示两个 Session 均值的范围，不是置信区间。输入 CSV 的 SHA256 保存在绘图回执中。全部长度均为实测，未从 32 Token 外推。

本地交付仅含轻量核心证据；完整运行记录以服务器持久盘目录为准。未向 GitHub Push，未修改论文 LaTeX。
