# 同一GPU严格续跑与证据说明

- 当前GPU UUID：GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e。
- 固定模型：/home/featurize/ate-local-models/InternVL3-8B；Revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9。
- 固定环境：/home/featurize/.venvs/ate-frozen/bin/python。
- 独立工作树：/home/featurize/work/ATE-internvl3-figure3-3000。
- 原仓库同名结果路径是指向本目录的链接，不覆盖历史结果。

先检查generation_complete.json与validation.json。完整结果存在时只核验，不重新启动GPU。
中断时保留protocol.json、source_manifest.json、sample_manifest.json、cpu_input_bindings.json、所有Checkpoint和失败记录。
只可在相同GPU UUID且GPU无计算进程时继续，不混合其他GPU/实例结果，也不覆盖或编辑冻结协议及源码。

在确认既有工作进程已经结束、GPU空闲且UUID相同之后，续跑命令：
    cd /home/featurize/work/ATE-internvl3-figure3-3000
    CUDA_VISIBLE_DEVICES=0 /home/featurize/.venvs/ate-frozen/bin/python -u -m vlm_research.internvl3_visual_writing_figure3_3000.run

程序先核对源码/模型/样本哈希并重新做必要Smoke；完整且哈希绑定一致的逐题31条件会跳过，部分记录只补缺失条件。
不要并发启动多个worker。出现数值、No-op、Mask、Cache或UUID错误时保留失败并停止，不自动修改协议重跑。

运行期间Checkpoint路径可链接到本地SSD，由独立存储助手异步镜像到同结果目录的checkpoints_persistent。
迁移记录见io_layout.json，镜像进度见persistence_progress.json；该操作不改变生成源码、协议或已有预测。
GPU进程结束后，助手核对全部SHA并恢复普通工作盘checkpoints目录，写persistence_complete.json。
本地SSD只是当前实例的快速暂存；持久证据以工作盘通过SHA验证的实体目录为准。

生成及持久化完成后，仅CPU后处理：
    CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/featurize/.venvs/ate-frozen/bin/python -u -m vlm_research.internvl3_visual_writing_figure3_3000.analyze
    /home/featurize/.venvs/ate-frozen/bin/python vlm_research/results/internvl3_visual_writing_figure3_3000/plot_figure3.py
    CUDA_VISIBLE_DEVICES='' /home/featurize/.venvs/ate-frozen/bin/python -u -m vlm_research.internvl3_visual_writing_figure3_3000.finalize

图表和统计源码另由analysis_source_freeze.json绑定，逐请求生成代码由protocol/source_manifest绑定。
run_support保留本次存储/CPU收尾助手，它们含本次PID，属于现场运行记录，不应原样用于其他实例。
未授权GitHub推送；本目录与分支均保留服务器，不上传权重、图片、密钥或系统缓存。
