# 新 RTX 4090 环境恢复

状态：**READY_WITH_WARNING（0.01015 数值差异已获用户接受）**。仅恢复环境及最小 Smoke，未启动正式评测。

Python 3.11.8；Torch 2.6.0+cu124 / CUDA runtime 12.4；Transformers 4.49.0；RTX 4090，驱动610.57.04。

模型：`/home/featurize/ate-local-models/llava-1.5-7b-hf`，Revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。14 个文件/三片权重通过官方内容哈希与历史验收；FP16/Eager 加载一次，峰值分配显存 13.8755 GiB。

数据实际根目录：`/home/featurize/ate-local-data/ate-recovery`（本地 ext4）；兼容入口 `/home/featurize/data/ate-recovery`。模型、数据、虚拟环境和缓存均不在 NFS 持续盘。

| 冻结视图 | 题数 | 状态 |
|---|---:|---|
| GQA-testdev-balanced | 12,578 | PASS，normalized/GQA-testdev-balanced |
| MME-paper | 2,374 | PASS，normalized/MME-paper |
| ScienceQA-IMG-paper | 2,017 | PASS，normalized/ScienceQA-IMG-paper |
| POPE-paper | 8,910 | PASS，normalized/POPE-paper |
| TextVQA-OCR | 5,000 | PASS，normalized/TextVQA-OCR |
| MMBench-paper | 4,377 | PASS，normalized/MMBench-paper |
| OCRBench-v1 | 1,000 | PASS，normalized/OCRBench-v1 |

六包所有恢复视图的 Input/Gold 文件哈希与3090冻结记录完全一致；七包整包 SHA256 及归档 CRC通过。OCRBench 1,000问/930图，官方ID、答案、顺序和路径逐项一致；MMBench 1,176 Circular Groups保持。MME已知单图SHA差异保留。TextVQA Paper-OCR与Image-only独立。

GQA默认使用官方对齐 Vicuna-v1 + Square-pad，cap32；旧HF-crop成绩仅索引历史，不跨协议求差。其他六项继续冻结Paper协议。另一台GPU正在完成新的GQA三组任务，本次只读索引进度，未干预其进程、数据库或输出。

最小验收：3个固定样本，Native/C3/Reaccess各3个完整答案；另做必要的三Token物理/同语义Dense参照与Native/HF/No-op一致性检查。视觉退出层、视觉KV驻留、L2–L5物理Attention跳算、Norm/MLP/Residual、原始位置RoPE、禁图概率0、真实Decode和有限输出全部通过。历史ScienceQA/POPE的C3完整答案与评分保持一致；小样本答案本身不构成Accuracy结果。

严格数值结果：GQA及ScienceQA两方法均PASS；POPE C3 RMSE=0.010154514573514462，Reaccess=0.010152915492653847，均超过冻结0.01，严格项记FAIL。短输出/argmax一致、结构检查PASS；用户明确接受该差异，未改阈值。保留历史失败记录。

具备继续HF Native / Physical C3 / Reaccess研究的基础环境条件。正式评测仍需后续单独授权。GPU已释放：1 MiB、0%，无计算进程。

进入环境：`source /home/featurize/ate-env.sh`。配置入口 `runtime_config.py`，不会在导入时启动实验。

独立分支：`codex/featurize-4090-recovery-20261108`。GitHub与Windows最终传输验证收据在发布后单独保存；未经收据验证不宣称备份成功。

详见 environment.json、dataset_restore.json、model_verification.json、physical_smoke_summary.json、runtime_paths.json 与 history_index.json。
