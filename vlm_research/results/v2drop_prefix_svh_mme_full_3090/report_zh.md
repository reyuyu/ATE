> 同步说明：本报告现随独立证据分支公开；实验结束时的“未 Push”状态保留于下方原文。完整预测、数据库与原始报告已在 Windows 完成 SHA256 备份。归档范围、压缩 Smoke 文件及当前发布说明见七项结果目录的 [README](../v2drop_prefix_svh_sevenbench_3090/README.md)。

# V²Drop-prefix + SVH：MME 全量正式评测

**状态：COMPLETE。** 完成 MME Paper 2,374/2,374 题、1,187/1,187 个同图双题组；14 类均完整。

| 条件 | Perception | Cognition | Total |
|---|---:|---:|---:|
| V²Drop-prefix + SVH | 1467.895358 | 312.500000 | 1780.395358 |
| llava-SVH 历史同协议对照（直接复用） | 1478.823229 | 348.928571 | 1827.751801 |
| Fusion − SVH | -10.927871 | -36.428571 | -47.356443 |

分数单位为 MME Score：每个子任务 Accuracy + Accuracy+；Accuracy+ 要求同图两题全部答对，再按原任务分组累加。Total 不是百分比，也不是将所有问题简单平均。逐题 Yes/No 解析、子任务评分与总分均调用冻结原评分器，并独立用计数公式复核通过。

逐题 Accuracy（仅辅助）78.685762%；空输出 0；截断 0；Yes/No 解析失败 2，全部保留计分。
相对历史 SVH：正确→错误 74 题，错误→正确 48 题。全部 2,374 题的 Prompt、Token IDs、FP32 预处理像素哈希（随后转相同 FP16）、Gold 与图像标识均与历史存档逐题匹配，因此可作本协议下的配对比较；未重跑 SVH、Native 或 V²Drop。

| Task | Questions | Pairs | Accuracy (%) | Accuracy+ (%) | MME Score | Δ vs SVH |
|---|---:|---:|---:|---:|---:|---:|
| existence | 60 | 30 | 95.0000 | 90.0000 | 185.0000 | -5.0000 |
| count | 60 | 30 | 75.0000 | 53.3333 | 128.3333 | -5.0000 |
| position | 60 | 30 | 73.3333 | 50.0000 | 123.3333 | -5.0000 |
| color | 60 | 30 | 93.3333 | 86.6667 | 180.0000 | +10.0000 |
| posters | 294 | 147 | 84.0136 | 70.0680 | 154.0816 | +2.3810 |
| celebrity | 340 | 170 | 74.1176 | 53.5294 | 127.6471 | -7.0588 |
| scene | 400 | 200 | 87.5000 | 76.5000 | 164.0000 | +2.2500 |
| landmark | 400 | 200 | 87.0000 | 74.5000 | 161.5000 | -6.0000 |
| artwork | 400 | 200 | 71.0000 | 48.0000 | 119.0000 | -5.0000 |
| OCR | 40 | 20 | 75.0000 | 50.0000 | 125.0000 | +7.5000 |
| commonsense_reasoning | 140 | 70 | 67.1429 | 42.8571 | 110.0000 | -1.4286 |
| numerical_calculation | 40 | 20 | 40.0000 | 15.0000 | 55.0000 | +5.0000 |
| text_translation | 40 | 20 | 57.5000 | 15.0000 | 72.5000 | -42.5000 |
| code_reasoning | 40 | 20 | 55.0000 | 20.0000 | 75.0000 | +2.5000 |

真实环境：NVIDIA GeForce RTX 3090, GPU-ca74e8a3-413d-9ddb-76ac-3dc7a1a9a292, 610.57.04, 1 MiB, 24123 MiB；torch 2.6.0+cu124、Transformers 4.49.0、FP16/Eager、batch=1、greedy、EOS=2、max_new_tokens=128。模型 `llava-hf/llava-1.5-7b-hf@b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；冻结基线 Commit `e281754ef17b3accc09d6938b5b9f020048b2aec`；新增独立执行器文件 SHA256 见 protocol.json，未修改冻结核心或之前的融合实验结果。

算法沿用上一轮已验收的 PrefixSVH，官方选择器只在 L3/L17 进行 576→288→144 物理剪枝，L20 前退出全部视觉 Hidden；P−1/Decode 全层不读取原始视觉 KV；L2–L5 答案 Attention 真实跳算，Norm/MLP/Residual 保留；持久视觉 KV 为零。不执行完整 V²Drop 的 L22 剪枝。
上一轮同 GPU、同执行器的 4 图严格数值门禁全部 PASS，本轮未降低门限或重复 GQA/TextVQA。首个 MME 同图双题组追加真实图像、剪枝、位置/RoPE、Attention、缓存与计分检查，PASS；这两题直接计入 2,374 条正式结果，没有重复生成。模型只加载一次。
MME 已知图片差异保留：landmark/03d5e3bfc958be38.jpg，涉及 :0/:1 两题。恢复后的 SHA 与历史 SVH 本次使用的数据一致，未替换图片、未删题；与采用另一图片字节版本的外部结果不能视为全部输入逐字节一致。具体旧/新哈希见 input_audit.json。
原始预测、QID、Gold、Prompt/像素绑定和选择索引保存于 predictions.sqlite / predictions.jsonl.gz；逐题 FULL 同步事务与 progress.json 支持断点恢复。全部结果重新读取后计分，独立复核 1,187 个双题组及 14 类加总。
没有下载模型/数据，没有运行其他 Benchmark、其他算法、层搜索或性能实验。模型进程已结束、GPU 已释放，服务器未关闭；没有 GitHub Push。Windows 真实 SFTP + SHA256 备份回执见 windows_backup_receipt.json。
