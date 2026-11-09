# InternVL3-8B Visual Writing Figure 3

完成31×3000＝93,000次生成；GPU GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e（RTX3090），BF16/Eager，Revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9，Transformers4.49.0，logits_to_keep=1。
3000题与四份历史清单的QID、顺序、Gold、任务和Image SHA严格一致，2409图片簇均匹配；没有重新抽样或筛掉错误答案。

**All-Writer 90.3333%（2710/3000）；No-Writer 63.3667%；L17–L27 OFF 88.4667%（2654/3000）。**
深层联合关闭的损失（All-Writer减限制条件）**+1.8667 pp**；整簇配对普通95%CI [+1.1807, +2.5848] pp；原对变错85题，原错变对29题。
本轮深层联合关闭的普通95%CI完全大于0，可检出质量损失，不能称为等价或无损。L17边界在GPU执行前固定，没有按结果调整。

## 最敏感的五个单层

| Layer | ACC % | 损失 pp | 普通95%CI pp | Bonferroni-28 CI pp | 对→错 / 错→对 |
| --- | ---: | ---: | --- | --- | ---: |
| L14 | 89.1333 | +1.2000 | [+0.4396, +1.9706] | [-0.0333, +2.4308] | 87 / 51 |
| L16 | 89.3667 | +0.9667 | [+0.3967, +1.5672] | [+0.0668, +1.9192] | 56 / 27 |
| L15 | 89.6333 | +0.7000 | [+0.1338, +1.2701] | [-0.1678, +1.6050] | 48 / 27 |
| L8 | 89.8667 | +0.4667 | [+0.0000, +0.9444] | [-0.2737, +1.2308] | 33 / 19 |
| L19 | 89.9333 | +0.4000 | [-0.1001, +0.9088] | [-0.4334, +1.2052] | 36 / 24 |

Top5按观察到的整体净损失排序；校正区间用于区分峰值与统计证据。
整体单层比较中，Bonferroni-28校正区间完全大于0的层：L16。点估计排序与校正后的统计证据不同，不能把Top5排序视为层之间差异已被显著性检验确认。

## 六任务差异

| Task | All-Writer % | 最大净损害层（含并列） | 单层损失 pp | L17–L27 OFF % | 联合损失 pp |
| --- | ---: | --- | ---: | ---: | ---: |
| ChooseAttr | 85.00 | Drop-L15 | +1.600 | 85.20 | -0.200 |
| ChooseCat | 89.80 | Drop-L8 | +1.600 | 89.20 | +0.600 |
| ChooseRel | 90.00 | Drop-L16 | +2.200 | 89.00 | +1.000 |
| CompareAttr | 88.80 | Drop-L9, Drop-L14, Drop-L16 | +1.400 | 84.80 | +4.000 |
| LogicalObj | 96.60 | Drop-L12 | +0.600 | 95.80 | +0.800 |
| QueryAttr | 91.80 | Drop-L14 | +2.200 | 86.80 | +5.000 |

## 集中性和统计限制

以下是事后探索性描述：Top5层占各层原对变错事件总数的比例。同一题可在不同层重复计数，不是互斥概率质量。
- LLaVA-1.5-7B：Top5损害事件占比 60.11%；净损失≥0.5pp的层 3/32。
- Qwen3-VL-8B：Top5损害事件占比 31.93%；净损失≥0.5pp的层 12/36。
- InternVL3-8B：Top5损害事件占比 49.15%；净损失≥0.5pp的层 3/28。
按上述探索性集中度距离，InternVL更接近LLaVA-1.5-7B；前三个整体净损失层为L14、L16、L15，六任务峰位见表。这个描述不把跨模型差异归因于架构。
最后11/28层（39.29%）联合关闭的各任务损失见上表，存在明显的任务差异。没有预设等价界限，不能用整体分数看起来相近来宣称深层无损。
整体负ΔACC（关闭后改善）的层：Drop-L0、Drop-L1、Drop-L2、Drop-L3、Drop-L5、Drop-L6、Drop-L7、Drop-L10、Drop-L13、Drop-L20、Drop-L21、Drop-L23、Drop-L24、Drop-L25。负值、损害及恢复案例全部保留。
Image SHA整簇Bootstrap20,000次、题数加权、同题配对。28项单层比较在Overall和各任务内分别给Bonferroni-28区间；六任务分析是探索性的，不声称覆盖全部196项比较的统一家族错误率。

## 正确性与历史复用

- 最短/典型/最长题官方Chat、独立Native/No-op全部有效Logits严格torch.equal，Token与视觉注入完全一致。
- Smoke及全部正式请求的被禁概率为0；非目标Mask、Image Query、RoPE、Norm/MLP/Residual及28层真实Cache通过检查。
- Drop-L27在全部3000题上与All-Writer Token和全部有效Logits哈希严格一致；正常/异常Hook清理通过。
- All-Writer不是Native Full：所有条件的原始P−1和全部Decode在28层均不能直接读原始Image Key。
- 历史InternVL Only Text=90.3333%与本轮All-Writer权限、输入和评分同义；只作参考，本轮未复用生成。
- LLaVA/Qwen历史曲线从固定CSV读取；LLaVA深层窗口证据Commit b13a21b07e84a0095731d10dc1e8854f514a7877；Qwen窗口证据Commit dfe0cbacd09de9dfeacff980b911e6aec22a7665。历史数据未重跑、未修改。
- 两次正式生成前的加载/官方Chat参数包装错误保存在engineering_attempts，没有放宽数值门槛，也未排除失败正式样本。
- Checkpoint逐请求写SSD、异步镜像工作盘；迁移SHA及协议哈希一致。最终所有Checkpoint恢复为工作盘实体目录，SHA已核验。

## 物理移除的结论

**本轮结果尚不能单独支持在L17入口物理移除视觉Hidden。** 本实验仅改Dense访问权限，视觉Hidden、视觉Query、Norm/MLP/Residual与真实视觉KV仍计算/保留。小损失不能证明删除Hidden、KV或视觉Query路径后等价；需要另行授权的物理路径验证，本轮未执行。

## 图与证据

- /home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/figure3_internvl_row.png / .pdf
- /home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/figure3_three_models.png / .pdf
- 原始绘图数据：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/plot_layerwise.csv；/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/plot_deep.csv。
- 英文图注：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/figure_caption_en.md；脚本：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/plot_figure3.py。
- 逐题预测：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/per_question.csv；Checkpoint：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000/checkpoints。
PNG≥300DPI；PDF文字、线条、热图矢量化。热图共享0–10%色标，超范围用三角及真实数值显式标注；没有平滑、删负值或裁掉峰值。
本轮附件只有任务文字，没有独立参考Figure3图片；按指定3×3排版和冻结Qwen六任务配色制图，不宣称逐像素复刻未提供的参考图。
各模型Prompt、视觉Token预算、架构及历史软件协议存在差异，已在英文图注及references披露；这是描述性功能比较，不是效率或跨模型因果结论。

## 结束状态

独立分支 codex/internvl3-visual-writing-figure3-3000。GitHub未推送（未获本轮授权）。GPU无计算进程，服务器未关闭。
