# Middle10：早层与深层Decode视觉回看恢复

## 协议与完整性
- 基线Commit：`0a27cef983b5ae1f20930f2660afd59c11bbac7e`。同500题、309图片SHA簇，10答案TextVQA共识软评分。
- Native与E各500条经输入、权重、源码、生成协议和首Logits绑定核验后复用；Early/Late各新增500条。
- 全部Prefill（包括P−1）完全Native；所有层Attention/MLP正常，只对真实Decode的Image来源进行Softmax前Hard Mask。
- Native：L0–L31；E：L11–L20；Early：L0–L20；Late：L11–L31。Dense功能性实验，无物理退出或KV优化。
- CPU18项及双卡各3道真实Smoke通过；Native/No-op严格一致、禁止概率为0、原始位置/Cache/Hook检查通过。
- 全部2000条记录的首Logits SHA均与同题Native一致。本轮不进行任何性能测试。

## 质量
| 条件 | Accuracy (%) | ΔNative (pp) | ΔE (pp) |
|---|---:|---:|---:|
| Native | 49.68 | +0.00 | +4.54 |
| E | 45.14 | -4.54 | +0.00 |
| Early | 46.38 | -3.30 | +1.24 |
| Late | 48.50 | -1.18 | +3.36 |

## 配对比较
20,000次Image SHA整簇Bootstrap，保留问题等权点估计；双侧探索性95%CI。
| 比较 | 差值pp [95%CI] | 软分上升/下降 | 满分→较低/较低→满分 |
|---|---|---:|---:|
| Early − E | +1.24 [-0.51, +3.13] | 16/9 | 6/11 |
| Late − E | +3.36 [+1.41, +5.45] | 22/6 | 5/20 |
| Early − Native | -3.30 [-5.10, -1.68] | 2/19 | 17/2 |
| Late − Native | -1.18 [-2.63, +0.25] | 7/16 | 10/5 |
| Late − Early | +2.12 [+0.12, +4.23] | 22/13 | 8/18 |
| E − Native | -4.54 [-6.81, -2.44] | 6/30 | 24/4 |

## 研究结论
- 点估计上，Late补充更有效。Late−Early为+2.12pp，95%CI [+0.12, +4.23]。
- 这些是普通、未经多重比较校正的探索性区间，不作为独立确认性结论；CI跨0也不能解释为等效或无损。
- Late相对E有22题软分提高、6题下降；相对Native仍有16题下降（其中10题从满分降为非满分），不能只报告净收益。
- Late−Native为-1.18pp，95%CI [-2.63, +0.25]。达到47.68%仅表示点估计过门槛，不能证明相差不超过2pp或非劣效。
- Early开放21层，Late开放21层；二者分别补回L0–L10和L21–L31，且共同保留L11–L20。本结论只适用于该固定配置和历史探索集。
- 全部首答案Logits不变；后续真实Decode对自己的生成前缀运行。首Token一致不代表完整答案一致。
- 达到47.68%点估计门槛的条件：Late。本轮仍立即停止，不测速。
- 后续物理方案仅建议：保持完整Native Prefill及P−1，L11–L31持久保留原始Image KV，L0–L10完成各层Prefill后只持久保留文本KV；Decode保持全层文本Attention，原始RoPE位置及因果关系不变。
- 因Native Prefill/P−1仍需32层读图，不能直接沿用原C3进入L20前删除视觉Hidden的路径。此建议是选择性Decode KV压缩，不承诺保留旧C3的Prefill退出收益；先验证数值与生成，再另行研究性能。本轮没有实现或测速。

## 异常与边界
- 各条件EOS、空答案、重复及截断计数见`accuracy_summary.csv`，全部损害及恢复案例见对应CSV。
- 质量图：`quality_recovery.png/pdf`；完整原子检查点和Smoke分别压缩归档；冻结500题清单为`sample_manifest.json.gz`。
- 不新增层组合，不调整Prompt、权重、精度或冻结数值容差；CI跨0不等于无损。
- Native/E/Early/Late的EOS数为499/499/499/498，截断数为1/1/1/2，空答案均为0；重复标记数为1/3/3/2。全部保留，没有删除异常样本。
- 正式双卡质量矩阵墙钟217.83秒；这是任务完成时间，不是推理性能基准。
- GPU Worker均已退出；资源释放与执行日志已保存。

## 复现
```bash
python -m vlm_research.middle10_decode_regions.prepare
# GPU0/1授权原文：确认允许，以后都确认。源码冻结后执行：
python -m vlm_research.middle10_decode_regions.run --freeze --devices 0,1 --authorization-text "确认允许，以后都确认"
python -m vlm_research.middle10_decode_regions.run --driver
python -m vlm_research.middle10_decode_regions.analyze
```
