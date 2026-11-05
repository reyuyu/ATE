# TextVQA：Decode L11–L24完整答案验证

只新增一个条件、500条完整Greedy答案；原500题全部保留，另取其中先前冻结的199题子集单独汇总，不重复生成、不混用分母。

## 固定协议
- Native Prefill/P−1在全部32层正常；只限制真实增量Decode的原始Image Key访问，开放L11–L24（含两端，共14层）。
- 所有Decode文本Attention、MLP、Residual、RoPE和独立真实KV保持原生；不是Attention输出置零，不做物理退出。
- LLaVA-1.5-7B固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager、Transformers4.49.0、H1 last-row、原Prompt、Greedy max32、自然EOS。
- 官方十答案TextVQA共识软评分。Native/E/Late的1500条历史记录经输入、权重、源码及首Logits绑定复用。
- 14项CPU测试、GPU0/1各3题Smoke通过；Native/No-op严格一致，Mask实际概率为0，独立Dense参考、KV及Hook验证通过。

## 得分：分别列出两种样本范围
| 条件 | 原500题 | 其中的原199题子集 |
|---|---:|---:|
| Native | 49.68% | 52.31% |
| E | 45.14% | 48.14% |
| Late | 48.50% | 50.25% |
| Decode-L11-L24 | 48.10% | 49.80% |

## 同题配对变化
按Image SHA整簇20,000次配对Bootstrap，固定seed 20261106；点估计问题等权。普通95%CI仅作探索描述，不宣称无损或非劣效。
| 范围 | L11–L24相对 | ΔAccuracy pp | 95%CI | 软分上升/下降题数 |
|---|---|---:|---|---:|
| full500 | Native | -1.58 | [-3.28, +0.02] | 8/16 |
| full500 | E | +2.96 | [+1.47, +4.64] | 17/1 |
| full500 | Late | -0.40 | [-1.60, +0.72] | 6/6 |
| previous199 | Native | -2.51 | [-5.27, +0.05] | 4/9 |
| previous199 | E | +1.66 | [-0.20, +3.90] | 4/1 |
| previous199 | Late | -0.45 | [-2.77, +1.63] | 3/3 |

- `per_question.csv`保存全部2000条新旧答案与Token；`harm_recovery_cases.csv`保留全部逐题增减。
- `accuracy_summary.csv`记录EOS、空答案、重复与截断，不剔除失败或负面样本。
- 没有性能测试、额外层组合、物理优化或训练。GPU0/1 Worker已退出并释放资源，本轮到此停止。

## 复现
```bash
python -m vlm_research.textvqa_decode_l11_l24.run --prepare
python -m vlm_research.textvqa_decode_l11_l24.run --freeze
python -m vlm_research.textvqa_decode_l11_l24.run --driver
python -m vlm_research.textvqa_decode_l11_l24.analyze
```
