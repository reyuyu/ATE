# TextVQA Decode深层视觉回看：单层Add/Drop

完成：199题、124个图片SHA簇，22个新条件，共4378条新增完整答案；复用597条Native/E/Late答案。
基线Commit：`5627bc822a7827ae11edfaec84c31f6b73398e25`；执行源码Commit见`execution_freeze.json`。

## 样本与协议
- 固定seed 20261106，按sha256(seed|Image SHA)排序原500题的图片簇，取题数最接近200的完整簇前缀；不使用模型输出选样。完整选择顺序见`selection.json`。
- 本子集Native=52.31%，E=48.14%，Late=50.25%；不能混用原500题总体分数。
- 固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager，Transformers4.49.0，原Prompt、官方十答案共识软评分、H1 last-row LM Head。
- 全部Prefill及P−1保持Native。Add-Li仅在E的L11–L20上增加Li；Drop-Li仅从Late的L11–L31删除Li。只改变真实Decode的Image来源Hard Mask，所有文本Attention/MLP正常。
- 每条件独立Cache、真实Greedy生成至EOS或32 Token。CPU16项及每卡3题×全部22条件GPU Smoke通过；每条首Logits SHA与同题Native一致。

## 逐层结果
下表CI均为按Image SHA整簇配对、20,000次Bootstrap的普通探索性95%CI；点估计保持问题等权。Drop负值表示删除该层后损害，正值表示删除后反而提高。
| 层 | Add Accuracy | Add−E pp [95%CI] | Drop Accuracy | Drop−Late pp [95%CI] |
|---:|---:|---|---:|---|
| 21 | 48.64% | +0.50 [+0.00, +1.55] | 50.15% | -0.10 [-2.01, +1.76] |
| 22 | 49.65% | +1.51 [+0.00, +3.48] | 50.40% | +0.15 [-1.39, +1.76] |
| 23 | 48.64% | +0.50 [+0.00, +1.55] | 50.00% | -0.25 [-1.82, +1.24] |
| 24 | 49.85% | +1.71 [-0.10, +3.96] | 50.65% | +0.40 [-0.36, +1.60] |
| 25 | 48.64% | +0.50 [-1.04, +2.16] | 50.15% | -0.10 [-1.59, +1.37] |
| 26 | 48.64% | +0.50 [+0.00, +1.55] | 49.95% | -0.30 [-0.93, +0.00] |
| 27 | 48.64% | +0.50 [+0.00, +1.55] | 49.75% | -0.50 [-1.55, +0.00] |
| 28 | 48.64% | +0.50 [+0.00, +1.55] | 49.75% | -0.50 [-1.55, +0.00] |
| 29 | 49.15% | +1.01 [+0.00, +2.54] | 50.30% | +0.05 [+0.00, +0.15] |
| 30 | 48.64% | +0.50 [+0.00, +1.55] | 50.25% | +0.00 [-1.49, +1.52] |
| 31 | 48.64% | +0.50 [+0.00, +1.56] | 49.80% | -0.45 [-1.88, +0.69] |

## 探索性排序
- Add点估计最高的三层：L24 (+1.71pp), L22 (+1.51pp), L29 (+1.01pp)。
- Drop损害点估计最大的三层（Late−Drop）：L27 (+0.50pp), L28 (+0.50pp), L31 (+0.45pp)。
- 全部22项自身基线比较的普通95%CI均包含0（包括端点为0）；目前没有明确的逐层确认性证据。
- Drop-L27与Drop-L28的损害来自同一道题：textvqa__39346（Late回答420，删除后回答42，软评分1→0）。不能把相同案例的重复翻转当作独立证据。
- L24在E背景下增加读取时提高1.71pp，但在Late背景下删除时也提高0.40pp；这一不对称不能用可加的层贡献解释，也不足以证明交互机制。
- Add是在E背景下测试单层增量，Drop是在Late背景下测试单层必要性；两者不要求相同排序。单层低损害不代表多层可联合删除，不能将这些结果当作层贡献可加。
- 这是历史探索队列的哈希子集和22项扫描，区间未经多重比较校正；不作统计确认、无损或非劣效声明。平坦曲线或CI跨0都不是无功能的证明。

## 案例、异常与审计
- `per_question.csv`保存全部答案、Token IDs、得分、对自身基线及Native的逐题变化；`harm_recovery_cases.csv`及正负案例CSV保留全部损害和恢复，不只展示净收益。
- `accuracy_summary.csv`保留每条件EOS、空答案、截断和重复计数；不静默跳样或剔除异常。
- 首Token由未改动的Native Prefill预测，故首Token一致是正确性要求而不是质量指标；Accuracy来自完整自由生成。
- GPU0/1按图片SHA分片；同图全部条件同卡。源码、输入、权重、协议、逐条原子检查点哈希及原始Smoke归档保留。
- 没有性能测试、物理退出、KV优化、额外层组合或训练。Worker已退出并释放GPU，本轮到此停止。

正式质量采集墙钟16.61分钟；这是实验作业耗时，不是推理性能测量。

## 图与复现
- `add_drop_delta.png/pdf`：Add−E、Drop−Late双曲线及配对CI。
- `add_drop_accuracy.png/pdf`：绝对Accuracy双曲线及同子集Native/E/Late参考线。
```bash
python -m vlm_research.textvqa_deep_add_drop.prepare
python -m vlm_research.textvqa_deep_add_drop.run --freeze
python -m vlm_research.textvqa_deep_add_drop.run --driver
python -m vlm_research.textvqa_deep_add_drop.analyze
```
