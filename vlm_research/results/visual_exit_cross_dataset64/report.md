# 跨数据集 Physical Exit 64题 Pilot

固定种子20261029，Phase4冻结500题中各选32张不同SHA256图片；未根据模型结果选题。
同一RTX3090、FP16/eager、Transformers4.49.0、原始Prompt和评分器。
三个条件均Greedy、无KV Cache，每步完整重算，最多32个Token，EOS提前结束。

| 数据集 | 题数 | Full | K16 | K24 |
|---|---:|---:|---:|---:|
| ScienceQA-IMG | 32 | 81.250% | 81.250% | 81.250% |
| TextVQA | 32 | 50.625% | 18.750% | 49.688% |

## 分数变化和翻转

- scienceqa_img K16：+0.00个百分点；原对→错0，原错→对0。
- scienceqa_img K24：+0.00个百分点；原对→错0，原错→对0。
- textvqa K16：-31.875个百分点；软评分上升0，下降12，不变20。
- textvqa K24：-0.938个百分点；软评分上升0，下降1，不变31。

## 最大退化案例

按评分下降排序，同分按Question ID排序，不事后挑选。

### scienceqa_img / K16
- 无评分退化样本。

### scienceqa_img / K24
- 无评分退化样本。

### textvqa / K16
- 34743：Full `Halloween`（1.00） → K16 `4th of july`（0.00）。
- 34821：Full `Lg`（1.00） → K16 `Samsung`（0.00）。
- 34930：Full `Anniversary ale`（1.00） → K16 `Ale`（0.00）。

### textvqa / K24
- 37087：Full `Beginning`（0.30） → K24 `Death`（0.00）。

## 结论及边界

本Pilot以最坏平均分下降衡量，textvqa更敏感；两个任务评分含义不同，仅描述本批样本，不能推断总体敏感性。
正确性Smoke：每个数据集1题，Full无缓存生成与原生有/无缓存生成Token均一致；K32初始Hidden/Logits精确一致；
K16/K24各Smoke生成前缀均通过原Mask参考和预先冻结数值容差；每步实际物理长度、原文本位置与RoPE复用均审计。
未修改模型权重、基础算法、容差、Prompt、评分器；没有训练、Cache算法或额外干预。
每组仅32题，没有等价性检验，不宣称性能等价或通用提升。实际计时含审计与重算，不作为稳定加速基准。

## 复查入口

- 执行代码 Commit：`da32d76b17905fa19e7854c6ee557207e0b53bd7`。
- `manifest.json`：固定样本、图片SHA256、原Prompt映射及源码哈希。
- `per_sample.csv`：192行条件记录，含逐题生成答案、软评分及配对变化。
- `smoke.json`：两个Smoke及各生成前缀的Mask/Physical等价审计。
- `validation.json`：64题完成状态、权重哈希、运行计时229.29秒（含模型加载和Smoke）。
- `cpu_validation.json`：155项CPU回归通过。
- `gpu_console.txt`：实际GPU运行日志；GPU1～3未使用，GPU0完成后已释放。

表格保留三位小数以避免50.625%的半舍入歧义；CSV和JSON保留原始精度。
