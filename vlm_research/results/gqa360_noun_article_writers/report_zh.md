# GQA360补测：Noun+Article（N∪D）

**名词＋冠词：303/360，Accuracy=84.167%。** 平均Writer 5.853个，平均R预算18.85%。相对S为-4.444pp，纠正4题、损害20题，配对95%CI [-7.222, -1.944]pp。

样本逐字复用Structural Receiver的同一批360题、六任务各60、360张唯一图片，sample_manifest.csv SHA256与父实验完全一致，不重采样、不按输出选择。只新增Noun+Article = 冻结NOUN/PROPN与依存修饰冠词a/an/the的N∪D；不加Semantic或SPACE/NL，不重跑解析器或改词性规则。原固定版本解析器信息及N、D逐题位置均保存在sample_plan.json，历史字节绑定见history_byte_audit.json。

本批有295道含冻结D位置；其余题N∪D自然等于N。全部360题保留，不能把本轮组合表现解释为冠词单独的贡献。

严格沿用LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager/Transformers4.49.0、原Prompt/图像/评分器、Greedy32与logits_to_keep=1。所有32层P−1/真实Decode禁读Image Key，仅所选R可直接读图；原生QKV/MLP/Residual/完整KV保持不变，每次独立真实KV。没有物理跳过或加速实验。

350条新增ND生成，10条ND历史记录因同题Writer集合、输入、协议完全相同而审计复用；另复用1440条All/S/Q/S+B对照。ND历史相同集合可能具有其他原标签，以实际位置为准；全部标签与SHA见计划。双GPU采集含加载/三道Smoke/ND Smoke墙钟176.3秒。两卡各自HF/No-op全部有效Logits严格torch.equal、历史All/S Token与首Token向量严格复现、实际Mask/独立Cache/32MLP/权重/输入/Hook清理检查通过。350条新增首Token及有效Logits逐步argmax、禁止概率0与非目标Mask精确、投影视觉Hash等于同题历史S+B记录，均复核通过。输出异常0条。

为避免/data空间不足，独立分支使用/data1稀疏Worktree，只读引用原工作区的冻结源码与历史文件。最初完整检出失败发生在任何GPU操作之前，未覆盖或清理其他人的文件。新NPZ原始数值保存在/data1/likai/ATE-resources/experiments/gqa360_noun_article_writers，索引保存SHA256与路径，Git保存JSON检查点及日志；不声称GitHub包含NPZ。

## 同一360题得分

| 条件 | 正确/360 | ACC | 平均R预算 | 来源 |
|---|---:|---:|---:|---|
| All-Writer | 327/360 | 90.833% | 100.00% | 历史复用 |
| Semantic-k | 319/360 | 88.611% | 25.95% | 历史复用 |
| Question-Only | 324/360 | 90.000% | 42.20% | 历史复用 |
| S+SPACE+NL | 314/360 | 87.222% | 32.75% | 历史复用 |
| Noun+Article | 303/360 | 84.167% | 18.85% | 350新增+10精确复用 |

## 配对比较

图片SHA整簇配对Bootstrap20000次，seed20261104，Question等权。下列95%区间为此次补测的探索性、未作多重校正；不作为确认性检验或非劣效证明。六任务完整结果见task_accuracy.csv。Native Full本轮未执行，All-Writer保留答案阶段禁读图协议，不能相互替代。

| ND−对照 | pp | 95%CI | 对照正确→ND错 / 对照错误→ND对 |
|---|---:|---|---:|
| All-Writer | -6.667 | [-10.000, -3.333] | 31/7 |
| Semantic-k | -4.444 | [-7.222, -1.944] | 20/4 |
| Question-Only | -5.833 | [-8.889, -3.056] | 26/5 |
| S+SPACE+NL | -3.056 | [-6.111, +0.000] | 21/10 |

本次只回答N∪D组合在同一360题上的功能表现，不证明名词或冠词各自的独立因果贡献，也不能外推3000题总体ACC。原360题有任务均衡与历史图片排除约束，属于内部探索样本。全部原始答案、Token、N/D/ND位置及正负翻转已保留。GPU释放状态见gpu_release.json；main及历史Structural结果未改。本轮到此停止。
