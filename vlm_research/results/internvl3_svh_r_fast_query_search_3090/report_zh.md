# InternVL3-SVH-R 快速中层回看探索

状态：COMPLETE_EXPLORATORY；EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。本轮不宣称数值等价。

**没有找到可在独立Holdout重复改善Query的配置。** Discovery挑出的两个候选在Holdout均低于新R0，不能据此认定某个最少回看层数有效，也不再扩大搜索。

Discovery 冻结质量候选 **R10**（K=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]），低成本候选 **R4-M**（K=[12, 13, 14, 15]）。
独立 Holdout Query：Native 43.269%，R0 29.808%，R10 26.923%，R4-M 27.885%。最佳质量相对R0：-2.885pp，95%CI [-7.478, +0.990]pp。
是否有重复改善应依据该独立配对差值和区间判断，不能用Discovery挑选结果代替Holdout验证。

## 1. 阶段A：固定64题全部七候选

| method | group | N | correct | accuracy_pct | mean_KV_capacity_MiB | anomalies |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | all | 64 | 32.000 | 50.000 | 3.207 | 0 |
| R0 | query | 48 | 18.000 | 37.500 | 3.170 | 0 |
| R0 | nonquery | 16 | 14.000 | 87.500 | 3.319 | 0 |
| R2 | all | 64 | 32.000 | 50.000 | 4.206 | 0 |
| R2 | query | 48 | 17.000 | 35.417 | 4.168 | 0 |
| R2 | nonquery | 16 | 15.000 | 93.750 | 4.319 | 0 |
| R4-M | all | 64 | 32.000 | 50.000 | 5.208 | 0 |
| R4-M | query | 48 | 18.000 | 37.500 | 5.172 | 0 |
| R4-M | nonquery | 16 | 14.000 | 87.500 | 5.316 | 0 |
| R4-L | all | 64 | 33.000 | 51.562 | 5.209 | 0 |
| R4-L | query | 48 | 18.000 | 37.500 | 5.172 | 0 |
| R4-L | nonquery | 16 | 15.000 | 93.750 | 5.319 | 0 |
| R6 | all | 64 | 33.000 | 51.562 | 6.207 | 0 |
| R6 | query | 48 | 18.000 | 37.500 | 6.171 | 0 |
| R6 | nonquery | 16 | 15.000 | 93.750 | 6.316 | 0 |
| R8 | all | 64 | 32.000 | 50.000 | 7.207 | 0 |
| R8 | query | 48 | 17.000 | 35.417 | 7.170 | 0 |
| R8 | nonquery | 16 | 15.000 | 93.750 | 7.316 | 0 |
| R10 | all | 64 | 35.000 | 54.688 | 8.207 | 0 |
| R10 | query | 48 | 20.000 | 41.667 | 8.170 | 0 |
| R10 | nonquery | 16 | 15.000 | 93.750 | 8.316 | 0 |
| Native | all | 64 | 39.000 | 60.938 | 17.744 | 0 |
| Native | query | 48 | 25.000 | 52.083 | 17.702 | 0 |
| Native | nonquery | 16 | 14.000 | 87.500 | 17.869 | 0 |

筛选规则在任何GPU生成前冻结：Query提升减去Non-Query损失、超额异常率和实际KV容量惩罚。全六个回看候选排名保存在selection_stage_A.json，只有前三个和R0补齐剩余56题。4层中心与后段只在共同StageA64上作首要同预算比较，不能把不同样本量误当同样本比较。

## 2. Discovery120

| method | group | N | correct | accuracy_pct | mean_KV_capacity_MiB | anomalies |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | all | 120 | 63.000 | 52.500 | 3.216 | 0 |
| R0 | query | 80 | 30.000 | 37.500 | 3.182 | 0 |
| R0 | nonquery | 40 | 33.000 | 82.500 | 3.282 | 0 |
| R4-M | all | 120 | 65.000 | 54.167 | 5.216 | 0 |
| R4-M | query | 80 | 31.000 | 38.750 | 5.182 | 0 |
| R4-M | nonquery | 40 | 34.000 | 85.000 | 5.282 | 0 |
| R4-L | all | 120 | 63.000 | 52.500 | 5.216 | 0 |
| R4-L | query | 80 | 29.000 | 36.250 | 5.183 | 0 |
| R4-L | nonquery | 40 | 34.000 | 85.000 | 5.282 | 0 |
| R10 | all | 120 | 67.000 | 55.833 | 8.214 | 0 |
| R10 | query | 80 | 33.000 | 41.250 | 8.181 | 0 |
| R10 | nonquery | 40 | 34.000 | 85.000 | 8.281 | 0 |
| Native | all | 120 | 76.000 | 63.333 | 17.754 | 0 |
| Native | query | 80 | 42.000 | 52.500 | 17.717 | 0 |
| Native | nonquery | 40 | 34.000 | 85.000 | 17.828 | 0 |

best_quality按Query最高选取；best_efficiency允许距最高Query 2.5pp且Non-Query不低于R0 2.5pp，再优先少层、低实测KV。规则细节见protocol.json，不根据Holdout重新选层。候选选择文件SHA256：a582d9bc4c2b9d518c1bea785dbfb8d4b20c0d8cf185081cef7048a9695c23b5。

## 3. 独立Holdout160

| method | group | N | correct | accuracy_pct | mean_KV_capacity_MiB | anomalies |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | all | 160 | 79.000 | 49.375 | 3.215 | 2 |
| R0 | query | 104 | 31.000 | 29.808 | 3.167 | 2 |
| R0 | nonquery | 56 | 48.000 | 85.714 | 3.304 | 0 |
| R10 | all | 160 | 75.000 | 46.875 | 8.211 | 0 |
| R10 | query | 104 | 28.000 | 26.923 | 8.161 | 0 |
| R10 | nonquery | 56 | 47.000 | 83.929 | 8.303 | 0 |
| R4-M | all | 160 | 77.000 | 48.125 | 5.215 | 2 |
| R4-M | query | 104 | 29.000 | 27.885 | 5.166 | 2 |
| R4-M | nonquery | 56 | 48.000 | 85.714 | 5.306 | 0 |
| Native | all | 160 | 93.000 | 58.125 | 17.752 | 0 |
| Native | query | 104 | 45.000 | 43.269 | 17.698 | 0 |
| Native | nonquery | 56 | 48.000 | 85.714 | 17.853 | 0 |

| candidate | group | N | difference_pp | CI95_low_pp | CI95_high_pp | wrong_to_right | right_to_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R10 | all | 160 | -2.500 | -5.696 | 0.000 | 1 | 5 |
| R10 | query | 104 | -2.885 | -7.478 | 0.990 | 1 | 4 |
| R10 | nonquery | 56 | -1.786 | -5.357 | 0.000 | 0 | 1 |
| R4-M | all | 160 | -1.250 | -3.846 | 1.170 | 1 | 3 |
| R4-M | query | 104 | -1.923 | -5.941 | 1.905 | 1 | 3 |
| R4-M | nonquery | 56 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| Native | all | 160 | 8.750 | 1.875 | 15.823 | 23 | 9 |
| Native | query | 104 | 13.462 | 2.913 | 23.423 | 23 | 9 |
| Native | nonquery | 56 | 0.000 | 0.000 | 0.000 | 0 | 0 |

Image SHA整簇配对Bootstrap 5000次，普通未校正95%CI；搜索、多组和子组比较均为探索性。区间跨0不证明等价；小样本不能证明全量GQA非劣效。Query各Object/Attribute/Color/Relation/Spatial等实际类别及Non-Query四类全部见query_subtypes.csv，全部新增错误见harm_cases.csv，恢复与异常见recovery_and_anomaly_cases.csv。没有数量子类时不虚构数量题。

## 4. 同机真实固定32 Token速度

| method | requests | E2E_ms | TTFT_ms | TPOT_ms | KV_live_MiB | KV_capacity_MiB | peak_allocated_MiB | peak_reserved_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 36 | 1115.8514687776915 | 140.64498383231694 | 31.4582737079153 | 19.404947916666668 | 19.404947916666668 | 15286.632242838541 | 15444.0 |
| R0 | 36 | 1082.5370513331209 | 120.78634111013444 | 31.024216458806013 | 4.6328125 | 4.6328125 | 15286.633138020834 | 15444.0 |
| R4-M | 36 | 1072.2275780822706 | 120.8691128049395 | 30.688982750881653 | 6.6328125 | 6.6328125 | 15286.633138020834 | 15444.0 |

12张预先冻结不同图片，每方法2次预热、每图3次正式重复，轮换顺序。包含真实Vision、Projector、Prefill、31次Decode及方法必要清理；同步墙钟，排除磁盘/Tokenizer/预处理/H2D和结果写盘。KV均由真实28层K/V张量与底层存储字节重算。质量正常EOS与测速固定32严格分开；质量EOS平均KV不能冒充固定32KV。GPU为GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e。不混入旧实例时间。零值Key/Value临时scratch参与实际计算，开销已计时；不宣称完全消除了被屏蔽Key的计算。

| 方法 | 相对Native E2E加速倍数 | 相对R0增加KV@32 MiB | 相对Native KV下降 |
|---|---:|---:|---:|
| Native | 1.000× | +14.772 | 0.00% |
| R0 | 1.031× | +0.000 | 76.13% |
| R4-M | 1.041× | +2.000 | 65.82% |

## 4b. 相同4层预算及Query子类

| base | candidate | group | difference_pp | CI95_low_pp | CI95_high_pp |
| --- | --- | --- | --- | --- | --- |
| R4-M | R4-L | all | -1.667 | -5.833 | 2.459 |
| R4-M | R4-L | query | -2.500 | -7.595 | 2.500 |
| R4-M | R4-L | nonquery | 0.000 | -7.500 | 7.500 |

| method | value | N | correct | accuracy_pct |
| --- | --- | --- | --- | --- |
| R0 | attribute | 10 | 4.000 | 40.000 |
| R0 | color | 17 | 1.000 | 5.882 |
| R0 | object | 19 | 5.000 | 26.316 |
| R0 | other | 16 | 3.000 | 18.750 |
| R0 | relation | 28 | 8.000 | 28.571 |
| R0 | spatial | 14 | 10.000 | 71.429 |
| R10 | attribute | 10 | 3.000 | 30.000 |
| R10 | color | 17 | 1.000 | 5.882 |
| R10 | object | 19 | 4.000 | 21.053 |
| R10 | other | 16 | 2.000 | 12.500 |
| R10 | relation | 28 | 8.000 | 28.571 |
| R10 | spatial | 14 | 10.000 | 71.429 |
| R4-M | attribute | 10 | 3.000 | 30.000 |
| R4-M | color | 17 | 1.000 | 5.882 |
| R4-M | object | 19 | 4.000 | 21.053 |
| R4-M | other | 16 | 3.000 | 18.750 |
| R4-M | relation | 28 | 8.000 | 28.571 |
| R4-M | spatial | 14 | 10.000 | 71.429 |
| Native | attribute | 10 | 4.000 | 40.000 |
| Native | color | 17 | 4.000 | 23.529 |
| Native | object | 19 | 8.000 | 42.105 |
| Native | other | 16 | 5.000 | 31.250 |
| Native | relation | 28 | 13.000 | 46.429 |
| Native | spatial | 14 | 11.000 | 78.571 |

4层中心与4层后段在Discovery上的差异只表示本样本观察；后段未进入最终Holdout，不能宣称中层位置普遍更优。Holdout恢复和损害是逐题配对统计，不以净平均掩盖Non-Query或Query类别中的错误。

## 5. Smoke与数值豁免

Native/No-op有效Logits严格torch.equal通过；8题新Native完整Token和历史记录一致，历史Native逐题图片SHA、Prompt、Token IDs、BF16像素Tensor、Gold、模型Revision、原生生成协议与源码绑定全部通过，故质量参考合法复用，新增Native质量生成0次。新R0全部重新生成，不用历史Physical-D顶替。

只对R0/R4-L/R8进行8题对应Dense-R/Physical-R对照；短/长输入做完整结构审计，其余仅轻量有效Logits记录。L20删除256视觉Hidden、L2–L5答案Attention真实不调用QKV/QK/AV/o_proj、所选层保留真实视觉KV、其他层无持久原始视觉KV、位置RoPE、禁用概率、非目标连接和Norm/MLP/Residual由冻结V2结构审计及逐题KV核对验证。缓存独立、Hook恢复，无NaN/Inf。Smoke状态：PASS_WITH_NUMERICAL_WAIVER。

- R0：{'answer_agreement': 8, 'N': 8, 'severe_outputs': 0}
- R4-L：{'answer_agreement': 8, 'N': 8, 'severe_outputs': 0}
- R8：{'answer_agreement': 8, 'N': 8, 'severe_outputs': 0}
- R0：相同生成前缀Logits MaxAbs最大2.062500，步均MeanAbs 0.250679，Top1一致20/20。前缀分歧后的Logits不作同条件数值比较。
- R4-L：相同生成前缀Logits MaxAbs最大2.000000，步均MeanAbs 0.251105，Top1一致23/23。前缀分歧后的Logits不作同条件数值比较。
- R8：相同生成前缀Logits MaxAbs最大2.984375，步均MeanAbs 0.316805，Top1一致22/22。前缀分歧后的Logits不作同条件数值比较。

历史9602和2d671的NUMERICAL_EQUIVALENCE_FAILED及C1/C2失败证据原封保留。本轮使用9602冻结V2原引擎，不导入C1/C2补零投影修复，不放宽旧协议后宣称PASS，而是用户在GPU前授权新探索性质量协议。Dense/Physical存在BF16形状相关数值差异，输出/质量变化不能全部归因于回看机制；这限制机制归因和跨实现可迁移性。本轮只有结构门槛PASS_WITH_NUMERICAL_WAIVER，绝非数值等价。

## 6. 完整性与资源

正式新增质量1152次，Native历史复用280题，固定32正式测速108请求，模型加载1次。全部异常计入评分，没有删题、换Gold、扩上限、重新挑层或运行其他Benchmark。独立官方/标量评分逐条相等；输入、协议、源码与检查点SHA绑定核验通过。逐题原始答案/Token IDs、实际KV、异常和进度保存在SSD与持久工作盘server_only。

服务器完整结果：/home/featurize/work/ATE-internvl3-svh-r-fast-query-search-3090/vlm_research/results/internvl3_svh_r_fast_query_search_3090/
SSD运行目录：/home/featurize/ate-runs/internvl3-svh-r-fast-query-search-3090
Git分支：codex/internvl3-svh-r-fast-query-search-3090
GPU释放核验：True；未关闭服务器。不制作Windows备份。

收尾工程事件：所有生成、测速及汇总已完成、模型已释放后，原run.py最后一条进度日志在SQLite关闭后查询而报错。原failure.json与完整日志保留，post_run_bookkeeping_incident.json记录CPU核验；未重跑任何生成。run_safe.py仅修正关闭顺序供后续复现，未参与本轮测量；原始运行源码SHA保持不变。
