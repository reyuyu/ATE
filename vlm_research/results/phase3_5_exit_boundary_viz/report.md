# Phase 3.5 — Exit Boundary vs ROI Attention Alignment

## 结论

现有缓存不支持一个跨Query、跨固定K的统一“退出收益越高，恰好前层ROI聚焦、下一层ROI失焦并变得分散”的规律。K16中ROI下降常见，但Gain与Delta_ROI相关方向为正；K24中正Gain图通常ROI反而上升，所有图的条件熵都上升，无法区分谁更受益。事后最佳K的ROI负相关是探索性现象，且同时伴随熵变化与Gain负相关，并不是“越受益越分散”。五历史案例仅作说明，未加入48图主统计。不将任何不一致强行解释为Attention Sink，也不作因果或新方法主张。

## 数据、缓存与层边界

只读Phase3.2的240图×8K分数/Manifest和Phase3.4的53图Full Attention、ROI/Sink清单、逐图层表及主层汇总。按question_id、img_id、问题、标注、候选和图像SHA256配对；48张原主图left/right各24，另有5历史案例，绝无重新采样。全部53张所需NPZ均存在，原始文件哈希与Phase3.4交付清单一致。重新从所选Attention行计算的指标与历史CSV最大差1.7764e−15；主48层汇总也一致。没有缺失项、网络访问、Torch/Transformers导入、GPU访问或模型Forward。

Physical Exit K是进入L_K前删除：ROI_before=Full的L(K−1)，ROI_after=Full的L_K。K24对应L23→L24；L_K之后Full与Physical是不同路径，此处只是用Full轨迹与退出结果作相关对照。组合图共用坐标x，层L_l位于x=l，边界K位于x=K−0.5；顶端标真实L0–L31，底端标K。K0和K32缺少两侧真实Full层，不计算边界差；完整8K分数保留在附表。

每个Head先在576视觉Patch内归一化，再求ROI Mass/(N_ROI/576)和H/log(576)，最后对32Head等权平均。Question是原缓存的真实问题Token行均值，Answer为P−1预测位置。Visual Mass保留原始量纲。辅助Sink Share只使用冻结非ROI候选，没有重新定义Sink或挑Head。空间图展示条件化Head均值地图（其非线性指标不等于Head指标均值），每个before/after对共用色标、标注原始质量，使用原已验证Resize/裁剪几何。

## 固定边界：全部48图

Gain=Physical Margin−Full Margin，严格Margin>0正确、0平局判错。主48在Full、K16、K24都正确33/48（left24/24、right9/24），无任何分类翻转。K16有37/48正Gain（36条>0.01nats），K24有33/48正Gain（均>0.01）；不能将这些Margin变化称为准确率提高。

| K | Query | ROI下降 | 熵上升 | 两者同时 | 正Gain中的ROI下降 | 正Gain中的两者同时 |
|---|---|---:|---:|---:|---:|---:|
| 16 | question | 40/48 | 42/48 | 35/48 | 31/37 | 27/37 |
| 16 | answer | 35/48 | 8/48 | 6/48 | 27/37 | 5/37 |
| 24 | question | 7/48 | 48/48 | 7/48 | 4/33 | 4/33 |
| 24 | answer | 3/48 | 48/48 | 3/48 | 2/33 | 2/33 |

若再要求前层ROI富集>1且下一层下降，正Gain子集K16的Q/A为29/37、15/37，K24两者均仅2/33；ROI下降的宽松描述不能自动升级为“从高度聚焦到失焦”。

K16的ROI下降在不受益组也很常见：Q为9/11、A为8/11；受益组分别31/37、27/37，不能因下降计数多就认为其特异对应收益。K24正Gain33图中Q/A只有4/2张ROI下降，而没有正Gain的15图分别3/1张；Q/A均无一张从ROI富集>1降到≤1。K16 Q/A从>1跨到≤1为14/10图，但同样不保证退出收益。

图级5000次Bootstrap，seed20261031，配对重采样整图；Spearman使用平均并列排名，常量重采样的未定义相关单独计数。以下为名义95%百分位区间，未多重比较校正，不作显著性/机制确认。

| K | Query | Pearson(Delta_ROI, Gain) [95%CI] | Spearman [95%CI] | Pearson(Delta_Entropy, Gain) [95%CI] |
|---|---|---|---|---|
| 16 | question | +0.212 [-0.100, +0.447] | +0.075 [-0.236, +0.378] | +0.131 [-0.181, +0.416] |
| 16 | answer | +0.323 [+0.018, +0.525] | +0.231 [-0.070, +0.506] | +0.329 [+0.095, +0.534] |
| 24 | question | -0.130 [-0.356, +0.147] | -0.094 [-0.375, +0.192] | +0.196 [-0.082, +0.453] |
| 24 | answer | -0.251 [-0.427, -0.029] | -0.208 [-0.491, +0.098] | +0.240 [-0.063, +0.503] |

K16 ROI相关为正，K24为负，方向随边界变化；四组Spearman区间均包含0。K24 Answer Pearson有负的名义区间，但绝大多数该组ROI变化本身仍为正，因此应理解为“ROI上升较少者倾向更多Gain”的弱描述，而非多数图ROI失焦或已证明机制。K24全部48图熵上升，属于共同层转换，不能单独解释Gain差异。

### Gain分组的Delta_ROI分布

| K | Query | Gain>0均值/中位数 | Gain≤0均值/中位数 | 均值差(正−非正)及95%CI |
|---|---|---|---|---|
| 16 | question | -0.2534 / -0.2056 | -0.6416 / -0.1817 | +0.3882 [-0.0955, +0.9455] |
| 16 | answer | -0.1581 / -0.0650 | -0.4167 / -0.2768 | +0.2586 [-0.0572, +0.6978] |
| 24 | question | +0.2035 / +0.1689 | +0.3260 / +0.1964 | -0.1225 [-0.3735, +0.1111] |
| 24 | answer | +0.2686 / +0.2060 | +0.5345 / +0.3804 | -0.2659 [-0.6464, +0.0244] |

全部极端点保留。四个分组均值差区间均跨0，中位数差及其Bootstrap区间、IQR保留在fixed_boundary_summary.csv；不要用均值差替代整个分布的结论。

## 五个独立历史案例

下表为Full→Physical K24的Margin及Full L23→L24的变化，不将案例混入主估计。

| ID | Full→K24 Margin | Q Delta_ROI / Delta_H | A Delta_ROI / Delta_H | 全8K最佳（含并列） |
|---|---|---|---|---|
| 04132006 | -0.0078125→+0.0312500 | -0.1226 / +0.0498 | +0.0075 / +0.0499 | 24, 28 |
| 15133706 | -0.0312500→+0.0390625 | +0.1092 / +0.0210 | +0.1892 / +0.0348 | 24 |
| 0216546 | +0.0703125→+0.0625000 | +0.0208 / +0.0558 | -0.0010 / +0.0393 | 32 |
| 02310173 | +0.0859375→+0.0859375 | +0.3008 / +0.0440 | +0.2663 / +0.0782 | 20, 24, 32 |
| 05480434 | +0.1406250→+0.1484375 | +0.0605 / +0.0321 | +0.1205 / +0.0381 | 24 |

- 04132006：K24恢复，Q ROI下降但A ROI上升；且L23 Q/A富集仅0.775/0.667，本来均低于均匀基线，不能称为“前一层高度ROI聚焦”。K24和K28并列最佳。
- 15133706：K24恢复，但Q和A的ROI都上升、熵也都上升，直接反驳将恢复要求为ROI下降的说法；L23富集0.598/0.535，同样不高。
- 0216546：K16翻错时Q ROI微增、A ROI下降且熵下降；全网格最佳是K32 Full，不是K24。其ROI510Patch，面积/裁剪限制与主队列不同。
- 02310173：K16翻错时Q/A ROI均下降，熵也均下降（Q−0.0193/A−0.0536）；ROI转移不等于分散。全网格K20/24/32并列最佳。
- 05480434：K16 Q ROI下降且熵上升，却Gain=−0.4140625并翻错，是预期模式但性能更差的反例；K24 Q/A ROI均上升，Gain仅+.0078125且仍原本正确。

本轮没有新FP32主干复验，也没有把FP16近零Margin分类当成稳健能力结论。Full→Exit跨0才计翻转；“最佳”是同一数据上已测试稀疏网格的观测最优。

## 探索性最佳K（与固定K分开）

只从[12,16,20,24,28]选最大Margin，绝对差≤1e−10视为浮点并列，取最小K并保存全部并列；未改严格Margin>0判定。K0/K8不作主最优候选，避免其严重left先验，K32也不参与此受限选择。选中数量：K12=9、K16=21、K20=8、K24=5、K28=5。两图13515212/00784767在K24/K28并列；另存对所有并列边界先按图平均的敏感性结果。

| Query | ROI下降 | 熵上升 | 两者同时 | Pearson ROI–Gain | Spearman ROI–Gain | Pearson熵–Gain |
|---|---:|---:|---:|---:|---:|---:|
| question | 29/48 | 37/48 | 22/48 | -0.372 | -0.499 | -0.588 |
| answer | 22/48 | 32/48 | 10/48 | -0.255 | -0.335 | -0.386 |

最佳点41/48正Gain，但主48仍无分类改变。ROI确有负相关；同时熵–Gain也为负，与“越受益越分散”不一致。不能把上述相关当独立确认：K_best和Gain均由同一Margin网格挑出，而且混合不同层转换，存在选择和层深混杂。并列平均敏感性方向相似，但不能消除事后选择偏差。所有相关区间在best_observed_summary.csv和best_tie_sensitivity_summary.csv。

## 低视觉质量与数值敏感性

在读取结果前固定：任一端点Head均值Visual Mass<1e−3，或任一端点任一Head质量<1e−4，即标记低质量风险，主统计不剔除。固定K16/K24所有主图均未触发；最佳K的Answer两图15394126、07329994（均K12）各有少数Head低于1e−4，端点层均值仍约0.029–0.050，不是全层视觉质量接近零。

另按固定1e−4/1e−3/1e−2阈值，仅保留before/after都合格的同一组Head重新计算，并按图等权，保存所有结果和空Head数。固定K16 ROI–Gain Pearson Q约+.212～+.215、A+.315～+.325；K24 Q−.130～−.127、A−.251～−.242，方向保持。较严格.01阈值下ROI下降计数可变（K16 Answer35→29，K24 Answer3→5），不能把细小符号视为稳定机制。最佳K Answer排除两标记图后n46，ROI Pearson−.256；限制到≥.01的同组Head后n48约−.157，强度减弱。

Gain>0主计数另与>1e−10、>.01nats对照；K16 37/37/36，K24 33/33/33。对K24的3个近零Gain不作改善解释。低质量筛选、并列处理或有限精度都不能把本轮变成因果证明。

## 图表索引（全部18张）

| 历史案例 | 32层ROI与8K Margin对齐 | 局部L12–18/L20–27 | K16/K24空间前后对照 |
|---|---|---|---|
| 04132006 | [层曲线与Margin](case_04132006_alignment.png) | [局部放大](case_04132006_local_boundaries.png) | [空间ROI Overlay](case_04132006_spatial.png) |
| 15133706 | [层曲线与Margin](case_15133706_alignment.png) | [局部放大](case_15133706_local_boundaries.png) | [空间ROI Overlay](case_15133706_spatial.png) |
| 0216546 | [层曲线与Margin](case_0216546_alignment.png) | [局部放大](case_0216546_local_boundaries.png) | [空间ROI Overlay](case_0216546_spatial.png) |
| 02310173 | [层曲线与Margin](case_02310173_alignment.png) | [局部放大](case_02310173_local_boundaries.png) | [空间ROI Overlay](case_02310173_spatial.png) |
| 05480434 | [层曲线与Margin](case_05480434_alignment.png) | [局部放大](case_05480434_local_boundaries.png) | [空间ROI Overlay](case_05480434_spatial.png) |

- [48图固定K16/K24散点](main48_fixed_boundary_scatter.png)
- [Gain分组Delta_ROI完整分布](main48_delta_roi_by_gain.png)
- [探索性最佳K分布及边界计数](main48_best_observed_k.png)

## 复现与交付

入口exit_boundary_analysis.py仅使用标准库、NumPy、Matplotlib、Pillow；不导入torch/transformers。9项数值单元测试通过，包含Head非线性聚合、稀疏网格并列、负相关Bootstrap已知值、ROI下降而熵下降的反例及层索引。主计算最终12.493秒（不含补充图排版/文档），未调用GPU。

```bash
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' python -m unittest test_exit_boundary_analysis -v
# 首次运行只对不存在的新目录；已存在会拒绝覆盖
CUDA_VISIBLE_DEVICES='' python exit_boundary_analysis.py
CUDA_VISIBLE_DEVICES='' python exit_boundary_analysis.py --verify-existing
```

boundary_changes.csv为53图×6有效内部K×Q/A共636行，含192行固定主边界；all_eight_k_scores.csv为53图×8K共424行。固定边界/最佳K/低质量敏感性各自CSV，summary.json、analysis_protocol.json、validation.json和checksums.json保存全部口径、数值与输入/输出哈希。analysis_audit.json另核验行键、图数量、分母和历史结果未改。图库与gallery.md列出所有图。旧Phase1–3.4结果未修改。到此停止，不补跑Forward、不开始其他阶段。
