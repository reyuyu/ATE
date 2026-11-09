# InternVL3-8B 追加关闭 L20–L27：3000题配对验证

追加请求在原 L17–L27 结果已知后提出，属于探索性分析；未搜索其他边界。原93,000条记录、协议和原图保持不变。
只新增3000次完整生成，复用同一3090上的All-Writer及Keep-L0-L16。原P−1和全部Decode仍在全部28层禁止访问Image Key；本轮只关闭R在L20–L27的写入权限，没有物理删除。

| 条件 | 正确/3000 | Accuracy % | 相对All-Writer损失 pp | 普通配对95%CI pp |
|---|---:|---:|---:|---|
| All-Writer | 2710 | 90.3333 | +0.0000 | [+0.0000, +0.0000] |
| Keep-L0-L16 | 2654 | 88.4667 | +1.8667 | [+1.1807, +2.5848] |
| Keep-L0-L19 | 2700 | 90.0000 | +0.3333 | [-0.1315, +0.7947] |

相对旧L17–L27 OFF，新窗口Accuracy差为 +1.5333 pp，95%CI [+0.8943, +2.1898] pp。
普通95%CI基于20,000次Image SHA整簇、同题配对、题数加权Bootstrap；新增比较未作多重校正，不能按确认性检验解读。CI跨0不能证明等价或无损。

| Task | All-Writer % | L17–L27 OFF % | L20–L27 OFF % |
|---|---:|---:|---:|
| ChooseAttr | 85.00 | 85.20 | 84.00 |
| ChooseCat | 89.80 | 89.20 | 90.40 |
| ChooseRel | 90.00 | 89.00 | 90.40 |
| CompareAttr | 88.80 | 84.80 | 86.60 |
| LogicalObj | 96.60 | 95.80 | 96.60 |
| QueryAttr | 91.80 | 86.80 | 92.00 |

最短/典型/最长输入官方Chat与Native/No-op全部有效Logits严格torch.equal；重载后的All-Writer与旧记录全部Token及Logits哈希严格一致。新增全部3000题的输入绑定完全一致，被禁概率为0，Cache/位置/独立评分通过。
新图(a)(b)复用原固定CSV，未修改曲线数据；InternVL(c)保留旧窗口并新增L20–L27，后者标注为post hoc。热图同时标示L17与L20。原LLaVA/Qwen数据不重跑。
Dense权限限制不能证明物理移除视觉Hidden、KV或Attention路径后的行为，不能声称实现加速。

结果：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_l20_l27_3000
父实验：/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000
GitHub未推送；GPU无计算进程，服务器未关闭。

最终复核：新增条件原对变错 30 题，原错变对 20 题。净Accuracy接近不表示逐题预测一致，也不能证明等价。All-Writer不是Native Full。
两个PNG/PDF均通过目视、DPI及矢量检查；完整源代码快照保存在source/。全部3000题QID、Image SHA、Gold、输入绑定、独立评分及检查点核验通过；GPU已释放。
