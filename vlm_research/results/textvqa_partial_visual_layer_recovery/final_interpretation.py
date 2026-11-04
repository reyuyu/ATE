import pathlib,json,csv
out=pathlib.Path('/data/lk/ATE-worktrees/textvqa-partial-visual-layer-recovery/vlm_research/results/textvqa_partial_visual_layer_recovery')
p=out/'report_zh.md';text=p.read_text();text+='''
## 阶段作用与迁移判定

两阶段分别恢复都有正向作用：C−B为+3.367pp，D−B为+2.667pp；在另一阶段已恢复的背景下仍有额外增益，E−C为+2.300pp、E−D为+3.000pp。上述七个预声明配对比较的Holm-8 p均小于0.05。不能仅凭C略高于D的点估计宣称P−1比Decode更重要，因为本轮没有预声明C−D的直接检验。

交互项−0.367pp的95%CI跨0（−1.667到+0.600pp），p=0.503，未发现非加性的统计证据；这不证明严格可加。E相对B有21题Score提高、2题降低；相比A则仅3题提高、59题降低。E在B基础上新出现1题截断，同时恢复了B的2题截断；重复3gram新出现3题、消失2题，故不能只看异常净数。

固定L14/L19在本批TextVQA中有可重复审计的恢复作用，但不能承接GQA中接近Full的结论：E仅27.500%，而真正Native Full为45.233%，差距−17.733pp、95%CI [−22.300, −13.400]，Holm-8 p=0.00040。该结果支持“部分恢复有效”，不支持“两层已足够恢复完整TextVQA能力”，也不证明这两层相对其他层具有特殊性。

最终布局由同目录replot_final.py从冻结CSV重现，修正数值标签与误差棒重叠，并在预算图中放大C/D的近邻点；没有更改数据、模型、统计或推理执行源码。
''';p.write_text(text)
print('INTERPRETATION_COMPLETE')
