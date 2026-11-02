# Qwen3-VL Query角色敏感性：固定L1–L6、240题正式探索

## 结论

五条件均为本轮独立真实KV Cache完整Greedy生成，不是首Token命中率。只将指定Query的o_proj输出置零，未跳过实际计算。

| 条件 | 正确/240 | Accuracy | ΔFull pp | 损害/纠正 | 首Token翻转 | 平均signed LogP Loss |
|---|---:|---:|---:|---:|---:|---:|
|Full|200/240|83.333%|+0.000|0/0|0|+0.00000|
|I-Bypass|192/240|80.000%|-3.333|9/1|10|+0.04661|
|R-Bypass|133/240|55.417%|-27.917|73/6|95|+2.76511|
|A-Bypass|198/240|82.500%|-0.833|2/0|2|+0.00049|
|I+A-Bypass|194/240|80.833%|-2.500|8/2|10|+0.03324|

## 图像簇配对统计

固定seed=20261103，按Image SHA256整簇配对Bootstrap 20,000次，保留题目权重。240张图各一题。六个预设比较加I+A交互共七项，同时报告普通和Bonferroni校正区间。任务分组仅作描述。

| 比较 | Δ pp | 普通95%CI | 七项校正95%CI | 损害/纠正 |
|---|---:|---|---|---:|
|I-Bypass minus Full|-3.333|[-5.833, -0.833]|[-7.083, 0.000]|9/1|
|R-Bypass minus Full|-27.917|[-34.167, -21.667]|[-36.667, -19.167]|73/6|
|A-Bypass minus Full|-0.833|[-2.083, 0.000]|[-2.917, 0.000]|2/0|
|I+A-Bypass minus Full|-2.500|[-5.000, 0.000]|[-6.250, 0.833]|8/2|
|R-Bypass minus I-Bypass|-24.583|[-31.250, -18.333]|[-33.750, -15.417]|68/9|
|R-Bypass minus A-Bypass|-27.083|[-33.750, -20.417]|[-36.250, -17.917]|73/8|

I+A交互 (IA−I−A+Full)：+1.667 pp，95%CI [0.000, 3.750]，校正CI [0.000, 5.000]。
联合独有损害（Full、I、A均正确但I+A错误）：0题；题号见joint_interaction.csv。

**R相对I/A是否明显更有害：是，两个预设配对比较的普通95%CI均低于0。**
I+A相对Full净变化为-2.500 pp，损害8题、纠正2题，不能仅凭净变化或CI跨0宣称严格无损。
低敏感性是本批数据、固定层段和指定干预下的行为结果；不是Attention不读取信息、不是可物理删除层，也不是已实现推理加速。

## 与LLaVA的同题对照

| 条件 | Qwen Accuracy | LLaVA Accuracy |
|---|---:|---:|
|Full|83.333%|85.833%|
|I-Bypass|80.000%|85.000%|
|R-Bypass|55.417%|57.500%|
|A-Bypass|82.500%|86.667%|
|I+A-Bypass|80.833%|85.417%|

LLaVA历史结果来自8d12f8c05f2b246d029ec4d9ffe45d1f857fa947及8019c6009bb1d8c9f2dc3ffd8c18bf4dfb8c1fd1，逐题ID和图片SHA一致。Qwen保留历史原生Prompt、Processor、BF16和DeepStack，LLaVA为FP16、L2–L5；因此这是角色模式的跨模型对照，不是控制所有架构与层数因素的实验。
本轮支持跨模型复现“R比I/A更敏感”的方向，但I+A是否仍低损害必须结合上表真实损害及区间单独判断。

## 协议与正确性

- 固定Qwen3-VL-8B-Instruct revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b；BF16/eager，Transformers4.57.6，原生分辨率、mRoPE、DeepStack保持不变。
- 原LLaVA样本清单字节不变，SHA256：258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c。未重采样、未按结果筛选。
- I使用实际image-pad位置；R严格位于vision_end之后、P−1之前，包括后图像模板文本，排除vision_end本身；A包括P−1与全部真实Decode Query。没有硬编码576。
- 只关闭0-based L1–L6的Attention o_proj目标行，不改Mask/其余行/MLP/Residual/KV。I/R只作用Prefill；A与I+A在真实Decode继续关闭。
- CPU范围/异常清理测试，以及输入长度最短、中位、最长三题GPU Smoke全部通过；全部返回有效logits在同形状logits_to_keep=1下Full/No-op逐位torch.equal。没有放宽容差。
- Smoke检查独立Cache对象及实际存储、每步36层KV长度、原生BF16残差、三路DeepStack特征与注入、MLP调用、历史A-only一致性、异常Hook清理、输入及权重不变。正式1200检查点再校验范围、有限性、Cache及原始Logits SHA。
- R、I、A干预Query数量不同，不能将总损害直接当作单Query贡献；低净损害不等于严格无损。无训练、无Kernel、无其他层段扫描。

实际I Query范围：128–1024；R Query范围：19–36。
正式生成耗时4.62分钟（包含结束模型内容校验），显存allocated峰值21088MiB、reserved峰值22128MiB。
全部负面结果共243个条件-题目行，见negative_cases.csv；真实原对变错见harm_cases.csv；完整答案、Token与LogP见per_question.csv。EOS/空答案/截断/重复诊断见accuracy_summary.csv。
生成源码Commit：`4dbdc3df9eb73154d5e8398cc9bd82608fa2cfea`；协议SHA256：`356f4b1f3223b914320783187cac6230f61564496adaaa72e153dbfcc383f27d`。
原始首步全词表Logits仅保存在服务器外部结果目录，不上传权重或大缓存；对应哈希随检查点交付。

![Qwen角色结果](figure_query_roles.png)

![跨模型同题对照](figure_cross_model.png)
