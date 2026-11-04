## 算法与绑定
C0是真Native Full；C1是冻结Fast-P2 All-Writer；C2在C1上采用Fixed-Q-Conditional-Assistant-15；C3-V2在C1上增加Answer-only物理跳算；C4-V2组合Conditional Writer与Answer-only跳算。旧C3/C4没有重命名或进入新推理路径。
L2–L5的Prefill仍为全部Token执行Norm、K/V和MLP；Q、QK/Softmax/AV、o_proj仅省去最后P−1一行。全部576个Image Query及此前Prefix/Question/R Query真实计算Attention。保留完整临时K/V，使用原逻辑RoPE；P−1不能影响此前因果Query。Decode这四层QKVO/Attention不执行，因此不分配其持久KV。
Conditional Writer仅控制L0–L19的R→Image Softmax前Hard Mask：全部Question Subword；首词what/which或Question Token数≥15时，加完整ASSISTANT四个Subword。其他R/P−1禁止直接读图，不能将这一Mask解释成跳过R计算。L20前物理移除视觉Hidden，后12层只计算文本，保留原位置。其余28层持久Cache原地存文本KV，不复制历史。
同旧实验完全相同的图片SHA、Question ID、Prompt IDs、输入Tensor、10个官方答案；Manifest和Input Specs的文件SHA逐位一致。使用原LLaVA revision、FP16/eager、Transformers4.49.0、Greedy32、自然EOS。数据来自历史本地冻结500题，不是全VQAv2的无偏随机测试。
