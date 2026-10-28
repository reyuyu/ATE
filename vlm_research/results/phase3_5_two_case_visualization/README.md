# 两案例：32层视觉 Attention 与 Physical Exit

仅CPU读取既有Phase3.4所选Attention行/已验证Processor几何和Phase3.2完整8K Margin；没有GPU、Forward、训练或额外样本。

## 图像与颜色口径

- 每张热图4×8，行优先L0–L31。真实CLIP shortest-edge336/BICUBIC/中心裁剪；复用实际整数缩放、裁剪偏移与裁后对象框，蓝绿色/品红边框分别对应官方对象1/2。ROI富集使用原576Patch正面积交集并集，而非框内语义分割。
- 先平均全部32Head，再对576视觉Patch归一化。四张热图、两类Query和两图共用同一对数色标，没有按层最大值拉伸；V是未归一化的视觉总Attention Mass，E是显示的Head均值地图的面积富集。对数颜色不能按线性强度阅读；小质量层的明亮条件热点不等于原始视觉总质量更高。
- E=(显示地图的ROI条件概率)/(ROI Patch数/576)，熵为显示地图的H/log576。此前Phase3.5统计先逐Head条件化再平均，属于不同非线性顺序；layer_metrics.csv同时保存旧口径以便核对，本次曲线与热图使用同一新口径。Question行是缓存中真实问题Token行均值，Answer行是P−1。
- 曲线上方为真实L0–L31，下方K点在共享坐标K−0.5；K24在L23和L24之间。主曲线保留完整8K，内嵌图放大K16–32弱Margin，Margin>0才正确，平局判错。原生Full=K32，没有将Full Attention当作退出后Attention。

全图公共对数色标：[1.54557264e-07, 0.373589769]，正值无截断；零值数量=0。

## 图表与实际观察

### 04132006

Is the couch pillow to the right or to the left of the curtains?

标准答案：**right**。Full Margin=-0.0078125；K24 Margin=+0.0312500。

- [Question→Image全部32层](04132006_question_32_layers.png)
- [Answer P−1→Image全部32层](04132006_answer_32_layers.png)
- [ROI曲线与Physical Margin对照](04132006_roi_exit_margin.png)

| Query | L23→L24 ROI富集 | L23→L24条件熵 | L23→L24原始视觉Mass |
|---|---|---|---|
| question | 1.0290→0.7451 | 0.8081→0.8815 | 0.0699→0.1134 |
| answer | 0.6903→0.7151 | 0.6595→0.7647 | 0.0489→0.0435 |

全8K Margin：K0=-2.0781250, K8=-1.0859375, K12=-0.7265625, K16=-0.0156250, K20=-0.0546875, K24=+0.0312500, K28=+0.0312500, K32=-0.0078125。

### 15133706

Is the young girl to the left or to the right of the boy?

标准答案：**right**。Full Margin=-0.0312500；K24 Margin=+0.0390625。

- [Question→Image全部32层](15133706_question_32_layers.png)
- [Answer P−1→Image全部32层](15133706_answer_32_layers.png)
- [ROI曲线与Physical Margin对照](15133706_roi_exit_margin.png)

| Query | L23→L24 ROI富集 | L23→L24条件熵 | L23→L24原始视觉Mass |
|---|---|---|---|
| question | 0.8608→1.0846 | 0.7961→0.8456 | 0.1096→0.1531 |
| answer | 0.6748→0.8454 | 0.7051→0.7633 | 0.0681→0.0629 |

全8K Margin：K0=-4.2226562, K8=-3.0468750, K12=-0.7500000, K16=+0.0234375, K20=-0.0312500, K24=+0.0390625, K28=-0.0546875, K32=-0.0312500。

04132006在K24恢复时Question ROI富集下降（此Head均值地图中约1.029→0.745），但Answer略升；两者条件熵都上升。15133706恢复时Question和Answer ROI富集都上升、熵也上升，说明更分散与ROI占比上升可以共存。两例并无一致的“ROI恰好下降”模式，不能把条件熵升高自动归因Sink，也不能解释为确定因果。

04132006的K24/K28并列全网格最高，15133706的K24最高。两例均为事后案例，Full和K24 Margin接近零，原FP16数值敏感性与有限候选协议的限制保留；本轮没有新FP32或模型验证。

## 复现

```bash
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py
CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py --verify-existing
```

已有目录拒绝覆盖。validation.json保存缓存/图像/脚本/六图SHA256、条件概率归一化、映射/历史指标复核结果和颜色口径。历史阶段及此前Phase3.5结果不变。
