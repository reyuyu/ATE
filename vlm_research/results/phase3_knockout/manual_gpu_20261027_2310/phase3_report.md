# Phase 3：Layer-wise Visual Attention Knockout 实测报告

以下数值来自本目录已完成的 7B GPU 运行；随机小模型测试不参与结果统计。

## 设置

模型 `llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；FP16/eager、batch=1、eval/inference_mode、use_cache=False。
层索引 >= K 的 LLM 层中，所有后图像文本 Query 到视觉 Key 的合法连接被置为 -inf；其他连接 Mask 不变。
答案损失只包括固定参考答案 Token，不包括 Prompt 或 EOS；KL 方向为 KL(Full || Knockout)。
候选评分累加完整候选词序列的条件 log P；候选间使用相同 padding 后输入长度，并共享此前答案 Token。

## Cutoff 结果

下表为每条样本的 mean answer NLL increase（nats/token）；不能将同图多问当作独立图片。

| Sample | K=0 | K=8 | K=12 | K=16 | K=20 | K=24 | K=28 | K=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| street_color | 0.049835 | 0.080673 | 0.020849 | 0.005807 | 0.002662 | -0.007311 | -0.004884 | 0.000000 |
| street_description | 0.705758 | 0.626013 | 0.544594 | 0.509531 | 0.247572 | 0.046651 | 0.002213 | 0.000000 |
| street_animal | -0.374953 | 0.207264 | 0.436407 | 0.214903 | 0.090259 | 0.068056 | 0.006731 | 0.000000 |
| street_letters_color | -0.182578 | 0.478413 | 0.766541 | 0.440326 | 0.091971 | 0.145456 | 0.105852 | 0.000000 |
| street_ocr | -0.234787 | 0.072161 | 0.262989 | 1.705725 | 0.791730 | 0.117914 | -0.095034 | 0.000000 |
| street_spatial_left | -0.652170 | 0.210136 | 0.026459 | 0.114700 | 0.013725 | 0.061682 | 0.028904 | 0.000000 |
| street_car_color | -1.934183 | 1.370306 | 1.559305 | 1.445778 | 0.011519 | -0.021647 | -0.050807 | 0.000000 |
| street_sign_shape | -1.194139 | 0.413497 | 0.404773 | 0.393542 | 0.160808 | 0.125072 | 0.020794 | 0.000000 |
| street_spatial_right_mirrored | -0.792551 | 0.037544 | -0.060233 | 0.121231 | 0.001932 | 0.038800 | 0.013204 | 0.000000 |

## 逐 Token、候选与数值校验

### street_color

[逐 Token CSV](samples/street_color/token_statistics.csv) · [完整指标 JSON](samples/street_color/metrics.json)

![NLL](samples/street_color/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_color/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_color/token_knockout_sensitivity_heatmap.png)

![Candidate margin](samples/street_color/cutoff_candidate_margin.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8)]。

### street_description

[逐 Token CSV](samples/street_description/token_statistics.csv) · [完整指标 JSON](samples/street_description/metrics.json)

![NLL](samples/street_description/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_description/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_description/token_knockout_sensitivity_heatmap.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[]。

### street_animal

[逐 Token CSV](samples/street_animal/token_statistics.csv) · [完整指标 JSON](samples/street_animal/metrics.json)

![NLL](samples/street_animal/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_animal/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_animal/token_knockout_sensitivity_heatmap.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (8, 12)]。

### street_letters_color

[逐 Token CSV](samples/street_letters_color/token_statistics.csv) · [完整指标 JSON](samples/street_letters_color/metrics.json)

![NLL](samples/street_letters_color/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_letters_color/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_letters_color/token_knockout_sensitivity_heatmap.png)

![Candidate margin](samples/street_letters_color/cutoff_candidate_margin.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (8, 12), (20, 24)]。

### street_ocr

[逐 Token CSV](samples/street_ocr/token_statistics.csv) · [完整指标 JSON](samples/street_ocr/metrics.json)

![NLL](samples/street_ocr/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_ocr/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_ocr/token_knockout_sensitivity_heatmap.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0, 28]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (8, 12), (12, 16), (28, 32)]。

### street_spatial_left

[逐 Token CSV](samples/street_spatial_left/token_statistics.csv) · [完整指标 JSON](samples/street_spatial_left/metrics.json)

![NLL](samples/street_spatial_left/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_spatial_left/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_spatial_left/token_knockout_sensitivity_heatmap.png)

![Candidate margin](samples/street_spatial_left/cutoff_candidate_margin.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (12, 16), (20, 24)]。

### street_car_color

[逐 Token CSV](samples/street_car_color/token_statistics.csv) · [完整指标 JSON](samples/street_car_color/metrics.json)

![NLL](samples/street_car_color/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_car_color/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_car_color/token_knockout_sensitivity_heatmap.png)

![Candidate margin](samples/street_car_color/cutoff_candidate_margin.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0, 24, 28]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (8, 12), (28, 32)]。

### street_sign_shape

[逐 Token CSV](samples/street_sign_shape/token_statistics.csv) · [完整指标 JSON](samples/street_sign_shape/metrics.json)

![NLL](samples/street_sign_shape/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_sign_shape/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_sign_shape/token_knockout_sensitivity_heatmap.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8)]。

### street_spatial_right_mirrored

[逐 Token CSV](samples/street_spatial_right_mirrored/token_statistics.csv) · [完整指标 JSON](samples/street_spatial_right_mirrored/metrics.json)

![NLL](samples/street_spatial_right_mirrored/cutoff_answer_nll_increase.png)

![Key tokens](samples/street_spatial_right_mirrored/cutoff_key_token_log_probability.png)

![Token sensitivity](samples/street_spatial_right_mirrored/token_knockout_sensitivity_heatmap.png)

![Candidate margin](samples/street_spatial_right_mirrored/cutoff_candidate_margin.png)

相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：[0, 12]。这是原始变化描述，不是显著性检验。
K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：[(0, 8), (12, 16), (20, 24)]。

## 资源与解释边界

正式 sweep/candidate forwards 最大 allocated：13649.285 MiB；各次同步 Forward 时间与显存见 JSON。校验阶段峰值另见首样本 correctness_validation.json。
总墙钟时间（含加载、校验、绘图）：42.201 秒。
新 QA 为用户委托 AI 目视复核，未声称人工核验；黑色汽车有蓝色反光，颜色标签存在歧义。
空间配对图为水平镜像派生图，文字也反转；不是新增独立自然图。原描述轨迹包含不能确认的 fire hydrant，不能作为完整真值。
当前干预测量固定答案历史下直接视觉访问被屏蔽后的预测变化，含 softmax 重分配和分布外效应。
某个 K 下损失接近 Full 只能说明这些样本/轨迹的边际变化较小；无预注册稳定阈值，不自动判定 Visual Fusion Saturation Depth。
不据此宣称文本吸收全部视觉信息、安全删除视觉 Token 或真实推理加速。未开展 Question-only/Answer-only、Decode、Pruning 或 Router。
