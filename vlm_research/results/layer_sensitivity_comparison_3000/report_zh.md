# 跨模型层敏感性：仅使用 GitHub 历史结果

主数据冻结 commit：`6a0dae9fed134624e732e80984080bfc17aa18b2`。全部 136 行通过校验；四组均为 3000 题、2409 个图片 SHA256 簇。

## 定义与基线

Visual Writing 衡量在既有 All-Writer 协议内关闭单层视觉写入后的完整答案 Accuracy 变化。P−1 Update 衡量在原始 Prompt 最后位置单层整个 Attention o_proj 输出更新置零后的变化；后续 Decode 原生。它不是仅屏蔽 P−1→Image Key，也不是纯视觉读取敏感性。

统一 Signed S = ACC_base − ACC_ablated = −delta_accuracy_pp，单位 pp；Drop-only = max(0, S)。正值为掉分，负值为涨分。普通与校正区间均由 [low,high] 转为 [−high,−low]，没有新估计 CI 或 p 值。

All-Writer 的写入隔离协议与 Native Full 的原生推理不同，必须分别配对，不能混用基线。

| 模型 | 干预 | 基线 | 基线 ACC (%) | 最大损害层 | S (pp) | 校正后掉分层 | 校正后涨分层 |
|---|---|---|---:|---|---:|---|---|
| LLaVA | Visual Writing | All-Writer | 87.033333 | L11 | +2.6000 | L11 | 无 |
| LLaVA | P-1 Update | Native Full | 87.566667 | L15 | +0.8333 | L26 | 无 |
| Qwen | Visual Writing | All-Writer | 87.900000 | L18 | +1.1667 | L8, L10, L11, L18 | 无 |
| Qwen | P-1 Update | Native Full | 88.433333 | L24 | +1.4000 | L24, L27, L34 | L35 |

## 数值和历史核验

LLaVA Writing / P−1 峰分别为 L11 / L15（相差 4 层）；Qwen 分别为 L18 / L24（相差 6 层）。峰是最大点估计，不自动等于校正后显著层。例如 LLaVA P−1 峰在 L15，校正后掉分证据只在 L26。

Qwen P−1 L35：S=-0.633333 pp；校正后区间[-1.280662, -0.033115] pp，原对→错 7，原错→对 26。区间全负，因此标记为显著 Accuracy 增益，绝不取绝对值。

Writing 历史 CSV 副本与原始冻结 CSV 的 Accuracy、差值、全部区间及翻转计数逐字段核对一致。Qwen Writing 显著掉分层与历史报告的 L8、L10、L11、L18 一致。LLaVA Writing 历史报告明确分列 NLL 和 Accuracy，但文字列出的显著层属于 NLL；本报告只使用 Accuracy 校正区间自动判断，未挪用 NLL 显著性。两模型 P−1 显著掉分层与历史跨模型报告一致。

历史置信区间采用图片 SHA256 整簇配对 Bootstrap（20,000 次），问题等权；Bonferroni 家族按原始模型和干预保持 32/36 层分别校正，不合并四组为新的家族。有限重采样使极端尾部分位数存在 Monte Carlo 误差。

## 图的读法与限制

Figure A：2×2 曲线，全部使用同一纵轴范围；浅色带为普通 95% CI，星号和 dagger 分别依据历史 Bonferroni 区间表示掉分和涨分。Figure A2：同模型叠加两类干预，线型和颜色同时区分，明确注明各自基线。

Figure B：四行共用绝对色标；仅此主图使用 Drop-only。Figure C：保留全部负值，色标以 0 为中心且正负范围对称。热图实际层号为 L0–L35，LLaVA 的 L32–L35 是 NaN 并显示灰色 N/A，不是 0。热图使用矢量网格，PDF 坐标轴、线条、文字保留矢量。

两类敏感性峰不重合，支持“在各自协议和基线下，完整答案 Accuracy 对两类干预的层依赖不同”的描述。不证明存在两个独立神经模块，不将模型差异单独归因于架构；原始 Prompt、精度、视觉布局等也可能不同。

Drop-only 为 0 或 CI 跨 0 不证明没有逐题损害或等价；对→错和错→对可能抵消，统一 CSV 保留两种计数。单层低敏感不证明联合关闭安全。Hard Mask 和 o_proj 输出旁路均不是物理加速实验，也不证明层可删除。本图比较完整答案 Accuracy，不是 Gold NLL 或首 Token 概率。

## CPU 复现

```bash
python -m pip install numpy matplotlib
python vlm_research/analysis/layer_sensitivity_plot.py
# 缓存齐全后完全离线：
python vlm_research/analysis/layer_sensitivity_plot.py --offline
```

源码只导入标准库、NumPy、Matplotlib；不连接服务器，不加载模型，不使用 GPU，不读取服务器 logits、checkpoint 或 NPZ。所有输入均为 GitHub 已提交 CSV/报告；source_manifest.json 保存固定 commit、原始 SHA256、Git blob SHA1、脚本版本及 SHA256。source_inputs 保留原始字节供离线复现；历史实验文件不改写。

## P−1 六任务补充图

Figure D 将六类 GQA 的 Signed Accuracy-drop Sensitivity 叠加在同一坐标系，LLaVA/Qwen 各一面板，颜色、线型和点形区分任务，共享纵轴。Figure E 显示各模型六任务×层的 Signed 热图，正负对称、两模型共用同一 pp 色标。LLaVA L32–L35 为 NaN/灰色 N/A。每条任务曲线使用自己的 Native Full（每类 500 题），12 个基线见 p1_task_baselines.csv；408 个任务层值及历史区间见 p1_task_sensitivity_unified.csv。任务均值与总体敏感性、翻转计数求和均经自动核验。任务结果仅作描述，不添加总体显著性星号；曲线不叠加六组 CI 以保持可读，原始区间已完整保留在 CSV。
