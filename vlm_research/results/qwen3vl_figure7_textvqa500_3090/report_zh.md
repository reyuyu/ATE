# Qwen3-VL Figure7：Decode视觉回看层预算曲线

完成30个固定条件；原500题保留，统一有效交集 **498/500题、308个图片簇**。共14940次成功生成，60条BLOCKED记录。OOM未补零、未换题；本报告不是完整500题分数。

Native Full **82.89%**；Decode全层禁图 **50.92%**。Prefill和P−1在全部36层保持原生；所有条件首输出Token一致。

本轮核心观察：**Qwen在k=4、9、13、18、22、27时均以Late得分最高；没有复现LLaVA Figure7的中层优先形态。** k=18时Late为81.51%，Middle为76.57%，Late−Middle为+4.94pp，普通95%配对CI[+2.46, +7.53]。Late18相对Native损失1.39pp；Late27与Native的均值相同，但均值相同不证明逐题或统计等价。

高预算差异很小：Middle31为83.03%，相对Native仅+0.14pp，95%CI[−0.37,+0.74]，不能声称优于Native。非单调现象保留：Early4低于全禁图、Early18低于Early13、Late31低于Late27；没有据此改层或修正结果。以上结论限于这一探索性共同有效子集、原生Qwen动态视觉预算与一个Random排列，不能将跨模型差异全部归因于视觉回看位置。

| k | Early | Middle | Late | Random |
|---:|---:|---:|---:|---:|
| 0 | 50.92% | 50.92% | 50.92% | 50.92% |
| 4 | 50.48% | 52.23% | 66.85% | 51.59% |
| 9 | 50.78% | 64.24% | 70.18% | 60.86% |
| 13 | 51.35% | 71.87% | 74.20% | 70.14% |
| 18 | 50.96% | 76.57% | 81.51% | 75.12% |
| 22 | 52.13% | 80.74% | 82.79% | 79.02% |
| 27 | 77.61% | 82.49% | 82.89% | 82.01% |
| 31 | 81.14% | 83.03% | 82.65% | 81.79% |
| 36 | 82.89% | 82.89% | 82.89% | 82.89% |

同预算配对差值（percentage points；图片SHA整簇20000次Bootstrap，普通95%CI，未经多重校正）：

| k | Middle−Early | Middle−Late | Middle−Random |
|---:|---:|---:|---:|
| 4 | +1.75 [-0.34, +3.79] | -14.62 [-18.09, -11.22] | +0.64 [-1.33, +2.57] |
| 9 | +13.45 [+10.26, +16.70] | -5.94 [-8.85, -3.03] | +3.37 [+1.03, +5.77] |
| 13 | +20.52 [+16.69, +24.43] | -2.33 [-5.08, +0.48] | +1.73 [-0.97, +4.45] |
| 18 | +25.60 [+21.50, +29.80] | -4.94 [-7.53, -2.46] | +1.45 [-1.41, +4.34] |
| 22 | +28.61 [+24.33, +32.95] | -2.05 [-4.03, -0.12] | +1.73 [-0.83, +4.30] |
| 27 | +4.88 [+2.64, +7.24] | -0.40 [-1.45, +0.61] | +0.48 [-1.52, +2.49] |
| 31 | +1.89 [+0.16, +3.81] | +0.38 [+0.00, +0.93] | +1.24 [-0.38, +2.89] |

协议与限制：
- 与Figure7对应的原始证据为9b540dc6d5b7cce53c930fdfb322479f7478c534，完全复用问题、原图、Image-only、十答案Gold与顺序。Figure7原始LLaVA500准确率不直接充当本轮Qwen基线。
- Qwen固定Revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，BF16/Eager，Transformers4.57.6；Qwen原生Chat、动态Processor、DeepStack与mRoPE。原图不缩小，视觉Token数不同于LLaVA的576。
- 36层采用用户指定的四等分及各区间中点，共9点：[0, 4, 9, 13, 18, 22, 27, 31, 36]；中点向下取整；Middle连续嵌套窗口start=(36+4−k)//2；Random固定单种子20261108，一次排列，不择优。
- 仅真实Decode禁止原始Image Key访问；Prefill及P−1全Native。没有SVH Attention跳算、Hidden退出或物理KV压缩，不能据此宣称FLOPs、显存或延迟节省。
- Smoke确认Native/No-op Logits逐位相等，HF原生生成与手工真实Decode Token一致；独立Cache克隆和独立重算Prefill的所有Smoke Logits/Token逐位相等；禁止图像概率严格为0。模型在正式工作进程只加载一次。
- OOM策略经用户确认，GitHub历史来源详见oom_policy_audit_zh.md。各点使用同一完整成功交集；OOM与分辨率相关，可能引入缺失偏差。不得把子集结果称为完整Figure7的500题复现。
- 36759的原图已实测在原生Vision Eager阶段OOM；36760与之共用相同SHA的图片，沿用该图容量失败证据，未再重复GPU尝试。逐条记录保留original_failed_key、generation_executed与原始错误；不把共享证据声称为独立测量。
- 全部离散点均为真实生成，无平滑、补点或结果驱动选层。区间为探索性普通95%CI；CI跨0不证明等价。只有一个随机排列，不代表随机选层总体期望。
- 不运行其他Benchmark、质量测试或性能Profiling。完整检查点保留本地SSD并归档到持久工作盘；GitHub仅同步轻量代码、协议、数据和图。

排除的原始题：
- 36759；图片SHA 37ec96f1731499c9c9ef9492b17b5e7cd47fd481c3ad0e1d2d53114c4ddc203f；capacity OOM; no zero score, no replacement
- 36760；图片SHA 37ec96f1731499c9c9ef9492b17b5e7cd47fd481c3ad0e1d2d53114c4ddc203f；capacity OOM; no zero score, no replacement

逐点CI、相对Native损失、相对全禁图恢复比例见accuracy_summary.csv；全部同预算配对比较见paired_comparisons.csv；EOS、空输出、截断及长度P50/P95/最大值同表保存。

图片SHA整簇配对方法：以共同有效集的图片簇有放回重采样，簇内全部问题一起纳入，每次按实际抽取的问题数计算软评分均值；全部条件使用同一重采样权重。

核心路径：
- 本地执行与完整SQLite：/home/featurize/ate-runs/qwen3vl-figure7-textvqa500
- 持久结果：/home/featurize/work/ATE-qwen3vl-figure7-textvqa500-3090/vlm_research/results/qwen3vl_figure7_textvqa500_3090
