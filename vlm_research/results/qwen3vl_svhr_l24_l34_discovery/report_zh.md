# Qwen3-VL-SVH-R：L24–L34 单候选结果

**COMPLETE：585条Discovery生成；不含Validation/Held-out，不测速。**

只新增11层集合L24–L34（0-based，不含L35）。P−1与全部真实Decode同层回看；
R写入L0–L21、主视觉Hidden在L22前退出、答案L1–L6跳算均保持冻结。
本轮是功能性参考质量测试，不能据此宣称物理加速或KV节省。

## 同题结果

| 任务 | 题目/组 | Native | SVH | L7–L35 | L0＋L7–L35 | L24–L34 |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA | 250 | 80.64% | 59.80% | 80.52% | 80.44% | 77.68% |
| GQA-Query | 120 | 45.83% | 30.00% | 45.00% | 45.83% | 43.33% |
| GQA-NonQuery | 60 | 78.33% | 76.67% | 80.00% | 80.00% | 78.33% |
| ScienceQA-IMG | 40 | 87.50% | 90.00% | 92.50% | 90.00% | 90.00% |
| MMBench | 30 | 83.33% | 83.33% | 83.33% | 83.33% | 83.33% |

## 中文结论

L24–L34的11层回看达到77.68%，较原SVH提高17.88pp，
仍低于同题Native 2.96pp，比L7–L35低2.84pp。
这说明缩减到11层仍可恢复大部分原始损失，但不能维持宽回看的全部质量。
该候选值得作为质量/层数折中保留；没有独立Validation或实际测速，
不能宣称满足−1pp质量目标、严格无损或真实计算加速。
本轮双卡执行墙钟178.79秒，含模型加载和GPU Smoke，不含CPU开发与统计。

ScienceQA / MMBench沿用原冻结P2输入；MMBench按完整Circular组计分。
TextVQA为原Discovery的250题，不能与先前498题Validation百分比混作同一对照。

## L24–L34 配对差

| 任务 | 对照 | 差值pp | 图片簇95%CI | 损害/恢复 |
|---|---|---:|---|---:|
| TextVQA | Native | -2.96 | [-6.02, -0.08] | 16/10 |
| TextVQA | SVH | +17.88 | [13.07, 22.96] | 6/55 |
| TextVQA | Wide29_L7_35 | -2.84 | [-5.76, -0.08] | 15/8 |
| TextVQA | Wide30_AllActive | -2.76 | [-5.72, 0.04] | 16/8 |
| GQA-Query | Native | -2.50 | [-6.20, 0.87] | 4/1 |
| GQA-Query | SVH | +13.33 | [5.00, 22.22] | 6/22 |
| GQA-Query | Wide29_L7_35 | -1.67 | [-5.00, 1.67] | 3/1 |
| GQA-Query | Wide30_AllActive | -2.50 | [-6.20, 0.87] | 4/1 |
| GQA-NonQuery | Native | +0.00 | [0.00, 0.00] | 0/0 |
| GQA-NonQuery | SVH | +1.67 | [0.00, 5.17] | 0/1 |
| GQA-NonQuery | Wide29_L7_35 | -1.67 | [-5.17, 0.00] | 1/0 |
| GQA-NonQuery | Wide30_AllActive | -1.67 | [-5.17, 0.00] | 1/0 |
| ScienceQA-IMG | Native | +2.50 | [0.00, 7.89] | 0/1 |
| ScienceQA-IMG | SVH | +0.00 | [0.00, 0.00] | 0/0 |
| ScienceQA-IMG | Wide29_L7_35 | -2.50 | [-7.89, 0.00] | 1/0 |
| ScienceQA-IMG | Wide30_AllActive | +0.00 | [0.00, 0.00] | 0/0 |
| MMBench | Native | +0.00 | [0.00, 0.00] | 0/0 |
| MMBench | SVH | +0.00 | [0.00, 0.00] | 0/0 |
| MMBench | Wide29_L7_35 | +0.00 | [0.00, 0.00] | 0/0 |
| MMBench | Wide30_AllActive | +0.00 | [0.00, 0.00] | 0/0 |

固定种子20261111，20,000次Image SHA整簇配对Bootstrap；题目等权，Circular整组。
TextVQA损害/恢复表示软分下降/上升，不是假定二值纠错。区间仅作探索性描述。

## 验证与局限

- 585条记录完整，无重复、无跳样、无本轮OOM；原始视觉预算/权重/Prompt/评分器不变。
- 两卡各3道真实Smoke，Native/No-op严格torch.equal；禁止视觉概率为0；非目标Prefill保持一致。
- 功能参考与选层存储的少量Smoke采用原0.02门槛；生成Token一致。不是全量物理质量验收。
- 历史严格数值状态继续为FAIL_RETAINED，不把新Smoke改写成旧算法全部数值PASS。
- 首次CPU复查有一个旧缓存用例失败，保留cpu_validation.json；该用例比较未初始化备用容量，
  可被NaN填充构造出假失败，但最初失败矩阵未捕获，不能事后证明它就是当时的原因。
  cpu_cache_checks.py以活跃行严格相等、全部原始字节严格相等、独立指针与源缓存不变进行补充。
  修正测试夹具后17项CPU检查通过；未修改原引擎、旧测试文件或任何GPU数值阈值。
- 使用物理GPU1/3，未触碰GPU0/2；释放记录见gpu_release_audit.json。
- 原Held-out的MME OOM历史及其PARTIAL状态不变，本轮没有续跑或删除其中样本。
- 不自动扩大条件或样本，不进行测速，不自动推送GitHub。

## 复现

代码分支：codex/qwen3vl-svhr-l24-l34-discovery。
运行环境：/data/lk/深度学习/.envs/ate-qwen3vl/bin/python。
冻结入口：l24_l34.py --freeze；执行：l24_l34.py --launch；汇总：summarize_l24_l34.py。
历史运行时由protocol.json绑定绝对路径及SHA256；不复制模型或历史大数组。
同目录已有launch.json时入口拒绝重复启动，故障时保留检查点，需明确恢复而非覆盖。
