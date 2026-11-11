# LLaVA-SVH-R L10–L25：TextVQA Image-only 128题
本轮L10–L25共16层回看，官方十答案软评分为45.312500%，软分合计58.0000/128。同GPU Native 50.156250%，差值-4.843750pp；相比上轮L0–L25 47.031250%，差值-1.718750pp。

|配置|题数|官方软评分Accuracy|来源|
|---|---:|---:|---|
|Native Full|128|50.156250%|复用同GPU 177c631521d5c9d0436768e11e529ec492c2cbc8|
|SVH-R L11–L24|128|43.515625%|同上|
|SVH-R L0–L25，跳过L2–L5|128|47.031250%|复用同GPU f7db13807e810c8dfd6cb5db0c18329d171c0745|
|SVH-R L10–L25|128|45.312500%|本轮新增128次|

## 逐题配对
本轮−Native：-4.843750pp，图片SHA整簇Bootstrap普通95%CI [-9.453125, -0.703125]pp；上升3题、下降11题、不变114题。
本轮−L11–L24：+1.796875pp，95%CI [-0.234375, +4.375000]pp。
本轮−L0–L25：-1.718750pp，95%CI [-5.703125, +2.109375]pp；上升4题、下降7题、不变117题。
按128个图片SHA整簇同题配对重采样20000次，seed20261112，三个比较均为普通未调整95%CI。本128题已用于多轮探索，不是独立确认集；不据此宣称最佳层、严格等价/非劣效或完整5000题成绩。

## 冻结配置
模型llava-hf/llava-1.5-7b-hf，Revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9；FP16/Eager、Torch2.6.0+cu124、Transformers4.49.0、Batch1、Greedy、正常EOS、max_new_tokens32、use_cache=True、logits_to_keep=1。
原128题Question ID、顺序、图片SHA、问题、十答案Gold、原生HF Chat及single-word-or-phrase后缀、CLIP336 shortest-edge/center-crop不变。Image-only，无OCR和Gold输入。仅复用本地已有模型、图片、prepared Tensor，未重新下载或全量安装。
P−1与真实Decode只在L10–L25共16层读取原始Image Key；L2–L5答案Attention持续真实物理跳算，R→Image仍仅L0–L19允许；进入L26物理删除576视觉Hidden，后六层仅处理非视觉Hidden。真实Compact KV只在L10–L25保留原视觉KV，L2–L5无持久KV；原逻辑RoPE、非目标文本Attention、32层Norm/MLP/Residual不变。前一轮L0–L25与本轮具有相同退出位置L26，区别是原视觉回看集合22层变16层；L11–L24历史配置退出位置L25，不能将该比较全部归因于层集合。
固定5ee2a0dbe78df336229b55fed14a4a6d0d880bed的26个核心文件及独立TerminalMiddleWriter字节不变，独立目录只配置KEEP与cutoff并核验，不修改冻结实现。

## 验收与完整性
CPU验证原cutoff25严格torch.equal回归、本轮cutoff26实际退出、32层真实KV和逻辑位置；原131份输入（128主集+3外部Smoke）图片/模型Tensor SHA与历史一致，Gold保持分离。既有三个条件384条原记录逐题核验checksum、原协议/源码/输入SHA、GPU UUID和Manifest，保留原Commit。
GPU短/中/长三题Smoke：Native/No-op有效Logits严格torch.equal；Native循环与原生HF generate完整Token一致；本轮Dense/Physical完整自然EOS Token一致。沿用先前v2数值门槛：正常EOS有效步Logits atol0.10/rtol0.003，Hidden relativeRMSE<=0.01、cosineerror<=1e-4；强制9步EOS后仅结构与有限性，仍保留数值差异，本轮EOS后原阈值失败0步，不声明固定长度严格数值等价。审计确认L26真实删行、目标投影跳算、禁止连接概率0、原RoPE与每层KV正确。
正式前最短/中位/最长3题Native完整Token和文本校准通过才复用既有基线。仅新增128次正式生成，模型加载一次，不重跑旧128题条件。
全部512条记录由官方TextVQA十答案共识软评分和独立leave-one-out公式重算一致；完整预测、Token IDs、Gold、原始SHA、32层KV和逐题损害均保存。本轮空输出1、截断1、EOS127、重复启发式3，全部计分；长度P50/P95/最大3.0/7.6499999999999915/32。
GPU始终GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27，结束已释放，无计算进程。不继续暂停的单层扫描，不运行InternVL、GQA、OCRBench、训练、其他窗口或任何性能测量。KV为正常EOS终点，不是KV@32，不作速度声明。

## 文件与同步
SSD完整本轮检查点：/home/featurize/ate-runs/llava-svh-r-l10-l25-textvqa128-3090/predictions.sqlite
持久化结果：/home/featurize/work/ATE-llava-svh-r-l10-l25-textvqa128-3090/vlm_research/results/llava_svh_r_l10_l25_textvqa128_3090/
源码：/home/featurize/work/ATE-llava-svh-r-l10-l25-textvqa128-3090/vlm_research/llava_svh_r_l10_l25_textvqa128/
独立GitHub分支：codex/llava-svh-r-l10-l25-textvqa128-3090。代码、轻量CSV、协议、报告、验证和SHA清单同步GitHub；SQLite与详细日志仅保存在服务器，artifact_manifest逐项区分。模型、图像、prepared Tensor、缓存和密钥不上传。未修改main或历史实验，不创建PR。
