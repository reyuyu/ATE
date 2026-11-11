# LLaVA-SVH-R L0–L25：TextVQA Image-only 128题
本轮L0–L25回看（L2–L5仍物理跳算，实际22层）十答案软评分：47.031250%，软分合计60.2000/128。同机Native为50.156250%，差值-3.125000pp，图片SHA簇配对Bootstrap普通95%CI [-6.876953, +0.234375]pp。

|配置|题数|Accuracy|来源|
|---|---:|---:|---|
|Native Full|128|50.156250%|复用同GPU Commit 177c631521d5c9d0436768e11e529ec492c2cbc8|
|SVH-R L11–L24|128|43.515625%|同上|
|SVH-R L0–L25（跳过L2–L5）|128|47.031250%|本轮新增128次|

## 冻结配置
模型llava-hf/llava-1.5-7b-hf，Revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9；FP16/Eager、Torch2.6.0+cu124、Transformers4.49.0、Batch1、Greedy、正常EOS、max_new_tokens32、use_cache=True、logits_to_keep=1。
保持原128题ID/顺序、图片SHA、问题、十答案Gold、原生HF Chat、single-word-or-phrase后缀及CLIP336 shortest-edge/center-crop，不输入OCR和Gold。原prepared仅三项模型Tensor，全部131份（含外部Smoke3题）CPU哈希核验。
P−1与真实Decode仅L0,L1,L6–L25共22层读取原始Image Key；L2–L5答案Attention继续真实跳算。R→Image固定L0–L19允许、L20起禁止；进入L26物理删除576视觉Hidden，L26–L31仅处理非视觉行。实际Compact KV只在开放22层保留视觉KV，L2–L5无持久KV。原逻辑RoPE、文本Attention、全部32层Norm/MLP/Residual保持原生。冻结26个核心依赖来自5ee2a0dbe78df336229b55fed14a4a6d0d880bed，独立TerminalMiddleWriter原样复用，未编辑冻结算法。

## 验收、复用与评分
CPU小模型测试包括原cutoff25严格torch.equal回归、本轮cutoff26实际退出、真实Cache和原位置。GPU外部短/中/长Smoke：Native/No-op有效Logits严格torch.equal、原生HF generate与Native循环完整Token一致；本轮Dense/Physical完整自然EOS Token一致，正常EOS有效步沿用atol0.10/rtol0.003，Hidden relativeRMSE<=0.01、cosineerror<=1e-4。另强制9步只审计EOS后结构与有限性，保留数值差异；EOS后原阈值失败0步，不声称固定长度数值等价。真实删除、跳算投影调用、禁图概率0、全部Norm/MLP/原RoPE及32层实际Cache均审计。
正式前最短/中位/最长3题Native完整Token和文本复现后，才复用同GPU历史Native与L11–L24各128条，全部保留原Commit、协议、源码和记录SHA。只新增本轮配置128次正式生成，模型加载一次。
官方TextVQA十答案共识软评分与独立leave-one-out公式重算完全一致。全部384条（128新+256复用）来源、协议、输入、记录checksum及实际KV核验。空输出0、截断0、EOS128、重复启发式1均计分，长度P50/P95/最大3.0/7.6499999999999915/11。
相对Native：软分上升2题，下降7题，不变119题。相对L11–L24：+3.515625pp，95%CI [-0.078125, +7.343750]pp。图片SHA整簇配对Bootstrap20000次，普通未校正区间；128题已用于多轮探索，并非独立确认集，不据此宣称最佳层或完整TextVQA成绩。
用户要求暂停的单层扫描已停止，已有检查点保留；本轮不继续扫描、不运行InternVL、GQA、OCRBench、训练或性能实验。KV记录为正常EOS终点，不是KV@32。本轮结束GPU已释放，无其他计算进程。

## 位置
SSD完整检查点：/home/featurize/ate-runs/llava-svh-r-l0-l25-textvqa128-3090/predictions.sqlite
持久化结果：/home/featurize/work/ATE-llava-svh-r-l0-l25-textvqa128-3090/vlm_research/results/llava_svh_r_l0_l25_textvqa128_3090/
源码：/home/featurize/work/ATE-llava-svh-r-l0-l25-textvqa128-3090/vlm_research/llava_svh_r_l0_l25_textvqa128/
GitHub独立分支：codex/llava-svh-r-l0-l25-textvqa128-3090。轻量产物和源码可追溯；SQLite和详细日志仅服务器，模型/数据/密钥不上传。artifact_manifest与SHA256SUMS列明各文件。
