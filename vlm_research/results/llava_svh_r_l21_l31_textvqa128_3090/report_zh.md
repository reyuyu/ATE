# LLaVA-SVH-R L21–L31：相同128题TextVQA Image-only
晚层L21–L31回看得分 36.718750%，相比同机Native 50.156250%变化 -13.437500pp；相比之前L11–L24的 43.515625%变化 -6.796875pp。

| 方法 | 题数 | 十答案软分Accuracy | 记录来源 |
|---|---:|---:|---|
| Native |128|50.156250%|同一GPU与输入，复用177c631521d5c9d0436768e11e529ec492c2cbc8|
| SVH-R L11–L24 |128|43.515625%|同上|
| SVH-R L21–L31 |128|36.718750%|本轮128次真实新增生成|

## 配对比较
晚层−Native：-13.437500pp，图片SHA簇配对Bootstrap普通95%CI [-19.921875, -7.265625]pp；软分上升4题，下降23题，不变101题。
晚层−原中层：-6.796875pp，CI [-12.109375, -1.796875]pp；上升4题，下降14题，不变110题。
按128个图片SHA整簇配对重采样20000次，seed20261112，两个探索性比较均为普通未调整95%CI。本数据集已用于探索，并非独立确认集；CI跨0不能证明等价/非劣效，不能由128题推断完整TextVQA结果。

## 配置与实现
模型llava-hf/llava-1.5-7b-hf，Revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/Eager，Torch2.6.0+cu124，Transformers4.49.0，Batch1，Greedy，logits_to_keep=1，use_cache=True，正常EOS，max_new_tokens32。
严格保留之前LLaVA的128题ID、顺序、RGB图片SHA、问题、十答案Gold、原生HF chat及“Answer the question using a single word or phrase.”后缀、CLIP336 shortest-edge/center-crop、Tokenizer。Image-only，不注入OCR或Gold，prepared文件仅模型Tensor。未修改Prompt、精度或评分。

P−1和真实Decode仅L21–L31共11层可访问原始576 Image Key；L2–L5答案Attention继续物理跳算，R→Image从L20开始禁止。全部32语言Block仍处理原视觉Hidden；退出放在L31之后、最终逐Token RMSNorm之前，因此不宣称Prefill Block视觉计算减少。仅L21–L31持久保留原视觉KV，L2–L5无持久KV，其余缓存只留文本与真实增量位置。保持原RoPE逻辑位置。冻结的skip实现Prefill K/V仍计算全部行，仅最后P−1 Q/Attention/o_proj不计算；这些层Decode完整QKV/Attention/o_proj不执行。

未编辑固定5ee2a0dbe78df336229b55fed14a4a6d0d880bed的26个核心依赖，新增独立TerminalMiddleWriter，只补上cutoff32的终端退出。CPU32层Mock验证原cutoff25/KEEP11..24与旧实现Tokens/Logits/Hidden严格torch.equal，晚层Dense/Physical CPU差异及实际退出由cpu_tests.json保存。

## GPU验收与复用
GPU始终GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27。三道外部Smoke覆盖短中长输入：Native/No-op有效Logits严格torch.equal、共享Native循环和原生HF generate完整Token一致、晚层Dense/Physical完整自然EOS Token一致，数值有效生成步使用原门槛atol0.10/rtol0.003，Prefill Hidden relativeRMSE<=0.01、cosine error<=1e-4。另9步强制生成检查真实Decode、投影调用、全部Norm/MLP、原RoPE、Mask禁图零概率与每层实际KV。
沿用上一轮已明确的v2检查范围：超出自然EOS的合成步不计入正常质量Logit门槛，仍保留全部数值差异；本轮此类门槛失败计0步，见smoke_numerical.csv。未抬高阈值，不宣称固定长度数值测试通过。
正式前确定性选最短/中位/最长三道主集题重跑Native，与原完整Token/文本3/3一致，才复用原Native和L11–L24各128条原记录。旧记录原protocol/source/Commit/checksum完整保留，未伪装成本轮新生成。

## 完整性与异常
共384条逐题记录：128新增+256复用，全部独立重算官方十答案软评分并用leave-one-out公式再次核验；源、协议、输入及checkpoint SHA校验通过。晚层空输出1题、截断1题、EOS 127题、重复启发式标记2题，全部纳入评分。长度P50/P95/最大为3.0/6.0/32。全部损害案例保留harm_cases.csv；不因答案错误或输出格式重跑/剔除。
未测速，KV统计为自然EOS实际终点，不是KV@32或性能改进。本轮模型加载一次，GPUWorker已退出，外部核验无计算进程。未运行GQA、OCRBench、训练、其他层搜索或任何额外质量Benchmark。

## 产物
本轮SSD完整检查点：/home/featurize/ate-runs/llava-svh-r-l21-l31-textvqa128-3090/predictions.sqlite
原始复用检查点：/home/featurize/ate-runs/llava-svh-r-textvqa128-3090/predictions.sqlite
持久盘结果：/home/featurize/work/ATE-llava-svh-r-l21-l31-textvqa128-3090/vlm_research/results/llava_svh_r_l21_l31_textvqa128_3090/
实现代码：/home/featurize/work/ATE-llava-svh-r-l21-l31-textvqa128-3090/vlm_research/llava_svh_r_l21_l31_textvqa128/
独立Git分支：codex/llava-svh-r-l21-l31-textvqa128-3090。GitHub仅上传轻量代码、CSV、协议、评分与验证证据；完整SQLite、详细日志在server_only，模型/图像/prepared Tensor/密钥不上传。SHA256SUMS与artifact_manifest.json区分上传及仅服务器文件。Git提交与远端逐文件回读结果存server_only/git_publish_receipt.json。
