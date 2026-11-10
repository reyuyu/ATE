# TextVQA Image-only复现与独立审计

本分支从03ae6763a02bca30ae7bb553e1ff1c006d8c2552建立，继承冻结V2物理引擎，不使用C1/C2。source_manifest.json为实际GPU执行前冻结的全部运行源码SHA；analysis_source_manifest.json为后续CPU分析、存储与发布源码SHA。全部历史结果保持不动。

模型OpenGVLab/InternVL3-8B，Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0，256视觉Token。数据必须是TextVQA Validation5000题，协议ATE_SVH_1TILE_SCORER_ALIGNED，原图+原问题+冻结指令，绝无OCR辅助文字。模型本地缓存/home/featurize/ate-local-models/InternVL3-8B。

当前运行目录/home/featurize/ate-runs/internvl3-svh-r-fast-textvqa-imageonly-3090，Python/home/featurize/.venvs/ate-frozen/bin/python。复现入口为vlm_research.internvl3_svh_r_fast_textvqa_imageonly.prepare、run、analyze，依次CPU准备、一次加载分阶段运行、独立CPU分析。设置PYTHONPATH为完整本分支源码或已准备的code目录，ATE_TEXT_RUN为新的空运行目录（results/checkpoints/logs/prepared子目录），CUDA_VISIBLE_DEVICES为冻结GPU UUID。旧worker依赖ATE_SEVEN_ROOT=/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090和ATE_SEVEN_RUN=/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090；父实验TextVQA Manifest与历史Native SQLite必须先恢复到其冻结路径，不能伪造匹配。

禁止覆盖本轮运行目录或把新机器时间混入现有结果。新GPU若UUID不一致，应另立协议，不静默改本轮UUID。所有样本/候选/排序/门槛/源码SHA均在GPU前冻结。sampling_freeze.json不含Gold/预测，正式96+96和另8题Smoke均来自问题文字/图像SHA确定性抽取，三组图片互斥；历史Native答案正文只在允许的阶段按Question ID读取，Holdout正文在最终selection冻结后才读取。

GitHub的per_question.csv包含全部912条正式质量记录：720新增SVH-R、192严格绑定的历史Native参考。Token IDs、原始输出、十答案Gold、官方软分、元回答标记均完整；speed_per_input.csv包含108次本轮TextVQA正式测速。input_bindings.json提供每题图像/Prompt/输入Token/BF16像素Tensor SHA；population_prompt_audit.csv记录5000题的逐题Prompt及Token审计。kv_layout_per_layer.csv.gz保留全部真实逐层K/V形状/字节，解压SHA见artifact_manifest.json。服务器server_only保留完整原始SQLite与日志；模型、原图和CPU像素缓存不上传GitHub。SHA256SUMS的路径相对于仓库根目录。

官方TextVQA Answer Processor与十答案leave-one-out独立实现逐条重算，源码来自冻结七Benchmark评分包。元回答检测不影响官方得分；全体异常原样保留。数值状态为探索性豁免，严禁据此宣称Dense/Physical数值等价。未来即使扩样，也不得根据本轮Holdout重新挑层、改Prompt或混合生成上限。
