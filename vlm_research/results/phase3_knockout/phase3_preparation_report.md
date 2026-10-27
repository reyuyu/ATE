# Phase 3：Layer-wise Visual Attention Knockout 进展报告

日期：2026-10-27（Asia/Shanghai）。**状态：实现与 CPU 正确性测试完成，7B GPU 科研实验受显存占用阻塞；本阶段验收尚未完成。**

## 已完成与未完成

| 项目 | 状态 |
| --- | --- |
| 检查实际 Transformers 4.49.0 LLaMA eager 源码与 Mask | 完成，源码 SHA256/forward 签名保存在 input_audit.json |
| 全局后图像文本 Query→视觉 Key Mask Hook | 完成，独立克隆、不改参数/位置/第三方源码 |
| 单层→32 层 K=32/K=0/K=16、因果/归一化/清理检查 | CPU 小型随机 LLaVA 通过；7B GPU 待运行 |
| Teacher-forced 答案 NLL、逐 Token log P、KL、完整候选序列评分 | 已实现并用解析数据和随机模型测试 |
| 真实 Processor、576 视觉位置、原 Phase 2.1 输入一致性 | CPU 审计通过，共 9 条样本 |
| 7B 单图全部 cutoff sweep | 未执行，显存不足，未尝试加载权重 |
| 5–10 条补充样本的 7B 测试 | 已准备 7 条；GPU Forward 未执行 |
| 真实 NLL/关键 Token/候选 Margin/敏感性图 | 待 GPU 结果；没有保存伪造曲线 |
| 候选 Visual Fusion Saturation Depth | 无可用实测结论 |

## 资源阻塞与环境

环境保留 Python 3.11.18、PyTorch 2.6.0+cu124、Transformers 4.49.0、CUDA runtime 12.4、驱动 595.99.02；完整依赖快照见 `environment_snapshot.json`。没有创建环境、下载权重、升级包、量化或 CPU offload。

目标模型仍为 `llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，7,063,427,072 参数，预定完整加载至单卡 FP16/eager。

`resource_preflight.json` 记录本轮实际检查：GPU 0 被 PID 2265913 占用约 17070 MiB，GPU 1 被 PID 286617 占用约 20012 MiB。驱动报告 free 约 6928 / 3920 MiB；进程内 CUDA 查询 GPU 0 可用约 6536 MiB，口径受驱动预留和 CUDA 上下文影响。均不足以容纳约 13.5 GiB 参数及验证工作区。程序要求至少 16000 MiB free，在加载前以退出码 2 停止；没有重复制造 OOM，没有操作其他任务。

因此**没有本轮 7B GPU Forward 时间、峰值显存、NLL 或概率变化数值**。第一/二/2.1 阶段历史性能不能冒充 Phase 3 性能。

## 实现定义与正确性

`knockout_probe.py` 复用未修改的 `load_model()`、`prepare_inputs()`、Phase 2.1 的 `locate_question_tokens()`、`encode_answer()` 与 `append_answer_inputs()`。

本机 `LlamaDecoderLayer` 使用 keyword 调用 `LlamaAttention.forward(hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs)`。eager 先将 additive mask 加入 QK 分数，再以 float32 softmax 转回 Query dtype。本实验只给语言模型各层 self_attn 安装 `register_forward_pre_hook(with_kwargs=True)`。

- 从实际 IDs 定位完整连续的 576 个视觉位置，Query 为最后一个视觉位置之后全部有效非视觉 Token；包含 Question、后续模板、Answer-start、已输入答案。
- 层索引 ≥ K 时克隆原始 `[1,1,S,S]` Mask，只将原合法的 Text→Visual 连接置为 `-inf`，使 softmax 后权重严格为零；其他 Mask 元素逐元素保持相同。其余 Attention 概率允许重新分配。
- 不物理删除 Token，不改 position_ids/cache_position，不改视觉 Query 的连接；拒绝 cache、Decode、非 eager、异常 Mask 和任何全屏蔽 Query Row。
- context manager 在正常或异常退出时移除所有本次 Hook。K=32 在全部层安装 no-op Hook，对照无 Hook Baseline。没有修改模型参数或第三方源码。
- 正确性验证先单层，再 K=32/0/16，检查所有层所有行有限/非负、行和误差≤0.002、未来与 padding 权重零、阻断权重零、视觉及此前 Query 的 Attention 哈希不变、cutoff 前层全部 Attention 不变，最后确认 Hook 清理和 output_attentions 开关没有改变 baseline logits。
- Attention 只在验证时返回，逐层审计后释放；Transformers ModelOutput 的 dict 与 attribute 引用一起清除，避免只清 attribute 留住完整矩阵。磁盘仅存检查摘要与哈希，不存完整矩阵。

`test_knockout.py` 在当前安装的真实 Transformers LLaVA/CLIP/LLaMA 实现上构造 **32 层、4 Query Heads/2 KV Heads、4 个视觉 Patch、hidden=32 的随机 CPU 模型**，用于验证同一 Hook 代码；另有 FP16 CPU eager 测试。其词表、权重、视觉网格和计算设备都不等于 7B 模型，不能把测试预测当作实验结果。CPU 检查记录见 `cpu_correctness_validation.json`；K=32 最大 logits 绝对误差为 0。

17 项新增测试覆盖共享 Mask 不被修改、位置从输入推导、合法连接保护、非法输入/后端/cache 拒绝、异常清理、参数/输入不变、FP16 softmax 零权重/归一化、逐 Token 偏移、KL 方向、多 Token 候选完整评分、真实 Tokenizer、完整输出管线及绘图。测试绘图使用临时目录内明确的 fixture，结束后删除，不进入科研结果目录。加上原阶段测试共 34 项，实测日志见 `unit_test_results.txt`。

## 位置与统计定义

所有 cutoff K=[0,8,12,16,20,24,28,32] 对同一样本使用完全相同的图片 Tensor、Prompt、参考答案和长度；不调用自由生成、不使用 KV Cache。

对于答案 a_i，读取 q=P−1+i 的 logits，目标位置为 P+i。颜色样本 P=595，完整长度 601；`red` 目标仍为 599，其预测 Query=598、输入为 `is`，不使用已输入 red 的 q599。描述样本 P=594，完整长度 644；两者全部 teacher-forced IDs 与 Phase 2.1 保存文件逐项一致。

NLL 同时保存 sum 和 mean，仅参考答案 Token，不包括 Prompt/EOS；ΔNLL=KO−Full。每个答案位置在完整词表上计算 **KL(Full || Knockout)**，采用 float64 CPU log-softmax 汇总 GPU 返回的 FP16 logits，避免直接概率取对数的下溢。

候选答案在完整替换文本中重新分词，确认候选之前的条件 Token IDs 完全相同。允许 SentencePiece 首 Token offset 含前导空格，拒绝跨越实际词边界。`light blue` 等多 Token 候选必须累加整段条件 log P，而不是首 Token；候选后缀不计分。所有候选右 padding 到共同长度并遮蔽 padding；不同 K 保持这些输入不变。该候选专用长度可能与主参考答案不同，必须在同一候选控制中比较。记录候选 Token 数、joint/mean log P，以及正确候选减最强竞争者的 log P Margin；joint 分数本身存在长度偏好。

## 小样本及复核来源

用户要求新增样本人工核验，随后明确回复“你自己确认吧”。按这一指示，由 AI 目视查看原始图与真实 CLIP 336×336 输入；新增样本记录 `annotation_source=assistant_visual_review_user_delegated`、`human_verified=false`。**未宣称有人类逐条标注。**

| ID | 类型 | 参考答案 | 来源/限制 |
| --- | --- | --- | --- |
| street_color | 颜色 | The stop sign is red. | 原用户参考；STOP 常红，语言先验强 |
| street_description | 描述 | 原第一阶段完整描述 | 原模型生成轨迹；fire hydrant 未证实，非真值 |
| street_animal | 物体 | A lion. | AI 目视复核石狮 |
| street_letters_color | 颜色属性 | White. | STOP 白色字母 |
| street_ocr | OCR | STOP. | 大交通牌；自然语言先验仍强 |
| street_spatial_left | 空间 | To the left. | 原图 STOP 在汽车左侧 |
| street_car_color | 颜色属性 | Black. | 黑/深色车身有蓝色反光，标签歧义须保留 |
| street_sign_shape | 形状 | An octagon. | 原图八角形；CLIP crop 截去最左边缘 |
| street_spatial_right_mirrored | 空间配对 | To the right. | 原图水平镜像，同问题、反向答案；文字也镜像 |

共 2 条原任务 + 7 条新增 QA；只有 1 张独立自然图和 1 张镜像派生图，不能按 9 张独立图解释。尚无颜色不同且控制良好的自然配对图。镜像只用于空间关系，不作为 OCR 样本。配置为 `configs/phase3_samples.json`，真实输入审计为 `input_audit.json`，实际 CLIP 图像为 `image_review/`。`test_images/phase3/SOURCE.md` 记录派生方法，原图和旧结果未修改。

## 继续运行

先确认 GPU 空闲；不必创建环境、下载或升级。

```bash
source /anaconda3/etc/profile.d/conda.sh
conda activate vlm_research
cd /home/zsy/ATE/vlm_research

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --preflight-only

# 先校验并完成原颜色问题的全部 cutoff
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --sample-ids street_color \
  --output-dir ./results/phase3_knockout_color_run

# 一次加载跑完颜色、描述与 7 条补充 QA
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --output-dir ./results/phase3_knockout_gpu_run
```

默认首样本通过完整 Hook/Mask 检查后才开始该样本正式 sweep；后续样本逐 cutoff 保存各层 Hook 审计、有限 logits、K=32 等价性。已有 run.json/samples 的目录拒绝覆盖，复跑/部分失败改用新目录。每样本保存 CSV/JSON 和实际曲线，运行全部完成后自动写该 GPU 运行目录的 phase3_report.md。离线 `--replot-only` 不加载模型。

## 当前能与不能下的结论

目前只能确认干预实现通过 CPU 正确性测试、真实输入位置审计通过；没有测出 7B 模型的预测变化。不同 K 的 NLL 是否稳定、普通词与视觉词是否不同、低 Attention Mass 是否仍有因果影响、是否出现反向改善或候选饱和深度，全部待 GPU 实测。

即使后续某个 K 的 NLL 接近 Full，也仅说明当前样本和固定答案历史下直接视觉访问的边际作用有限。Knockout 同时导致 softmax 重分配与分布外输入连接，不能直接解释为纯粹、唯一的视觉因果贡献，更不能证明文本完整吸收视觉信息、安全删 Token 或推理加速。

停在 Phase 3 资源阻塞状态；未开展 Question-only/Answer-only Knockout、Decode 干预、Token Pruning、Router 或训练。
