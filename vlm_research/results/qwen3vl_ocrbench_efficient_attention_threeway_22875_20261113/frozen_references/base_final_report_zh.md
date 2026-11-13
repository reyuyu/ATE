# Qwen3-VL-8B OCRBench 内存高效 Attention 验收

状态：**COMPLETE_WITH_NUMERICAL_WARNING**。新后端 Native 与 Physical-Original 均完成原1000题，官方评分和独立复算逐题一致；旧Eager OOM和旧数值失败保持不变。

**质量结论：SVH严重退化，不能视为质量恢复。** Native为868分，Physical-Original仅38分，下降830分；其中833题由对变错、3题由错变对。结构检查通过和OOM修复，不等于质量可用或严格数值等价。

| 条件 | 得分 /1000 | 覆盖 | 空输出 | 截断 | 自然EOS | 峰值分配 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Native | 868 | 1000/1000 | 0 | 0 | 1000 | 21.109 |
| Physical-Original | 38 | 1000/1000 | 0 | 88 | 912 | 20.003 |

SVH−Native：-830 分；配对改善3题，恶化833题。截断和错误输出全部保留，没有补零或跳题。

## OOM 是否解决

0738 原图4961×7016、grid=[1,304,214]，65056个Vision Patch合并为16264个LM图像Token，总长16307，四项Tensor SHA与旧冻结布局一致。旧Vision Eager的[1,16,65056,65056] BF16 Scores理论126.131866 GiB；语言Eager Scores约15.850 GiB，FP32 Softmax约31.700 GiB。新后端不物化这些Scores，Vision与Language均仅启用SDPA_EFFICIENT_ATTENTION。

Profiler实测aten::_scaled_dot_product_efficient_attention及CUTLASS fmha_cutlassF_bf16_aligned_64x128_rf_sm80 CUDA Kernel。MATH、FLASH、CUDNN全部关闭，三种兼容性探针与官方HF显式KV分组展开实现逐位一致。默认Processor、图像预算、模型和生成上限没有改动。

| 0738 条件 | 输出 | Vision峰值 GiB | Prefill峰值 GiB | Live KV MiB | Storage Capacity MiB |
|---|---|---:|---:|---:|---:|
| Native | UNIHAKKA INTERNATIONAL SDN BHD | 20.003 | 21.109 | 2294.438 | 2294.438 |
| Physical-Original | company that issued this receipt | 20.003 | 19.498 | 5.625 | 19.922 |

## 物理语义与数值限制

36层、R读取Image L0–L21、进入L22前物理删除视觉Hidden、L1–L6答案Attention真实跳算、P−1与Decode禁止原始Image Key、三次DeepStack及原mRoPE逻辑位置均保留。L22后Q/K/V与MLP处理文本行；30个非跳算层持久KV只保留文本和已生成Token，6个跳算层无持久KV。实际KV Live与底层Storage Capacity均逐题保存并复核。没有Dense Key补零或Math回退。

仅新增隔离Attention适配和Mask存储执行器；原冻结Execution.context、CompactKV及原始36个源码文件未修改。把三份大Mask改为一份对齐存储Mask，L22后使用小型紧凑文本Mask。独立CPU216层Mask值精确一致；FP64显式Softmax测试的紧凑/跳算误差≤4.45e-16。短题与未修改冻结Physical在同一SDPA后端的有效Logits逐位一致。

**严格数值门禁仍FAIL**：Physical对同后端Dense-F的max_abs分别0.28125、0.5、0.75；最高单步mean_abs分别0.049899、0.049799、0.167922，超过原0.02门槛。门槛未修改，也未将历史NUMERICAL_EQUIVALENCE_FAILED改成PASS。三题共14步Greedy argmax（包括EOS）均一致，结构和功能Key集合检查全部通过，因此按本任务允许浮点差异的条款，以单独的带数值警告状态继续正式质量评测。不能据此宣称全局数值等价或与Eager逐题输出相同。

Dense-F仅为同前缀功能诊断，未计入正式预测。旧Eager与新SDPA的六条重叠输出比较见old_eager_output_comparison.json；旧有效Logits未保存，不能虚构误差。Native与SVH是不同算法，不要求相同输出。

## 协议核对

| 项目 | 本轮核验 |
|---|---|
| 模型 | Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，复用已验证下载回执及文件大小；小配置/Tokenizer Git Blob哈希复核 |
| 输入 | 原1000题/930张图，原Question/Gold/顺序和SHA；所有input_ids、attention_mask、pixel_values、image_grid_thw哈希与旧布局完全一致 |
| Processor/Prompt | 原默认全分辨率、原Chat Template及Short Answer后缀；不设新max_pixels/token预算 |
| 生成 | BF16，Batch1，Greedy，自然EOS，上限128，logits_to_keep=1 |
| 软件/GPU | PyTorch2.6.0+cu124，Transformers4.57.6，RTX3090 24GB；TF32关闭，原单线程设置 |
| Attention唯一变化 | Vision/Text统一PyTorch SDPA Efficient；精确KV分组展开及Mask stride对齐，不升级依赖 |
| 评分 | 冻结OCRBench v1官方AST及独立注释逻辑逐题2000/2000一致 |
| 历史保护 | 原预测、协议、失败记录、checkpoint及冻结36源码未修改；旧Eager预测零混入 |

## 保存与资源

模型、图片、环境、推理及原始逐题文件在/home/featurize本地ext4盘。持续盘本任务目录保存配置、报告和压缩逐题checkpoint，未放模型或图片。新后端Native/SVH各3条Smoke预测已直接复用，正式新生成各997条；模型仅加载一次。

完整逐题原始记录位于predictions/；轻量逐题输出、逐题独立评分表、类别汇总和所有阶段显存CSV同时保存。CPU审计未初始化CUDA，GPU进程已退出，释放证据见gpu_release.json。Windows备份及最终压缩包SHA将在归档回执中另行核验。

本任务未修改论文、未创建GitHub提交/分支/PR、未Push、未关闭服务器。

请求计数：正式质量预测共2000条，包含复用的6条新后端Gate预测；另有4次功能诊断（3次同前缀Dense-F、1次未修改冻结Physical短题）未计入质量结果。3种后端兼容性探针不是模型生成请求。

代码身份：工作区HEAD `e281754ef17b3accc09d6938b5b9f020048b2aec`；冻结Qwen研究源码Commit `7e18895a6ccd57c4cb3aaedd5d512789e7467b59`；历史Eager证据Commit `a4fb52e3aa4d57385bd1108832c58a28471ecea3`。新执行器是独立、未提交的源码，精确SHA见两个source_binding、run_metadata及SOURCE_SHA256清单，不声称其已有GitHub Commit。

## 负结果的离线复核与归因限制

两条件1000题四项输入Tensor哈希逐题相同，Gold匹配且官方与独立评分逐题一致，所有文本与原Token IDs解码一致；未发现输入或评分映射错误。Native/SVH共同正确35题、Native单独正确833题、SVH单独正确3题、共同错误129题。重复输出及固定诊断例见prediction_pattern_audit.json。

三题Physical与同后端Dense-F参考的Greedy相同，短题与未修改冻结Physical的全部有效Logits逐位一致，支持已检查输入上的适配没有改变功能定义。但严格数值门禁失败，全量Dense-F质量没有运行，因此不能把830分损失完全归因于某一因素，也不能宣称已排除全部数值影响。此处原因仍UNRESOLVED；本轮不追加GPU生成、不修改层号或Prompt追分。

各阶段耗时包含验收检查，Gate另含Profiler开销，不作为正式效率或TPOT对比成绩。
