# ATE 跨服务器交接：当前状态与统计实验恢复

交接日期：2026-10-28。仓库：[reyuyu/ATE](https://github.com/reyuyu/ATE)。当前实验代码基线：[`bcc3c3da2ff7d416cd988c83264fd3ffc18afaed`](https://github.com/reyuyu/ATE/commit/bcc3c3da2ff7d416cd988c83264fd3ffc18afaed)。初始交接提交7970be9只新增恢复资料；后续2026-10-28已完成单案例六条件GPU实验，结果在独立`completed_20261028_2245/`，旧指纹清单保持不变。

**先读本页 → [数据/模型下载与跨机路径适配](DATASET_SETUP.md) → [可直接发给新服务器 Codex 的提示词](CODEX_PROMPT.md)。** 不依赖旧聊天、旧机知识库或私钥。原图需要在新机下载完整GQA目录，不能仅拿案例图片替代统计数据。当前单案例六条件 GPU 实验已完成；历史统计实验已有真实结果，不需要全部重跑。

## 1. 项目目标与已经完成的阶段

研究LLaVA视觉Attention、视觉Token物理退出与答案/问题表征变化。只作可验证的实验观察，不宣称新退出算法、通用安全退出层或已证明Attention Sink导致性能退化。

| 阶段 | 状态/数据范围 | 首选入口或报告（相对vlm_research） |
| --- | --- | --- |
| Phase1 | 单卡FP16推理、环境/结构/实测完成 | `vlm_common.py`；`results/phase1_report.md` |
| Phase2 /2.1 | 32层Attention与真实Q/答案预测Token对齐完成 | `attention_probe.py`、`attention_position_probe.py`；对应results报告 |
| Phase3 | 街景及镜像9条QA全局Mask扫描完成；不是9张独立图 | `knockout_probe.py`；`results/phase3_knockout/phase3_report.md` |
| Phase3.1 | 12张GQA，Full/Mask/Physical的CPU/GPU校验及pilot完成 | `physical_exit.py`、`physical_exit_pilot.py`；`results/phase3_1_physical_exit_pilot/report.md` |
| Phase3.2 | 新增240独立图、8K真实Physical配对完成，1920行 | `fixed_k_exit.py`；`results/phase3_2_fixed_k_exit/report.md` |
| Phase3.3 | 同240图定向后缀Mask完成，3120行 | `directed_suffix.py`、`directed_suffix_probe.py`；`results/phase3_3_directed_suffix/report.md` |
| Phase3.4 | 240图ROI审计；48主图+5独立历史案例观察及小规模因果诊断完成 | `roi_sink_common.py`、`roi_sink_probe.py`；`results/phase3_4_roi_sink_diagnostic/report.md` |
| Phase3.5 | 缓存离线边界统计与两案例全32层可视化完成 | `exit_boundary_analysis.py`、`two_case_attention_viz.py`；`results/phase3_5_exit_boundary_viz/`及`phase3_5_two_case_visualization/` |
| 单案例Attention继承 | **完成**。101项CPU回归、六条件7B及96层审计通过 | `attention_inheritance.py`、`attention_inheritance_pilot.py`；`results/attention_inheritance_15133706/completed_20261028_2245/report.md` |

旧阶段commit与源码指纹在各结果配置/知识文献记录中。代码和历史结果只读；不要重复运行准备脚本来“恢复”已经冻结的样本。

## 2. 单案例GPU实验：已完成

只研究 `question_id="15133706"`，`img_id="2324597"`，Girl/Boy。

- Question：`Is the young girl to the left or to the right of the boy?`
- Gold：`right`，候选顺序固定`[left,right]`，各为单Token；不加EOS评分。
- Prompt长度603，Teacher-forcing长度604；目标Query **P−1=602**，候选Token位置603不能作为预测Query。
- 576视觉Token精确位于原始输入位置5～580，由`input_ids == image_token_index`定位，不靠猜。
- 在L24～L31，仅替换P−1→Image的相对分布。逐Head继承**原始Full**的L23或L24，λ=`[0,0.5,1]`，共六条件；不是层层动态取得参考。
- 每Head：`m=sum(a_current_visual)`，`p_ref=a_full_source/sum(a_full_source)`，替换`m*((1−λ)*a_current_visual/m + λ*p_ref)`。只在Softmax后、乘当前层V前修改；保留当前视觉Mass、非视觉权重和其他Query。λ0跳过写入。
- 保持FP16/eager、batch1、eval/inference_mode、use_cache=False。无训练、Decode、KV Cache、量化或Physical删除。

[完整报告](../results/attention_inheritance_15133706/completed_20261028_2245/report.md) · [图](../results/attention_inheritance_15133706/completed_20261028_2245/margin_comparison.png) · [验收](../results/attention_inheritance_15133706/completed_20261028_2245/validation.json)。以下六条件来自同一次完成运行，非拼接旧尝试。

| 条件 | Margin=logP(right)−logP(left) | 结果 |
| --- | ---: | --- |
| Native Full | −0.03125000 | 与历史精确一致，left错误 |
| L23 λ0 / .5 / 1 | −0.03125000 / +0.02343750 / +0.08593750 | 非零λ转为right |
| L24 λ0 / .5 / 1 | −0.03125000 / −0.03125000 / −0.03906250 | 未转正 |
| Physical K24 | +0.03906250 | **Phase3.2历史**，本轮未重跑 |

两source λ0全词表logits精确Full；所有条件Question最终hidden不变、非目标Attention局部bitwise不变；结束后原生Full精确，库源码hash未改。峰值allocated13640.731MiB/reserved13734MiB；15次诊断forward合计2.743861秒，不能当稳定速度基准。L23非零λ提高二候选相对Margin，但right绝对概率下降，单个FP16弱Margin案例不证明一般能力或Sink/Physical机制。

历史首次L23 λ.5在L25因低Mass下FP16条件L1超过固定.002阈值停止，后续曾被显存资源门槛阻止；旧失败/partial完整保留。公式不变，校验改为逐元素等于公式FP16舍入值及dtype推导界：`u=eps/2,d=tiny*eps,E=uΣ|x|+576d/2; L1<=2E/Σround(x)+2e−6`；.002保留告警，行和及Mass容差不变。**校正版现已通过真实7B全部六条件**；8层审计超过旧告警，全部通过推导界。旧根`report.md/validation.json/CSV`只表示历史partial，不是当前完成状态；不要运行partial汇总器覆盖完成结果。

`VisualAttentionInheritance`用模型实例私有forward/globals包装，finally恢复，无全局softmax替换、无第三方源码修改。局部“非目标不变”指**该层编辑前后**bitwise相同，非指后层权重始终等于Full；跨层Head编号对应也不证明语义对应。

## 3. 数据、缓存和权重：Git中有什么

**Git已提供**：全部Python入口/测试、requirements、模型文件哈希、官方CSV副本、冻结Manifest、逐样本checkpoint/CSV/JSON、ROI/Sink定义、53图Attention缓存及已有图表。正常完整clone约数百MB；不要sparse clone漏掉NPZ/检查点。

**Git未提供**：GQA原始图片、14.1GB模型权重、Conda环境、私钥。按[下载说明](DATASET_SETUP.md)恢复MMEB发行的完整GQA子目录（不是GQA全基准全集）、固定模型，并校验；只下载图片任务archive，不需要视频/训练数据。

`handoff/artifact_manifest.json`记录交接时全部已跟踪Python源码、配置和结果数据（JSON/CSV/NPZ/gzip/TXT）的SHA256，不含PNG/PDF或交接文件本身。`verify_recovery.py`只用CPU校验这些文件，以及可选的252张原图/模型。初始交接没有新增推理；此后的完成子目录是新增真实结果，不在旧指纹清单中，旧清单不重写。工具实测见`handoff/validation.json`：1143文件指纹、252图历史SHA/尺寸、源CSV785图解码/尺寸、15个模型文件/三片权重SHA校验通过；缺图、覆盖旧审计及缺少必需参数的失败路径也已验证。源785中另外533图无历史哈希基准，不能当作已有785图模型实验。

## 4. 统计实验必须继承的数据划分与协议

### 固定队列

1. `results/phase3_2_fixed_k_exit/manifest.json`：240张、left/right各120、每图一个问题；种子20261029；排除Phase3.1的12图。已经冻结，不重采样。其`validation_samples`为原12图，恢复图片共252张。
2. `results/phase3_4_roi_sink_diagnostic/manifest.json`：`main_question_ids`固定48图、各标签24；`case_question_ids`固定5图。**案例不能混入48主统计**，且不能把48样本视为完整240图Attention覆盖。
3. 原始ChooseRel源CSV有964题/785图，带上游`correct_question`筛选，不代表无偏GQA全集。已有240图统计不等于全GQA评测。
4. 五历史案例：`04132006`、`15133706`是K24原错变对；`0216546`、`02310173`、`05480434`是K16原对变错。题号必须字符串保留前导零。

### 原始分数与干预定义

- Full vs Physical：`K=[0,8,12,16,20,24,28,32]`。K24=进入L24之前删除视觉，最后视觉计算层L23；真实隐藏状态减少576，后层保留原逻辑位置/RoPE。
- Phase3.3：Q-only、A-only、Q+A、Global，K16/24/32后缀Mask；A-query是P−1，不是答案位置。Mask不删除Token、不属于加速。
- 评分：完整候选序列全词表log-softmax之和，**不加EOS**；当前left/right为单Token，多Token实现不要退化为首Token。`Margin=logP(gold)−logP(wrong)`，`Drop=Full−Intervened`，`Gain=Intervened−Full=−Drop`。
- 严格`Margin>0`才正确，**tie算错**。Full-correct→wrong用Full>0且干预<=0，单独记录正→负和正→零；不能只数净准确率。
- Q Drift使用真实问题Token的最终RMSNorm hidden逐Token Cosine Distance；不能直接解释为语义失败。
- Full概率极低与二候选归一化概率是不同量；二选一准确率不等于自由生成GQA准确率。

### 缓存与ROI口径

- `attention_rows/<qid>_task.npz`：`attention`为FP16 **[32 layers,32 heads,2 query groups,576 patches]**；group0=每Head真实Question行的均值；group1=P−1行。只存选定行，没有完整Attention矩阵。
- `*_neutral.npz`和`sink_manifest.json`保存中性提示稳定热点定义/候选与随机控制；不用答案或结果重设候选。ROI交集独立记录；因果Sink条件只取非ROI候选。
- `sample_layer_attention.csv`、`layer_attention_summary.csv`及压缩Head CSV是统计复用入口。32层真实CLIP坐标为336×336 resize+center crop后24×24行优先；复用已验证`geometry`和ROI Patch集合，不能把原图框直接映射Patch。
- Phase3.4/Phase3.5统计**逐Head条件化再平均**；后来的两案例全景热图**先Head平均再条件化**。这是非线性聚合差异，不能把两种Enrichment/Entropy数字混成同一个指标。当前继承干预必须使用逐Head参考，不能输入Head均值图。
- ROI是边界框而非分割；背景不等于无关，热点不自动等于有害Sink；条件化图需要同时看原始Visual Mass，尤其低Mass/FP16敏感性。

### 现有真实统计作为恢复自检

| 条件 | 正确数/240 | Full原对→错 | Full原错→对 |
| --- | ---: | ---: | ---: |
| Full | 187 | 0 | 0 |
| Physical K16 | 187 | 3 | 3 |
| Physical K20 | 186 | 1 | 0 |
| Physical K24 | 189 | 0 | 2 |

Full left115/120、right72/120，明显左右偏好；K0/K8全部预测left，不能解释为视觉已不必要。K24两恢复例Margin很弱，不能由微小翻转证明机制。Phase3.3 K16 Q/A/Q+A/Global正确数分别189/184/188/186；历史Global与Physical存在近零Margin精度例外，读原correctness报告，不强行抹平。

Phase3.4主48的Full=33/48。中层Q→ROI有面积富集及相对匹配背景更大Margin损伤；深层SinkMask没有主样本准确率收益，也没有恢复K24两案例。Phase3.5不支持统一“更好的退出边界刚好ROI下降”规律，15133706 L23→L24的Q/A ROI反而上升。描述性相关、定向Mask效应、Physical整体效应不可互相替代。

## 5. 新服务器恢复顺序（单案已完成，不默认重跑）

1. clone仓库，确认包含bcc3c3d及当前交接提交；先读原报告/源码，再按数据页下载完整GQA目录和固定模型。
2. 运行`verify_recovery.py`，保存新机审计。通过不代表GPU实验已完成。
3. 激活/按固定版本建立环境；独立路径适配解决图片、模型、CPU解释器三处旧绝对路径。原文件/Manifest不改，适配器保存路径映射/源码hash和新机器环境。
4. 先跑7项新增CPU测试：`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest test_attention_inheritance -v`。
5. 全工程101项测试含真实Processor依赖，若模型目录不同，在**独立测试launcher中**先把`vlm_common.default_model`映射到实际同revision目录，再discover；不要为过测试重写历史配置。若依赖缺失，逐项列明，不能谎称全套通过。
6. GPU手动预检：RTX4090/24GB为原基线；至少16000MiB空闲门槛。只暴露一张卡；物理GPU号可不同，通过`CUDA_VISIBLE_DEVICES`映射到cuda:0。不终止其他进程。
7. 先读取已完成单案结果；只在用户明确要求独立复验时运行同案例的Native Full、两source×三λ、末尾Full恢复检查。使用全新目录，如`results/attention_inheritance_15133706/remote_<server>_<timestamp>/`；不要使用默认旧根目录。
8. 当前runner是**逐条件落盘但不支持从已有配置目录自动续算**；它会拒绝覆盖。失败保留已完成数据，下次用新目录重新执行这个案例。不能把原1条件与新机结果无条件拼成完整六条件。
9. 如果不同硬件使历史Full/cache严格bitwise检查失败，保留差异与机器环境，先调查、单独做跨机数值审计；不删除断言/改阈值来追求转正。新机λ0必须与**同机Native Full**精确一致，跨机历史一致性另报告，不能混为一项。
10. 完成后保存六条件CSV/JSON、配置、rawlogits、各层审计、峰值allocated/reserved与真实耗时、对比图/report，更新索引、提交并push，核验远端commit与文件。未完成必须明确partial、原因与恢复方式。

旧机完全相同路径环境的原始命令（**新机须先做路径适配**）：

```bash
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python attention_inheritance_pilot.py \
  --output-dir results/attention_inheritance_15133706/completed_retry
```

注意`.gitignore`忽略大部分`*.log`；提交新日志用`.txt`或仅对该运行的已检查日志`git add -f`，不要广泛强制加入缓存/权重/凭据。

## 6. 如何继续统计性工作，而不是只接手这个案例

**先复用已有结果做离线统计，不重扫7B来重建已存在数据**。按question_id一对一join；每qid每条件唯一，校验共同分母、左右标签、输入/Prompt/候选一致。数据入口：

- `results/phase3_2_fixed_k_exit/per_sample_results.csv`、`summary_by_k.csv/json`、`checkpoints/run/<qid>.json`：240×8 Physical分数、完整候选logP、逐TokenQ drift/层长度。
- `results/phase3_3_directed_suffix/per_sample_results.csv`、`summary_by_condition.csv/json`和检查点：240图定向路径条件。
- `results/phase3_4_roi_sink_diagnostic/manifest.json`、`sink_manifest.json`、`sample_layer_attention.csv`、`attention_rows/`：48主图/5case的Attention与ROI/Sink；`per_sample_results.csv`及paired summary为因果Mask数据。
- `results/phase3_5_exit_boundary_viz/boundary_changes.csv`、`fixed_boundary_summary.csv`、`best_observed_*.csv`及`all_eight_k_scores.csv`：固定K边界与事后稀疏网格最佳K，二者分开。

只核验已生成离线结果、不覆盖历史：

```bash
CUDA_VISIBLE_DEVICES='' python exit_boundary_analysis.py --verify-existing
CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py --verify-existing
```

后续用户要求扩大继承实验时，数据已可恢复，但**尚未实现240图继承runner、也没有240图继承分数**。单案正确性已通过；收到扩展授权后，在新脚本/新目录冻结协议：预先固定同240图或同48主图、source/λ/层范围、同图Full配对与量化容差；不要按单例正向结果选择有利图/Head/λ。新配置需绑定原Manifest、图片hash、环境、源码、权重及条件网格，检查断点兼容、去重、完整性。

统计最少保留：总体及left/right正确率、准确率变化、Full-correct保持率、原对→错/原错→对/tie、Margin Drop mean/median/P90/P95/max及正Drop比例、Q drift、低Mass敏感性、真实算力开销。Bootstrap按**图**重采样并保持所有条件配对，不把32Head或32层当作独立样本。原对/原错只作预定义子集，不能只报Full-correct；案例只单列。多个探索性条件的区间需说明未/如何多重比较校正；平均改善不掩盖尾部退化。

本交接是让后续统计实验可继续，不代表本轮已经授权或执行240图新干预。单案例现已完成，本轮到此停止；之后遵循用户在新会话给出的统计目标。无Router、Gate、GRPO、训练、Decode或其他数据集，除非用户另有明确指令。
