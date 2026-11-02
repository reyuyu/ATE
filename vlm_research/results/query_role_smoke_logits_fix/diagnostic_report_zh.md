# Query-Role Smoke：严格计算路径修复完成

**结论：根因已隔离，16项CPU测试、3/3道冻结GPU Smoke通过。正式实验仍为0/960，没有产生I/R/A机制结论。本轮停止。**

分支：`codex/query-role-smoke-logits-fix`。基于Partial Commit
`094e1d7916d1e18351bffaf950c68280ba29bc85`，未修改main。
实际执行源码Commit：`382a94ed01d5c2c8b060b4b43bf0529026cb9310`。
最终交付Commit以分支Git记录为准。

## 1. 为什么Logits不一致？

**确实由LM Head计算形状不同造成，已用同一个真实Hidden Tensor隔离复现，不是根据FP16作出的猜测。**

对安装的Transformers **4.49.0** 源码和实际Forward参数进行了核验：

1. `generation/utils.py:2073`检查Forward是否支持`logits_to_keep`，默认设置为1。
2. LLaVA的`prepare_inputs_for_generation`将该参数传给语言模型准备逻辑，LLaVA Forward再将其传给Llama。
3. `models/llama/modeling_llama.py:858`先切Hidden States，再执行`lm_head`，不是计算后切Logits。
4. 原自定义生成器没有传该参数，采用默认0，Prefill计算全部位置。

冻结第1题`00150746`，Prompt长度606，词表大小32064：

|路径|LM Head输入shape|返回Logits shape|
|---|---|---|
|HF原生|1×1×4096|1×1×32064|
|旧自定义Full|1×606×4096|1×606×32064|
|修复后Full|1×1×4096|1×1×32064|

比较两条旧路径的末位置：

- LM Head前的4096维Hidden **逐位相同**。
- 首次Logits差异发生在 **t=0 / Prefill**，11978/32064个元素不同。
- 最大绝对误差 **0.0078125**，平均绝对误差 **0.0004199991809453317**。
- 两者Top-1均为Token **19941**，完整生成Token均为`[19941, 2]`。
- t=1 Decode的实际矩阵形状相同，Logits也逐位相同。

进一步只重放LM Head，不再次执行Decoder：把旧Full捕获的**同一块Hidden数据**分别按全部606行和末1行投影，分别精确复现旧Full和HF的Logits，且复现上述差异。这将本题差异隔离到LM Head矩阵计算形状，而非Attention Hook、输入、RoPE或Cache数值。

实际原生/自定义Forward关键参数、Cache初始状态、位置、Mask、输入哈希与每步数值见`logits_comparison.json`和`comparisons/`。原生Prefill显式传入位置和空DynamicCache，自定义由原生Forward推导位置并从None开始；不将这些参数表示形式说成完全相同。其末Hidden及匹配形状后的Logits均严格相同。

## 2. 修复方式与严格对照

唯一生成器改动：新增`logits_to_keep`参数，默认1，**显式传入模型Forward**；诊断时允许0以保留旧路径。没有修改模型、FP16、Eager、Prompt、Tokenizer、图片、角色定义、L2–L5、评分或第三方源码。

所有同形状判定均使用`torch.equal`，不使用`allclose`替代，不新增或放宽容差。

第1题诊断通过：

- HF原生与修复Full的每步有效Logits、完整Token精确一致。
- `logits_to_keep=1`下Full重复、Full/No-op全部返回的有效Logits精确一致。
- `logits_to_keep=0`下Full重复、Full/No-op的**全部606个Prefill位置及Decode Logits**精确一致。
- 跨形状差异单独作为诊断记录，不将其当作同形状一致性检查。

所有比较先保存JSON，再执行严格断言；不同向量另保存在本地诊断数组中。没有采集完整Attention矩阵。

## 3. 三道冻结GPU Smoke

|Question ID|任务|Prompt长度|结果|
|---|---|---:|---|
|00150746|QueryAttr|606|PASS|
|11697551|ChooseRel|624|PASS|
|11127723|CompareAttr|611|PASS|

每题均验证：HF/Full逐位一致、完整Token一致、Full重复一致、I/R/A三个No-op一致；真实I/R/A Hook的目标输出严格零、非目标行没有被直接修改；只注册L2–L5输出消融；32层MLP正常执行且输出有限；Cache各层长度和真实绝对位置正确；各条件Cache不共享存储且不改变保留的Full Cache；输入哈希与参数版本不变；正常和异常路径Hook均清理。

三题都正常EOS结束，无NaN、Inf、OOM、空答案或截断。真实角色Smoke每个条件生成长度均为2（一个内容Token及EOS），因此**长Decode及240题总体尚未验证**，不能把本轮正确性检查说成正式结果。

## 4. 数据、模型与GPU

- 原240题Manifest字节及SHA256不变：`258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c`。
- 旧Partial目录的协议、报告、失败和全部其他文件均原样保留，新协议绑定其完整哈希。
- 模型Revision仍为`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；运行前后核验全部绑定模型文件SHA256。
- 新源码、旧样本、模型、安装的Transformers及评分源码在新`protocol.json`中冻结。
- 只使用物理GPU1，UUID `GPU-b08489af-0a8a-eb48-55ba-8346733af417`，进程内cuda:0，只有一张可见GPU；未终止外部进程。
- 模型加载、单题诊断和三题Smoke合计 **27.48秒**，不是正式实验耗时。
- Torch峰值allocated **14253.65 MiB**、reserved **14400 MiB**。末次nvidia-smi进程采样为14740 MiB，不冒充连续采样峰值。
- 进程已退出；释放核验GPU1为 **27 MiB、0%利用率、无计算进程**，详见`gpu_release.json`。

## 5. 是否可以恢复正式采集？

**本次基线一致性阻塞已修复，原科学协议无需改变。**但未获本轮授权启动正式采集，仍是0/960，不产生Accuracy或角色必要性推断。

后续若恢复，必须使用新源码版本、新执行冻结及已通过的匹配形状验证，不能冒充旧冻结版本直接续跑。旧`query_role_attention.run gpu`仍绑定旧源码哈希，其Smoke还含历史全位置LM Head对照，**不要原样调用**；应先接入新版Smoke，确保历史对照也匹配LM Head形状，再冻结正式执行入口。此项是恢复正式入口的工程接线要求，不是本轮三道Smoke未通过。

仍需保留的边界：三题短序列不保证所有240题无运行时异常；跨形状、跨硬件或不同软件版本不能假定逐位一致；当前输出置零仍执行原Attention计算，不代表真实跳层或加速。

## 交付索引

- 根因及差异：`logits_comparison.json`、`lm_head_isolation.json`、`comparisons/`。
- CPU/GPU：`cpu_validation.json`、`gpu_smoke_validation.json`、`smoke_checkpoints/`、`gpu_execution.txt`。
- 源码审阅与绑定：`source_diff_review.md`、`protocol.json`、`postrun_integrity.json`。
- 资源：`environment.json`、`gpu_preflight.json`、`gpu_release.json`。
- 本地差异数组：`diagnostic_array_manifest.json`记录路径及哈希，不向GitHub提交二进制数组。

**停止条件已满足：仅修复和三题Smoke，本轮到此结束。**
