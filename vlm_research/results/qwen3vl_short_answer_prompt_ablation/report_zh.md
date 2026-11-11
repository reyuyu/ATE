# Qwen3-VL短答案Prompt独立实验

**独立协议：QWEN3VL_SHORT_ANSWER_PROMPT_ABLATION。方案B，非作者官方复现。SVH算法、Processor、BF16/Eager、视觉预算及评分器未变。原始ATE结果未覆盖。**

## 1. 当前Prompt
GQA已有 `Answer the question using a single word or phrase.`；ScienceQA/MMBench只有 `Please select the correct answer from the options above. `，并未要求仅答字母。15个真实完整模板和Token绑定见 current_prompt_audit_zh.md / prompt_examples.jsonl。

## 2. Validation结果
以下只报告预先隔离的Validation。每任务Prompt由Discovery冻结，未用Validation调参。MMBench为150个完整Circular组（全部旋转必须正确），不是逐行准确率。

|任务|Prompt|方法|严格分数%|增强诊断%（仅SQA）|均值/P50/P95 Token|解析失败%|截断%|
|---|---|---|---:|---:|---|---:|---:|
|GQA|P0|Native|55.33|N/A|2.32/2/4.0|0.00|0.00|
|GQA|P0|SVH|49.33|N/A|2.23/2/3.5|0.00|0.00|
|ScienceQA-IMG|P0|Native|58.00|59.33|56.66/7/128.0|42.00|40.67|
|ScienceQA-IMG|P0|SVH|13.33|52.67|91.63/115/128.0|86.67|45.33|
|ScienceQA-IMG|P2|Native|98.67|98.67|2.00/2/2.0|0.00|0.00|
|ScienceQA-IMG|P2|SVH|98.67|98.67|2.00/2/2.0|0.00|0.00|
|MMBench|P0|Native|74.67|N/A|20.81/6/128.0|10.33|9.96|
|MMBench|P0|SVH|64.00|N/A|37.41/8/128.0|19.38|17.93|
|MMBench|P2|Native|85.33|N/A|2.00/2/2.0|0.00|0.00|
|MMBench|P2|SVH|85.33|N/A|2.00/2/2.0|0.00|0.00|

## 3. 同Prompt下SVH−Native及Prompt收益

|任务|比较|指标|变化pp|图片簇95%CI|损害/纠正|
|---|---|---|---:|---|---|
|GQA|P0:SVH-Native|FROZEN_STRICT_ACCURACY|-6.00|[-12.00,+0.00]|15/6|
|ScienceQA-IMG|P0:SVH-Native|FROZEN_STRICT_ACCURACY|-44.67|[-52.67,-36.67]|67/0|
|ScienceQA-IMG|P0:SVH-Native|CONSERVATIVE_DIAGNOSTIC|-6.67|[-14.00,+0.67]|N/A/N/A|
|ScienceQA-IMG|P2:SVH-Native|FROZEN_STRICT_ACCURACY|+0.00|[+0.00,+0.00]|0/0|
|ScienceQA-IMG|P2:SVH-Native|CONSERVATIVE_DIAGNOSTIC|+0.00|[+0.00,+0.00]|N/A/N/A|
|ScienceQA-IMG|Native:P2-P0|FROZEN_STRICT_ACCURACY|+40.67|[+32.67,+48.67]|0/61|
|ScienceQA-IMG|Native:P2-P0|CONSERVATIVE_DIAGNOSTIC|+39.33|[+31.33,+47.33]|N/A/N/A|
|ScienceQA-IMG|SVH:P2-P0|FROZEN_STRICT_ACCURACY|+85.33|[+79.33,+90.67]|0/128|
|ScienceQA-IMG|SVH:P2-P0|CONSERVATIVE_DIAGNOSTIC|+46.00|[+38.00,+54.00]|N/A/N/A|
|MMBench|P0:SVH-Native|CIRCULAR_ACCURACY|-10.67|[-15.89,-6.00]|16/0|
|MMBench|P2:SVH-Native|CIRCULAR_ACCURACY|+0.00|[+0.00,+0.00]|0/0|
|MMBench|Native:P2-P0|CIRCULAR_ACCURACY|+10.67|[+5.37,+16.56]|1/17|
|MMBench|SVH:P2-P0|CIRCULAR_ACCURACY|+21.33|[+14.86,+28.38]|0/32|

## 4. 每任务冻结选择与解释

- GQA冻结选择P0。Native严格分数变化+0.00pp，SVH变化+0.00pp。同Prompt差距由-6.00pp变为-6.00pp（差距变化+0.00pp）。Native/SVH平均长度分别由2.32/2.23变为2.32/2.23。
- ScienceQA-IMG冻结选择P2。Native严格分数变化+40.67pp，SVH变化+85.33pp。同Prompt差距由-44.67pp变为+0.00pp（差距变化+44.67pp）。Native/SVH平均长度分别由56.66/91.63变为2.00/2.00。
  增强解析诊断：Native 59.33→98.67%，SVH 52.67→98.67%。截断输出仍判不确定；增强分数不替代严格正式分数。
- MMBench冻结选择P2。Native严格分数变化+10.67pp，SVH变化+21.33pp。同Prompt差距由-10.67pp变为+0.00pp（差距变化+10.67pp）。Native/SVH平均长度分别由20.81/37.41变为2.00/2.00。

Prompt带来的格式/长度收益属于通用输出协议改善，不代表SVH视觉交接能力本身提高。即使差距变小，也要检查Native是否同时受损；CI跨0不等于无损或等价。本轮不做非劣效认定。

## 5. 同卡配对真实性能
每任务2个预冻结Discovery输入（每卡1个），每条件1次预热、3次交错正式重复，冷KV、每请求重算Vision。以下为小样本工程诊断，不代表全部150题平均。自然EOS保留原cap；fixed32只在测速中忽略EOS，恰好生成32 Token，不参与质量评分。P0全体长度来自严格绑定历史结果，性能表P0为本轮重新测量。

|任务|Prompt|方法|模式|TTFT ms|TPOT ms|E2E ms|相对同方法P0加速|同Prompt Native/SVH|
|---|---|---|---|---:|---:|---:|---:|---:|
|GQA|P0|Native|natural_EOS|230.4|72.64|303.6|1.000|1.000|
|GQA|P0|SVH|natural_EOS|222.4|73.68|298.8|1.000|1.016|
|GQA|P0|Native|fixed32|233.3|73.71|2525.0|1.000|1.000|
|GQA|P0|SVH|fixed32|226.4|73.98|2527.8|1.000|0.999|
|ScienceQA-IMG|P0|Native|natural_EOS|205.0|72.50|4585.7|1.000|1.000|
|ScienceQA-IMG|P0|SVH|natural_EOS|205.6|74.47|1698.1|1.000|2.700|
|ScienceQA-IMG|P2|Native|natural_EOS|198.5|70.02|269.1|17.039|1.000|
|ScienceQA-IMG|P2|SVH|natural_EOS|201.7|72.25|276.6|6.140|0.973|
|ScienceQA-IMG|P0|Native|fixed32|200.0|65.40|2233.4|1.000|1.000|
|ScienceQA-IMG|P0|SVH|fixed32|190.7|68.12|2309.8|1.000|0.967|
|ScienceQA-IMG|P2|Native|fixed32|189.4|66.17|2246.5|0.994|1.000|
|ScienceQA-IMG|P2|SVH|fixed32|196.7|67.62|2299.9|1.004|0.977|
|MMBench|P0|Native|natural_EOS|160.2|65.97|1086.1|1.000|1.000|
|MMBench|P0|SVH|natural_EOS|167.2|66.14|932.7|1.000|1.164|
|MMBench|P2|Native|natural_EOS|171.8|64.33|236.6|4.590|1.000|
|MMBench|P2|SVH|natural_EOS|183.2|69.18|254.7|3.663|0.929|
|MMBench|P0|Native|fixed32|162.5|64.77|2175.9|1.000|1.000|
|MMBench|P0|SVH|fixed32|166.1|66.75|2242.2|1.000|0.970|
|MMBench|P2|Native|fixed32|171.4|64.68|2181.8|0.997|1.000|
|MMBench|P2|SVH|fixed32|173.4|67.93|2285.9|0.981|0.954|

自然EOS时间下降不能当作SVH算子加速；fixed32比较剥离生成长度差异。完整字段含输出吞吐、峰值显存、Vision/Prefill，见 performance_comparison.csv。

## 6. 有效性与限制
按Image SHA整簇配对Bootstrap 20,000次；MMBench先计算完整Circular组再按图像簇重采样。所有区间均为探索性，未宣称多重比较确认性发现。三任务Discovery/Validation全局图片SHA隔离。P0复用逐记录验证Prompt、Token IDs、输入Tensor、Gold、模型Revision、cap与协议哈希。
历史严格Physical数值验收仍为 FAIL_RETAINED、FULL_59未完成。新GPU Smoke仅验证Native/No-op严格相等、SVH重复运行严格相等、输入/Mask/位置/KV/MLP/Hook结构守恒；不将其冒充历史Dense/Physical严格门禁通过。
本轮TextVQA无新增GPU生成，保留之前CPU审计结论。main、历史源码、原始预测、Checkpoint、评分器均只读；未自动推送。
GPU资源释放见 gpu_release_audit.json。原始逐题文件 per_question.csv、worker0/1.sqlite、bindings0/1.sqlite；完整P0绑定 p0_reused.sqlite。

## 7. 差距缩小的直接配对检验

差中之差定义：(新Prompt SVH−Native)−(P0 SVH−Native)。正值表示SVH相对差距缩小；按相同图片簇配对Bootstrap，不是用两条独立CI相减。

|任务|指标|差距变化pp|探索性95%CI|
|---|---|---:|---|
|ScienceQA-IMG|score|+44.67|[+36.67,+52.67]|
|ScienceQA-IMG|diagnostic_score|+6.67|[-0.67,+14.00]|
|MMBench|score|+10.67|[+6.00,+15.89]|

GQA的P1在Discovery中，Native为68.00%（P0 70.00%），SVH为57.33%（P0 58.00%），因此按冻结规则保留P0。Validation没有再运行GQA P1；不能将保留P0的验证分数解释为P1已获独立验证。

每条旋转记录的Native/SVH四格表及ScienceQA“内容正确、格式失败”数量见 native_svh_contingency_rows.csv；MMBench正式主分数与CI仍采用完整Circular组，不能混用分母。

## 8. 自然EOS时间改善与计时边界

- GQA保留原P0，无Prompt变更收益主张。
- ScienceQA-IMG Native，两个固定输入的平均自然EOS E2E：4585.7→269.1ms，减少94.1%，加速17.04倍。
- ScienceQA-IMG SVH，两个固定输入的平均自然EOS E2E：1698.1→276.6ms，减少83.7%，加速6.14倍。
- MMBench Native，两个固定输入的平均自然EOS E2E：1086.1→236.6ms，减少78.2%，加速4.59倍。
- MMBench SVH，两个固定输入的平均自然EOS E2E：932.7→254.7ms，减少72.7%，加速3.66倍。

计时沿用原引擎统一边界：预处理完的CPU输入开始，Vision及Language Prefill/真实Decode都重新计算，冷KV；不包含磁盘图片读取、Processor、模型载入及输入H2D初始化。E2E为该生成引擎实际墙钟，不是完整服务请求延迟。TTFT为首轮前向CUDA事件时间，TPOT不含首Token；tokens/s包含EOS，所有Token长度统计都沿用相同口径。

P1/P2增加了原生Prefill文本位置，在冻结SVH规则下这些位置也可能接收视觉信息。因此本轮不能把收益全部归因于解析器，或断言视觉信息交接能力已修复；未做区分格式指令与新增接收位置的额外机制消融。

总体建议：ScienceQA/MMBench可将已冻结并留出验证的P2作为新的独立短答协议候选；GQA保留P0。原ATE统一HF成绩保持原值。历史物理数值FAIL未解决，本轮结果不构成该门禁的重新通过。

## 9. 执行与资源

执行源码和样本在GPU前冻结于本地Commit `48b8616965`。本轮新增4,568条质量生成、严格绑定复用3,436条P0质量记录；Discovery共4,896条、Validation共3,108条记录。另有192条正式计时记录（不含预热）。两卡常驻模型的完整GPU阶段约15.30分钟。数据均为历史研究队列，不宣称独立外部测试。

Native/No-op与SVH重复运行检查通过，9项CPU回归通过。GPU1和GPU3在结束后均回到5MiB/0%，未影响GPU0/2的外部任务。没有推送远端，也没有六项全量重评测。

MMBench采用冻结ATE内部EN Dev/Circular exact_matching版本，不冒充作者V1.1。ScienceQA P2严格分数与保守增强解析均为98.67%；Prompt改善不是把增强解析结果替换成正式成绩。

完整新增质量记录、输入绑定、历史复用证据和SHA256见本目录SQLite及CSV。最终独立完整性检查见 validation.json；原始大规模Physical严格数值失败仍单独保留。
