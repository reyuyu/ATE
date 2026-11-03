"""Chinese read-only report, explicitly separating observations and causal unknowns."""
from .common import *

def main():
    cases=read(OUT/'case_annotations.json')['cases'];summary=read(OUT/'readonly_summary.json');status=read(OUT/'causal_validation_status.json')
    assert status['status']=='NOT_RUN_NO_IDLE_GPU'
    lines=['# Semantic-k 错误案例归因：真实漏选、接收位置与答案形式','',
    '**交付范围：只读案例分析完成；补词因果验证未执行。** 启动前检查时四张GPU均已有其他用户的计算进程，依照任务边界没有加载模型、共享占用卡、等待轮询或抢占任务。不得把本文的候选解释当作“补回已恢复正确”的证据。','',
    '**主要发现：** 12个预先锁定的错误案例中，4例存在真实词法核心候选遗漏，合计7个完整词组；8例保留全部原词法核心候选。1例自然k退化为0，并存在名词 `can` 被固定停用词表当成功能词的歧义。另有1例 `Panda` 对 `panda bears` 的严格答案形式不匹配。因此，Semantic-k的损害不能统一解释为漏掉某一类“重要词”，也不能据此确定只要增加预算就能解决。上述计数只描述已选案例，不外推3000题比例。','',
    '## 1. 直接查看','',
    '- [HTML案例画廊](gallery.html)：16张卡，含任务/损害或纠正筛选、ID搜索及逐Token展开表；图片已嵌入，HTML可离线打开。',
    '- [案例PDF](case_cards.pdf)：16页，每页1题，包含原图、完整问题、全部指定答案、实际Token高亮及候选说明。',
    '- [逐题表](selected_cases.csv)、[实际词组位置与Subword数](token_group_audit.csv)、[词类覆盖审计](token_feature_coverage.csv)。',
    '- [锁定的补词方案](candidate_plan.json)及[补词状态表](supplementation_results.csv)：231个计划标签均为 `NOT_RUN_NO_IDLE_GPU`，新答案与正确性留空，不能当作231次失败。此数包含条件性替换及不可行项；实际新增生成数为0。','',
    '## 2. 基线和选例的冻结规则','',
    f'正式结果参考Commit：`{BASE}`；独立分支：`codex/semantic-writer-error-cases`。未修改main或历史结果。原始3000题整体中，All-Writer为2611/3000（87.03%）、Semantic-k为2514/3000（83.80%）；184题原对→错、87题原错→对，净少97题。此处复用正式评分，不重新生成原有答案，也不重新拟合选择器。','',
    '规则在读取案例池前提交冻结，见 `selection_rules.json` 和首个规则Commit `db474b1bac`。按六任务顺序，每任务2例；优先图片不重叠子集，再覆盖新错误类型、真实遗漏词类和预算区间，最终以固定SHA256排序破同分；案例间图片不重复。固定必含 `001020372`，它属于历史Semantic240探索图片，因此12例中11例来自确认子集，1例为明确标注的必含例外。4个纠正反例全部来自确认子集、覆盖4种任务。','',
    '选择仅依赖已存在答案与Manifest，不依赖补词结果；后续没有因补词成功与否更换案例。完整逐题Token集合、Word分组、Question字符Offset、输入Token IDs和图片SHA均从正式结果逐项复核。Question-Random展示全部三个冻结种子20261103/20261107/20261111，不挑选有利种子。','',
    '## 3. 六任务的12个主要错误案例','',
    '| ID | 任务 | Gold | All-Writer | Semantic-k | 核心候选实际遗漏 | k/R |','|---|---|---|---|---|---|---:|']
    for c in cases:
        if c['kind']!='harm':continue
        spec=c['spec'];s=c['sample'];miss='、'.join(g['decoded'] for g in c['semantic_core_dropped_groups']) or '无'
        lines.append(f"| {c['question_id']} | {s['task']} | {s['answer']} | {c['rows']['All-Writer']['text']} | {c['rows']['Semantic-k']['text']} | {miss} | {len(spec['writers']['Semantic-k'])}/{len(spec['relay_positions'])} |")
    lines += ['','### 3.1 确实存在可定位的规则遗漏，但还不是因果修复','',
    '- `14236062`：遗漏 `or` 和 `colored`。`cream`已选，但 `cream colored` 的末词未选；它们原本是核心候选，因整词后缀预算可行性缩减2 Token而被排除。',
    '- `09662582`：遗漏视觉类别选项 `train`，预算缩减1 Token；`truck`和`or`已选。它是一个具体、可检验的接收位置遗漏候选。',
    '- `03567478`：遗漏电源状态词 `on`，预算缩减1 Token；`any`、`TVs`、`laptops`和`or`实际已选。',
    '- `10763612`：问题为 `On which side is the can?`。词法核心候选仅 `On/which/side`，总3 Token；R末端的完整 `ASSISTANT` 组为4 Token，为同时匹配历史Question-Last与R-Last整词后缀对照，原选择规则找不到正预算，故k降到0。Semantic-k、Question-Last-k、各Random-k都与No-Writer写入集合相同。这是对照可行预算约束导致的退化，不是模型推断出的“无需看图”。此外，词表将 `can` 作为模态动词停用，但本题是罐子名词；这是无词性消歧词表的规则设计风险。','',
    '以上均为真实输入规则行为，原实现按冻结规则执行，不能不加区分地称为运行程序Bug。预算截断、词性歧义是有希望修正的选择器设计问题；本轮未验证其修正能否恢复答案，更没有验证一个改进规则在独立样本上的收益。','',
    '### 3.2 八例没有遗漏任何原核心候选','',
    '已知案例 `001020372` 中，`zebra/to/left/of/bench/black/or/white` 全部绿色选中。漏选的是 `Is`、冠词和问号等。它不能用“漏了 left/of/or”解释。两道ChooseRel错误也完整保留left/right和对象锚点；`0481307`完整保留On/which/side。两个CompareAttr错误中的What/common及目标对象均已选，而答案从Color变成Material。','',
    '这些现象说明“保留全部词法核心候选”并不能保证与All-Writer一致，但尚不能在功能词、边界/模板接收、多位置协作或预算不足之间作因果区分。给功能词开放读图，测的是该位置作为视觉接收者的作用，不是重新把这个词加入文本；这些词在全部条件下原本就存在、仍参与非图像Attention/MLP/Residual。','',
    '### 3.3 将视觉错误与答案形式分开','',
    '`02760157`的图像是熊猫，Question为 `Are these penguins or panda bears?`。panda与bears均已选，Semantic输出Panda，All输出Panda bears，官方Gold为panda bears。冻结评分器仍判Semantic错误；本分析保留该标签，但不将它直接称为识别成了错误动物，也不通过改评分器挽回ACC。','',
    '`0160501`是是非问题，Gold Yes，Semantic输出Door；Do未直接读图，而or/door/window/open均已选。这是可见的答案类型偏移，不能仅由它推断视觉信息缺失或某个特定词唯一承担视觉存储。','',
    '## 4. 真实词类和位置覆盖','',
    '| 检查对象 | 本批实际观察 | 可以和不可以推断的内容 |','|---|---|---|',
    '| 功能词/冠词 | 各题存在未选辅助动词、冠词等；详见逐题颜色 | 是候选遗漏，不等于已经证明功能词必须读图 |',
    '| 数量/逻辑连接 | `any`在03567478已选；多个`or/and`已选，但14236062的or确实被裁掉 | 不能预设整类词都漏选 |',
    '| 否定 | 错误案例没有因此可证实的否定遗漏；纠正例03678636的not确实已选 | 没有观察到该类不等于它在其他样本不重要 |',
    '| 比较/疑问 | common、What、which等在多例已选；10763612因k=0丢掉On/which/side | 需要按实际位置区分，不能统一归因为漏疑问词 |',
    '| 问号/问题结束位置 | 12/12主要错误例的问号未直接读图 | 4/4纠正反例同样遗漏问号，故共现不能证明因果 |',
    '| 指令/模板R | 12/12错误例、4/4纠正例均未直接读图 | 灰色是协议事实；是否恢复正确尚无新实验 |','',
    '## 5. 四个纠正反例','',
    '| ID | 任务 | Gold | All-Writer | Semantic-k | 约束解释 |','|---|---|---|---|---|---|']
    for c in cases:
        if c['kind']!='rescue':continue
        s=c['sample'];lines.append(f"| {c['question_id']} | {s['task']} | {s['answer']} | {c['rows']['All-Writer']['text']} | {c['rows']['Semantic-k']['text']} | 模板和问号均未读图，仍纠正 |")
    lines += ['','其中03678636的否定词not实际被选中；03539455的左右关系词也已选。反例不证明这些词足够，但阻止把同样的省略模式一律解释成错误根因。','',
    '## 6. 补词验证状态、边界和已锁定方案','',
    '首次只读检查时GPU空闲；完成CPU锁例与计划冻结后，启动检查的四张GPU均已有其他用户进程（见 `gpu_selection.json`）。即使某卡还有剩余显存，也不共享已有任务。按要求至此停止GPU工作，本轮没有新GPU Smoke、没有加载7B模型、没有生成新答案，也没有设置后台轮询。','',
    'CPU锁定内容包括：每题最多6个单组遗漏位置、全部未选Question、全部未选模板、单组等新增预算的位置对照；对单组恢复正确者才允许运行预先锁定的等预算替换。优先保留问号、指令开始Answer及R末端ASSISTANT组，再按冻结词类顺序填入其他Question候选。固定候选不覆盖全部遗漏词组；例如14236062的colored仅包含在全Question补回中，未另增第7个候选。','',
    '对照位置取指令内部的完整词组，操作性含义是不同接收位置，不声称这些位置已被证明与视觉无关。等预算替换优先移除一个同大小完整原选组；不可行时允许完整组子集且明确标注。k=0的同预算正增补不可能，方案中如实标注不可行。即使将来某个特定替换失败，也不能推导所有等预算配置都失败。','',
    '作为只读可核对的上限，Semantic集合联合全部遗漏Question与模板位置等于历史All-Writer的R全集，其本批12题均正确；这是复用已存在的条件，不是新做了“所有补词恢复12/12”的实验。尚不知道只补Question、只补模板或单补词时会怎样。','',
    '## 7. 最终归因分类：候选计数与验证计数分开','',
    '| 要求的类别 | 本批只读候选/已验证数量 | 当前可作的结论 |','|---|---|---|',
    '| 疑似重要Token漏选 | 4/12有核心候选遗漏（7组）；0例完成补词验证 | 可定位具体遗漏，不知道哪些能修复 |',
    '| 疑似模板/问题边界接收位置有作用 | 12/12有遗漏候选；0例验证其作用；4/4纠正反例也有同类省略 | 只能作为重叠候选，不可归因为已证实模板必要 |',
    '| 多Token共同作用，单补词无效 | 无法判定；12例均未测试 | 不能把未运行计成单补词失败 |',
    '| 增加预算有效但等预算替换无效 | 无法判定；0次新增预算/替换测试 | 不能判定更多接收者是唯一出路 |',
    '| 补词仍不能恢复，原因尚不明确 | 无法判定；0次补词测试 | 未执行不等于补词未恢复 |','',
    '**12个主要案例均处于“因果归因待验证”状态。** 第一、二行是可重叠的只读候选计数，不是互斥失败类型；后三行不填虚构的失败数。另单列1例答案表面形式不匹配和1例答案类型偏移，它们也不与上述候选互斥。任何计数都不外推为全3000题机制比例。','',
    '## 8. 审计与可复现文件','',
    '- 本轮17项CPU测试通过（6项新候选计划约束测试和11项继承协议测试），未初始化CUDA。历史GPU验证和实际Mask审计仅复核原记录，不冒充本轮GPU测试。',
    '- 16张原图SHA与历史样本完全一致；96条指定历史完整答案重新评分，与原CSV、检查点、Token IDs逐项一致。原始Logits数组SHA及禁止概率/非目标Mask审计均复核通过。',
    '- `case_manifest.json`、`candidate_plan.json`、`selection_validation.json`提供选例与候选冻结绑定；`historical_case_outputs.csv`保留复用来源；`readonly_case_attribution.csv`和`token_feature_coverage.csv`保留实际观察。',
    '- `supplementation_results.csv`留空未产生的答案，`causal_validation_status.json`明确未执行。没有丢弃失败例、重选题目、改Prompt、改层或修改评分器。',
    '- PDF逐页渲染检查与HTML浏览器交互检查见 `visual_review.json`、`html_browser_validation.json`；渲染脚本见 `../../semantic_writer_error_cases/render.py`。PDF一页一题；HTML的图像内嵌且另附原文件。',
    '- 交付完成状态为 `READ_ONLY_COMPLETE`，而不是GPU补词实验COMPLETE。资源记录注明本轮未占用GPU，不声称释放了其他用户的GPU。','',
    '## 9. 回答核心问题','',
    '**目前确实发现可改进的选择器规则风险：整词对照预算会裁掉train/on等本来入选的核心组，甚至使短问题k=0；无词性消歧的can停用规则也会排除视觉对象名词。这些是具体、可审计的遗漏，不是事后编造词类解释。但本轮没有空闲GPU，不能声称已经证实“补回即可修复”。另外8/12错误例保留全部原核心候选，其中还有严格评分的表面形式问题，因此“只修核心词漏选”也不足以解释全部现象。究竟还需增加接收者数量，还是用更好的等预算位置替换即可，需要冻结方案的因果验证才能区分；当前证据不允许二选一。**','',
    '本轮按资源边界交付只读结果后停止；未扩展3000题搜索、未训练选择器或Router、未做Kernel/KV优化，也不自动等待GPU继续运行。','']
    assert not (OUT/'report_zh.md').exists();(OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print('CHINESE READONLY REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
