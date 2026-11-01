"""Concise Chinese delivery, explicit stopped coverage and frozen negative results."""
import csv
from datetime import datetime
from pathlib import Path

from .common import ROOT,OUT,START,DEADLINE,TASKS,read,write,sha,command


def rows(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def table(items,columns):
    lines=['|'+'|'.join(label for _,label in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for item in items:
        lines.append('|'+'|'.join(str(item[key]) for key,_ in columns)+'|')
    return lines


def make():
    coverage=read(OUT/'qwen/coverage.json')
    gate=read(OUT/'llava_relay/extension_decision.json')
    peaks=read(OUT/'qwen/peaks.json')
    fullpeak=next(p for p in peaks if p['cohort']=='all' and p['task']=='Overall')
    pilotpeak=next(p for p in peaks if p['cohort']=='pilot240' and p['task']=='Overall')
    r=rows(OUT/'llava_relay/accuracy_summary.csv')
    inter=rows(OUT/'llava_relay/interaction.csv')
    now=datetime.now().astimezone()
    elapsed=(now-datetime.fromisoformat(START)).total_seconds()
    scan_seconds=max(read(p)['formal_seconds'] for p in (OUT/'qwen').glob('all_gpu*/complete.json'))
    relay_seconds=read(OUT/'llava_relay/gpu3_shard0of1/complete.json')['seconds']
    llava_rescue_seconds=read(OUT/'llava_rescue/gpu3/complete.json')['seconds']
    qwen_rescue_seconds=read(OUT/'qwen_rescue/gpu3/complete.json')['seconds']
    lines=['# Visual Pathways：视觉来源敏感性、中继交互与稀疏补偿','',
        '**总体状态：partial。InternVL 原生 Logits Smoke 失败后按规则停止；没有 InternVL 正式曲线或准确率。其余完成范围如下。**','',
        '## 覆盖与执行','',
        f'- Qwen：{coverage["questions"]}/3,000 题，{coverage["conditions"]:,}/108,000 个逐层条件；其中严格复用历史 {coverage["historical_questions"]} 题 × 36 层，新增 {coverage["new_questions"]} 题 × 36 层。状态：{coverage["status"]}。',
        '- InternVL：0/240 个正式样本，未启动 3,000 题扩展；三卡均在官方 Chat 与独立 Full 首位置 Logits 精确一致性检查处失败。输入 Token ID 和视觉注入 Embedding 已验证一致，但这不能替代输出一致性。失败日志全部保留，没有放宽容差或继续评分。',
        '- LLaVA 交互：同一批 240 题 / 235 个图片 SHA256 簇，A/B/C/D 共 960 次配对完整生成。',
        '- 稀疏补偿：LLaVA 240 题 × 3 条件，其中 FullImageAccess 严格复用交互 C，新增 480 次生成；Qwen 240 题 × 3 条件，新增 720 次生成。',
        '- 未扩展 Qwen/InternVL 的 L2–L5 交互：LLaVA 未通过事先冻结的正向交互门槛；InternVL 另有 Smoke 停止限制。没有增加层组合、训练或加速算法。',
        '- Qwen/InternVL 历史固定 revision、原生 Chat Template、BF16/eager；Qwen DeepStack 原生。LLaVA 固定 revision、FP16/eager、Transformers 4.49.0、原始 GQA Prompt。模型文件校验和、输入展开及环境见各 protocol/validation 文件。',
        f'- 本轮仅使用物理 GPU0/2/3；GPU1 排除。分支 `{command("git","branch","--show-current")}`，独立 Worktree；没有修改或推送 main。',
        f'- 计时起点 {START}；报告生成 {now.isoformat()}，已用 {elapsed/3600:.2f} 小时；总预算截止 {DEADLINE}。最终资源与交付审计时间见 delivery_audit.json。',
        f'- 实测正式运行：Qwen 2,760 个新增样本两卡并行约 {scan_seconds/60:.2f} 分钟；LLaVA 交互 {relay_seconds/60:.2f} 分钟；LLaVA/Qwen 稀疏补偿分别 {llava_rescue_seconds/60:.2f}/{qwen_rescue_seconds/60:.2f} 分钟。后几项与大扫描并行，不能相加作为总墙钟。','',
        '## 1. 单层 Image 来源必要性','',
        '每次只在一个目标层对原始 Prefill 最后 Query P−1 → 实际 Image Key 加 Softmax 前 −inf；不是 o_proj OFF，也不是 Keep-only。目标是每题原生 Full 自己的首 Token Top1；使用全词表 FP32 log_softmax。主指标为先逐题取绝对值再求均值，不是平均 Signed Loss 的绝对值。Top1 Flip 不是完整答案 Accuracy。','',
        '低敏感仅针对当前原生残差、其他层与间接路径背景下的直接来源读取，不表示该层没有视觉信息，也不表示整个 Attention 更新可以省略。','',
        f'Qwen 全部已完成样本的主指标峰为 L{fullpeak["absolute_peak_layer"]}，均值 {fullpeak["absolute_peak_value"]:.6f} nat；前三层为 {fullpeak["top3_layers"]}，占全部逐层平均绝对变化之和 {100*fullpeak["top3_share"]:.2f}%。这一比例仅是描述性的集中程度，不是可跳层比例或独立贡献分解。',
        f'原 240 题的峰为 L{pilotpeak["absolute_peak_layer"]}；单峰并不代表所有任务共享同一峰，六任务完整曲线与样本数如下。','']
    taskpeaks=[p for p in peaks if p['cohort']=='all' and p['task']!='Overall']
    lines+=table([dict(task=p['task'],n=p['questions'],peak='L'+str(p['absolute_peak_layer']),
        value=f'{p["absolute_peak_value"]:.6f}',share=f'{100*p["top3_share"]:.2f}%') for p in taskpeaks],
        [('task','任务'),('n','题数'),('peak','Mean Absolute 峰层'),('value','峰值 nat'),('share','Top3 描述性占比')])
    hist=rows(OUT/'llava_historical_imageoff_layer_summary.csv')
    histoverall=[x for x in hist if x['task']=='Overall']
    hp=max(histoverall,key=lambda x:float(x['mean_absolute_logp_change']))
    total=sum(float(x['mean_absolute_logp_change']) for x in histoverall)
    top3=sorted(histoverall,key=lambda x:-float(x['mean_absolute_logp_change']))[:3]
    lines+=['',f'同题 240 问的历史 LLaVA Image OFF 主峰为 L{hp["layer"]}，前三层 {[int(x["layer"]) for x in top3]} 占比 {100*sum(float(x["mean_absolute_logp_change"]) for x in top3)/total:.2f}%；Qwen 的同批峰位与它不同。图中保留各自的全部层，不对齐或平移峰。原生 Prompt、Tokenizer/目标 Token、精度和网络深度不同，因此跨模型比较是层级模式的描述，不是严格控制架构以外因素的因果比较。',
        '对“不同 VLM 是否共享少数视觉热点”的回答：可观察到非均匀层级分布及模型/任务依赖峰，但不能仅凭最大峰声称强稀疏；应同时查看 Top3 占比与非峰背景。InternVL 本轮无有效结果，**不能声称已验证 Qwen 与 InternVL 的共同规律**。',
        '图：`figures/qwen_overall_sensitivity.pdf`、`qwen_six_tasks_absolute/signed/flip.pdf`、`cross_model_paired240.pdf`。跨模型图仅使用完全相同的 240 问，LLaVA 明确标为历史数据；不为 InternVL 填零或画假曲线。','',
        '## 2. L2–L5 死亡区是否苏醒','',
        'A：原生中继/原生 P−1；B：原生中继/P−1 L2–L5 o_proj OFF；C：中继 OFF/原生 P−1；D：中继 OFF/P−1 L2–L5 o_proj OFF。中继 OFF 只屏蔽 P−1 之前全部非视觉 Prefill Query → Image 的合法连接；P−1 直接读图正常，Decode 完全原生。其他 Prefill Query 不做 o_proj 关闭，视觉 Token、MLP、残差和原始位置保留。','']
    lines+=table([dict(condition=x['condition'],acc=f'{float(x["accuracy_percent"]):.3f}%',
        delta=f'{float(x["delta_full_pp"]):+.3f}',ci=f'[{float(x["delta_ci_low"]):.3f}, {float(x["delta_ci_high"]):.3f}]',
        harm=x['full_correct_to_wrong'],gain=x['full_wrong_to_correct'],loss=f'{float(x["mean_full_token_logp_loss"]):.6f}',flip=x['first_token_flip_count']) for x in r],
        [('condition','条件'),('acc','完整 Accuracy'),('delta','ΔA pp'),('ci','配对95%CI pp'),('harm','原对→错'),('gain','原错→对'),('loss','固定 Full 首Token Loss nat'),('flip','首Token翻转')])
    lines+=['']
    for x in inter:
        lines.append(f'- `{x["metric"]}`：{float(x["mean"]):+.6f}，95%CI [{float(x["ci_low"]):+.6f}, {float(x["ci_high"]):+.6f}]。')
    lines+=['',
        '交互定义：(D 的固定 Full Token Loss − C 的 Loss) − (B 的 Loss − A 的 Loss)，正数才表示中继受损后 Skip 损害增加。事前扩展门槛为该交互的图片簇配对双侧95%CI整体大于0。',
        f'本轮门槛是否通过：**{gate["extend_to_other_models"]}**。LogP 交互点估计 {gate["mean"]:+.6f} nat，95%CI [{gate["ci_low"]:+.6f}, {gate["ci_high"]:+.6f}]，不是正向“苏醒”。Accuracy 交互 {gate["accuracy_interaction_pp"]:+.3f} pp，区间跨0，不能据此认定 Skip 改善 Accuracy 或两者等价。',
        '非零负向 LogP 交互说明该干预的影响并非完全不受路径背景影响，但方向不支持“原先低敏感的更新变得更必要”，Accuracy 交互也没有明确证据。',
        '**结论：在本批样本与固定中继干预下，没有观察到 L2–L5 死亡区苏醒；固定 Token LogP 交互反而指向更小的 Skip 损害。** 这是对当前具体假说的反向证据，不是对所有信息路径、所有模型的普遍否定。图：`figures/llava_relay_interaction.pdf`。','',
        '## 3. 固定稀疏热点能否补偿中继损伤','',
        '三个条件均为 Relay OFF，只改变原始 P−1 的直接 Image 读取；Decode 原生。FullImageAccess 不是原生 Full，而是中继受损、P−1 全层可读图的对照。KeepSparse 仅保留 LLaVA L14/L19 或 Qwen L26，不是只保留这些完整 Transformer 层。','']
    for model,label in [('llava','LLaVA'),('qwen','Qwen')]:
        folder=OUT/(model+'_rescue')
        items=rows(folder/'accuracy_summary.csv');pairs=rows(folder/'paired_comparisons.csv');base=read(folder/'native_full_reference.json')
        lines += [f'### {label}','',f'原生 Full（中继正常）Accuracy：{base["accuracy_percent"]:.3f}%。','']
        lines+=table([dict(condition=x['condition'],acc=f'{float(x["accuracy_percent"]):.3f}%',loss=f'{float(x["mean_full_token_logp_loss"]):.6f}',
            flip=x['first_token_flip_count'],eos=x['eos'],truncated=x['truncated']) for x in items],
            [('condition','Relay OFF 条件'),('acc','Accuracy'),('loss','固定原生Full Token Loss nat'),('flip','首Token翻转'),('eos','EOS'),('truncated','截断')])
        lines+=['']
        lines+=table([dict(comparison=x['comparison'],delta=f'{float(x["delta_accuracy_pp"]):+.3f}',
            ci=f'[{float(x["ci_low"]):.3f}, {float(x["ci_high"]):.3f}]',harm=x['reference_correct_to_wrong'],gain=x['reference_wrong_to_correct']) for x in pairs],
            [('comparison','探索性配对比较'),('delta','ΔAccuracy pp'),('ci','95%CI pp'),('harm','参照对→错'),('gain','参照错→对')])
        lines+=['']
    lines += ['LLaVA 固定 L14/L19 未能补偿到完整直接视觉访问；相对全关的小幅增益区间跨0，不能宣称有效恢复。Qwen L26 相对全关有小幅描述性恢复，但仍低于完整直接视觉访问的点估计，远未恢复到原生 Full；探索性未校正区间不是确认性证明，也不能把区间接触0认定等价。',
        '**单层 OFF 的敏感峰不能证明该层 Keep-only 足够。** 这些预先固定热点在中继受损背景下并未充分恢复完整答案表现。图：`figures/sparse_visual_rescue.pdf`。','',
        '## 研究结论与边界','',
        '1. 不同架构和任务的视觉来源敏感峰不同；本轮支持“非均匀、模型/任务依赖”，但 InternVL 未通过验证，不能完成两新模型的普适性论证。',
        '2. 中继关闭明显损害最终答案，表明指定视觉信息路径干预有实际预测影响；Hard Mask 同时导致概率重分配，这不等于某种视觉语义的独立贡献。',
        '3. 本轮没有支持“Dead Attention Can Wake Up”的关键正交互；固定 LogP 的反向交互削弱这一具体论文叙事。不能为了故事改层、改样本或追加组合。',
        '4. 稀疏热点仅有有限/不确定的恢复，本轮未证实足以补偿受损中继；敏感峰与充分性必须分开。Decode 仍可看图，完整答案结果可能含后续补偿，不能当作纯首 Token 因果分解。',
        '5. 首 Token Loss/Flip 不是最终 Accuracy；所有来源 Mask 都重新归一化其余合法 Key。观察性曲线、局部因果干预与最终生成评分分别报告，不宣称 Attention 冗余、整层可跳过或实际加速。','',
        '## 统计与审计','',
        '- 图片 SHA256 整簇配对 Bootstrap，10,000次、seed=20261102、题目加权；同图多题一起重采样。层、Head、重叠条件不视为独立图片。曲线区间为逐点探索性95%CI，未用来宣称96/36项扫描的确认性显著性。',
        '- 240题是复用的历史探索队列，部分候选层来自先前相关研究；它不是新的独立确认集。没有根据本轮3,000题新峰位调整固定LLaVA L14/L19或Qwen L26候选。',
        '- Task2 的固定 LogP 交互为事前单一主比较；Accuracy 及其他生成比较辅助/探索性。Task3 的多项比较全部探索性，未进行事后非劣效或等价认定。',
        '- 成对正确性完全相同或没有观察到翻转时，经验Bootstrap区间可能退化为[0,0]；它不能覆盖未观察到的稀有损害，也不表示总体损害率为0。本轮没有据此认定无损或等价。',
        '- CPU Hook/Mask/Bootstrap 检查、真实 GPU No-op/目标概率为0/非目标 Query/原生 DeepStack/Cache隔离/Hook清理见验证记录；正式采集没有 NaN、Inf、OOM、空答案或截断。',
        '- InternVL 首次 Smoke 失败严格保留并停止，不计为正式样本。Qwen 两处启动工程问题（权重加载模式、Smoke 输入设备）在数值实验前修复，原始日志与协议修订均保留，没有放宽数值标准；详见 attempts_audit.json。',
        '- JSON 条件数组字段口径：顶层 `protocol.json/task2/conditions` 为 `[relay_on, skip]`；`llava_relay/protocol.json/conditions` 为 `[relay_off, skip]`。执行代码和本报告 A/B/C/D 表一致，另存 condition_definitions.json 供审阅；没有改动冻结协议或逐题结果。',
        '- CSV、逐题原子检查点、源/模型/图片绑定、环境、执行日志、PNG/矢量PDF及哈希均在当前结果目录。资源最终快照和交付审计时间见 `delivery_audit.json`。','',
        '## 下一步建议','',
        '不基于本轮增加层组合或样本。若继续 InternVL，应先单独解决原生首 Logits 与官方 Chat 的精确一致性（当前怀疑 LM-head 投影形状不同，但未经本轮复核，不能当作已证实原因），重新通过 Smoke 后再授权采集。对论文故事应优先如实呈现负交互与热点补偿受限，而不是继续寻找正面条件。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    (OUT/'README.md').write_text('# Visual Pathways: Seven-Hour Budget\n\n'
        'Overall status: **partial** (InternVL stopped at exact native-logit smoke).\n\n'
        f'Qwen: {coverage["questions"]}/3,000 questions, {coverage["conditions"]:,} source-OFF rows.\n\n'
        'LLaVA relay interaction and LLaVA/Qwen sparse rescue: 240/240 each.\n\n'
        'See [report_zh.md](report_zh.md) for the Chinese report, limitations and negative results.\n'
        'Integrity, frozen protocols, raw checkpoints, CSVs, PNG/PDF figures and resource release audit are retained.\n')
    write(OUT/'condition_definitions.json',dict(parent_task2_array_fields=['relay_on','skip'],
        relay_protocol_array_fields=['relay_off','skip'],
        task2={'A':dict(relay_on=True,skip=False),'B':dict(relay_on=True,skip=True),
               'C':dict(relay_on=False,skip=False),'D':dict(relay_on=False,skip=True)},
        skip_layers=[2,3,4,5],skip_query='Original Prefill P-1 only',decode='Native for all experiments'))
    write(OUT/'report_status.json',dict(overall='partial',reason='InternVL stopped on required exact-native-logit smoke',
        qwen=coverage,llava_relay_complete=True,llava_rescue_complete=True,qwen_rescue_complete=True,
        no_task2_extension=not gate['extend_to_other_models'],report_time=now.isoformat(),elapsed_seconds=elapsed))
    print('Wrote Chinese report; overall partial due InternVL stop.',flush=True)


if __name__=='__main__':make()
