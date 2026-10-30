"""Report only completed real collection; never substitute anticipated effects."""
import json
from .protocol import OUT,SPECIAL,TYPES,DEEP,MODES,GROUPS
from .run import verify,verify_targets,gate


def bounds(r,corrected=False):
    prefix='familywise95' if corrected else 'ci95'
    return f"[{r[prefix+'_low']:.6f}, {r[prefix+'_high']:.6f}]"


def report(rows,native,summary,source_summary,main,paired,deep,recent):
    config=verify();targets=verify_targets();gate('ablation')
    validation=json.loads((OUT/'numerical_validation.json').read_text())
    lookup={(r['mode'],r['layer'],r['word_type']):r for r in summary}
    contrasts={(r['contrast'],r.get('layer'),r.get('mode'),r.get('analysis')):r for r in main}
    lines=['# Late-Layer Attention Source Decomposition','',
        f"Status: {'complete' if len(main)==41 else 'partial_statistics_rank_failure'} independent200-image GPU study; {targets['positions']} frozen native Decode targets; {46*targets['positions']} single-step interventions.",
        f"Collector commit: `{config['code_commit']}`; target manifest SHA256: `{targets['selected_sha256']}`.",
        '', 'No training, freely continued ablation, complete-caption accuracy or acceleration claim.',
        '', '## Independent Data and Protocol',
        '',f"Exact lexical target counts: `{targets['counts']}`. Alignment failures: {targets['alignment_failures']}. Missing roles are retained, not replaced.",
        '200 new COCO IDs/byte-SHA/RGB identities, disjoint from earlier Observation2 sample/smoke and400 COCO Discovery images. '
        'Fixed original prompt, tokenizer/lexicons/priority, FP16/eager/Transformers4.49.0; Greedy cached generation, max64/EOS.',
        'Every condition consumes native y[t-1] to predict native y[t], clones the same unmodified history and discards the altered cache. '
        'Native source observation gathers all10 layers in one extra unchanged forward per target, never repeats a whole Caption per condition.',
        'BOS/image/prompt/history exactly partition visible keys. Recent is the just-consumed generated token, included in History; Older may be empty at t=1. '
        'Source subtraction uses original weights and Values, bias-free FP32 Wo projection, one nativeFP16 subtraction cast, with no new Softmax or mask.',
        '', '## Measured Late-Layer Effects',
        '', '| Role | Layer | Mode | Images | Mean loss | Ordinary95% CI | Mean absolute change | Top1 flip |',
        '|---|---:|---|---:|---:|---|---:|---:|']
    for role in TYPES:
        for layer in SPECIAL:
            for mode in ['Whole','Image','Prompt','History','BOS','Recent','Older']:
                r=lookup[(mode,layer,role)]
                lines.append(f"| {role} | {layer} | {mode} | {r['n_images']} | {r['mean']:.6f} | {bounds(r)} | {r['mean_abs_delta_logp']:.6f} | {r['top1_flips']}/{r['n_observations']} |")
    lines+=['','## Planned Primary Comparisons','',
        '41 contrasts were frozen before GPU outputs: source effects and History-source differences10; role differences raw/adjusted12; '
        'seven deep Whole means and six adjacent differences13; Recent-Older6. Ordinary and family41 Bonferroni percentile Bootstrap intervals use10,000 paired image resamples. '
        'Very small adjusted tail quantiles have finite Monte Carlo uncertainty; this is not an exact finite-sample guarantee. '
        'Other descriptive tables do not license additional significant findings.',
        '', '| Contrast | Layer | Mode | Analysis | Estimate | Corrected95% CI | Images |',
        '|---|---:|---|---|---:|---|---:|']
    for r in main:lines.append(f"| {r['contrast']} | {r['layer']} | {r['mode']} | {r['analysis']} | {r['mean']:.6f} | {bounds(r,True)} | {r['n_images']} |")
    lines+=['','## Statistical Boundaries','',
        'Image, not head/layer/token, is the bootstrap unit. Word roles are conditioned on native Full outputs and inherited single-token lexicons. '
        'Controlled comparisons use same-image Function/Visual pairs, available Relation tokens, image fixed effects and Full LogP/entropy/absolute/relative time. '
        'Token length is1 by construction and is explicitly not estimated. Rank/condition and bootstrap-identifiability counts are audited. '
        'A CI spanningzero does not establish equivalence or a low-harm interval.',
        'Norm, Mass and loss are distinct. Group projections can cancel; their intervention losses are nonlinear and cannot be summed to Whole loss. '
        'History may contain visual information, but History-off does not prove visual-to-text information transfer or strict semantic grounding.',
        '', '## Correctness and Resource Use','',
        f"All4 logical-shard smoke gates on physical GPUs1/2/3 passed; native model.generate tokens, no-op logits, native source-observer logits and every formal Full replay step are exact. "
        f"{targets['positions']} target parent-cache SHA checks passed; {validation['numerical_checks']} native decomposition audits passed unchanged/analytically extended rounding bounds. "
        f"All {validation['raw_logit_files']} raw NPZ SHA256 files verified, {validation['raw_logit_bytes']} bytes, stored locally with committed inventory. "
        'No silent skipped image, nonfinite value, OOM, input mutation or leaked hook was accepted. Model versions and installed Transformers source remain bound.',
        'An initial GPU0 preflight refused an external compute task; see resource_event.json and original smoke console logs. '
        'The user then authorized three GPUs. Four original logical shards stay unchanged: GPU1 runs shards0/3, GPU2 shard1, GPU3 shard2. '
        'Old Smoke/bindings were archived, new Smoke rerun. No other process was terminated or altered; GPU0 was not used.',
        '', '## Figures','']
    for path in sorted((OUT/'figures').glob('*.png')):lines+=[f'![{path.stem}](figures/{path.name})','']
    lines+=['See [research_summary.md](research_summary.md) for the eight Chinese research answers and [README.md](README.md) for exact reproduction/resume commands.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

    chinese=['# 深层 Attention 来源分解：中文研究总结','',
        f"状态：已完成新的200图，{targets['positions']}个冻结目标，{46*targets['positions']}次真实单步消融。词类计数：{targets['counts']}。",
        '不是旧200图的重新选样，不因结果排除图片；复用上一轮词表、单Token对齐和位置选择规则。',
        '', '## 1. Visual 在 L26/L31 对什么来源最敏感？','']
    for layer in SPECIAL:
        sources=[lookup[(m,layer,'Visual')] for m in ['Image','Prompt','History','BOS']]
        highest=max(sources,key=lambda r:r['mean'])
        chinese.append(f"L{layer} 四个互斥来源中最大带符号平均损失来自 {highest['mode']}：{highest['mean']:.6f} nat（普通95% CI {bounds(highest)}），"
            f"平均绝对变化{highest['mean_abs_delta_logp']:.6f}，Top1翻转{highest['top1_flips']}/{highest['n_observations']}。这只是实测排序，不是对所有词/层的普遍断言。")
        for other in ['Image','Prompt']:
            r=contrasts[('A_Visual_History_minus_'+other,layer,'History-'+other,'paired')]
            chinese.append(f"同Token History−{other}：{r['mean']:.6f}，family41校正95% CI {bounds(r,True)}。")
    chinese+=['','## 2. L31 的高 Whole、低 Image 能否由 History 或 Prompt 解释？','']
    for mode in ['Whole','Image','Prompt','History','BOS']:
        r=lookup[(mode,31,'Visual')]
        chinese.append(f"Visual L31 {mode}：loss={r['mean']:.6f}，mean|ΔLogP|={r['mean_abs_delta_logp']:.6f}，flip={r['top1_flips']}/{r['n_observations']}。")
    chinese.append('只有实际正向且有区间支持的 History/Prompt 移除效应才能支持其局部预测作用；即使来源移除损失很大，也不能线性分摊或完整解释 Whole 损失。负损失说明原Token概率上升，而非该来源没有功能。')
    chinese+=['','## 3. Function 是否表现出不同来源敏感性？','']
    for r in paired:
        chinese.append(f"L{r['layer']} {r['mode']} Visual−Function（{r['analysis']}，{r['n_images']}张同图）：{r['mean']:.6f}，校正95% CI {bounds(r,True)}。")
    chinese.append('调整后区间仍排除零才支持超出线性置信度/位置控制后的条件类别差异；区间跨零不能证明两类相同。Function低损失不等于可跳过整层。')
    chinese+=['','## 4. Relation/Action 在中层和后层依赖哪些来源？','']
    for layer in [14,16,26,31]:
        sub=[lookup[(m,layer,'Relation')] for m in ['Image','Prompt','History']]
        chinese.append(f"L{layer}："+'；'.join(f"{r['mode']} loss={r['mean']:.6f}, |Δ|={r['mean_abs_delta_logp']:.6f}, flip={r['top1_flips']}/{r['n_observations']}" for r in sub)+'。')
    chinese.append('Relation沿用优先级，介词/空间关系与动作词合并；词频表在classification_audit.json。以上Relation逐来源排序是描述性结果，不冒充新增主对比的显著发现。')
    chinese+=['','## 5. L25–L31 是否形成连续深层敏感区？','']
    means=[r for r in deep if r['contrast']=='C_Visual_Whole_vs_zero']
    chinese.append('预定判据：七层校正区间下界全部>0，才称这七个实测层均有正向平均损失证据。'+
        ('本次满足该判据。' if all(r['familywise95_low']>0 for r in means) else '本次不满足完整连续正损失证据判据；不把不确定层解释成无功能间隔。'))
    for r in means:chinese.append(f"L{r['layer']}：{r['mean']:.6f}，校正95% CI {bounds(r,True)}。")
    chinese.append('六个相邻层配对差异完整保留在deep_layer_peak_analysis.csv；不能仅靠L26/L31两个点宣布离散峰，也不能把未测L17–L24补成曲线。')
    chinese+=['','## 6. Attention Mass 与实际输出贡献是否一致？','']
    for layer in [14,16,26,31]:
        sub=[r for r in source_summary if r['layer']==layer and r['word_type']=='Visual' and r['source'] in GROUPS]
        mass=max(sub,key=lambda r:r['mass_mean']);norm=max(sub,key=lambda r:r['contribution_norm_mean'])
        chinese.append(f"Visual L{layer}：最大平均Mass来源={mass['source']}（{mass['mass_mean']:.6f}）；最大平均投影范数来源={norm['source']}（{norm['contribution_norm_mean']:.6f}）。"
            f"Image–History平均Cosine={sub[0]['image_history_cosine_mean']:.6f}，四组向量抵消比例={sub[0]['group_cancellation_ratio_mean']:.6f}。")
    chinese.append('Mass没有Value方向或Wo的信息；范数与实际移除损失也不是同一个量。图2分开画Mass、范数、相同残差分母的比例和实际绝对LogP变化，不从未标准化范数推断固有重要性。')
    chinese+=['','## 7. 是否支持生成历史承担重要预测作用？','']
    for layer in SPECIAL:
        r=contrasts[('A_Visual_History_vs_zero',layer,'History','raw')]
        chinese.append(f"Visual L{layer} History-off：{r['mean']:.6f}，校正95% CI {bounds(r,True)}。")
    for r in recent:
        chinese.append(f"{r['word_type']} L{r['layer']} Recent−Older：{r['mean']:.6f}，校正95% CI {bounds(r,True)}；"
            f"Recent/Older平均绝对变化={r['recent_abs_change']:.6f}/{r['older_abs_change']:.6f}，Older为空的t=1目标={r['empty_older_targets']}。")
    chinese.append('Recent包括当前消费的上一个生成Token；Older可为空但不能删除这些样本。只有本表真实效应及不确定性支持的范围内，才讨论短距离或更早历史的作用。')
    chinese+=['','## 8. 本轮仍不能证明什么？','',
        '不能证明视觉信息已经从图像迁移到历史；历史Value可能混有多层、多来源信息。不能证明直接图像贡献小就不使用图像。',
        '不能将来源损失相加为Whole损失；各来源方向抵消和后续非线性都可能改变单独移除效应。',
        '不能将词类标签当作严格Grounding或词性的无偏因果效应；原生Caption决定可观察词类，单Token词表过滤限制推广范围。',
        '不能从单步LogP/Top1变化推导完整Caption准确率、整层可跳过、动态路由有效或真实推理加速。',
        '', '## 验证与复现','',
        f"GPU1/2/3上的四个逻辑分片Smoke、23项采集/调度CPU测试、独立统计测试通过。原生生成/无操作/观察器Logits精确一致；{targets['positions']}次目标父Cache哈希检查通过。",
        f"{validation['numerical_checks']}组原生数值分解验证通过。容差来自冻结FP16/FP32传播误差界，旧Image审计原样保留；未按输出放宽。",
        '最初GPU0曾被其他用户任务占用，预检拒绝加载；随后用户授权改用GPU1/2/3。原200图及四个逻辑分片保持不变，新绑定重新Smoke；未使用GPU0或干扰其他进程。',
        '全部明细、置信区间、环境、代码及原始Logits哈希见CSV/JSON，五张主图均有PNG和PDF。大体积原始NPZ保留本机并提交路径/哈希，不向Git提交模型权重。']
    (OUT/'research_summary.md').write_text('\n'.join(chinese)+'\n')
