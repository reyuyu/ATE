from .common import *


def bound(row, field='mean'):
    return f"{row[field]:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"


def report(checkpoints, variations, pairs, adjusted, consistency, correlations):
    coverage = read(OUT / 'token_coverage.json')
    completion = read(OUT / 'collection_completion.json')
    lines = ['# COCO逐词Decode Attention来源观察', '',
        '## A. L31来源是否逐词变化？', '',
        f"严格复用历史200张COCO图片，{coverage['tokens']}个原生生成Token，其中{coverage['decode_tokens']}个为真实增量Decode；"
        f"{coverage['multi_token_words']}个多Token词和{coverage['nonfirst_lexical_subwords']}个非首子词均保留。",
        '以下数值为每张图内部变化的图片级均值与普通95% Bootstrap区间；首Token Prefill不混入增量主分析。', '',
        '| 来源 | 每图平均Mass | 每图范围(max−min) |', '|---|---|---|']
    for source in MAIN:
        mean = next(r for r in variations if r['analysis'] == 'non_eos_decode' and r['layer'] == 31 and r['source'] == source and r['metric'] == 'mean')
        spread = next(r for r in variations if r['analysis'] == 'non_eos_decode' and r['layer'] == 31 and r['source'] == source and r['metric'] == 'range')
        lines.append(f"| {source} | {bound(mean)} | {bound(spread)} |")
    lines += ['', '范围刻画同一Caption内非EOS位置的实际变化，不等于不同词类的因果差异。完整含EOS/不含EOS、32层和每图最小/最大/均值/标准差均在CSV。', '',
        '## B. 词类关联与位置控制', '',
        '主分析使用真实词的首子词预测位置，Function/Visual/Relation沿用旧词典及优先级。Ambiguous并入Unknown但保留原标签；Visual只是词表候选，不等于已验证视觉Grounding。',
        '原始均值采用同图词类均值配对；调整模型采用图片固定效应，每张Caption等权，控制绝对/相对生成位置、原生Token LogP/Entropy及词的Token长度。',
        'Generated History合法Key数量严格等于t，因此t同时控制历史长度；不再加入完全共线的Key数量列。另报告单位Key Mass和非BOS条件占比。', '',
        '| 结果 | 比较 | 原始同图差异 [95%CI] | 调整后系数 [95%CI] |', '|---|---|---|---|']
    for outcome in ['BOS', 'IMAGE', 'GENERATED_HISTORY', 'nonbos_IMAGE', 'nonbos_GENERATED_HISTORY', 'image_history_ratio']:
        for comparison, variable in [('Visual-Function', 'Visual_vs_Function'), ('Relation-Function', 'Relation_vs_Function')]:
            raw = next(r for r in pairs if r['outcome'] == outcome and r['contrast'] == comparison)
            fit = next((r for r in adjusted if r['outcome'] == outcome and r['variable'] == variable), None)
            lines.append(f"| {outcome} | {comparison} | {bound(raw)}; n={raw['n_images']} | {bound(fit, 'effect') if fit else '不可识别，见审计'} |")
    significant = [r for r in adjusted if r['variable'] in ['Visual_vs_Function', 'Relation_vs_Function'] and
                   r['outcome'] in ['IMAGE', 'GENERATED_HISTORY', 'nonbos_IMAGE', 'nonbos_GENERATED_HISTORY'] and
                   (r['ci95_low'] > 0 or r['ci95_high'] < 0)]
    lines += ['', '调整后的描述性区间排除0的项目：'+('；'.join(f"{r['outcome']}/{r['variable']}" for r in significant) if significant else '本批未观察到这些来源指标的明确条件类别差异')+'。',
        '这些普通区间未作确认性多重比较保证；不能把关联解释为词类驱动注意力的因果效应，也不能从区间跨0宣称分配恒定。', '',
        '## C. Mass相近但输出贡献改变？', '',
        '预先固定的描述性“Mass相近”：相邻增量位置四来源JS≤1e−4 nat且非BOS三来源JS≤1e−3 nat。',
        '“贡献变化”：同来源投影向量范数相对变化≥20%，或相邻向量Cosine≤0.90。阈值只用于可复核描述，不是等价性或必要性检验。', '',
        '| 层 | 来源 | 相近Mass位置对 | 贡献变化位置对 | 涉及变化图片 | 图片平均变化率 [95%CI] |', '|---|---|---:|---:|---:|---|']
    for r in consistency:
        if not r['include_eos'] and r['source'] in ['IMAGE', 'GENERATED_HISTORY']:
            lines.append(f"| L{r['layer']} | {r['source']} | {r['eligible_pairs']} | {r['changed_pairs']} | {r['images_with_change']} | {bound(r)}; n={r['n_images']} |")
    late = [r for r in consistency if r['layer'] == 31 and not r['include_eos'] and r['source'] in ['IMAGE', 'GENERATED_HISTORY']]
    lines.append('L31在上述预声明口径下'+('存在' if any(r['changed_pairs'] > 0 for r in late) else '未观察到')+'“Mass相近而投影贡献变化”的真实位置对。范数和方向仍是观察量，不是独立因果必要性。')
    lines += ['', '## D. 是否L31特有？', '']
    for layer in SELECTED:
        rows = [r for r in consistency if r['layer'] == layer and not r['include_eos'] and r['source'] in ['IMAGE', 'GENERATED_HISTORY']]
        lines.append(f"L{layer}："+'；'.join(f"{r['source']}变化{r['changed_pairs']}/{r['eligible_pairs']}对，{r['images_with_change']}图" for r in rows)+'。')
    if any(r['layer'] in [14, 26] and not r['include_eos'] and r['source'] in ['IMAGE', 'GENERATED_HISTORY'] and r['changed_pairs'] > 0 for r in consistency):
        lines.append('这种分离并非L31独有；在实测L14/L26也可观察。只在L14/L26/L31做了投影分解，不能推广为所有深层均具有同样贡献结构。')
    else:
        lines.append('本次未在另外两个实测层观察到相同口径的事件；样本和阈值限制使其不能证明L31独有。')
    lines += ['', '## E. Attention Allocation vs Functional Necessity', '',
        '支持点：同一Caption内分配和来源输出贡献可以分别量化；上述配对数据直接检验它们是否同步变化，而不是预设BOS/Image/History必须动态切换。',
        '反证与限制：若原始/非BOS分配或位置控制后词类差异仍明显，则“分配始终不变”过强；若Mass与贡献呈明显同向关联，也不能只强调分离案例。',
        '本轮没有移除Attention或来源，没有新的因果必要性/质量实验；Wo(PV)范数不等于去掉该来源造成的LogP损害。来源向量可能抵消，历史Value也可能携带视觉信息。',
        '因此不能据此宣称视觉语义检索、某层可安全跳过或已经实现计算加速。应将本轮观察与历史独立记录的消融损害严格区分。', '',
        '## 正确性、资源与文件', '',
        f"200图原生与只读采集的全部生成Token及有效Logits严格一致；八图GPU Smoke通过。所有32层、32Head和每步均覆盖，行和容差沿用{read(OUT/'protocol.json')['numerical']['row_atol']}。",
        '实际来源位置动态构建、互斥完备；生成采用两次独立原生Cache，未保留/修改其他条件Cache。原生输入与权重版本未变，Hook和eager包装均清理。',
        f"与历史Caption轨迹一致：{coverage['historical_caption_matches']}/200；历史逐Logits数值差异逐图记录，不隐去跨运行差异。",
        f"GPU1采集耗时{completion['gpu_seconds']/60:.2f}分钟；峰值显存见collection_completion.json及environment.json。",
        'per_token.csv保留消费/预测Token与词类、词对齐、LogP/Entropy；per_token_layer_mass.csv保留32层质量、非BOS比例、Key数量和相邻JS。',
        'per_head_mass/*.csv.gz保存每图每步每层每Head全部五类Mass（9位有效数字可还原FP32）；完整词表原生Logits及未平均Mass在本机raw_path指向的NPZ并有SHA256绑定。',
        'per_token_contributions.csv保留三层全部来源投影范数、相对残差比例、方向及相邻向量变化；模型/数据/源码绑定、CPU/GPU验证和每图检查点随分支保存。', '',
        '图1代表图片在GPU前固定为原Manifest第一图，不按效果选取。首Token Prefill和后续增量位置用虚线分隔；多子词的后续位置标记·2/·3。F/V/R/U/P/E为词类。',
        '图2保留全部生成Token，统一Image/History色标，标记L31。图3上排使用同时具备三类词的同图集合，下排是各自可用同图配对区间。',
        '图4上排每点为一图一来源的均值，纵轴为投影范数/该层残差范数；下排使用预声明近Mass阈值和贡献变化阈值，排除EOS。',
        'Bootstrap只重采样图片SHA256簇，不将Token/Head/Layer当作独立样本；相同词的其他子词及EOS分析在subword_and_eos_summary.csv。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT / 'analysis_completion.json', dict(complete=True, images=200,
          validation='native and observed all token logits exact', coverage=coverage,
          all_four_pdf_present=all((OUT/(f+'.pdf')).exists() for f in [
              'figure1_caption_timeline', 'figure2_layer_token_heatmap', 'figure3_word_role_distribution', 'figure4_mass_vs_contribution'])))
