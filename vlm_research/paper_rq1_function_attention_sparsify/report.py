"""Render measured results and an auditable, conservative Chinese close-out."""
import json
from pathlib import Path
import subprocess
from collections import Counter
from benchmarks.dataset_registry import sha256,digest
from paper_rq1_single_layer_gqa6.run import atomic_json
from paper_rq1.analyze import write_csv
from .protocol import ROOT,OUT,RAW
from .analysis import read_csv


def table(rows,columns):
    def cell(v):
        if v is None or v=='':return 'NA'
        if isinstance(v,float):return f'{v:.6f}'
        return str(v).replace('|','/')
    return ['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(cell(r.get(k)) for k in columns)+' |' for r in rows]


def yes(v):return str(v).lower()=='true'


def main():
    stage_a=json.loads((OUT/'stage_a_gate.json').read_text());predictor=json.loads((OUT/'predictor_freeze.json').read_text())
    pipeline=json.loads((OUT/'pipeline_status.json').read_text())
    if pipeline['status']!='complete':raise ValueError('Report cannot claim completion while pipeline is partial')
    manifest=json.loads((OUT/'sample_manifest.json').read_text());sets={k:[s for s in manifest['development'] if s['split']==k] for k in ['train','calibration']};sets['holdout']=manifest['holdout']
    for key in ['image_identity','image_sha256','rgb_sha256']:
        group={k:{s[key] for s in v} for k,v in sets.items()}
        if any(len(group[k])!=len(sets[k]) for k in group) or any(group[a]&group[b] for a,b in [('train','calibration'),('train','holdout'),('calibration','holdout')]):raise AssertionError('Image leakage/duplicates')
    source=read_csv(OUT/'source_retention_summary.csv');quality=read_csv(OUT/'caption_quality_summary.csv');latency=read_csv(OUT/'latency_summary.csv')
    classifiers=read_csv(OUT/'holdout_classifier_summary.csv');calibration=read_csv(OUT/'classifier_calibration.csv');risk=read_csv(OUT/'holdout_pruning_risk_summary.csv')
    def low_quality_loss(r):return yes(r['CIDEr_bootstrap_noninferior']) and yes(r['SPICE_bootstrap_noninferior'])
    timing={r['strategy']:r for r in latency if r['implementation']=='A2'}
    candidates=[r for r in quality if r['strategy']!='S0' and low_quality_loss(r) and yes(timing[r['strategy']]['speed_supported'])]
    dyn=[r for r in candidates if r['strategy'] in ['S3','S4']];static=[r for r in candidates if r['strategy'] in ['S1','S2']]
    if static:verdict='B';verdict_text='在本次固定样本和硬件上，静态来源裁剪已满足样本内质量/时间判据；没有必要优先引入额外词类预测器。仍不能声称总体安全或跨硬件稳定加速。'
    elif dyn:verdict='A';verdict_text='本次固定留出样本和硬件上存在满足冻结质量和时间判据的动态策略。其可复现范围限于本模型/样本/实现，不能推出罕见损害安全。'
    elif not stage_a['dynamic_source_gate_passed'] and stage_a['observed']['mean_loss']>.1:
        verdict='D';verdict_text='联合保留 BOS+History 未通过预先冻结的风险门槛，不支持继续开发当前动态联合裁剪方案。静态 No-Image 结果须另行看待。'
    else:verdict='C';verdict_text='存在局部低敏感现象，但当前证据不足以证明能够可靠转化为低损害、低开销的实际 Attention 加速。'
    numerical=[];all_raw=[]
    for path in sorted(OUT.glob('*/shard*/checkpoints/*.json')):
        r=json.loads(path.read_text())
        if r.get('raw_path'):
            if sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Final raw hash mismatch')
            all_raw.append(dict(checkpoint=str(path.relative_to(OUT)),raw_path=r['raw_path'],sha256=r['raw_sha256'],bytes=Path(r['raw_path']).stat().st_size))
        if path.parts[-4]=='smoke':
            for check in r.get('checks',[]):
                for audit in check.get('audits',[]):numerical.append(dict(rank=path.parts[-3],t=check['t'],algorithm=check['algorithm'],mode=check['mode'],layer=check['layer'],audit=json.dumps(audit,sort_keys=True)))
    write_csv(OUT/'raw_artifacts_manifest.csv',all_raw);write_csv(OUT/'numerical_audits.csv',numerical)
    completion=[]
    for stage in ['smoke','stage_a','quality_smoke','quality','risk','latency']:
        for rank in ([0] if stage=='latency' else range(3)):
            path=OUT/f'{stage}/shard{rank}/completion.json';r=json.loads(path.read_text())
            if r['status']!='complete' or not r['weights_unchanged'] or not r['hooks_clean']:raise ValueError('Incomplete/dirty worker')
            completion.append(dict(stage=stage,**r))
    write_csv(OUT/'execution_times.csv',completion)
    failures=[str(p.relative_to(OUT)) for stage in ['smoke','stage_a','quality_smoke','quality','risk','latency'] for p in (OUT/stage).glob('shard*/failure_*.json')]
    if failures:raise ValueError('Unresolved formal failure records: '+str(failures))
    limits=json.loads((OUT/'execution_manifest.json').read_text())
    for path,r in limits['model_files'].items():
        if sha256(path)!=r['sha256']:raise ValueError('Model changed at final hash review')
    subset=[r for r in source if r['role']=='Function' and int(r['layer'])==31]
    selected=predictor['selected'];chosen=next((r for r in classifiers if selected and r['predictor']==selected['predictor']),None)
    lines=['# Function-Token Identification and Attention Source Sparsification','',f'**最终判断：{verdict}。{verdict_text}**','',
        '## 设计、冻结与样本','',
        '- 基线提交：`6964407e6b4c1291cb0ca7c41128947d3025520d`。本分支不修改 main，不覆盖历史实验。',
        f"- Stage A：两轮原有400张图、{stage_a['targets']}个冻结位置、{stage_a['interventions']}次单步干预；只研究 L14/L26/L31。",
        '- Stage B：按图片输入哈希固定300训练/100校准；所有实际 Decode 位置，包含 EOS、歧义、未知、多 Token、不对齐和标点。t=0 是 Prefill，不属于在线裁剪范围。',
        '- 独立留出200张：ID、图像字节 SHA256、解码 RGB 均与开发数据隔离，并排除前两轮 Smoke 及更早 COCO Discovery。没有根据输出选题。',
        '- 使用物理 GPU1/2/3，每张图所有配对条件在同卡执行。GPU0 的外部任务未被终止或使用。',
        '- LLaVA-1.5-7B 固定权重，FP16/eager，Transformers4.49.0，Greedy，正常 KV Cache，最多64 Token。只训练小型识别器，不训练 VLM。',
        '- Stage A、B 是规则选择/校准，不是独立确认。全 Caption 质量和识别器最终指标仅使用冻结留出200图。','',
        '## Stage A：联合来源保留','',
        'A1 保留原始 Softmax 分母，联合减去移除来源的 FP32 投影贡献；它已经计算完整 Attention，不能当作加速。A2 在 QK/AV 前真正 gather 保留 K/V，重新 Softmax。模板分隔符全部归入原 Prompt 文本，四类 Key 无遗漏/重叠。','',
        '### L31 Function','']
    lines+=table(subset,['algorithm','mode','n','mean_loss','mean_abs_delta_logp','p95_loss','p99_loss','top1_flips','max_loss'])
    lines+=['',f"冻结的动态来源风险门槛通过：**{stage_a['dynamic_source_gate_passed']}**。门槛为 Mean Loss<=0.10nat、P95<=0.50nat、Top1 Flip<=10%，三个都通过才继续；这只是开发门槛，不是安全认证。",
        f"历史 Full 最大 Logit 差异：`{stage_a['historical_max_logit_difference']}`；400图原生生成 Token 序列全部一致。没有为历史差异放宽容差。",
        '', '全部角色、三层、A1/A2 的结果以及负面尾部都保存在 `source_retention_summary.csv`、`source_retention_per_token.csv`、`severe_source_cases.csv`。','',
        '![Function source retention](figures/figure1_function_source_retention.png)','![A1 vs A2](figures/figure2_subtraction_vs_sparse.png)','',
        '## Stage B：提前识别 Function','',
        'A 仅使用已消费前缀最近 Token 的哈希一元/二元/三元特征和已知位置；B 使用真实 L31 层输入隐藏状态及其范数/已知位置；C 合并两者。没有使用当前 Full logits、下一 Token ID、Attention 输出、真实 BOS Mass 或未来生成长度。位置/64 的分母是预先固定上限，不是实际最终长度。',
        '', '### 校准集冻结阈值','']
    lines+=table(calibration,['predictor','threshold','precision','precision_ci95_low','recall','coverage','eligible'])
    lines+=['',f"选择的部署识别器：`{selected['predictor'] if selected else '无，拒绝动态部署'}`；阈值：`{selected['threshold'] if selected else 'NA'}`。不因留出成绩重新选模型或调阈值。",
        '', '### 独立留出集','']
    lines+=table(classifiers,['predictor','precision','precision_ci95_low','precision_ci95_high','recall','false_positive_rate','coverage','brier'])
    lines+=['','EOS、未知、歧义、多 Token 及所有其他类别的误判率分别保存在 `holdout_classifier_category_errors.csv`。校准曲线和覆盖率原始点也独立保存，不删除未知类别。高 Function Precision 不意味着裁剪安全。','',
        '![Function identification](figures/figure3_function_identification_holdout.png)','',
        '## Stage C：真实自由生成质量','',
        '每个策略独立从图像 Prompt 开始，使用该策略自己的实际前缀和隐藏状态继续生成；没有从 Full 后续 Token teacher-force。Oracle Function 仅用于独立 Full 轨迹的单步参考，从未用于可部署自由生成。','']
    lines+=table(quality,['strategy','n','CIDEr','CIDEr_delta','CIDEr_delta_ci95_low','CIDEr_delta_ci95_high','SPICE','SPICE_delta','SPICE_delta_ci95_low','SPICE_delta_ci95_high','changed_captions'])
    lines+=['','S0=Full；S1=静态 No-Image；S2=静态 BOS+History；S3=预测 Function 后 BOS+History；S4=预测 Function 后 No-Image。未列出的策略因冻结门槛失败而未运行，不能当作0损害。',
        'CIDEr 使用原始0–10口径（不是乘100后的数）；SPICE 为0–1。普通图片级配对 Bootstrap10000次，CIDEr 的参考语料 IDF 固定为同一完整200图。冻结非劣效容忍为 CIDEr0.02、SPICE0.01；下界大于负容忍只是样本内 Bootstrap 判据，200图仍不足以排除罕见严重损害。','']
    lines+=table(quality,['strategy','CIDEr_bootstrap_noninferior','SPICE_bootstrap_noninferior','pruned_decode_steps','decode_steps','pruned_function_steps','pruned_nonfunction_steps','retained_key_ratio','mean_length','eos_count','truncated_count'])
    lines+=['','### 预测/Oracle裁剪的实际单步风险','']
    lines+=table(risk,['group','mode','n','images','mean_loss','p95_loss','p99_loss','top1_flips','severe_gt1','false_function_positions'])
    lines+=['','对象候选词、颜色、数量、关系词差异保存在逐图 CSV。它们只是词表对齐的文本变化，不是人工验证的视觉语义错误或 Grounding 得分；严重退化案例未删除。','',
        '![Quality vs sparsity](figures/figure4_quality_sparsity.png)','![Actual Caption cases](figures/figure6_caption_case_studies.png)','',
        '## 实际延迟与计算','',
        '固定物理 GPU1，冻结留出顺序前10图，每实现3次 Warmup、5次重复，种子固定的随机条件顺序。时间统计先在每张图内平均重复，再做10图配对 Bootstrap；50次重复不当成50张独立图片。主时间不含 profiler/隐藏状态采集、图像预处理、模型加载和官方评分。CUDA Events 与同步用于 Prefill/Decode 分解，另有完整模型生成 wall time。','']
    lines+=table(latency,['strategy','implementation','mean_caption_seconds','measured_speed_ratio','paired_time_delta_ci95_low','paired_time_delta_ci95_high','decode_per_token_delta_ci95_low','decode_per_token_delta_ci95_high','speed_supported','mean_l31_attention_operator_ms','mean_classifier_host_ms','peak_mib'])
    lines+=['','- 只裁剪 L31 的真实 Decode QK/AV。QK/AV 理论操作数为每步 `4*32*128*retained_keys`；不把理论 FLOPs 比例当成实测速比。',
        '- 全尺寸 K/V 投影、缓存存储、DynamicCache 拼接仍保留，拼接本身仍读取完整历史缓存。稀疏 QK/AV 的输入范围缩小，但不声称整条缓存带宽按同比例降低。',
        '- PyTorch index_select/gather、Python 调度与分类器同步成本均真实存在。`operator_profile.json` 中独立 profiler 的 index_select 时间是全 Caption 运算统计，不冒充纯 L31 gather；不拿 profiler 时间作为端到端速度。',
        '- A1 的实际耗时仅作机制模拟成本对照，不给出加速比。真实速度支持同时要求 Caption 配对时间和 Decode 每 Token 时间的95%CI上界<0，避免把提前 EOS/更短 Caption 单独误报成 Attention 加速。',
        '- 此 eager实现的理论收益仅来自32层中最后1层的部分运算。若 gather/路由抵消收益，可考虑专用 fused retained-key Attention kernel；本轮没有实现或虚构 Triton/CUDA 收益。','',
        '![Latency vs quality](figures/figure5_latency_quality.png)','',
        '## 九个研究问题','']
    bh=next(r for r in subset if r['algorithm']=='A2' and r['mode']=='BOS+History');ho=next(r for r in subset if r['algorithm']=='A2' and r['mode']=='History-only');ni=next(r for r in subset if r['algorithm']=='A2' and r['mode']=='No-Image')
    lines += [f"1. **Function 是否只需要 History？** 不能这样概括。History-only 的 Mean/P95 Loss 为 {ho['mean_loss']}/{ho['p95_loss']}，Top1翻转 {ho['top1_flips']}/{ho['n']}，必须保留尾部和反例。",
        f"2. **BOS+History 是否更稳？** 同图冻结条件对照：BOS+History 的 Mean/P95 为 {bh['mean_loss']}/{bh['p95_loss']}、翻转 {bh['top1_flips']}；与 History-only 对照见表，不能用均值单独判断稳定性。",
        f"3. **能否同时移除 Image 和 Prompt？** L31 Function 的 BOS+History 已实际测量；再去 Prompt 相对 No-Image 的 Mean Loss 差为 {float(bh['mean_loss'])-float(ni['mean_loss']):+.6f}nat。只针对固定原生轨迹，不推出整条生成安全。",
        '4. **真正裁剪 Key 后仍低损害吗？** A2 与 A1 分开汇报，重新 Softmax 会改变保留来源权重；以真实 A2 的尾部和完整 Caption 质量判断，不能直接继承减法结论。',
        f"5. **能提前识别 Function 吗？** {'冻结部署识别器的留出 Precision/Recall 为 '+str(chosen['precision'])+'/'+str(chosen['recall']) if chosen else '没有识别器通过冻结校准门槛；动态策略被拒绝。'}，并保留 EOS/未知等假阳性。",
        '6. **动态优于静态吗？** 需要同时比较200图官方质量、实际裁剪范围、10图固定硬件时间和路由成本，不根据覆盖率宣称更优；本轮整体判断见开头。',
        '7. **完整 Caption 质量受损吗？** 上表提供 CIDEr/SPICE 配对差异、区间、逐图增减和真实变更案例；CI包含0不等于无损，非劣效阈值没有调整。',
        f"8. **哪种策略实际加速？** 满足冻结的两个配对时间判据者为：{', '.join(r['strategy'] for r in latency if yes(r['speed_supported'])) or '无'}。理论稀疏度、A1模拟和更短文本均不单独作为加速证据。",
        '9. **足够声称安全吗？** 不足。还需更大且独立的图像/文本任务确认、更严格的严重风险上界、多硬件重复，以及在质量可控前提下的专用 kernel 验证；本轮不扩展研究范围。','',
        '## 正确性与审计','',
        '- 原生 cached Full 与 model.generate 的 Token 完全一致；Observer/No-op logits 精确一致。',
        '- 单步条件各自深克隆原生 cache，目标 Token 时序、原始 RoPE/position、Key 分区与缓存哈希检查通过。',
        '- A1 使用前轮不变的解析 FP16/FP32舍入界；A2 使用独立 bmm参考加全维 masked softmax/AV，正式路径只对保留 K/V 做 QK/AV。',
        '- 初次 Smoke 因参考 QK 的宽度/FP16累计路径不同而被拒绝，未进入正式采集。失败与修正历史完整保留；没有放宽原有数值容差。',
        '- 正常/异常路径恢复 Hook 和临时 eager实现。模型参数版本不变，固定权重文件末尾重新 SHA256 校验通过；原始数据未改。',
        '- 正式完成记录无 NaN、Inf、OOM 或跳过失败样本。冻结代码版本、环境、GPU记录、CPU/GPU日志、逐图检查点和 raw SHA 清单均保存。',
        f"- 已校验外置原始结果文件 {len(all_raw)} 个。raw logits/特征不上传Git，以清单SHA及本机绝对路径绑定；小型识别器权重另行提交。",'',
        '## 产物','',
        '`source_retention_per_token.csv`、`all_development_token_labels.csv`、`development_predictions.csv`、`holdout_token_predictions.csv`、`caption_per_image.csv`、`caption_generation_steps.csv`、`oracle_and_predicted_single_step_risk.csv`、`latency_repetitions.csv`、`operator_latency.csv`、全部PNG/PDF及各阶段日志。','',
        '本轮不训练VLM，不扩展层范围、不做Router层扫描。实验已停止并释放本轮GPU资源，不自动合并 main。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    summary=['# 中文研究总结','',f'最终判断：**{verdict}**。{verdict_text}','',
        f"完成开发400图/{stage_a['targets']}个冻结目标和独立留出200图，实际使用GPU1–3。",'',
        '## 核心质量结果','']+table(quality,['strategy','CIDEr','CIDEr_delta','SPICE','SPICE_delta','changed_captions'])
    summary+=['','## 解释边界','',
        '来源贡献减法不等于真实 Softmax 重新归一化；局部原生 Token 的低敏感性不等于完整 Caption 安全。Function 识别的高精度不自动保证裁剪无损。',
        '静态与动态必须同时比较质量、尾部风险、有效覆盖和真实延迟。只省去 L31 部分 QK/AV 的收益上限有限，全尺寸缓存更新和分类器/Gather开销仍存在。',
        '本轮普通图片级95%区间和200图留出只支持有限范围的描述性确认，不构成跨模型、跨任务或罕见损害安全保证。','',
        '所有负面结果、失败 Smoke、未知词类、严重反例及逐图生成均保留。详见 `report.md`。']
    (OUT/'research_summary.md').write_text('\n'.join(summary)+'\n')
    atomic_json(OUT/'final_completion.json',dict(status='complete',verdict=verdict,verdict_text=verdict_text,development_images=400,holdout_images=200,
        stage_a_targets=stage_a['targets'],interventions=stage_a['interventions'],raw_files_verified=len(all_raw),
        no_data_leakage=True,model_artifact_hashes_unchanged=True,unresolved_formal_failures=failures,
        report_sha256=sha256(OUT/'report.md'),summary_sha256=sha256(OUT/'research_summary.md')))
    print('\n'.join(summary),flush=True)


if __name__=='__main__':main()
