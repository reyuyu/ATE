"""Report and final immutable delivery audit, only after complete generation/replay."""
import argparse,csv,datetime,os,subprocess
import numpy as np
from .common import *

def table(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def number(r,k):return float(r[k])
def item(rows,pop,c):return next(r for r in rows if r['population']==pop and r['condition']==c)
def fmt(r):return f"{number(r,'accuracy_percent'):.3f}%（ΔAcc {number(r,'delta_accuracy_pp'):+.3f}pp，95%CI [{number(r,'delta_accuracy_ci_low'):+.3f}, {number(r,'delta_accuracy_ci_high'):+.3f}]）；ΔNLL {number(r,'delta_gold_nll'):+.4f} nat/token [{number(r,'delta_nll_ci_low'):+.4f}, {number(r,'delta_nll_ci_high'):+.4f}]"
def put(path,text):
    path=Path(path);assert not path.exists();path.write_text(text,encoding='utf-8',newline='\n')

def report():
    p=verify();assert read(OUT/'statistics_validation.json')['passed'] and read(OUT/'figure_validation.json')['passed'];summary=table('condition_summary.csv');paired=table('paired_comparisons.csv');swaps=table('image_swap_summary.csv');confirm=read(OUT/'confirmatory_tests.json');design=read(OUT/'input_design_audit.json');split=read(OUT/'split_integrity.json')
    lines=['# Receiver-Selective Visual Writing: Which Tokens Need to See?','',
        '本实验属于 *The Visual Handoff: Efficient Visual Writing and Lightweight Answer Reading*。全部3000题均执行冻结协议；先报告全样本描述性结果，再报告与历史Semantic240图片不重叠的预先固定确认性子集。','',
        '## 1. 协议与实际样本','',
        f"- 执行源码冻结Commit：`{p['code_commit']}`；原始数据Manifest SHA256：`{MANIFEST_SHA}`。",
        '- LLaVA-1.5-7B，revision `'+REVISION+'`；FP16、Eager、Transformers4.49.0；Greedy最多32 Token，每条件、每指标独立真实KV Cache；自由生成与Teacher Forcing均使用 `logits_to_keep=1`。',
        '- 3000道冻结GQA，六任务各500题，2409个Image SHA256簇。原Prompt、图像预处理、Tokenizer、Gold及评分器保持一致；原生Full不作为All-Writer。',
        f"- 44个逻辑条件、132000条逐题标签记录；同题完全相同写入集合去重为{p['unique_evaluations']}组独立生成与Gold NLL。去重不跨问题，不丢弃标签；主实验未以历史240题准确率替代任何3000题数据。",
        '- 所有条件在32层均屏蔽原始P−1及所有真实Decode Query→原始576个Image Key；只改变Prefill R的开放接收位置。R为图像之后、P−1之前的非视觉文本，包含问题及模板。只做Softmax前Hard Mask；Image Query、其他连接、32层MLP、Residual和完整KV保持原生。',
        '- 两张RTX3090以UUID固定，一卡一模型；按Image SHA256整簇分片，每卡1500题，同图不跨Worker。其他GPU任务未被干预。','',
        '## 2. 接收Token规则与预算审计','',
        '- Semantic-k逐字节复用历史STOP词表及完整选择器，240道重叠问题的完整自然选择结果逐项一致。其规则保留词法候选（包含疑问词、空间关系等），不能简单改称仅名词选择。Question-Only Random是新增的关键控制。',
        '- 曲线以实际Question tokenizer token数的20/40/60/80/100%为目标，先向下取整，再取不超过目标的最大完整问题后缀Token预算；它构成所有策略可实现的共同预算。Semantic-first在精确整词预算下最大化词法候选Token数，确定性优先较早组；Question-Last为完整后缀；随机策略为固定随机优先级后首个可行整词子集，并非声称对全部子集均匀采样。',
        f"- 随机种子：{SEEDS}。曲线使用前两种子；自然预算使用三种子。先在每题内平均随机策略，再进行图片整簇统计，绝不把三种子当9000道独立题。",
        f"- 15000个逐题非零名义预算点中，{design['stats']['curve_budget_shortfalls']}个因整词共同可行性低于向下取整目标。原始Semantic-k在{design['stats']['natural_k_zero']}题为k=0；全部保留且与No-Writer按集合去重，未删除或改选。",
        '- `100% Question`仅开放问题Token，绝不等于`All-Writer = 100% R`。曲线横轴为逐题k/R后取平均，连接预算为全部32层576个Image Key下的理论可访问比例，不是实际FLOPs、显存或时延节省。',
        '- 位置分层随机按Question Token索引的前/中/后三段精确匹配Semantic的逐段Token数，整词不拆分；两种子全部3000题均可行。',
        '- 不嵌套题数：`'+str(design['non_nested_questions'])+'`。除Question-Last外，部分曲线集合不嵌套，因此相邻预算差异不能解释为逐Token边际因果增益。所有逐题集合、实际k和预算偏差在 `token_manifests/`。','',
        '## 3. 两套指标与统计设计','',
        '- 主机制指标为固定官方Gold完整答案的平均Teacher-Forced NLL（nat/token）。将一个ASCII空格和原Gold附加到展开后的原生Prompt，严格检查Prompt前缀不变、只保留原始BOS；Gold续写不含额外BOS或EOS。3430个Gold Token，326个多Token答案；所有3000题的续写IDs与旧输入映射一致。',
        '- 每个Gold Token在原始P−1或真实后续Decode位置预测。保存原始FP16全词表Logits，以CPU FP64计算log_softmax和答案Token均值；符号保留，负ΔNLL不截断。它与首Token signed LogP Loss是不同指标。Greedy答案独立生成，不将Gold缓存混入生成缓存。',
        '- 每个条件报告完整答案Accuracy、All原对→错/原错→对、首Token变化和异常。20,000次Image SHA256整簇配对Bootstrap，seed='+str(BOOTSTRAP_SEED)+'；每次保留被抽图片的全部问题，按题加权。所有曲线区间是普通点位95%CI，不是同时置信带。',
        f"- 确认子集按特定Semantic240的Question ID及全部Image SHA256排除，共{split['confirmatory_questions']}题、{split['confirmatory_images']}图簇，排除{split['excluded_questions']}题；不是预设2656题。",
        '- 两项确认检验预先固定为：(1) Semantic-k减Question-Only Random三种子均值的Gold NLL优效；(2) Semantic-k减All-Writer的Accuracy非劣效，界限−2pp。Bonferroni控制两项单侧检验家族α=.05，每项α=.025；分别要求NLL上97.5%界<0和Accuracy下2.5%界>−2pp。数值端点与普通双侧95%CI相同，但推断角色不同。−1pp仅敏感性判据；CI跨0不构成非劣效证明。','',
        '## 4. 自然Semantic-k的独立复现','',
        '| 条件 | 全3000题 Accuracy | 相对All ΔAcc(pp)及95%CI | ΔGold NLL及95%CI(nat/token) | 对→错 / 错→对 | 实际R预算 |',
        '|---|---:|---|---|---:|---:|']
    natural=['All-Writer','No-Writer','Semantic-k','Question-Last-k','R-Last-k','Question-Random-k mean3','R-Random-k mean3','Question-Stratified-k mean2','Drop-Semantic']
    for c in natural:
        r=item(summary,'all3000_descriptive',c);lines.append(f"| {c} | {number(r,'accuracy_percent'):.3f}% | {number(r,'delta_accuracy_pp'):+.3f} [{number(r,'delta_accuracy_ci_low'):+.3f}, {number(r,'delta_accuracy_ci_high'):+.3f}] | {number(r,'delta_gold_nll'):+.4f} [{number(r,'delta_nll_ci_low'):+.4f}, {number(r,'delta_nll_ci_high'):+.4f}] | {number(r,'All_correct_to_condition_wrong_count_or_seed_mean'):.3f} / {number(r,'All_wrong_to_condition_correct_count_or_seed_mean'):.3f} | {100*number(r,'actual_R_budget_mean'):.2f}% |")
    lines += ['','随机均值行的对→错/错→对为种子内逐题事件数的平均，可为非整数；每个固定种子的实际整数损害题数和全部输出保留在逐题表中。','',
        '## 5. 预算—质量曲线','',
        '![Four-panel main figure](figures/receiver_selective_visual_writing_main.png)','',
        '| 策略 | 名义Question预算 | 实际R预算 | Accuracy | ΔAcc(pp) | ΔGold NLL(nat/token) |','|---|---:|---:|---:|---:|---:|']
    for policy in ['Semantic-first','Question-Last','Question-Random mean2','R-Random mean2']:
        for b in BUDGETS:
            r=item(summary,'all3000_descriptive',f'{policy}-Q{b}');lines.append(f"| {policy} | {b}% | {100*number(r,'actual_R_budget_mean'):.2f}% | {number(r,'accuracy_percent'):.3f}% | {number(r,'delta_accuracy_pp'):+.3f} | {number(r,'delta_gold_nll'):+.4f} |")
    lines += ['','每个点的Accuracy和ΔNLL区间详见 `budget_curve.csv`；全部44固定标签和种子均值见 `condition_summary.csv`。没有拟合平滑曲线、删去负面点或按任务重选预算。','',
        '## 6. 确认性结果','',
        f"Semantic-k确认子集：{fmt(item(summary,'image_disjoint_confirmatory','Semantic-k'))}。",
        f"主检验1，Semantic-k−Question-Only Random均值：ΔGold NLL={confirm['primary_NLL_difference']:+.5f} nat/token；Bonferroni单侧上界={confirm['nll_superiority_upper_bound']:+.5f}；{'达到' if confirm['nll_superiority_rejected'] else '未达到'}预设词法选择优效判据。",
        f"主检验2，Semantic-k−All-Writer：ΔAccuracy={confirm['primary_accuracy_difference_pp']:+.3f}pp；Bonferroni单侧下界={confirm['accuracy_NI_lower_bound_pp']:+.3f}pp；{'达到' if confirm['accuracy_NI_minus2_rejected'] else '未达到'}−2pp非劣效判据。−1pp敏感性：{'达到' if confirm['accuracy_minus1_sensitivity'] else '未达到'}。",
        '两项结果以 `confirmatory_tests.json` 为准；全样本、任务和其他预算比较均为描述/探索性，不用于替换预设主比较。','',
        '## 7. 图像交换控制','',
        '预先从相同问题、相同完整Prompt IDs、不同图片与不同Gold中构建40组，无图片跨组重复；40组均位于旧Semantic240图片之外。五条件端点全部来自本轮同协议主实验的严格审计复用，换图指标单独统计，不增加主样本数。',
        '| 条件 | 两端均正确/40 | 随图改变/40 | 端点正确/80 | 首Token JS(nat) | 首Token TV |',
        '|---|---:|---:|---:|---:|---:|']
    for r in swaps:lines.append(f"| {r['condition']} | {r['both_correct_groups']} | {r['answer_changed_groups']} | {r['endpoint_correct']} | {number(r,'first_JS_nat'):.6f} | {number(r,'first_TV'):.6f} |")
    lines += ['','No-Writer在全部40组相同文本/视觉布局下，所有返回有效Greedy Logits（含EOS预测）严格`torch.equal`，不是只比较答案或首Token。不同Gold前缀的Teacher-Forcing后续Logits不作错误的跨图等值比较。每组两个图簇作为关联区块进行20,000次Bootstrap，保持两端配对；每组图SHA均不出现在其他组。JS/TV来自真实首Token完整分布。','',
        '## 8. 验证、负面案例与文件说明','',
        '- 57项CPU测试通过；统计实现另有4项合成数据测试。两卡各四道真实7B Smoke，含冻结三题及一个预先选定多Token Gold题；原生与多种No-op的全部有效Greedy和Gold Logits严格相等，跨卡原生Logits严格相等。',
        '- 正式记录逐次检查32层实际禁止概率为0、非目标Mask精确不变、BOS/自环合法、Cache长度与绝对位置准确；MLP在Smoke中逐层检查；权重版本、输入Hash及Hook清理在每次条件中验证。完整原始Audit矩阵及所有Gold Logits可按SHA复核，所有保存NLL均重新计算且逐值一致。',
        '- `per_question/`按六任务拆分132000条完整标签记录，`per_question_index.json`提供总数与SHA；`token_manifests/`为全部冻结集合；`shard*/checkpoints/`为109518份不可变科学记录；`accuracy_harm_cases.csv`保留All对→错，`all_incorrect_cases.csv`保留全部错误输出。',
        '- `bootstrap/`保存原始20,000次重采样均值，`paired_comparisons.csv`保存所有预定对比；随机种子先逐题平均。`figures/`提供PNG和矢量PDF，作图输入CSV及脚本完整保留。',
        '- 全词表原始数组位于服务器资源目录 `'+str(RAW)+'`，每份路径、字节数、SHA见检查点及 `raw_logits_hashes.json`。GitHub保存答案、Token IDs、Gold逐Token NLL和验证摘要，不声称包含这些服务器数组。','',
        '## 9. 研究回答与边界','',
        '**视觉写入是否随接收预算增加出现平台？** 上述全曲线完整展示各预算点；不能用Accuracy单指标、个别高点或CI跨0判定平台。请结合两套指标及 `paired_comparisons.csv` 中相邻名义预算的完整探索性对比解释趋势；不嵌套集合不能解释为逐Token边际增益。',
        '**同预算词法选择是否优于问题内部随机及位置选择？** 自然预算的确认性机制主判据'+('支持' if confirm['nll_superiority_rejected'] else '尚不支持')+'相对Question-Only Random的Gold NLL优势；Accuracy与Question-Last/分层位置控制的结果必须同时查看，不能把Question内部选择与避免模板Token的效应混为一谈。',
        '**少量关键Token能否在图片不重叠确认集保留接近All-Writer的回答能力？** 按预先固定−2pp非劣效检验，'+('本确认集达到该界限' if confirm['accuracy_NI_minus2_rejected'] else '本确认集未达到该界限，不能宣称已证明接近或非劣效')+'；−1pp结果如上单独列示。该结论只针对本模型、冻结数据与本干预协议。',
        '本轮研究功能性接收Token稀疏，不涉及层×Token搜索、Router、Kernel、KV剪枝，也没有证明实际计算或时延节省。完成后停止，不追加实验。','']
    put(OUT/'report_zh.md','\n'.join(lines))

    anomaly_lines=['','## 10. 输出异常汇总','','| 条件 | 非EOS | 空答案 | 截断 | 重复3gram |','|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        rr=item(summary,'all3000_descriptive',c)
        anomaly_lines.append('| '+c+' | '+' | '.join(f"{number(rr,k):.0f}" for k in ['non_EOS_count_or_seed_mean','empty_count_or_seed_mean','truncated_count_or_seed_mean','repeated_trigram_count_or_seed_mean'])+' |')
    with (OUT/'report_zh.md').open('a',encoding='utf-8',newline='\n') as f:f.write('\n'.join(anomaly_lines)+'\n')

def complete():
    p=verify();assert_no_stop();assert read(OUT/'visual_layout_review.json')['passed'],'Visually inspect the actual exported main/six-task PDF and PNG first'
    for name in ['replay_validation.json','statistics_validation.json','image_swap_validation.json','figure_validation.json']:assert read(OUT/name)['passed']
    for name,h in read(OUT/'postprocessing_freeze_v2.json')['sources'].items():assert sha256(ROOT/name)==h
    for name,info in p['model_files'].items():assert sha256(name)==info['sha256']
    for name,h in read(OUT/'checkpoint_hashes.json').items():assert sha256(OUT/name)==h
    for name,h in read(OUT/'raw_logits_hashes.json').items():assert sha256(name)==h
    pids=[]
    for rank in range(2):
        c=read(OUT/f'shard{rank}/completion.json');assert c['passed'] and c['binding']==digest(p);pids.append(c['pid'])
    # Process disappearance, not low memory alone, confirms this experiment released its models.
    process_check=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,args'],capture_output=True,text=True)
    assert process_check.returncode==1,process_check.stdout
    state=hardware();state.update(passed=True,experiment_worker_pids=pids,experiment_workers_exited=True,selected_uuids=[g['uuid'] for g in p['gpus']],
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unrelated_processes_not_killed=True)
    atomic_json(OUT/'gpu_release.json',state)
    with np.load(OUT/'metrics.npz') as z:assert z['accuracy'].shape==(3000,44) and z['nll'].shape==(3000,44) and np.isfinite(z['nll']).all()
    replay=read(OUT/'replay_validation.json');assert replay['canonical_evaluations']==109518 and replay['logical_records']==132000
    atomic_json(OUT/'validation.json',dict(passed=True,binding=digest(p),all_generation_and_gold_metrics_complete=True,canonical_evaluations=109518,logical_evaluations=132000,
        raw_gold_nll_recomputed_exactly=True,raw_logits_in_GitHub=False,all_44_labels_complete=True,all3000_samples_retained=True,
        model_full_SHA256_reverified=True,sources_unchanged=True,all_artifact_hashes='completion.json',resource_release_audited=True,figures_visually_reviewed=True))
    artifacts={}
    for f in sorted(OUT.rglob('*')):
        if not f.is_file() or f.name in ['worker.lock','completion.json'] and f.parent==OUT:continue
        if f.name=='worker.lock':continue
        assert f.stat().st_size<98_000_000,('Split before GitHub push',str(f),f.stat().st_size)
        artifacts[str(f.relative_to(OUT))]=sha256(f)
    atomic_json(OUT/'completion.json',dict(status='COMPLETE',passed=True,binding=digest(p),code_commit=p['code_commit'],questions=3000,logical_conditions=44,
        logical_generation_and_NLL_records=132000,independent_canonical_generations=109518,independent_canonical_gold_evaluations=109518,
        confirmatory_questions=2640,confirmatory_images=2169,swap_pairs=40,all_statistics_and_figures_complete=True,artifacts=artifacts,
        raw_server_arrays='raw_logits_hashes.json; arrays are not in GitHub',no_further_experiments=True))
    print('COMPLETE',p['code_commit'],len(artifacts),'artifacts',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['report','complete']);a=parser.parse_args();globals()[a.stage]()
