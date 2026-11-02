"""Paired whole-image inference and a bounded Chinese report, never model execution."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import torch

from first_answer_attention.engine import first_metrics
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.statistics import bootstrap, mcnemar_exact
from .common import (ROOT, OUT, CONDITIONS, TASKS, LAYERS, SEED, read, write,
                     csv_write, sha256, digest, verify)
from .run import samples, checkpoint


def divergence(a, b):
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return i + 1, x == 2 or y == 2
    return '', False


def pair_flags(full, skip, text, joint):
    return dict(joint_unique_harm=int(full == skip == text == 1 and joint == 0),
        joint_unique_rescue=int(full == skip == text == 0 and joint == 1),
        joint_harm_after_skip_ok=int(full == skip == 1 and joint == 0),
        joint_harm_after_text_ok=int(full == text == 1 and joint == 0),
        both_singles_harm=int(full == 1 and skip == text == 0))


def load_records():
    p = verify()
    binding = digest(p)
    completion = read(OUT/'completion.json')
    assert completion['status'] == 'complete' and completion['formal_generations'] == 960
    assert completion['binding'] == binding
    ss = samples()
    rr = {s['question_id']:{c:checkpoint(s,c,binding) for c in CONDITIONS} for s in ss}
    assert all(r is not None for cs in rr.values() for r in cs.values())
    assert len(list((OUT/'checkpoints').glob('*/*.json'))) == 960
    return p, ss, rr


def analyze():
    p, ss, rr = load_records()
    cs = list(CONDITIONS)
    scores = np.array([[rr[s['question_id']][c]['generation']['score'] for c in cs] for s in ss],dtype=float)
    identities = [s['image_sha256'] for s in ss]
    assert len(ss) == len(set(identities)) == 240
    long_rows, pairs, audit_rows, logits_files = [], [], [], []
    for s in ss:
        records = rr[s['question_id']]
        gs = {c:r['generation'] for c,r in records.items()}
        full = gs['Full']
        pair = dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],
                    question=s['question'],gold=s['answer'],**pair_flags(*(gs[c]['score'] for c in cs)))
        for c in cs:
            r, g = records[c], gs[c]
            d, eos_div = divergence(full['token_ids'],g['token_ids'])
            row = dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],
                gold=s['answer'],condition=c,text=g['text'],prediction=g['prediction'],score=g['score'],
                token_ids=str(g['token_ids']),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],
                repeated_output=g['repeated_output'],length=g['length'],decode_steps=g['decode_steps'],
                full_correct_to_wrong=int(full['score']==1 and g['score']==0),
                full_wrong_to_correct=int(full['score']==0 and g['score']==1),
                first_top1_flip=int(g['first_top1_flip']),first_logp_loss=g['first_logp_loss'],
                full_first_token=g['full_first_token'],first_token=g['first_token'],
                full_first_token_logp=g['full_first_token_logp'],condition_full_token_logp=g['intervened_full_token_logp'],
                first_divergence_token_ordinal=d,first_divergence_involves_eos=eos_div,
                token_sequence_changed=g['token_ids']!=full['token_ids'],text_changed=g['text']!=full['text'],
                joint_unique_harm=pair['joint_unique_harm'] if c=='Joint' else 0,
                prompt_ids_sha256=r['prompt_ids_sha256'],first_logits_path=r['first_logits_path'],
                first_logits_sha256=r['first_logits_sha256'],generation_sha256=r['generation_sha256'])
            long_rows.append(row)
            pair.update({c+'_score':g['score'],c+'_text':g['text'],c+'_token_ids':str(g['token_ids']),
                         c+'_harm':row['full_correct_to_wrong'],c+'_rescue':row['full_wrong_to_correct']})
            logits_files.append(dict(question_id=s['question_id'],condition=c,path=r['first_logits_path'],
                sha256=r['first_logits_sha256'],bytes=Path(r['first_logits_path']).stat().st_size,
                committed_to_git=False))
            for step in g['steps']:
                for mask, prob in zip(step['masks'],step['probabilities']):
                    layer = mask['layer']
                    assert layer == prob['layer']
                    audit_rows.append(dict(question_id=s['question_id'],task=s['task'],condition=c,
                        t=step['t'],absolute_query=step['absolute_query'],local_query=step['local_query'],
                        cache_before=step['cache_before'],cache_after=step['cache_after'],
                        consumed_token=step['consumed_token'],predicted_token=step['predicted_token'],
                        predicts_eos=step['predicted_token']==2,layer=layer,
                        image_mask_applied=mask['applied'],image_probability_max=prob['image_probability_max'],
                        image_mass=prob['image_mass'],row_sum_max_error=prob['row_sum_max_error'],
                        attention_output_zero_verified=CONDITIONS[c][0] and layer in LAYERS,
                        image_keys=576,nonimage_mask_unchanged=True))
        pairs.append(pair)
    csv_write(OUT/'per_question.csv',long_rows)
    csv_write(OUT/'paired_per_question.csv',pairs)
    csv_write(OUT/'attention_kv_audit.csv',audit_rows)
    csv_write(OUT/'raw_logits_manifest.csv',logits_files)
    accuracy, tasks = [], []
    for task in ['All'] + TASKS:
        chosen = [i for i,s in enumerate(ss) if task=='All' or s['task']==task]
        subset, im = scores[chosen], [identities[i] for i in chosen]
        for k,c in enumerate(cs):
            harm = ((subset[:,0]==1)&(subset[:,k]==0)).astype(float)
            rescue = ((subset[:,0]==0)&(subset[:,k]==1)).astype(float)
            gs = [rr[ss[i]['question_id']][c]['generation'] for i in chosen]
            values = np.column_stack([subset[:,k],subset[:,k]-subset[:,0],harm,rescue,
                [g['first_top1_flip'] for g in gs],[g['first_logp_loss'] for g in gs]])
            stat = bootstrap(values,im,[1]*6,repetitions=20000,seed=SEED)
            row = dict(task=task,condition=c,n=len(chosen),correct=int(subset[:,k].sum()),
                accuracy_pct=100*stat[0]['mean'],accuracy_ci95_low_pct=100*stat[0]['ci95_low'],
                accuracy_ci95_high_pct=100*stat[0]['ci95_high'],delta_full_pp=100*stat[1]['mean'],
                delta_ci95_low_pp=100*stat[1]['ci95_low'],delta_ci95_high_pp=100*stat[1]['ci95_high'],
                harm=int(harm.sum()),rescue=int(rescue.sum()),first_top1_flips=sum(g['first_top1_flip'] for g in gs),
                mean_first_logp_loss=stat[5]['mean'],logp_loss_ci95_low=stat[5]['ci95_low'],
                logp_loss_ci95_high=stat[5]['ci95_high'],eos=sum(g['eos'] for g in gs),
                truncated=sum(g['truncated'] for g in gs),empty=sum(g['empty'] for g in gs),
                repeated_output=sum(g['repeated_output'] for g in gs),
                mcnemar_exact_p=mcnemar_exact(int(harm.sum()),int(rescue.sum())),
                degenerate_delta_bootstrap=stat[1]['degenerate_bootstrap'],clusters=stat[1]['clusters'])
            (accuracy if task=='All' else tasks).append(row)
    csv_write(OUT/'accuracy_summary.csv',accuracy)
    csv_write(OUT/'task_accuracy.csv',tasks)
    comparisons, task_comparisons = [], []
    for task in ['All']+TASKS:
        chosen=[i for i,s in enumerate(ss) if task=='All' or s['task']==task]
        f,a,t,j=scores[chosen].T
        unique=np.array([pairs[i]['joint_unique_harm'] for i in chosen],dtype=float)
        names=['Joint-Full','Joint-A-Skip','Joint-Text-Only','Interaction I','Joint unique harm rate']
        values=np.column_stack([j-f,j-a,j-t,j-t-a+f,unique])
        stats=bootstrap(values,[identities[i] for i in chosen],[1]*5,repetitions=20000,seed=SEED)
        for name,stat in zip(names,stats):
            row=dict(task=task,comparison=name,n=len(chosen),estimate_pp=100*stat['mean'],
                ci95_low_pp=100*stat['ci95_low'],ci95_high_pp=100*stat['ci95_high'],
                clusters=stat['clusters'],repetitions=20000,seed=SEED,
                degenerate_bootstrap=stat['degenerate_bootstrap'],exploratory=True)
            (comparisons if task=='All' else task_comparisons).append(row)
    csv_write(OUT/'paired_comparisons.csv',comparisons)
    csv_write(OUT/'task_paired_comparisons.csv',task_comparisons)
    unique_tasks=[dict(task=t,n=40,joint_unique_harm=sum(r['joint_unique_harm'] for r in pairs if r['task']==t),
        joint_full_harm=sum(r['Joint_harm'] for r in pairs if r['task']==t),
        joint_full_rescue=sum(r['Joint_rescue'] for r in pairs if r['task']==t)) for t in TASKS]
    csv_write(OUT/'joint_error_tasks.csv',unique_tasks)
    figures(accuracy,tasks,comparisons,unique_tasks)
    report(p,accuracy,tasks,comparisons,pairs,unique_tasks)
    print('Analysis complete:960 rows,240 paired records,3 PNG/PDF figures',flush=True)


def interval(row, prefix=''):
    return f"[{row[prefix+'ci95_low_pp']:+.3f}, {row[prefix+'ci95_high_pp']:+.3f}]"


def figures(acc,tasks,comparisons,unique_tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    colors=['#666666','#1b9e77','#377eb8','#c44e52']
    names=list(CONDITIONS)
    def save(fig,name):
        fig.savefig(directory/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for i,r in enumerate(acc):
        value=r['accuracy_pct'];low=r['accuracy_ci95_low_pct'];high=r['accuracy_ci95_high_pct']
        axes[0].bar(i,value,color=colors[i],width=.65)
        axes[0].errorbar(i,value,yerr=[[value-low],[high-value]],fmt='none',color='black',capsize=4)
        axes[0].text(i,value+.4,f"{r['correct']}/240",ha='center',fontsize=9)
    axes[0].set(xticks=range(4),xticklabels=names,ylabel='Complete-answer accuracy (%)',ylim=(0,105),title='Four independently generated conditions')
    for i,r in enumerate(acc[1:]):
        value=r['delta_full_pp']
        axes[1].errorbar(value,i,xerr=[[value-r['delta_ci95_low_pp']],[r['delta_ci95_high_pp']-value]],
            fmt='o',color=colors[i+1],capsize=4)
    axes[1].axvline(0,color='.55',lw=1)
    axes[1].axvline(-1,color='.6',lw=1,ls=':')
    axes[1].set(yticks=range(3),yticklabels=names[1:],xlabel='Accuracy difference vs Full (pp)',title='Paired image-cluster 95% intervals')
    save(fig,'figure1_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for i,r in enumerate(comparisons[:4]):
        value=r['estimate_pp']
        axes[0].errorbar(value,i,xerr=[[value-r['ci95_low_pp']],[r['ci95_high_pp']-value]],fmt='o',color='#c44e52',capsize=4)
    axes[0].axvline(0,color='.5',lw=1)
    axes[0].set(yticks=range(4),yticklabels=[r['comparison'] for r in comparisons[:4]],
        xlabel='Accuracy contrast (pp)',title='Joint losses and 2x2 interaction')
    vals=[r['joint_unique_harm'] for r in unique_tasks]
    axes[1].bar(range(6),vals,color='#c44e52')
    axes[1].set(xticks=range(6),xticklabels=TASKS,ylabel='Questions',ylim=(0,max(vals+[1])+1),title=f'Joint-only Full-correct losses: {sum(vals)}')
    axes[1].tick_params(axis='x',rotation=35)
    for i,v in enumerate(vals):axes[1].text(i,v+.08,str(v),ha='center')
    save(fig,'figure2_joint_interaction')
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    x=np.arange(6)
    for k,c in enumerate(names[1:]):
        rows=[next(r for r in tasks if r['task']==t and r['condition']==c) for t in TASKS]
        loc=x+(k-1)*.24
        axes[0].bar(loc,[r['delta_full_pp'] for r in rows],width=.22,color=colors[k+1],label=c)
        axes[1].bar(loc,[r['rescue'] for r in rows],width=.22,color=colors[k+1])
        axes[1].bar(loc,[-r['harm'] for r in rows],width=.22,color=colors[k+1],hatch='//',alpha=.65)
    for ax in axes:ax.axhline(0,color='.4',lw=.8)
    axes[0].set(ylabel='Accuracy difference vs Full (pp)',title='Six GQA tasks, 40 questions per task')
    axes[0].legend(ncol=3)
    axes[1].set(ylabel='Rescues (+) / harms (-)',xticks=x,xticklabels=TASKS)
    save(fig,'figure3_task_gains_losses')


def report(protocol,acc,tasks,comp,pairs,unique_tasks):
    by={r['condition']:r for r in acc}
    cm={r['comparison']:r for r in comp}
    manifest=read(OUT/'sample_manifest.json')
    exp1=Path(manifest['experiment_one_manifest'])
    comparison=dict(experiment_one_available_at_selection=manifest['experiment_one_available_at_freeze'],
                    available_at_reporting=exp1.exists())
    if exp1.exists():
        other=read(exp1)['samples']
        a={(s['question_id'],s['image_sha256']) for s in samples()}
        b={(s['question_id'],s['image_sha256']) for s in other}
        comparison.update(same_samples=a==b,overlap_questions_images=len(a&b),experiment_one_sha256=sha256(exp1))
    else:
        comparison.update(same_samples=None,reason='Experiment1 manifest still unavailable; do not claim identical cohort')
    write(OUT/'experiment_one_cohort_comparison.json',comparison)
    lines=['# 实验二：Dead Attention × Text Relay 联合可组合性',
        '', '**状态：complete。240题、六类各40题、240张互异图片；四条件全部重新生成，共960条完整答案。**',
        '历史来源上的探索性研究，不是独立确认集或严格非劣效认证。', '', '## 主要结果', '',
        '|条件|正确/240|完整答案Accuracy|ΔFull(pp)及配对95%CI|原对→错|原错→对|首Token Flip|平均signed LogP Loss(nat)|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in acc:
        lines.append(f"|{r['condition']}|{r['correct']}/240|{r['accuracy_pct']:.3f}%|{r['delta_full_pp']:+.3f} [{r['delta_ci95_low_pp']:+.3f}, {r['delta_ci95_high_pp']:+.3f}]|{r['harm']}|{r['rescue']}|{r['first_top1_flips']}|{r['mean_first_logp_loss']:+.6f}|")
    lines += ['', 'Full固定首Token的signed LogP Loss使用FP32全词表log_softmax；负值允许，不能解释为Gold概率或Accuracy。首Token Flip不是完整VQA准确率。',
        '', '|联合比较|估计(pp)|图像簇配对95%CI|', '|---|---:|---|']
    for r in comp:
        lines.append(f"|{r['comparison']}|{r['estimate_pp']:+.3f}|{interval(r)}|")
    unique=sum(r['joint_unique_harm'] for r in pairs)
    lines += ['', '`I=(AccJoint−AccTextOnly)−(AccASkip−AccFull)`；负值表示联合有超出加性期望的损害，正值相反。',
        f'Joint独有损害（Full及两个单独干预都正确、Joint错误）：**{unique}题**。其他Joint损害/纠正也完整保留，不以净均值代替逐题风险。',
        '', '## 直接回答', '',
        f"1. **两个单独低敏感现象是否复现？** A-Skip相对Full为{by['A-Skip']['delta_full_pp']:+.3f}pp，损害{by['A-Skip']['harm']}题、纠正{by['A-Skip']['rescue']}题；Text-Only为{by['Text-Only']['delta_full_pp']:+.3f}pp，损害{by['Text-Only']['harm']}题、纠正{by['Text-Only']['rescue']}题。应以这些观察量限定“低敏感”，不将其写成严格无损。",
        f"2. **Joint是否有额外损害？** 相对Full为{cm['Joint-Full']['estimate_pp']:+.3f}pp，CI {interval(cm['Joint-Full'])}；相对A-Skip为{cm['Joint-A-Skip']['estimate_pp']:+.3f}pp，相对Text-Only为{cm['Joint-Text-Only']['estimate_pp']:+.3f}pp；交互I为{cm['Interaction I']['estimate_pp']:+.3f}pp，CI {interval(cm['Interaction I'])}。逐题Joint独有损害{unique}题。CI包含0既不能证明额外损害，也不能证明安全等价。",
        '3. **是否集中在某些任务？** 下表报告每类净变化及损害/纠正，joint_error_tasks.csv列出Joint独有损害。每类仅40题，不把事后任务差异解释为确认性集中效应。',
        '4. **是否支持联合用于短答案VQA？** 本轮只能给出固定LLaVA/GQA/32Token协议下的可组合性探索证据；即使净Accuracy接近Full，也必须考虑Joint独有损害、逐题翻转和置信区间。不能据此宣布可安全部署或对所有短答案无损。',
        '5. **统计性低损害与严格无损？** 所有点估计、普通95%区间和零翻转观察都不是严格无损证明；−1pp仅是后续验证的参考，不作本轮非劣效认证。零差异的退化Bootstrap区间同样不证明总体零风险。',
        '6. **未来哪些计算可以尝试真实跳过？** 只能将L2–L5答案Query的Attention输出更新、以及答案Query到原始Image Key的直接读取视为待验证的候选，不是本轮已省略的计算。Prefill其他Query的视觉读取和文本中继不能由本轮结果推断可删除。',
        '7. **哪些KV仍需保留？** 本轮全部层原始图像、Prompt文本及真实生成前缀KV均完整维护。非目标Prefill Query需要视觉KV；后续答案Query仍需可读的文本/生成KV，L2–L5的KV正常更新。Mask不能证明图像KV可安全物理删除，也不能证明对应Q/K/V投影都可省略。',
        '8. **是否值得开发真实加速？** 本轮不实现Kernel或KV剪枝，不报告FLOPs、显存、延迟或吞吐收益。是否推进需依据这里的逐题风险与任务分布，并单独验证真实跳过后的Cache语义和数值行为；本次到此停止。',
        '', '## 六任务损益', '', '|任务|Full|A-Skip|Text-Only|Joint|Joint对→错 / 错→对|Joint独有损害|', '|---|---:|---:|---:|---:|---|---:|']
    for task in TASKS:
        rs={r['condition']:r for r in tasks if r['task']==task}
        uq=next(r['joint_unique_harm'] for r in unique_tasks if r['task']==task)
        vals='|'.join(f"{rs[c]['accuracy_pct']:.1f}%" for c in CONDITIONS)
        lines.append(f"|{task}|{vals}|{rs['Joint']['harm']} / {rs['Joint']['rescue']}|{uq}|")
    lines += ['', '## 图表', '',
        '![四条件Accuracy及配对差异](figures/figure1_accuracy.png)',
        '![联合损害与交互](figures/figure2_joint_interaction.png)',
        '![六任务损益](figures/figure3_task_gains_losses.png)',
        '', '## 协议、样本和验证', '',
        f"- 历史基准{protocol['base_commit']}；冻结执行代码{protocol['code_commit']}；独立分支codex/attention-relay-joint-240，main及实验一Worktree未修改。",
        '- LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager、Transformers4.49.0，Greedy/max32、原Prompt/官方完整答案精确评分器。',
        '- A-Skip在L2–L5仅置零原始P−1及每个真实Decode Query的o_proj输出；不是全Prefill关闭。V-Off在全部32层只对这些答案Query的576个原始Image Key做Softmax前−inf；其他Prefill Query原生读图。Joint同时安装两种历史Hook，不修改历史实现或第三方源码。',
        '- t=0为原始P−1，t≥1为真实增量Decode，包括EOS预测位置；每个条件从空Cache独立维护真实前缀，未使用反事实Cache、跨条件Cache或旧Full/Text-Only答案。所有32层KV逐步长度与实际位置严格检查。',
        '- 20,000次整Image SHA256簇配对Bootstrap，seed20261102，题目加权，同一指标族共用图像抽样。全部比较探索性普通双侧95%区间，不作多重比较后的确认性或非劣效宣称。Token/层/Head不是独立统计单位。',
        f"- 样本选择时实验一Manifest可用：{manifest['experiment_one_available_at_freeze']}。本轮来源：{manifest['source_manifest']}。报告时逐题集合比较：{comparison}。未等待或依照运行结果重选样本。",
        '- CPU测试及至少三题真实7B Smoke先于正式采集；Full/No-op所有有效Logits、完整Token及初始Cache指纹精确一致，Full与原生HF Greedy一致。两个单独条件与历史引擎逐步Logits一致；Joint逐层实际概率归零及输出置零同时通过。非目标Prefill Attention/Hidden/Logits精确不变，异常Hook清理和恢复原生通过。',
        '- 模型仅加载一份、四条件串行、只用物理GPU0和冻结UUID。未终止或抢占其他任务；GPU释放信息见gpu_release.json。无训练、无物理Attention跳过、无额外条件扫描。',
        f"- 全部条件EOS/截断/空答案/重复输出计数见accuracy_summary.csv；正式异常总数{sum(r['truncated']+r['empty']+r['repeated_output'] for r in acc)}（保留所有样本，不静默排除）。失败记录如存在保存在failures/，不放宽数值容差。",
        '- 首Token原始Logits仅保留于服务器本地，不虚称提交GitHub：raw_logits_manifest.csv逐文件列出绝对路径、SHA256和字节数。GitHub包含所有960份检查点、完整Token/文本、审计、CSV及PNG/PDF，不包含这批大型原始Logits。',
        '', '## 解释边界', '',
        'Attention输出置零与Hard Mask仍执行原生Attention及KV更新；本轮是功能干预，不是真实算子优化。低净Accuracy损害不等于逐题无损；交互CI包含0不等于两路径独立或可安全同时删除。若出现负交互，也只能支持条件性功能耦合，不能单独定位补偿语义或唯一中介。历史重复使用数据、短答案/语言先验、Hard Mask重归一化以及有限样本均限制外推。',
        '', '完成本实验后停止，不追加Query角色扫描、跨模型实验、训练、KV剪枝或Kernel实现。', '']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def audit():
    p,ss,rr=load_records()
    maximum_error=0.; zero_rows=zero_outputs=queries=0
    for s in ss:
        logits={c:torch.from_numpy(np.load(rr[s['question_id']][c]['first_logits_path'],allow_pickle=False)) for c in CONDITIONS}
        for c,(skip,off) in CONDITIONS.items():
            r=rr[s['question_id']][c];g=r['generation']
            assert tensor_hash(logits[c]) == r['first_logits_tensor_sha256']
            assert torch.isfinite(logits[c]).all()
            assert all(g[k] == v for k,v in first_metrics(logits['Full'],logits[c]).items())
            assert g['prediction'] == gqa_prediction(g['text']) and g['score'] == int(g['prediction']==s['answer'])
            assert g['eos'] == (g['token_ids'][-1]==2) and g['length'] == len(g['token_ids'])
            n=len(s['input_mapping']['prompt_ids'])
            assert len(g['steps']) == g['length'] and g['initial_cache_empty'] and g['own_cache'] and g['own_prefix']
            assert g['attention_output_edits'] == (4*g['length'] if skip else 0)
            assert len(g['attention_output_events']) == (4*g['length'] if skip else 0)
            for e in g['attention_output_events']:
                assert e['layer'] in LAYERS and e['active'] and not e['noop']
            zero_outputs+=g['attention_output_edits']
            for t,step in enumerate(g['steps']):
                assert step['t']==t and step['absolute_query']==n+t-1
                assert step['cache_after']==n+t and step['cache_before']==(n+t-1 if t else 0)
                assert step['consumed_token']==(g['token_ids'][t-1] if t else None)
                assert step['predicted_token']==g['token_ids'][t]
                assert len(step['masks']) == len(step['probabilities']) == 32
                for k,(m,prob) in enumerate(zip(step['masks'],step['probabilities'])):
                    assert m['layer']==prob['layer']==k and m['applied']==off
                    if off:
                        assert prob['image_probability_max']==0.
                        zero_rows+=1
                    if not t and off:assert m['blocked_connections_per_head']==576
                    maximum_error=max(maximum_error,prob['row_sum_max_error'])
                queries+=1
    with (OUT/'per_question.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==960 and len({(r['question_id'],r['condition']) for r in rows})==960
    with (OUT/'accuracy_summary.csv').open() as f:acc=list(csv.DictReader(f))
    for r in acc:
        c=r['condition'];gs=[rr[s['question_id']][c]['generation'] for s in ss]
        assert int(r['correct']) == sum(g['score'] for g in gs)
        assert float(r['accuracy_pct']) == 100*(sum(g['score'] for g in gs)/240)
        assert int(r['first_top1_flips']) == sum(g['first_top1_flip'] for g in gs)
    scores=np.array([[rr[s['question_id']][c]['generation']['score'] for c in CONDITIONS] for s in ss],dtype=float)
    f,a,t,j=scores.T
    expected={'Joint-Full':j-f,'Joint-A-Skip':j-a,'Joint-Text-Only':j-t,
              'Interaction I':j-t-a+f,
              'Joint unique harm rate':((f==1)&(a==1)&(t==1)&(j==0)).astype(float)}
    with (OUT/'paired_comparisons.csv').open() as stream:comparisons=list(csv.DictReader(stream))
    for r in comparisons:
        assert float(r['estimate_pp']) == 100*float(expected[r['comparison']].mean())
        assert float(r['ci95_low_pp']) <= float(r['estimate_pp']) <= float(r['ci95_high_pp'])
    for n,b in p['model_files'].items():assert sha256(n)==b['sha256']
    # Only new package and result paths may differ from the historical baseline.
    import subprocess
    changed=subprocess.check_output(['git','diff','--name-only',p['base_commit'],'--'],text=True).splitlines()
    assert all(n.startswith(('vlm_research/attention_relay_joint/','vlm_research/results/attention_relay_joint_240/')) for n in changed)
    for name in ['figure1_accuracy','figure2_joint_interaction','figure3_task_gains_losses']:
        for ext in ['png','pdf']:assert (OUT/f'figures/{name}.{ext}').stat().st_size>1000
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*'))
        if f.is_file() and f.name not in ['worker.lock','verification.json'] and '.tmp' not in f.name}
    write(OUT/'verification.json',dict(passed=True,questions=240,unique_images=240,formal_generations=960,
        real_query_steps=queries,actual32layer_masked_rows_exact_zero=zero_rows,
        actual_attention_output_zero_edits=zero_outputs,row_sum_max_error=maximum_error,
        hook_cleanup_and_parameter_versions=True,raw_first_logits_all_finite_and_recomputed=True,
        model_sha256_rechecked=True,only_new_paths_changed=True,source=p['source'],artifacts=artifacts))
    print('Final audit passed:960 independent checkpoints, real Mask/o_proj/KV and raw logits verified',flush=True)
