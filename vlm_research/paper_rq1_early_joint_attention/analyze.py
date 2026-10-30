"""Replay scores, preserve negative outcomes and render only complete core data."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import textwrap
import time
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from benchmarks.dataset_registry import digest,freeze_json,sha256
from paper_rq1.scorers import gqa_prediction
from .protocol import ROOT,OUT,MODEL_PATH,CONDITIONS,PROTOCOL,TASKS
from .run import verify,checkpoint_valid
from .statistics import bootstrap,mcnemar_exact

CONTRASTS=[('S1','S5'),('S2','S6'),('S3','S7'),('S4','S8'),('S2','S1'),('S6','S5')]


def write_csv(name,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def anomaly(ids,text,eos):
    tokens=[i for i in ids if i!=2];runs=0;last=None;longest=0
    for token in tokens:
        runs=runs+1 if token==last else 1;last=token;longest=max(longest,runs)
    four=Counter(tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3)))
    return dict(max_identical_token_run=longest,repeated_fourgram_count=sum(n-1 for n in four.values()),
        repetition_flag=longest>=4 or any(n>=3 for n in four.values()),empty_answer=not bool(text.strip()),
        missing_eos=not eos,non_ascii_fraction=sum(ord(c)>127 for c in text)/max(1,len(text)))


def load():
    config=verify();binding=digest(config);manifest=json.loads((OUT/'sample_manifest.json').read_text())
    environments=[];status=[]
    for rank in range(3):
        folder=OUT/f'generation/shard{rank}';done=json.loads((folder/'completion.json').read_text())
        if done['status']!='complete' or done['binding']!=binding:raise ValueError('Require all matching completed workers')
        status.append(done);environments.extend(json.loads(p.read_text()) for p in folder.glob('environment_*.json'))
    keys=['python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','parameter_dtypes','binding','code_commit']
    if len({digest({k:e[k] for k in keys}) for e in environments})!=1:raise ValueError('Resume/worker environments differ')
    freeze_json(OUT/'environment.json',dict(passed=True,attempt_environments=environments,source_commit=config['code_commit']))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    data={};per=[];paths=[];seen=set()
    for s in manifest['samples']:
        key=(s['task'],s['question_id'])
        if s['question_id'] in seen:raise ValueError('Duplicate frozen question')
        seen.add(s['question_id']);values={}
        if sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Image changed since freeze')
        for condition in CONDITIONS:
            p=OUT/f"generation/shard{s['rank']}/checkpoints/{s['task']}__{s['question_id']}/{condition}.json"
            if not checkpoint_valid(p,binding,s,condition):raise ValueError('Missing condition')
            row=json.loads(p.read_text());g=row['generation'];ids=g['token_ids']
            text=tokenizer.decode(ids,skip_special_tokens=True).strip()
            if text!=g['text'] or gqa_prediction(text)!=g['prediction'] or int(g['prediction']==s['answer'])!=g['score']:
                raise ValueError('Official score/text replay mismatch')
            if not 1<=len(ids)<=32 or any(i==2 for i in ids[:-1]) or g['eos']!=(ids[-1]==2):raise ValueError('EOS/length invalid')
            if not g['own_prefill'] or not g['own_prefix']:raise ValueError('Wrong cache ownership')
            values[condition]=dict(g,seconds=row['seconds'],**anomaly(ids,text,g['eos']))
            paths.append(dict(path=str(p.relative_to(OUT)),sha256=sha256(p),bytes=p.stat().st_size))
        baseline=values['S0']
        for c,g in values.items():
            if c in ['S3','S7'] and g['token_ids'][0]!=baseline['token_ids'][0]:raise ValueError('Decode-only first token changed')
            per.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                condition=c,name=CONDITIONS[c]['name'],layers=json.dumps(CONDITIONS[c]['layers']),mode=CONDITIONS[c]['mode'],
                question=s['question'],gold=s['answer'],full_text=baseline['text'],text=g['text'],prediction=g['prediction'],
                score=g['score'],full_score=baseline['score'],delta_score=g['score']-baseline['score'],
                correct_to_wrong=int(baseline['score']==1 and g['score']==0),wrong_to_correct=int(baseline['score']==0 and g['score']==1),
                first_token_changed=g['token_ids'][0]!=baseline['token_ids'][0],answer_changed=g['prediction']!=baseline['prediction'],
                text_changed=g['text']!=baseline['text'],token_ids=json.dumps(g['token_ids']),full_token_ids=json.dumps(baseline['token_ids']),
                length=g['length'],eos=g['eos'],truncated=g['truncated'],seconds=g['seconds'],
                **anomaly(g['token_ids'],g['text'],g['eos'])))
        data[key]=values
    actual=list((OUT/'generation').glob('shard*/checkpoints/*/*.json'))
    if len(actual)!=len(manifest['samples'])*len(CONDITIONS):raise ValueError('Extra/duplicate/missing checkpoints')
    if sum(s['completed'] for s in status)!=len(actual):raise ValueError('Worker completion counts disagree')
    freeze_json(OUT/'generation_coverage.json',dict(passed=True,questions=len(data),images=len({s['image_id'] for s in manifest['samples']}),
        conditions=len(CONDITIONS),generations=len(actual),question_ids_unique=True,all_scores_independently_replayed=True,
        complete_for_frozen_core=True,complete_for_requested_3000_six_tasks=False))
    write_csv('checkpoint_manifest.csv',paths);write_csv('per_question_generation.csv',per)
    return manifest,data,per


def summarize(manifest,data):
    summary=[];taskwise=[];comparisons=[];task_contrasts=[];bootrows=[];changes=[];anomalies=[]
    for scope in ['Overall',*TASKS]:
        selected=[s for s in manifest['samples'] if scope=='Overall' or s['task']==scope]
        if not selected:
            for c in CONDITIONS:taskwise.append(dict(task=scope,condition=c,n_questions=0,status='unavailable_independent_samples'))
            continue
        score=np.array([[data[(s['task'],s['question_id'])][c]['score'] for c in CONDITIONS] for s in selected],dtype=float)
        columns=[score[:,0]]+[score[:,i]-score[:,0] for i in range(1,11)]
        families=[1]+([8]*8+[2]*2 if scope=='Overall' else [60]*10)
        if scope=='Overall':
            columns += [score[:,list(CONDITIONS).index(a)]-score[:,list(CONDITIONS).index(b)] for a,b in CONTRASTS]
            families += [6]*6
        intervals=bootstrap(np.stack(columns,axis=1),[s['image_id'] for s in selected],families,
            repetitions=PROTOCOL['statistics']['repetitions'])
        for i,c in enumerate(CONDITIONS):
            selected_g=[data[(s['task'],s['question_id'])][c] for s in selected]
            first=intervals[0];delta=intervals[i] if i else None
            harm=int(np.sum((score[:,0]==1)&(score[:,i]==0)));gain=int(np.sum((score[:,0]==0)&(score[:,i]==1)))
            native=[data[(s['task'],s['question_id'])]['S0'] for s in selected]
            n=len(selected);margin=1 if scope=='Overall' else 3
            row=dict(task=scope,condition=c,name=CONDITIONS[c]['name'],n_questions=n,n_images=len({s['image_id'] for s in selected}),
                accuracy_pp=float(score[:,i].mean()*100),full_accuracy_pp=float(score[:,0].mean()*100),
                delta_accuracy_pp=float((score[:,i]-score[:,0]).mean()*100),correct_to_wrong=harm,wrong_to_correct=gain,
                full_correct=int(score[:,0].sum()),full_wrong=int(n-score[:,0].sum()),
                harm_fraction_of_full_correct=harm/int(score[:,0].sum()) if score[:,0].sum() else None,
                correction_fraction_of_full_wrong=gain/int(n-score[:,0].sum()) if n-score[:,0].sum() else None,
                text_changed=sum(g['text']!=f['text'] for g,f in zip(selected_g,native)),
                answer_changed=sum(g['prediction']!=f['prediction'] for g,f in zip(selected_g,native)),
                answer_change_rate_pp=100*sum(g['prediction']!=f['prediction'] for g,f in zip(selected_g,native))/n,
                first_token_changed=sum(g['token_ids'][0]!=f['token_ids'][0] for g,f in zip(selected_g,native)),
                mean_length=float(np.mean([g['length'] for g in selected_g])),
                eos_rate_pp=100*sum(g['eos'] for g in selected_g)/n,truncated=sum(g['truncated'] for g in selected_g),
                repetition_flagged=sum(g['repetition_flag'] for g in selected_g),empty_answer=sum(g['empty_answer'] for g in selected_g),
                mcnemar_exact_unadjusted=mcnemar_exact(harm,gain),
                mcnemar_caveat='Descriptive question-level exact test; repeated-image dependence not modeled; paired image bootstrap is primary.',
                status='complete_frozen_observed_mixture' if scope=='Overall' else 'complete_frozen_task',
                ni_margin_pp=margin,ni_scope='pooled_noninferiority' if scope=='Overall' else 'heterogeneity_risk_screen_only')
            if delta:
                row.update({k+'_pp':100*delta[k] for k in ['ci95_low','ci95_high','family95_low','family95_high']})
                row.update(ordinary_noninferior=100*delta['ci95_low']>-margin,
                    family_noninferior=100*delta['family95_low']>-margin,comparison_family=delta['family'],
                    degenerate_bootstrap=delta['degenerate_bootstrap'])
                bootrows.append(dict(task=scope,comparison=c+'-S0',**delta))
            else:
                row.update(full_ci95_low_pp=first['ci95_low']*100,full_ci95_high_pp=first['ci95_high']*100,
                    ordinary_noninferior=None,family_noninferior=None)
            (summary if scope=='Overall' else taskwise).append(row)
            changes.append({k:row[k] for k in ['task','condition','n_questions','correct_to_wrong','wrong_to_correct','delta_accuracy_pp','answer_changed','answer_change_rate_pp','text_changed','first_token_changed']})
            anomalies.append({k:row[k] for k in ['task','condition','n_questions','mean_length','eos_rate_pp','truncated','repetition_flagged','empty_answer']})
        if scope=='Overall':
            for (a,b),interval in zip(CONTRASTS,intervals[11:]):
                row=dict(task=scope,condition_a=a,condition_b=b,direction='a minus b',n_questions=len(selected),
                    delta_accuracy_pp=interval['mean']*100,
                    **{k+'_pp':interval[k]*100 for k in ['ci95_low','ci95_high','family95_low','family95_high']},
                    family=interval['family'])
                comparisons.append(row);bootrows.append(dict(task=scope,comparison=a+'-'+b,**interval))
        else:
            for a,b in CONTRASTS:
                va=score[:,list(CONDITIONS).index(a)];vb=score[:,list(CONDITIONS).index(b)]
                task_contrasts.append(dict(task=scope,condition_a=a,condition_b=b,n_questions=len(selected),
                    accuracy_a_pp=float(va.mean()*100),accuracy_b_pp=float(vb.mean()*100),delta_a_minus_b_pp=float((va-vb).mean()*100),
                    a_correct_b_wrong=int(np.sum((va==1)&(vb==0))),a_wrong_b_correct=int(np.sum((va==0)&(vb==1))),
                    inference='Descriptive paired point contrast only; no additional significance/noninferiority claim.'))
        print('Statistics complete',scope,len(selected),flush=True)
    for name,rows in [('joint_ablation_accuracy_summary.csv',summary),('taskwise_accuracy_summary.csv',taskwise),
        ('paired_comparisons.csv',comparisons),('bootstrap_statistics.csv',bootrows),('answer_change_analysis.csv',changes),
        ('generation_anomaly_analysis.csv',anomalies),('taskwise_paired_comparisons.csv',task_contrasts)]:write_csv(name,rows)
    return summary,taskwise,comparisons


def historical_audit():
    old=ROOT/'results/paper_rq1_single_layer_gqa6'
    generations=csv_read(old/'full_generation_80_layer_summary.csv')
    early=[r for r in generations if int(r['layer']) in [2,3,4,5]]
    snapshot=ROOT/'results/paper_rq1_attention_time/short_answer_snapshot/confirmation_summary.csv'
    if not snapshot.exists():snapshot=ROOT/'results/paper_rq1_attention_time/confirmation_summary.csv'
    matching=[]
    if snapshot.exists():
        matching=[r for r in csv_read(snapshot) if 'textvqa' in str(r).lower() and 'Decode-only' in str(r)]
    sources=[old/'report.md',old/'full_generation_80_layer_summary.csv',
        ROOT/'results/bos_cheap_discovery/report.md',ROOT/'results/paper_rq1_attention_time/short_answer_snapshot/report.md',
        ROOT/'results/paper_rq1_decode_layerwise_quality/report.md']
    if snapshot.exists():sources.append(snapshot)
    audit=dict(source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        original_singlelayer_generation_subset_questions=480,
        individual_L2_to_L5_rows=early,all_individual_early_net_accuracy_changes_zero=all(float(r['delta_accuracy_pp'])==0 for r in early),
        historical_textvqa_decode_rows=matching,
        scope='Historical480 exploratory single-layer P-1 generation is not independent joint confirmation. '
        'Prior Combined in paper_rq1 was P-1 Prefill plus Decode, not this study entire-Prefill Combined.')
    freeze_json(OUT/'historical_protocol_audit.json',audit)


def figures(manifest,data,summary,taskwise,contrasts):
    sys.path.insert(0,'/data/lk/ATE-resources/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    by={r['condition']:r for r in summary};labels=['P-1','Prefill-all','Decode','Combined'];colors=['#26747c','#bd4f46','#55904b','#ad812c']
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def errors(rows,low='ci95_low_pp',high='ci95_high_pp'):
        means=np.array([r['delta_accuracy_pp'] for r in rows]);return np.stack([means-np.array([r[low] for r in rows]),np.array([r[high] for r in rows])-means])
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    conditions=list(CONDITIONS)[:9];x=np.arange(len(conditions));acc=[by[c]['accuracy_pp'] for c in conditions]
    axes[0].bar(x,acc,color=['#777777']+colors+colors)
    axes[0].set_xticks(x,['Full']+[f'0-4\n{s}' for s in labels]+[f'2-5\n{s}' for s in labels],rotation=40,ha='right')
    axes[0].set_ylabel('Complete-answer accuracy (%)');axes[0].axhline(by['S0']['accuracy_pp'],color='black',linestyle='--',linewidth=1)
    selected=[by[c] for c in conditions[1:]]
    axes[1].errorbar(np.arange(8),[r['delta_accuracy_pp'] for r in selected],yerr=errors(selected,'family95_low_pp','family95_high_pp'),fmt='o',color='#26747c',label='Bonferroni95%, family8')
    axes[1].errorbar(np.arange(8),[r['delta_accuracy_pp'] for r in selected],yerr=errors(selected),fmt='none',color='black',capsize=3,label='Ordinary95%')
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].axhline(-1,color='#bd4f46',linestyle='--',label='NI boundary -1pp')
    axes[1].set_xticks(np.arange(8),conditions[1:]);axes[1].set_ylabel('Paired accuracy change (pp)');axes[1].legend(fontsize=8)
    fig.suptitle(f'Early joint Attention ablation: actual independent cohort N={len(manifest["samples"])} (not balanced3000)')
    save(fig,'figure1_joint_accuracy')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    task_lookup={(r['task'],r['condition']):r for r in taskwise};cs=['S0','S1','S2','S4','S5','S6','S8']
    for ax,task in zip(axes.flat,TASKS):
        available=[task_lookup[(task,c)] for c in cs]
        if not available[0]['n_questions']:
            ax.text(.5,.5,'No independent samples\nNo estimate',ha='center',va='center',transform=ax.transAxes);ax.set_title(task+' (N=0)');continue
        ax.bar(np.arange(len(cs)),[r['accuracy_pp'] for r in available],color=['#777777','#26747c','#bd4f46','#ad812c','#26747c','#bd4f46','#ad812c'])
        ax.axhline(available[0]['accuracy_pp'],color='black',linestyle='--',linewidth=.7)
        ax.set_xticks(np.arange(len(cs)),cs);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)')
        ax.set_title(task+f' (N={available[0]["n_questions"]})'+(' - descriptive only' if available[0]['n_questions']<20 else ''))
    fig.suptitle('Actual task coverage; S4/S8 use entire-Prefill plus Decode ablation')
    save(fig,'figure2_taskwise_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,pair,title in zip(axes,[('S1','S2'),('S5','S6')],['L0-L4','L2-L5']):
        full=[data[(s['task'],s['question_id'])]['S0']['score'] for s in manifest['samples']]
        transitions=Counter((data[(s['task'],s['question_id'])][pair[0]]['score'],data[(s['task'],s['question_id'])][pair[1]]['score']) for s in manifest['samples'])
        matrix=np.array([[transitions[(a,b)] for b in [0,1]] for a in [0,1]])
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=len(manifest['samples']))
        for a in [0,1]:
            for b in [0,1]:ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>len(full)*.5 else 'black',fontsize=18)
        contrast=next(r for r in contrasts if r['condition_a']==pair[1] and r['condition_b']==pair[0])
        ax.set_xticks([0,1],['Wrong','Correct']);ax.set_yticks([0,1],['Wrong','Correct']);ax.set_xlabel('Prefill-all-only');ax.set_ylabel('P-1-only')
        ax.set_title(f'{title}: all minus P-1 = {contrast["delta_accuracy_pp"]:+.2f}pp\nordinary95% [{contrast["ci95_low_pp"]:+.2f}, {contrast["ci95_high_pp"]:+.2f}]')
    save(fig,'figure3_p1_vs_prefill_all')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,cs,title in zip(axes,[['S1','S2','S3','S4'],['S5','S6','S7','S8']],['L0-L4','L2-L5']):
        rows=[by[c] for c in cs]
        ax.bar(range(4),[r['delta_accuracy_pp'] for r in rows],color=colors)
        ax.errorbar(range(4),[r['delta_accuracy_pp'] for r in rows],yerr=errors(rows,'family95_low_pp','family95_high_pp'),fmt='none',color='black',capsize=4)
        ax.axhline(0,color='black',linewidth=.7);ax.axhline(-1,color='#bd4f46',linestyle='--');ax.set_xticks(range(4),labels)
        ax.set_ylabel('Accuracy change (pp), family8 intervals');ax.set_title(title)
    fig.suptitle('Distinct temporal scopes; Combined is not assumed additive')
    save(fig,'figure4_temporal_scopes')
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained');cs=list(CONDITIONS)[1:];x=np.arange(10)
    axes[0].bar(x-.18,[by[c]['correct_to_wrong'] for c in cs],width=.36,color='#bd4f46',label='Correct to wrong')
    axes[0].bar(x+.18,[by[c]['wrong_to_correct'] for c in cs],width=.36,color='#26747c',label='Wrong to correct')
    axes[0].set_xticks(x,cs);axes[0].set_ylabel('Questions');axes[0].legend()
    axes[1].plot(x,[by[c]['delta_accuracy_pp'] for c in cs],'-o',color='#26747c',label='Net accuracy change (pp)')
    axes[1].plot(x,[by[c]['answer_change_rate_pp'] for c in cs],'-s',color='#ad812c',label='Answer change rate (%)')
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xticks(x,cs);axes[1].set_ylabel('Percentage points / percent');axes[1].legend()
    save(fig,'figure5_harm_and_correction')
    candidates={};kinds=['harm','improvement','unchanged']
    for s in sorted(manifest['samples'],key=lambda s:s['question_id']):
        values=data[(s['task'],s['question_id'])]
        for c in ['S4','S8','S1','S5','S2','S6']:
            f,g=values['S0'],values[c]
            kind='harm' if f['score']>g['score'] else 'improvement' if f['score']<g['score'] else 'unchanged' if f['text']==g['text'] else 'changed_same_score'
            if kind in kinds and (kind,c) not in candidates:candidates[(kind,c)]=(s,c,kind)
    cases=[];seen=set()
    for kind in kinds:
        for (k,c),value in candidates.items():
            if k==kind and value[0]['question_id'] not in seen:
                cases.append(value);seen.add(value[0]['question_id'])
                if sum(v[2]==kind for v in cases)==2:break
    fig=plt.figure(figsize=(17,12),layout='constrained');grid=fig.add_gridspec(4,3,height_ratios=[2,1.3,2,1.3]);case_rows=[]
    for index,(s,c,kind) in enumerate(cases):
        col=index%3;row=(index//3)*2;ax=fig.add_subplot(grid[row,col]);caption=fig.add_subplot(grid[row+1,col]);caption.axis('off')
        with Image.open(s['image_path']) as image:ax.imshow(image.convert('RGB'))
        ax.axis('off');ax.set_title(f'{kind}: {s["task"]} / Q{s["question_id"]}')
        values=data[(s['task'],s['question_id'])];full=values['S0'];g=values[c]
        message=f'Q: {s["question"]}\nGold: {s["answer"]}\nFull ({full["score"]}): {full["text"]}\n{c} ({g["score"]}): {g["text"]}\n{CONDITIONS[c]["name"]}'
        wrapped='\n'.join(textwrap.fill(line,55) for line in message.splitlines())
        caption.text(0,1,wrapped,transform=caption.transAxes,va='top',fontsize=10)
        case_rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],kind=kind,condition=c,
            question=s['question'],gold=s['answer'],full=full['text'],changed=g['text'],full_score=full['score'],changed_score=g['score']))
    fig.suptitle('Deterministic illustrative examples; not a prevalence estimate')
    save(fig,'figure6_real_cases');write_csv('selected_cases.csv',case_rows)
    freeze_json(OUT/'figure_sources.json',dict(figures=[f'figure{i}' for i in range(1,7)],
        source_files={n:sha256(OUT/n) for n in ['per_question_generation.csv','joint_ablation_accuracy_summary.csv','taskwise_accuracy_summary.csv','paired_comparisons.csv','selected_cases.csv']},
        smoothing=False,all_png_and_pdf=True))


def report(manifest,summary,taskwise,contrasts):
    by={r['condition']:r for r in summary};selection=json.loads((OUT/'selection_audit.json').read_text())
    code=json.loads((OUT/'execution_manifest.json').read_text())['code_commit']
    primary_pass=[c for c in list(CONDITIONS)[1:9] if by[c]['family_noninferior']]
    findings=dict(frozen_core_status='complete',requested_balanced_six_task_status='partial_due_independent_data_shortfall',
        questions=len(manifest['samples']),images=selection['images'],generations=len(manifest['samples'])*11,
        primary_pooled_corrected_ni_pass=primary_pass,pooled_mixture='Actual question-weighted1859 cohort, not balanced3000',
        s1_delta_pp=by['S1']['delta_accuracy_pp'],s4_delta_pp=by['S4']['delta_accuracy_pp'],s8_delta_pp=by['S8']['delta_accuracy_pp'],
        no_cross_task_generalization=True,supplementary_risk_checks='not_run')
    freeze_json(OUT/'main_findings.json',findings)
    lines=['# Early-Layer Joint Attention Ablation for Short-Answer VQA','',
        '## 完成状态与边界','',
        f'冻结核心矩阵已完成：{len(manifest["samples"])}题 × 11条件，共{len(manifest["samples"])*11}次真实完整生成。',
        '**原计划六类各500题确认仍为 partial（独立样本不足）**：未补造、重复采样、混入训练集或回填历史图片。',
        'LogicalObj 无独立题；CompareAttr 仅6题。总体是实际可用题目的加权混合，不能表述为六类平衡3000题结论。',
        '历史排除规则较保守：排除了历史清单中已预留但可能未执行的图片。精确 ID/SHA256 去重不是感知近重复证明。',
        '',f'模型：LLaVA-1.5-7B `{PROTOCOL["model_revision"]}`；FP16/eager，Transformers4.49.0。',
        f'冻结采集代码：`{code}`。物理GPU1/2/3，同图全部条件同卡；未使用/终止GPU0其他任务。',
        '保留原短答案 Prompt、正常KV Cache、Greedy、最多32新Token、EOS=2和官方GQA主准确率转换。',
        '同时安装多层 o_proj Hook；MLP/LayerNorm/残差正常；每条件独立Prefill及自己的真实生成前缀/Cache。',
        '**这里是输出置零的功能消融，不是真实计算跳过；没有速度收益主张。**','',
        '## 数据覆盖','', '| 任务 | 请求 | 实际题数 | 独立图片数 |','|---|---:|---:|---:|']
    for task in TASKS:
        selected=[s for s in manifest['samples'] if s['task']==task]
        lines.append(f'| {task} | 500 | {len(selected)} | {len({s["image_id"] for s in selected})} |')
    lines+=['','与历史图片ID/SHA256重叠为0；全部问题来自官方Balanced验证集，按输入哈希固定，不参考模型表现。',
        '每题均保存生成Token、完整答案、归一化预测、评分和条件检查点；重跑评分/Token解码与保存结果逐一核对。',
        '', '## 总体完整答案准确率','',
        '| 条件 | 准确率% | 相对Full pp | 普通95%CI pp | 校正95%CI pp | 对→错 | 错→对 | 校正NI |',
        '|---|---:|---:|---|---|---:|---:|---|']
    for c,r in by.items():
        ci='—' if c=='S0' else f'[{r["ci95_low_pp"]:+.2f}, {r["ci95_high_pp"]:+.2f}]'
        adj='—' if c=='S0' else f'[{r["family95_low_pp"]:+.2f}, {r["family95_high_pp"]:+.2f}]'
        ni='不适用' if c=='S0' else '通过' if r['family_noninferior'] else '未通过'
        lines.append(f'| {c}: {CONDITIONS[c]["name"]} | {r["accuracy_pp"]:.2f} | {r["delta_accuracy_pp"]:+.2f} | {ci} | {adj} | {r["correct_to_wrong"]} | {r["wrong_to_correct"]} | {ni} |')
    lines+=['','200,000次按图配对Bootstrap，以图聚类的sum/count比率保留重复图片的题目权重。',
        '主S1–S8共8比较，辅助S9/S10共2比较，各任务10×6共60比较，六个直接对比单独family6。使用双侧Bonferroni分位数区间；不同family之间不主张统一同时覆盖。',
        '总体校正下界须严格高于−1pp；任务−3pp只用于异质性风险筛查，不是安全保证。',
        '极端尾部分位数有Monte Carlo误差，不是精确有限样本覆盖保证；零翻转时Bootstrap退化，不能解释为总体风险为零。',
        '题目级未校正精确McNemar另存CSV，仅描述；重复图片相关性不由该检验建模，主要推断使用图聚类Bootstrap。',
        '普通区间包含0、不显著下降和得分相近，都不等于非劣效；总体通过也不能外推每类安全。','',
        '## 子任务完整结果','', '| 任务/题数 | 条件 | 准确率% | Δpp | 对→错 / 错→对 | 校正95%CI pp |','|---|---|---:|---:|---:|---|']
    for r in taskwise:
        c=r['condition']
        if not r['n_questions']:
            if c=='S0':lines.append(f'| {r["task"]}/0 | 无独立数据 | — | — | — | — |')
            continue
        ci='—' if c=='S0' else f'[{r["family95_low_pp"]:+.2f}, {r["family95_high_pp"]:+.2f}]'
        lines.append(f'| {r["task"]}/{r["n_questions"]} | {c} | {r["accuracy_pp"]:.2f} | {r["delta_accuracy_pp"]:+.2f} | {r["correct_to_wrong"]} / {r["wrong_to_correct"]} | {ci} |')
    lines+=['','## 预设直接配对比较','', '| a−b | Δ准确率 pp | 普通95%CI | family6校正95%CI |','|---|---:|---|---|']
    for r in contrasts:lines.append(f'| {r["condition_a"]} − {r["condition_b"]} | {r["delta_accuracy_pp"]:+.2f} | [{r["ci95_low_pp"]:+.2f}, {r["ci95_high_pp"]:+.2f}] | [{r["family95_low_pp"]:+.2f}, {r["family95_high_pp"]:+.2f}] |')
    lines+=['','同题、同任务、同模式的直接配对点估计和不一致数量另见taskwise_paired_comparisons.csv。该分解仅描述，不增加未预注册的显著性或非劣效结论。']
    lines+=['','## 研究问题','',
        f'1. **L0–L4 联合P−1**：S1 Δ={by["S1"]["delta_accuracy_pp"]:+.2f}pp，损害{by["S1"]["correct_to_wrong"]}题、纠正{by["S1"]["wrong_to_correct"]}题；总体校正NI {by["S1"]["family_noninferior"]}。只能描述当前独立可用样本。',
        f'2. **扩展完整Prefill**：S2 Δ={by["S2"]["delta_accuracy_pp"]:+.2f}pp；S6 Δ={by["S6"]["delta_accuracy_pp"]:+.2f}pp。S2−S1及S6−S5见直接配对表；不能把单Query低损害等同于整段Prefill冗余。',
        f'3. **全程L0–L4**：S4 Δ={by["S4"]["delta_accuracy_pp"]:+.2f}pp，校正NI {by["S4"]["family_noninferior"]}。'+('通过仅涉及当前混合样本的预设统计判据，仍需独立数据和真实跳过验证。' if by['S4']['family_noninferior'] else '因此不能声称前五层Attention可安全删除。'),
        f'4. **L2–L5对照**：S8 Δ={by["S8"]["delta_accuracy_pp"]:+.2f}pp，校正NI {by["S8"]["family_noninferior"]}。四种同模式对比全部保留；两组分别5层和4层，不能将差异完全归因于L0–L1。辅助S9/S10也不消除所有层数混杂。',
        '5. **任务异质性**：逐任务区间、损害数量及异常生成全部列出。CompareAttr六题统计功效极低、LogicalObj不可估计，不声称六类均无损。',
        f'6. **阶段边界**：S3/S7首Token严格保留Full，但完整答案可能因后续词或EOS改变。S3截断{by["S3"]["truncated"]}题，S7截断{by["S7"]["truncated"]}题；Combined不作Prefill与Decode可加性假设。','',
        '## 假说结论边界','',
        ('D型有限证据：部分方案在实际独立可用GQA混合样本通过校正NI：'+', '.join(primary_pass)+'。不等于所请求六类平衡确认完成。' if primary_pass else '没有主要条件在实际可用混合样本通过校正NI；不能宣称早层联合安全。'),
        'A/B/C的具体支持须结合S1与S2/S4、S5与S6/S8的负向结果及直接配对区间，不以“单层低敏感”替代联合证据。',
        '历史480题是探索性单层P−1生成，不能相加推断联合结果；原时间窗口Combined只改Prefill P−1，本轮Combined改全部Prefill，协议不同。',
        'TextVQA历史L0–L2 Decode-only下降约4.78pp为已有风险反例，本轮没有复跑该任务，不能将GQA发现推广到所有短答案VQA。',
        '', '## 正确性与资源','',
        '18项采集CPU测试、9项统计/汇总/终止边界CPU测试；三GPU各两个预设真实问题Smoke。Full与HF原生生成Token一致，全部11条件No-op有效Logits精确一致。',
        '各条件独立HF生成驱动复核；S1多层P−1同时置零、S2全Prefill/正常Decode、S3原生Prefill/首Token、S4/S8全程指定层均检查。',
        '32层Attention/MLP调用完整，未目标Query局部输出不变；各条件Cache独立；阶段用真实cache_position/长度定位。',
        '跨三卡同题Full有效Logits和干预生成Token精确一致；正常及异常Hook清理通过；参数版本、输入哈希和第三方代码/模型文件绑定检查通过。',
        '正式生成所有有效Logits检查有限；失败尝试如有全部保存在failure_inventory.json中，不隐藏已修复失败。',
        '', '## 生成异常与案例','',
        '每条件保存EOS、长度、截断、重复Token与重复四元组标记、空答案。重复标记只是可复核启发式，不将其自动称为无意义文本。',
        '见generation_anomaly_analysis.csv、逐题CSV及selected_cases.csv；每任务具体退化、改善及答案不变案例保留在task_error_cases.csv。',
        '', '## 图表','']
    for i,name in enumerate(['joint_accuracy','taskwise_accuracy','p1_vs_prefill_all','temporal_scopes','harm_and_correction','real_cases'],1):
        lines += [f'![Figure {i}](figures/figure{i}_{name}.png)','']
    lines+=['## 未执行部分','', 'POPE Adversarial、ScienceQA-IMG、TextVQA补充风险检查均未执行。优先完成并严格校验核心GQA，不声称跨数据集泛化。',
        '原六类各500题确认因独立官方验证图像不足而未完成；不回填旧样本、不放宽NI、不追加层窗口或调参。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    short=['# 中文研究总结','',f'冻结可用核心完成：{len(manifest["samples"])}题、11条件；请求的六类平衡确认仍partial。','',
        '| 条件 | Accuracy% | Δpp | 对→错 | 错→对 | 校正NI |','|---|---:|---:|---:|---:|---|']
    for c in ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']:
        r=by[c];short.append(f'| {c} | {r["accuracy_pp"]:.2f} | {r["delta_accuracy_pp"]:+.2f} | {r["correct_to_wrong"]} | {r["wrong_to_correct"]} | {r.get("family_noninferior")} |')
    short+=['', '通过方案仅是实际可用GQA混合样本的统计证据，非所有短答案任务安全、非真实计算省略、非推理加速。',
        '完整子任务、阶段对比、CI、错误案例与局限见report.md。LogicalObj无独立题、CompareAttr仅6题；未执行跨任务补充。']
    (OUT/'research_summary.md').write_text('\n'.join(short)+'\n')


def main():
    started=time.perf_counter();manifest,data,per=load();historical_audit()
    summary,taskwise,contrasts=summarize(manifest,data)
    cases=[]
    for task in TASKS:
        for c in list(CONDITIONS)[1:]:
            current=[r for r in per if r['task']==task and r['condition']==c]
            for field,kind in [('correct_to_wrong','harm'),('wrong_to_correct','improvement')]:
                for row in sorted([r for r in current if r[field]],key=lambda r:r['question_id'])[:3]:cases.append(dict(row,case_kind=kind))
    write_csv('task_error_cases.csv',cases)
    figures(manifest,data,summary,taskwise,contrasts);report(manifest,summary,taskwise,contrasts)
    failure_paths=list((OUT/'failures').glob('*.json'))+list((OUT/'attempts').glob('*failure.json'))
    failures=[dict(path=str(p.relative_to(OUT)),sha256=sha256(p),**json.loads(p.read_text())) for p in sorted(failure_paths)]
    freeze_json(OUT/'failure_inventory.json',dict(failed_attempts=len(failures),records=failures,
        unresolved_failures=0,all_frozen_conditions_replayed=True))
    completion=dict(status='complete_frozen_available_core',requested_six_task_confirmation='partial_data_shortfall',
        questions=len(manifest['samples']),generations=len(per),supplementary='not_run',analysis_seconds=time.perf_counter()-started,
        collector_commit=json.loads((OUT/'execution_manifest.json').read_text())['code_commit'])
    timing=[]
    for stage in ['smoke','generation']:
        for rank in range(3):
            value=json.loads((OUT/f'{stage}/shard{rank}/completion.json').read_text())
            timing.append(dict(stage=stage,physical_gpu=rank+1,seconds=value['seconds'],completed=value['completed'],
                caveat='Worker stage measured after model loading; GPU workers overlap in wall time.'))
    timing.append(dict(stage='analysis',seconds=completion['analysis_seconds'],caveat='CPU scoring replay, bootstrap, figures and report.'))
    write_csv('execution_times.csv',timing)
    freeze_json(OUT/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':main()
