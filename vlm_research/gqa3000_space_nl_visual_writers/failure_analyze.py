"""CPU-only complete-record audit and prespecified recovery comparisons."""
import collections,csv,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .failure_probe import P,OUT,ROOT,NAMES,PAIRS,TASKS,read,sha256,digest,csvout,atomic_json,verify,stamp,gqa_prediction

def category_flags(v,g):
    wrong=v['stratum']!='C';ok=lambda c:bool(g[c]['correct'])
    return dict(NL_fix_joint_fail=wrong and ok('S+NL') and not ok('S+SPACE+NL'),SPACE_fix_NL_fail=wrong and ok('S+SPACE') and not ok('S+NL'),joint_only_fix=wrong and ok('S+SPACE+NL') and not ok('S+SPACE') and not ok('S+NL'),both_template_controls_fix=wrong and ok('S+Template-Control-1') and ok('S+Template-Control-2'),either_template_control_fixes=wrong and (ok('S+Template-Control-1') or ok('S+Template-Control-2')),none_of_five_fix=wrong and not any(ok(c) for c in NAMES[2:]),S_correct_boundary_harm=not wrong and any(not ok(c) for c in NAMES[2:5]))

def bootstrap(ids,cases,records):
    images=sorted({cases[q]['sample']['image_sha256'] for q in ids});members=[[i for i,q in enumerate(ids) if cases[q]['sample']['image_sha256']==im] for im in images]
    mat=np.array([[records[q,c]['generation']['correct'] for c in NAMES] for q in ids],dtype=np.int64)
    ix=np.random.default_rng(20261104).integers(0,len(images),size=(20000,len(images)));sizes=np.array([len(m) for m in members]);ag=np.array([mat[m].sum(0) for m in members]);boots=ag[ix].sum(1)/sizes[ix].sum(1)[:,None]*100
    return mat,boots,len(images)

def main():
    p=verify();plan=read(P/'sample_plan.json');cases=plan['cases'];done=read(P/'gpu_completion.json');assert done['passed']
    workers=[read(P/f'worker{i}_completion.json') for i in range(2)];assert all(d['passed'] for d in workers)
    assert sum(d['new_generations'] for d in workers)==p['new_generations'] and sum(d['labels'] for d in workers)==p['logical_labels']
    for n,info in p['model_files'].items():assert sha256(n)==info['sha256'],n
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h,n
    records={};longrows=[];wide=[];rawindex=[];anomalies=[];audit_count=0
    for q,v in cases.items():
        s=v['sample'];d=v['design'];feature=set();gs={}
        for c in NAMES:
            path=P/f'worker{v["worker"]}'/'records'/q/(c+'.json');assert path.exists(),'NOT_GENERATED: '+str(path)
            r=read(path);g=r['generation'];w=v['writers'][c]
            assert r['writers']==w==g['writer_positions'] and g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
            assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['audit']['passed']
            if r['reused']:
                assert c in NAMES[:2] and sha256(ROOT/r['source']['path'])==r['source']['sha256'];old=read(ROOT/r['source']['path']);assert old['generation']==g
                assert sha256(r['historical_raw']['path'])==r['historical_raw']['sha256']
                with np.load(r['historical_raw']['path']) as z:assert int(z['first_logits'].argmax())==g['token_ids'][0]
            else:
                assert r['worker']==v['worker'] and r['plan_sha256']==p['plan_sha256']
                assert sha256(r['raw']['path'])==r['raw']['sha256'] and r['input_hashes']==d['input_hashes'] and r['MLP_calls']==[g['length']]*32 and r['weights_unchanged'] and r['hook_cleanup']
                with np.load(r['raw']['path']) as z:
                    a=z['audit'];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
                    assert np.array_equal(z['first_logits'],z['valid_logits'][0]) and z['valid_logits'].shape[0]==g['length'] and z['valid_logits'].argmax(-1).tolist()==g['token_ids']
                    meta=json.loads(str(z['record_metadata']));assert meta=={k:r[k] for k in meta}
                feature.add(r['projected_visual_sha256']);audit_count+=1
                rawindex.append(dict(question_id=q,worker=v['worker'],condition=c,checkpoint=str((P/f'worker{v["worker"]}'/'checkpoints'/q/(digest(w)+'.json')).relative_to(P)),checkpoint_sha256=sha256(P/f'worker{v["worker"]}'/'checkpoints'/q/(digest(w)+'.json')),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256'],raw_bytes=r['raw']['bytes']))
            records[q,c]=r;gs[c]=g
            if not g['eos'] or g['empty'] or g['truncated'] or g['repeated_trigram']:anomalies.append(dict(question_id=q,condition=c,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeat=g['repeated_trigram']))
        assert len(feature)==1
        a=gs[NAMES[0]];b=gs[NAMES[1]]
        for c in NAMES:
            g=gs[c];w=v['writers'][c]
            longrows.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],stratum=v['stratum'],question=s['question'],gold=s['answer'],condition=c,status='REUSED_AUDITED' if records[q,c]['reused'] else 'GENERATED_VERIFIED',answer=g['text'],prediction=g['prediction'],correct=g['correct'],token_ids=json.dumps(g['token_ids']),All_correct=a['correct'],S_correct=b['correct'],harm_vs_S=int(b['correct'] and not g['correct']),fix_vs_S=int(not b['correct'] and g['correct']),harm_vs_All=int(a['correct'] and not g['correct']),fix_vs_All=int(not a['correct'] and g['correct']),answer_change_vs_S=int(g['text']!=b['text']),unchanged_score_changed_answer=int(g['text']!=b['text'] and g['correct']==b['correct']),first_token_flip_vs_S=int(g['token_ids'][0]!=b['token_ids'][0]),writer_positions=json.dumps(w),k=len(w),R=len(d['R']),writer_ratio=len(w)/len(d['R']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],worker=v['worker']))
        row=dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],stratum=v['stratum'],question=s['question'],gold=s['answer'],All_answer=a['text'],S_answer=b['text'],S_SPACE_answer=gs[NAMES[2]]['text'],S_NL_answer=gs[NAMES[3]]['text'],S_B_answer=gs[NAMES[4]]['text'],control1_answer=gs[NAMES[5]]['text'],control2_answer=gs[NAMES[6]]['text'],correct_per_condition=json.dumps({c:gs[c]['correct'] for c in NAMES}),writer_positions=json.dumps(v['writers']),writer_budget=json.dumps({c:dict(k=len(w),R=len(d['R']),ratio=len(w)/len(d['R'])) for c,w in v['writers'].items()}),first_token_flip=json.dumps({c:int(gs[c]['token_ids'][0]!=b['token_ids'][0]) for c in NAMES}),selection_sha256=v['selection_sha256'],preference_tier=v['preference_tier'],worker=v['worker'],**category_flags(v,gs))
        wide.append(row)
    assert audit_count==p['new_generations'] and len(longrows)==p['logical_labels'] and len(wide)==p['questions']
    csvout(P/'per_question.csv',wide);csvout(P/'per_condition.csv',longrows);csvout(P/'checkpoint_index.csv',rawindex)
    fields=list(wide[0]);recovered=[r for r in wide if r['stratum']!='C' and any(json.loads(r['correct_per_condition'])[c] for c in NAMES[2:])]
    nonmono=[r for r in wide if r['NL_fix_joint_fail']];harms=[r for r in wide if r['S_correct_boundary_harm']]
    csvout(P/'recovered_cases.csv',recovered,fields);csvout(P/'nonmonotonic_cases.csv',nonmono,fields);csvout(P/'harm_cases.csv',harms,fields)
    csvout(P/'all_unrecovered_cases.csv',[r for r in wide if r['none_of_five_fix']],fields)
    csvout(P/'all_new_condition_failures.csv',[r for r in longrows if r['condition'] in NAMES[2:] and not r['correct']],list(longrows[0]))
    summaries=[];pairs=[];recoveries=[];harm_summary=[];task_rows=[]
    groups=[(label,[q for q,v in cases.items() if label=='ALL_SELECTED' or (v['stratum'] in ['A','B'] if label=='AB' else v['stratum']==label)]) for label in ['AB','A','B','C','ALL_SELECTED']]
    groups += [(t+'_'+st,[q for q,v in cases.items() if v['sample']['task']==t and (v['stratum'] in ['A','B'] if st=='AB' else v['stratum']==st)]) for t in TASKS for st in ['AB','A','B','C']]
    for group,ids in groups:
        if not ids:continue
        mat,boots,nimg=bootstrap(ids,cases,records)
        for j,c in enumerate(NAMES):
            rr=[r for r in longrows if r['question_id'] in ids and r['condition']==c];ci=np.percentile(boots[:,j],[2.5,97.5]);sd=boots[:,j]-boots[:,1];sc=np.percentile(sd,[2.5,97.5]);ad=boots[:,j]-boots[:,0];ac=np.percentile(ad,[2.5,97.5])
            r=dict(group=group,condition=c,n=len(ids),image_clusters=nimg,correct=int(mat[:,j].sum()),accuracy_percent=float(mat[:,j].mean()*100),CI95_low=ci[0],CI95_high=ci[1],delta_vs_S_pp=float((mat[:,j]-mat[:,1]).mean()*100),delta_vs_S_CI95_low=sc[0],delta_vs_S_CI95_high=sc[1],delta_vs_All_pp=float((mat[:,j]-mat[:,0]).mean()*100),delta_vs_All_CI95_low=ac[0],delta_vs_All_CI95_high=ac[1],harm_vs_S=sum(x['harm_vs_S'] for x in rr),fix_vs_S=sum(x['fix_vs_S'] for x in rr),harm_vs_All=sum(x['harm_vs_All'] for x in rr),fix_vs_All=sum(x['fix_vs_All'] for x in rr),answer_changes_vs_S=sum(x['answer_change_vs_S'] for x in rr),answer_changed_score_unchanged=sum(x['unchanged_score_changed_answer'] for x in rr),first_token_flips_vs_S=sum(x['first_token_flip_vs_S'] for x in rr),mean_k=float(np.mean([x['k'] for x in rr])),mean_R_ratio=float(np.mean([x['writer_ratio'] for x in rr])))
            if '_' in group and group!='ALL_SELECTED':task_rows.append(r)
            else:summaries.append(r)
            if group.endswith('AB') or group in ['A','B'] or group.endswith('_A') or group.endswith('_B'):
                recoveries.append(dict(group=group,condition=c,S_error_denominator=len(ids),recovered=r['fix_vs_S'],recovery_percent=100*r['fix_vs_S']/len(ids),CI95_low=ci[0],CI95_high=ci[1]))
            if group=='C':harm_summary.append(dict(condition=c,n=len(ids),harm_count=r['harm_vs_S'],harm_percent=100*r['harm_vs_S']/len(ids),harm_CI95_low=100-ci[1],harm_CI95_high=100-ci[0],answer_changed_score_unchanged=r['answer_changed_score_unchanged'],harm_difference_vs_Control1=r['harm_vs_S']-sum(x['harm_vs_S'] for x in longrows if x['stratum']=='C' and x['condition']==NAMES[5]),harm_difference_vs_Control2=r['harm_vs_S']-sum(x['harm_vs_S'] for x in longrows if x['stratum']=='C' and x['condition']==NAMES[6])))
        if group in ['AB','A','B','C']:
            for left,right in PAIRS:
                j=NAMES.index(left);k=NAMES.index(right);ci=np.percentile(boots[:,j]-boots[:,k],[2.5,97.5]);adj=np.percentile(boots[:,j]-boots[:,k],[.5,99.5]);primary=group=='AB'
                pairs.append(dict(group=group,left=left,right=right,n=len(ids),image_clusters=nimg,delta_pp=float((mat[:,j]-mat[:,k]).mean()*100),CI95_low=ci[0],CI95_high=ci[1],primary_five_family=primary,Bonferroni99_low=adj[0] if primary else '',Bonferroni99_high=adj[1] if primary else '',right_correct_left_wrong=int(((mat[:,k]==1)&(mat[:,j]==0)).sum()),right_wrong_left_correct=int(((mat[:,k]==0)&(mat[:,j]==1)).sum())))
    csvout(P/'accuracy_summary.csv',summaries);csvout(P/'recovery_summary.csv',recoveries);csvout(P/'paired_comparisons.csv',pairs);csvout(P/'negative_control_summary.csv',harm_summary);csvout(P/'task_accuracy.csv',task_rows)
    cats=list(category_flags(next(iter(cases.values())),{c:records[next(iter(cases)),c]['generation'] for c in NAMES}))
    representatives=[]
    for cat in cats:
        eligible=sorted([r for r in wide if r[cat]],key=lambda r:(r['selection_sha256'],r['question_id']))
        if eligible:representatives.append(dict(category=cat,**eligible[0]))
    csvout(P/'representative_cases.csv',representatives,['category']+fields)
    plot(summaries)
    release=read(P/'gpu_release.json');assert release['all_own_children_exited']
    validation=dict(passed=True,questions=len(cases),unique_images=len(cases),logical_records=len(longrows),new_raw_records_verified=audit_count,history_reused=p['history_reuse'],all_outputs_present=True,missing_outputs=[],all_generated_argmax_ids_rechecked=True,all32_MLP=True,all_forbidden_actual_probabilities_zero=True,all_non_target_masks_exact=True,all_weights_inputs_hooks_exact=True,projected_features_equal_within_question=True,raw_metadata_equals_checkpoint=True,model_dependency_scorer_SHA256_exact=True,plan_sha256=p['plan_sha256'],manifest_sha256=p['manifest_sha256'],analysis_source_sha256=sha256(__file__),shard_question_counts=p['worker_questions'],shards_no_overlap=True,quick_and_diagnostic_images_excluded=True,anomalies=anomalies,smokes={str(i):read(P/f'worker{i}/smoke_validation.json') for i in range(2)},gpu_release=release,bootstrap_repetitions=20000,bootstrap_seed=20261104,category_counts={c:sum(bool(r[c]) for r in wide) for c in cats},utc=stamp())
    atomic_json(P/'validation.json',validation)
    write_report(p,plan,done,workers,summaries,pairs,harm_summary,representatives,validation)
    atomic_json(P/'completion.json',dict(status='FAILURE_PROBE_COMPLETE',questions=len(cases),S_error_questions=p['stratum_counts']['A']+p['stratum_counts']['B'],negative_controls=p['stratum_counts']['C'],new_generations=audit_count,history_reuse=p['history_reuse'],condition_labels=len(longrows),validation_passed=True,seconds=done['seconds'],frozen_execution_commits=sorted({w['code_commit'] for w in workers}),full3000_started=False,main_modified=False,utc=stamp()))
    print(json.dumps(dict(recovery=[r for r in recoveries if r['group']=='AB'],negative_control=harm_summary,primary_pairs=[r for r in pairs if r['primary_five_family']],categories=validation['category_counts']),ensure_ascii=False),flush=True)

def plot(summary):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});short=['SPACE','NL','SPACE+NL','Control-1','Control-2'];x=np.arange(5);fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for group,offset,color in [('A',-.25,'#4583b4'),('B',0,'#db9342'),('AB',.25,'#418974')]:
        rr=[next(r for r in summary if r['group']==group and r['condition']==c) for c in NAMES[2:]];y=np.array([r['accuracy_percent'] for r in rr]);lo=np.array([r['CI95_low'] for r in rr]);hi=np.array([r['CI95_high'] for r in rr]);axes[0].bar(x+offset,y,width=.24,label=f'{group} (n={rr[0]["n"]})',color=color);axes[0].errorbar(x+offset,y,yerr=[y-lo,hi-y],fmt='none',ecolor='#333',capsize=2)
    axes[0].set_title('Recovery among historical Semantic-k errors');axes[0].set_ylabel('Recovery rate (%)');axes[0].legend(fontsize=8);axes[0].set_ylim(0,max(45,axes[0].get_ylim()[1]))
    rr=[next(r for r in summary if r['group']=='C' and r['condition']==c) for c in NAMES[2:]];y=np.array([r['harm_vs_S']*100/r['n'] for r in rr]);lo=np.array([100-r['CI95_high'] for r in rr]);hi=np.array([100-r['CI95_low'] for r in rr]);axes[1].bar(x,y,color=['#4583b4','#4583b4','#db9342','#888','#888']);axes[1].errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',ecolor='#333',capsize=3)
    for i,r in enumerate(rr):axes[1].annotate(f'{r["harm_vs_S"]}/{r["n"]}',(i,y[i]),xytext=(0,7),textcoords='offset points',ha='center')
    axes[1].set_title('Harm on originally correct controls (n=24)');axes[1].set_ylabel('Harm rate (%)');axes[1].set_ylim(0,max(30,axes[1].get_ylim()[1]+8))
    for ax in axes:ax.set_xticks(x,short,rotation=18);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.015,'Image-cluster bootstrap 95% intervals; selected failure/control samples, not GQA3000 population estimates.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(P/'recovery_and_harm.png',dpi=220);plt.close(fig)

def write_report(p,plan,done,workers,summary,pairs,harms,reps,val):
    lookup={(r['group'],r['condition']):r for r in summary};n=p['stratum_counts']['A']+p['stratum_counts']['B'];nl=lookup['AB','S+NL'];sp=lookup['AB','S+SPACE'];b=lookup['AB','S+SPACE+NL'];c1=lookup['AB',NAMES[5]];c2=lookup['AB',NAMES[6]];joint=next(r for r in pairs if r['group']=='AB' and r['left']==NAMES[4] and r['right']==NAMES[3])
    text=['# Semantic-k失败题的SPACE/NL定向恢复验证','','**状态：FAILURE_PROBE_COMPLETE。只执行本轮冻结的小规模定向实验，未启动3000题矩阵。**','',
        f'冻结138道、138张独立图片：A组{p["stratum_counts"]["A"]}道（All对/S错）、B组{p["stratum_counts"]["B"]}道（All错/S错）、C组24道（S对）。主要分母是{n}道历史S错误题；这不是随机总体ACC评估，不能将恢复率换算为3000题ACC提升。','',
        '## 选样与冻结','',
        '原3000题S正确2514、错误486，完整错误名单见historical_S_errors.csv。先打印每任务A/B/C原始候选数，再按固定任务顺序、A/B/C顺序分配；优先避开pilot120/historical12题及图片，各层按SHA256("space-nl-failure-recovery-v1|task|stratum|question_id|image_sha256")升序，图片跨所有层全局去重。Quick24和Diagnostic4的全部题及图片均硬排除。抽样不使用新增生成、Gold NLL或补位成功率。','',
        'LogicalObj原始仅7道A、8道B；排除已测图片后A剩6道、B剩8道，因此总数比目标144少6，不跨层补齐。该任务的2道A和1道B只能从优先排除的旧pilot/history备选中选入，preference_tier=1逐题记录；其余135道均为tier0。所有本轮图片互不重复。各层原始/排除后/优先池/实际数量见candidate_counts.csv。即使未测试过边界干预，按历史S错误条件选样也不是独立外部验证。','',
        f'基线commit `{p["base_commit"]}`；冻结执行commit `{workers[0]["code_commit"]}`。输入sample_manifest.csv SHA256=`{p["manifest_sha256"]}`；sample_plan.json SHA256=`{p["plan_sha256"]}`；原3000题token_plan SHA256=`{p["parent_plan_sha256"]}`。全部3000题原All/S记录和本轮选中记录均通过既有审计；本轮再对选中历史JSON核对提交字节、原始Logits文件SHA与首Token。','',
        '## 条件与真实Token','',
        '只改变Prefill R→原始Image Key的Softmax前Hard Mask。S严格复用原集合；SPACE原生位置581/ID29871，NL位置582/ID13；模板单Token为Instruction中的Answer，双Token为Answer+the，均来自原输入，逐题无S重叠、位于R，单/双新增预算严格为1/2。SPACE与NL的原生offset可能覆盖同一换行字符，但它们是两个不同Tokenizer位置；没有向Prompt插入标签。逐题ID、piece、decode与offset见token_audit.json。','',
        '所有32层P−1和每个真实Decode Query始终禁读图像Key，所选Prefill R可以在全部32层直接读图。模型LLaVA-1.5-7B，revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/Eager/Transformers4.49.0，Greedy32、logits_to_keep=1，使用原GQA归一化完整答案评分。每条件独立真实KV/前缀，原生QKV、MLP、Residual与完整KV保留，无o_proj旁路。只评价功能效果，不代表计算节省。','',
        '## 恢复与负面控制','','| 条件 | A恢复 | B恢复 | 合并恢复率 | 合并95%CI | C损害 | C答案改变但评分不变 |','|---|---:|---:|---:|---|---:|---:|']
    for c in NAMES[2:]:
        a=lookup['A',c];z=lookup['B',c];r=lookup['AB',c];h=lookup['C',c]
        text.append(f'| {c} | {a["fix_vs_S"]}/{a["n"]} | {z["fix_vs_S"]}/{z["n"]} | {r["fix_vs_S"]}/{n} ({r["accuracy_percent"]:.2f}%) | [{r["CI95_low"]:.2f}, {r["CI95_high"]:.2f}]% | {h["harm_vs_S"]}/24 | {h["answer_changed_score_unchanged"]} |')
    text+=['','All-Writer在A/B/C中的正确数分别是56/56、0/58及'+str(lookup['C','All-Writer']['correct'])+'/24；S分别0/56、0/58及24/24。这些数由选样定义决定。accuracy_summary.csv额外保留所选样本的条件ACC、相对All/S差值、损害与纠正、输出与首Token变化、实际R预算；ALL_SELECTED仅为定向混合样本描述，不能当成总体ACC。六任务在task_accuracy.csv按A、B、AB和C分别统计，未根据结果重选任务或预算。','',
        '## 五项预设配对比较','',
        '按Image SHA整簇配对Bootstrap20000次，seed=20261104。此次各题图片唯一，但重采样单位仍显式按图片构建。下表主比较家族仅为合并A+B，给出普通探索性95%区间及五比较Bonferroni99%区间。分层、任务和负面控制为探索性描述，不将CI跨0解释成非劣效。单位均为恢复率百分点。','','| 左−右 | 差值pp | 95%CI | 调整99%CI | 右对→左错 / 右错→左对 |','|---|---:|---|---|---:|']
    for r in pairs:
        if r['primary_five_family']:text.append(f'| {r["left"]} − {r["right"]} | {r["delta_pp"]:+.3f} | [{r["CI95_low"]:+.3f}, {r["CI95_high"]:+.3f}] | [{r["Bonferroni99_low"]:+.3f}, {r["Bonferroni99_high"]:+.3f}] | {r["right_correct_left_wrong"]}/{r["right_wrong_left_correct"]} |')
    text+=['','## 直接回答','',
        f'1. NL修复{nl["fix_vs_S"]}/{n}题，恢复率{nl["accuracy_percent"]:.2f}%；A为{lookup["A","S+NL"]["fix_vs_S"]}/56，B为{lookup["B","S+NL"]["fix_vs_S"]}/58。',
        f'2. SPACE修复{sp["fix_vs_S"]}/{n}题（{sp["accuracy_percent"]:.2f}%），NL与SPACE净差{nl["fix_vs_S"]-sp["fix_vs_S"]:+d}题；差异的不确定性见配对CI。',
        f'3. 联合修复{b["fix_vs_S"]}/{n}题。NL已修复但联合破坏{joint["right_correct_left_wrong"]}题；NL失败但联合修复{joint["right_wrong_left_correct"]}题；相对NL净{joint["right_wrong_left_correct"]-joint["right_correct_left_wrong"]:+d}题。完整非单调案例见nonmonotonic_cases.csv。',
        f'4. 同预算模板Control-1修复{c1["fix_vs_S"]}题，Control-2修复{c2["fix_vs_S"]}题。NL−Control-1、SPACE−Control-1及联合−Control-2的直接配对与调整区间见上表；只有调整区间整体大于0时才将本批优势称为多重校正后有证据，点估计更高本身不足以宣称明显优于模板。',
        '5. C组24道上，'+ '；'.join(f'{c}损害{lookup["C",c]["harm_vs_S"]}题' for c in NAMES[2:])+ '。同预算控制的损害差见negative_control_summary.csv；24题只能发现本批负效应，不能精确估计总体损害率。',
        '6. 边界Token的有效性必须同时看单Token、联合、同预算模板及负面控制。个别恢复说明指定权限干预可改变该题输出和评分，不能证明其唯一存储视觉信息。Native Full Attention Mass偏高也不构成功能有效性的充分条件；本轮结果不支持仅凭Mass选择接收者。是否有值得继续研究的方向，应依据本批效应与区间提出独立验证假说，不能将本轮条件抽样外推为总体提升。本轮到此停止，不追加Token规则搜索。','',
        '## 正负案例（预设类别，按冻结SHA首例）','',
        '类别允许重叠，完整成员由per_question.csv中的布尔列保留。代表题按冻结selection_sha256最小选择，不以答案好坏挑展示；未恢复和负面案例同样列出。类别数量：'+json.dumps(val['category_counts'],ensure_ascii=False)+'。']
    for r in reps:
        text+=['',f'### {r["category"]}：{r["question_id"]} / {r["task"]} / {r["stratum"]}', '',r['question'], '',f'Gold：{r["gold"]}。','','| 条件 | 真实回答 | 正确 | Writer位置 |','|---|---|---:|---|']
        answers=[r[k] for k in ['All_answer','S_answer','S_SPACE_answer','S_NL_answer','S_B_answer','control1_answer','control2_answer']];correct=json.loads(r['correct_per_condition']);writers=json.loads(r['writer_positions'])
        for c,answer in zip(NAMES,answers):text.append(f'| {c} | {answer.replace("|","/")} | {correct[c]} | {writers[c]} |')
    text+=['','## 执行、审计与资源释放','',
        f'两GPU UUID：`{p["gpus"][0]["uuid"]}`、`{p["gpus"][1]["uuid"]}`；每卡69题、345条新生成，一卡一模型、独立目录/文件锁/原子JSON与NPZ，Image SHA确定性排序交替分片，所有正式题与图片互斥。每Worker只为自身分片准备模型输入并写自身检查点；共享冻结计划只读。固定三道Smoke属于验证输入，不计入正式样本。两卡均完成各自三道真实7B Smoke后，CPU关卡才统一放行主实验。','',
        f'两卡Smoke均通过HF/No-op所有有效Logits严格torch.equal、历史All/S完整答案Token及首Token向量严格复现、独立KV、32MLP与Hook正常/异常清理。{p["new_generations"]}条新增逐层记录全部复核：禁止实际概率0，非目标Mask完全不变，原输入和权重版本不变、投影视觉Hash同题一致，实际Writer严格匹配计划，原始有效Logits逐步argmax等于保存Token。Attention行和沿用冻结FP16审计界限0.002，没有改精度或放宽容差。12项原CPU测试和本轮实际七条件集合/预算/分片/Mask审计通过。','',
        f'历史复用{p["history_reuse"]}条All/S，新增{p["new_generations"]}条，共{p["logical_labels"]}个条件记录，全部有答案与检查点，无NOT_GENERATED。输出异常标签数{len(val["anomalies"])}，详细字段见validation.json。双GPU采集墙钟{done["seconds"]:.1f}秒（{done["seconds"]/60:.2f}分钟，含模型加载、两卡各自Smoke及主生成，不含CPU准备、统计、提交）。worker0/1时间见各自completion。','',
        '原始首Token/全部有效Logits、逐层审计数组保存在服务器资源目录；checkpoint_index.csv给出逐文件绝对路径、字节数、SHA256，Git交付JSON检查点及可验证索引，不声称GitHub包含NPZ。history_byte_audit.json绑定所有历史复用JSON的提交字节。必要CPU/Worker日志、两卡资源选择/预检/释放记录均保留。','',
        '资源释放：'+json.dumps({k:val['gpu_release'][k] for k in ['all_own_children_exited','selected_devices_no_compute_processes','selected_uuids']},ensure_ascii=False)+'。gpu_release.json记录结束时实际显存与进程，不能当作未来空闲保证。main及旧Quick24/3000历史输出未改写。','',
        '![Recovery and harm](recovery_and_harm.png)','']
    (P/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
