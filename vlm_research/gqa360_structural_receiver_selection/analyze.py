"""Recompute GQA accuracy, audit every record, and report all fixed contrasts."""
import csv,json,collections,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import P,ROOT,OUT,NAMES,PAIRS,TASKS,read,sha256,digest,csvout,atomic_json,verify,stamp,gqa_prediction

def base_for(c):return NAMES[2] if c in [NAMES[2],NAMES[3],NAMES[4]] else NAMES[1] if c in [NAMES[1],NAMES[5],NAMES[6]] else NAMES[0]
def bootstrap(ids,cases,records):
    images=sorted({cases[q]['sample']['image_sha256'] for q in ids});members=[[i for i,q in enumerate(ids) if cases[q]['sample']['image_sha256']==im] for im in images]
    mat=np.array([[records[q,c]['generation']['correct'] for c in NAMES] for q in ids]);sums=np.array([mat[m].sum(0) for m in members]);sizes=np.array([len(m) for m in members])
    ix=np.random.default_rng(20261104).integers(0,len(images),size=(20000,len(images)));boots=np.empty((20000,7))
    for a in range(0,20000,500):
        j=ix[a:a+500];boots[a:a+500]=100*sums[j].sum(1)/sizes[j].sum(1)[:,None]
    return mat,boots,len(images)

def flags(g):
    ok=lambda c:bool(g[c]['correct'])
    return dict(S_correct_SB_wrong=ok(NAMES[1]) and not ok(NAMES[6]),Q_correct_QB_wrong=ok(NAMES[2]) and not ok(NAMES[3]),Q_correct_QPost_wrong=ok(NAMES[2]) and not ok(NAMES[4]),STail_correct_SB_wrong=ok(NAMES[5]) and not ok(NAMES[6]),SB_correct_STail_wrong=ok(NAMES[6]) and not ok(NAMES[5]),S_correct_STail_wrong=ok(NAMES[1]) and not ok(NAMES[5]),Q_equal_budget_answer_change=g[NAMES[3]]['text']!=g[NAMES[4]]['text'],S_equal_budget_answer_change=g[NAMES[5]]['text']!=g[NAMES[6]]['text'])

def main():
    p=verify();plan=read(P/'sample_plan.json');cases=plan['cases'];done=read(P/'gpu_completion.json');workers=[read(P/f'worker{i}_completion.json') for i in range(2)];assert done['passed'] and all(w['passed'] for w in workers)
    assert sum(w['new_generations'] for w in workers)==1440 and sum(w['reused_generations'] for w in workers)==1080 and sum(w['labels'] for w in workers)==2520
    for n,info in p['model_files'].items():assert sha256(n)==info['sha256'],n
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h,n
    records={};long=[];wide=[];index=[];anomalies=[];audits=0;smokeindex=[]
    for rank in range(2):
        W=P/f'worker{rank}';assert read(W/'smoke_validation.json')['passed'] and read(W/'structural_smoke_validation.json')['passed']
        refs=[read(f)['raw'] for f in sorted((W/'smoke_cases').glob('*.json'))]
        ss=read(W/'structural_smoke_validation.json');refs+=[r['raw'] for r in ss['Q_historical_replay']]+[r['raw'] for r in ss['four_frozen_new_writer_conditions']]
        assert len(refs)==10
        for r in refs:
            assert sha256(r['path'])==r['sha256'];smokeindex.append(dict(worker=rank,raw_path=r['path'],raw_sha256=r['sha256'],raw_bytes=r['bytes']))
    for q,v in cases.items():
        d=v['design'];s=v['sample'];gs={};visual=set()
        for c in NAMES:
            path=P/f'worker{v["worker"]}'/'records'/q/(c+'.json');assert path.exists(),'NOT_GENERATED: '+str(path)
            r=read(path);g=r['generation'];w=v['writers'][c]
            assert w==r['writers']==g['writer_positions'] and g['correct']==int(gqa_prediction(g['text'])==s['answer']) and g['prediction']==gqa_prediction(g['text'])
            assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
            if r['reused']:
                assert c in NAMES[:3] and sha256(ROOT/r['source']['path'])==r['source']['sha256'];old=read(ROOT/r['source']['path']);assert old['generation']==g and old['input_hashes']==d['input_hashes'] and old['image_sha256']==s['image_sha256'] and old['writers']==w
                assert sha256(r['historical_raw']['path'])==r['historical_raw']['sha256']
                with np.load(r['historical_raw']['path']) as z:assert int(z['first_logits'].argmax())==g['token_ids'][0]
            else:
                assert r['worker']==v['worker'] and r['plan_sha256']==p['plan_sha256'] and r['input_hashes']==d['input_hashes'] and r['MLP_calls']==[g['length']]*32 and r['weights_unchanged'] and r['hook_cleanup']
                assert sha256(r['raw']['path'])==r['raw']['sha256']
                with np.load(r['raw']['path']) as z:
                    a=z['audit'];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
                    assert np.array_equal(z['first_logits'],z['valid_logits'][0]) and z['valid_logits'].shape[0]==g['length'] and z['valid_logits'].argmax(-1).tolist()==g['token_ids']
                    m=json.loads(str(z['record_metadata']));assert m=={k:r[k] for k in m}
                audits+=1;visual.add(r['projected_visual_sha256']);cp=P/f'worker{v["worker"]}'/'checkpoints'/q/(digest(w)+'.json')
                index.append(dict(question_id=q,condition=c,worker=v['worker'],checkpoint=str(cp.relative_to(P)),checkpoint_sha256=sha256(cp),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256'],raw_bytes=r['raw']['bytes']))
            records[q,c]=r;gs[c]=g
            if not g['eos'] or g['empty'] or g['truncated'] or g['repeated_trigram']:anomalies.append(dict(question_id=q,condition=c,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeat=g['repeated_trigram']))
        assert len(visual)==1
        for c in NAMES:
            g=gs[c];base=base_for(c);b=gs[base];w=v['writers'][c]
            long.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,status='REUSED_AUDITED' if records[q,c]['reused'] else 'GENERATED_VERIFIED',historical_label=records[q,c]['source']['historical_label'] if records[q,c]['reused'] else '',answer=g['text'],prediction=g['prediction'],correct=g['correct'],token_ids=json.dumps(g['token_ids']),corresponding_baseline=base,baseline_answer=b['text'],baseline_correct=b['correct'],harm_vs_base=int(b['correct'] and not g['correct']),fix_vs_base=int(not b['correct'] and g['correct']),harm_vs_All=int(gs[NAMES[0]]['correct'] and not g['correct']),fix_vs_All=int(not gs[NAMES[0]]['correct'] and g['correct']),answer_change_vs_base=int(g['text']!=b['text']),first_token_flip_vs_base=int(g['token_ids'][0]!=b['token_ids'][0]),writers=json.dumps(w),added_positions=json.dumps(sorted(set(w)-set(v['writers'][base]))),k=len(w),R=len(d['R']),writer_ratio=len(w)/len(d['R']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],worker=v['worker']))
        wide.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],answers=json.dumps({c:gs[c]['text'] for c in NAMES}),correct_per_condition=json.dumps({c:gs[c]['correct'] for c in NAMES}),token_ids=json.dumps({c:gs[c]['token_ids'] for c in NAMES}),writer_positions=json.dumps(v['writers']),structural_positions=json.dumps(d['structural_positions']),writer_budget=json.dumps({c:dict(k=len(w),R=len(d['R']),ratio=len(w)/len(d['R'])) for c,w in v['writers'].items()}),first_token_flip=json.dumps({c:int(gs[c]['token_ids'][0]!=gs[base_for(c)]['token_ids'][0]) for c in NAMES}),selection_sha256=v['selection_sha256'],preference_tier=v['preference_tier'],worker=v['worker'],**flags(gs)))
    assert audits==1440 and len(long)==2520 and len(wide)==360
    csvout(P/'per_question.csv',wide);csvout(P/'per_condition.csv',long);csvout(P/'checkpoint_index.csv',index);csvout(P/'smoke_raw_index.csv',smokeindex)
    fields=list(wide[0]);lf=list(long[0]);csvout(P/'harm_cases.csv',[r for r in long if r['condition'] in NAMES[3:] and r['harm_vs_base']],lf);csvout(P/'recovered_cases.csv',[r for r in long if r['condition'] in NAMES[3:] and r['fix_vs_base']],lf)
    cats=list(flags({c:records[next(iter(cases)),c]['generation'] for c in NAMES}));nonmono=cats[:6]
    csvout(P/'nonmonotonic_cases.csv',[r for r in wide if any(r[c] for c in nonmono)],fields)
    csvout(P/'all_new_condition_failures.csv',[r for r in long if r['condition'] in NAMES[3:] and not r['correct']],lf)
    csvout(P/'required_case_categories.csv',[dict(category=c,**r) for c in cats for r in wide if r[c]],['category']+fields)
    summary=[];tasks=[];pairs=[]
    for group,ids in [('all360',list(cases))]+[(t,[q for q,v in cases.items() if v['sample']['task']==t]) for t in TASKS]:
        mat,boots,nim=bootstrap(ids,cases,records)
        for j,c in enumerate(NAMES):
            rr=[r for r in long if r['condition']==c and r['question_id'] in ids];base=base_for(c);k=NAMES.index(base);ci=np.percentile(boots[:,j],[2.5,97.5]);delta=boots[:,j]-boots[:,k];dc=np.percentile(delta,[2.5,97.5])
            row=dict(group=group,condition=c,n=len(ids),image_clusters=nim,correct=int(mat[:,j].sum()),accuracy_percent=float(mat[:,j].mean()*100),CI95_low=ci[0],CI95_high=ci[1],corresponding_baseline=base,delta_vs_base_pp=float((mat[:,j]-mat[:,k]).mean()*100),delta_vs_base_CI95_low=dc[0],delta_vs_base_CI95_high=dc[1],delta_vs_All_pp=float((mat[:,j]-mat[:,0]).mean()*100),delta_vs_S_pp=float((mat[:,j]-mat[:,1]).mean()*100),delta_vs_Q_pp=float((mat[:,j]-mat[:,2]).mean()*100),harm_vs_base=sum(r['harm_vs_base'] for r in rr),fix_vs_base=sum(r['fix_vs_base'] for r in rr),harm_vs_All=sum(r['harm_vs_All'] for r in rr),fix_vs_All=sum(r['fix_vs_All'] for r in rr),answers_changed_vs_base=sum(r['answer_change_vs_base'] for r in rr),first_token_flips_vs_base=sum(r['first_token_flip_vs_base'] for r in rr),mean_k=float(np.mean([r['k'] for r in rr])),mean_R_ratio=float(np.mean([r['writer_ratio'] for r in rr])),eos_count=sum(r['eos'] for r in rr),empty_count=sum(r['empty'] for r in rr),truncated_count=sum(r['truncated'] for r in rr),repeated_trigram_count=sum(r['repeated_trigram'] for r in rr),reused_count=sum(r['status']=='REUSED_AUDITED' for r in rr))
            (summary if group=='all360' else tasks).append(row)
        for left,right in PAIRS:
            j=NAMES.index(left);k=NAMES.index(right);delta=boots[:,j]-boots[:,k];ci=np.percentile(delta,[2.5,97.5]);adj=np.percentile(delta,[.5,99.5]);primary=group=='all360'
            pairs.append(dict(group=group,left=left,right=right,n=len(ids),image_clusters=nim,delta_pp=float((mat[:,j]-mat[:,k]).mean()*100),CI95_low=ci[0],CI95_high=ci[1],prespecified_five_family=primary,primary_exploratory_comparison=primary and [left,right]==PAIRS[0],Bonferroni99_low=adj[0] if primary else '',Bonferroni99_high=adj[1] if primary else '',right_correct_left_wrong=int(((mat[:,k]==1)&(mat[:,j]==0)).sum()),right_wrong_left_correct=int(((mat[:,k]==0)&(mat[:,j]==1)).sum()),answers_changed=sum(records[q,left]['generation']['text']!=records[q,right]['generation']['text'] for q in ids),first_token_flips=sum(records[q,left]['generation']['token_ids'][0]!=records[q,right]['generation']['token_ids'][0] for q in ids)))
    csvout(P/'accuracy_summary.csv',summary);csvout(P/'task_accuracy.csv',tasks);csvout(P/'paired_comparisons.csv',pairs)
    # Separate Pareto point estimates from uncertainty; never interpolate strategies.
    pareto=[r['condition'] for r in summary if not any(o['mean_R_ratio']<=r['mean_R_ratio']+1e-14 and o['accuracy_percent']>=r['accuracy_percent'] and (o['mean_R_ratio']<r['mean_R_ratio']-1e-14 or o['accuracy_percent']>r['accuracy_percent']) for o in summary)]
    reps=[]
    for c in cats:
        pool=sorted([r for r in wide if r[c]],key=lambda r:(r['selection_sha256'],r['question_id']))
        if pool:reps.append(dict(category=c,**pool[0]))
    csvout(P/'representative_cases.csv',reps,['category']+fields);plot(summary,tasks)
    release=read(P/'gpu_release.json');assert release['all_own_children_exited']
    validation=dict(passed=True,questions=360,unique_images=360,task_counts={t:60 for t in TASKS},logical_records=2520,new_raw_records_verified=1440,history_reused=1080,all_outputs_present=True,missing_outputs=[],all_generated_argmax_ids_rechecked=True,all32_MLP=True,all_forbidden_actual_probabilities_zero=True,all_non_target_masks_exact=True,all_weights_inputs_hooks_exact=True,projected_features_equal_within_question=True,raw_metadata_equals_checkpoint=True,model_dependency_scorer_SHA256_exact=True,Question_Only_historical_actual_writer_sets_equals_Q_all3000=True,all_native_token_ids_offsets_question_bounds_exact=True,all_two_token_budgets_exact=True,plan_sha256=p['plan_sha256'],manifest_sha256=p['manifest_sha256'],analysis_source_sha256=sha256(__file__),shard_question_counts=p['worker_questions'],shards_no_overlap=True,all_prior_boundary_tested_images_excluded=True,all_selected_images_outside_Semantic240_and_Pilot120=True,anomalies=anomalies,smokes={str(i):dict(base=read(P/f'worker{i}/smoke_validation.json'),structural=read(P/f'worker{i}/structural_smoke_validation.json')) for i in range(2)},smoke_raw_archives_verified=len(smokeindex),gpu_release=release,bootstrap_repetitions=20000,bootstrap_seed=20261104,case_category_counts={c:sum(bool(r[c]) for r in wide) for c in cats},point_estimate_pareto_conditions=pareto,utc=stamp())
    atomic_json(P/'validation.json',validation)
    atomic_json(P/'analysis_bundle.json',dict(summary=summary,tasks=tasks,pairs=pairs,representatives=reps,pareto=pareto))
    report(p,plan,done,workers,summary,pairs,reps,validation)
    atomic_json(P/'completion.json',dict(status='GQA360_STRUCTURAL_RECEIVER_COMPLETE',questions=360,new_generations=1440,history_reuse=1080,condition_labels=2520,validation_passed=True,seconds=done['seconds'],frozen_execution_commits=sorted({w['code_commit'] for w in workers}),full3000_started=False,NativeFull_generated=False,main_modified=False,utc=stamp()))
    print(json.dumps(dict(summary=summary,primary_pairs=[r for r in pairs if r['prespecified_five_family']],pareto=pareto,categories=validation['case_category_counts']),ensure_ascii=False),flush=True)

def plot(summary,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'});short=['All','S','Q','Q+B','Q+Post2','S+Tail2','S+B']
    fig,ax=plt.subplots(figsize=(10.5,5.8));palette=['#6b7280','#4584b1','#5c7f96','#e39c43','#945c9d','#389484','#c16748']
    for i,r in enumerate(summary):
        x=100*r['mean_R_ratio'];y=r['accuracy_percent'];ax.errorbar(x,y,yerr=[[y-r['CI95_low']],[r['CI95_high']-y]],fmt='o',color=palette[i],capsize=3,label=f'{i+1}. {r["condition"]} ({r["correct"]}/360)');ax.annotate(str(i+1),(x,y),xytext=(8,9 if i%2==0 else -16),textcoords='offset points',fontsize=10)
    ax.set_xlabel('Mean accessible R receivers (%)');ax.set_ylabel('Complete-answer GQA accuracy (%)');ax.set_title('Structural visual receivers: fixed natural GQA360');ax.set_xlim(0,105);ax.grid(alpha=.18);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9,frameon=False)
    fig.text(.5,.015,'Image-cluster bootstrap 95% intervals. Discrete strategies; no interpolation or population claim.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(P/'accuracy_writer_budget.png',dpi=220);fig.savefig(P/'accuracy_writer_budget.svg');plt.close(fig)
    data=np.array([[next(r['delta_vs_base_pp'] for r in tasks if r['group']==t and r['condition']==c) for c in NAMES] for t in TASKS]);vmax=max(5,float(np.abs(data).max()));fig,ax=plt.subplots(figsize=(11,5.5));im=ax.imshow(data,aspect='auto',cmap='RdBu',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(7),short);ax.set_yticks(range(6),TASKS)
    for i in range(6):
        for j in range(7):ax.text(j,i,f'{data[i,j]:+.1f}',ha='center',va='center',color='white' if abs(data[i,j])>.65*vmax else '#222')
    ax.set_title('Task accuracy delta vs corresponding baseline (60 questions per task)');fig.colorbar(im,ax=ax,label='Accuracy difference (pp)')
    fig.text(.5,.01,'Q / Q+B / Q+Post2 relative to Q; S / S+Tail2 / S+B relative to S; All relative to itself.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(P/'task_accuracy_delta.png',dpi=220);fig.savefig(P/'task_accuracy_delta.svg');plt.close(fig)

def report(p,plan,done,workers,summary,pairs,reps,val):
    lookup={r['condition']:r for r in summary};pr=[r for r in pairs if r['prespecified_five_family']];A,S,Q,QB,QP,ST,SB=[lookup[c] for c in NAMES]
    def verdict(pair):
        return '调整99%区间支持左侧较高' if pair['Bonferroni99_low']>0 else '调整99%区间支持右侧较高' if pair['Bonferroni99_high']<0 else '调整99%区间跨0，优劣仍不确定'
    text=['# Structural Receiver Selection：Question、Boundary与Question-tail（GQA360）','','**状态：GQA360_STRUCTURAL_RECEIVER_COMPLETE。** 本轮仅完成冻结360题、四个新增策略，没有启动3000题矩阵、增加条件或修改Token规则。','',
        '## 研究设置与选样','',
        '从原GQA3000取六任务各60题，共360题、360张不同图片。选样只使用任务、图像SHA、Token可行性与历史图片排除规则；不读取S/All正确性、NLL、Attention Mass或恢复结果作为选择输入。先硬排除Quick24/Diagnostic4/Failure Probe全部图片，优先排除Semantic240与Noun+Article Pilot120图片，再按SHA256("structural-receiver-gqa360-v1|task|question_id|image_sha256")在固定任务顺序中升序选择并全局去重。360张图片全部在优先未测池内，无回填；这是内部探索样本，不是独立外部验证。全部3000候选及排除/未入选理由见candidate_audit.csv。','',
        'CPU阶段逐题核验原Prompt、Tokenizer IDs、冻结Offset与Question范围：3000题均满足定义，Token不可行排除数为0。Question-final-token严格为Q最后一个真实Token；PostQuestion-NL为紧邻其后的原生ID13；Instruction-Answer是再后一个独立单Token。S+Tail新增final+post，S+B新增位置581/582；Q+Post新增post+Answer，Q+B新增581/582。四策略均严格多2个Token，没有臆造、替换或插入Prompt。逐题角色、ID、piece、decode、offset、原文与S/Q/R成员关系见token_audit.csv。','',
        f'基线commit `{p["base_commit"]}`；冻结执行commit `{workers[0]["code_commit"]}`。sample_manifest.csv SHA256=`{p["manifest_sha256"]}`；sample_plan.json SHA256=`{p["plan_sha256"]}`。原模型revision `{p["model_revision"]}`，模型文件在执行前及统计时两次完整SHA256核验，见model_preflight_sha256.json与validation.json。','',
        '## 严格区分各条件','',
        'Native Full开放原生全部连接；本轮未执行Native Full，不能将历史3000题87.567%作为本360题数值。All-Writer为全部Prefill R读图、所有32层P−1/真实Decode禁读原始Image Key。Question-Only只开放完整Q，Semantic-k只开放冻结自然S。历史Question-Only实际来自Semantic-first-Q100，已对3000题逐题确认Writer位置等于完整Q（包括末尾标点），并对所选360题审计原Prompt/input/image/Writer/Token/答案/归一化评分/独立KV协议和提交字节，不能仅凭名称复用。','',
        '新增四条件只有R→Image Key的Softmax前Hard Mask不同；全部32层P−1/Decode始终禁读图，Image Query、QKV、MLP、Residual和完整KV计算保持原生，每条件独立真实KV及前缀。LLaVA-1.5-7B，FP16/Eager/Transformers4.49.0，Greedy32，logits_to_keep=1，沿用原GQA完整答案评分，不计算Gold NLL。不涉及物理视觉Token删除或实际加速。Q+Post的两个新增位置中一个是词法Instruction-Answer，因此其效应不能称为两个纯边界的效应。','',
        '## 完整答案Accuracy与正负翻转','','| 条件 | 来源 | 正确/360 | ACC | 相对All pp | 对应基线 | 对应差值pp | 原对→错 / 原错→对 | 平均k | 平均k/R |','|---|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in summary:text.append(f'| {r["condition"]} | {"历史审计复用" if r["condition"] in NAMES[:3] else "新增生成"} | {r["correct"]}/360 | {r["accuracy_percent"]:.3f}% | {r["delta_vs_All_pp"]:+.3f} | {r["corresponding_baseline"]} | {r["delta_vs_base_pp"]:+.3f} | {r["harm_vs_base"]}/{r["fix_vs_base"]} | {r["mean_k"]:.3f} | {100*r["mean_R_ratio"]:.2f}% |')
    text+=['','全部答案改变数、首Token翻转、相对All/S/Q的差值与EOS/截断/空答案/重复三元组计数见accuracy_summary.csv及per_condition.csv。Writer比例先逐题k/R再求均值，不将Question100%误称为R100%。','',
        '## 五个预先固定的配对比较','',
        'Image SHA整簇配对Bootstrap20000次、seed=20261104，Question等权；本轮每图只有1题，但仍按图片构建簇。第一项为主要探索性比较，其余为预设辅助比较；同时提供普通95%及五比较Bonferroni99%区间。不以跨0证明无损或非劣效。单位pp。','','| 左−右 | 差值 | 95%CI | 调整99%CI | 右对→左错 / 右错→左对 |','|---|---:|---|---|---:|']
    for r in pr:text.append(f'| {r["left"]} − {r["right"]} | {r["delta_pp"]:+.3f} | [{r["CI95_low"]:+.3f}, {r["CI95_high"]:+.3f}] | [{r["Bonferroni99_low"]:+.3f}, {r["Bonferroni99_high"]:+.3f}] | {r["right_correct_left_wrong"]}/{r["right_wrong_left_correct"]} |')
    text+=['','## 七项研究判断','',
        f'1. Q+B为{QB["correct"]}/360，Q为{Q["correct"]}/360，净{QB["correct"]-Q["correct"]:+d}题（{pr[0]["delta_pp"]:+.3f}pp）；{verdict(pr[0])}。这里比较同批360题的审计历史Q，而非直接对比3000题历史百分比。',
        f'2. 相同新增2个Token，Q+Post为{QP["correct"]}/360，Q+B为{QB["correct"]}/360；Q+B−Q+Post={pr[2]["delta_pp"]:+.3f}pp，{verdict(pr[2])}。',
        f'3. S+Tail为{ST["correct"]}/360，S+B为{SB["correct"]}/360；S+Tail−S+B={pr[3]["delta_pp"]:+.3f}pp，{verdict(pr[3])}。',
        '4. 新策略对各自基线的纠正/损害分别为：'+ '；'.join(f'{r["condition"]}纠正{r["fix_vs_base"]}、损害{r["harm_vs_base"]}' for r in summary[3:])+ '。不能只报告纠正数。',
        '5. 本批点估计Accuracy–平均R预算的非支配前沿为：'+ '、'.join(val['point_estimate_pareto_conditions'])+'。这是离散策略之间的样本前沿，不是经过独立验证的最优策略；最佳取舍取决于质量目标与预算约束，不能仅凭最高ACC或最低预算确定。',
        f'6. Question-conditioned位置的证据分别来自Q+Post对Q+B、S+Tail对S+B两个等预算比较。Q+Post相对前置边界净{QP["correct"]-QB["correct"]:+d}题，S+Tail相对前置边界净{ST["correct"]-SB["correct"]:+d}题；结合上表区间判断，不能仅从因果时序或答案变化宣布后置Receiver必然更有用，也不能把Answer词法Token的作用归于纯边界。',
        '7. 可将样本前沿上的固定策略作为后续更大规模图像不重叠验证的候选；本轮360题只用于探索与候选选择，其置信区间和多任务结果不用于事后搜索Token位置。若候选的净收益不明确，后续应验证效应而非预设提升。本轮未执行扩样或新条件。','',
        '## 六任务与非单调案例','',
        '六任务全部条件的ACC、对应基线差值和翻转见task_accuracy.csv；没有根据任务表现改策略。必需五类案例及额外S→Tail损害、等预算输出变化全部保存在required_case_categories.csv。类别允许重叠：'+json.dumps(val['case_category_counts'],ensure_ascii=False)+'。harm_cases/recovered_cases保留各新策略相对对应S/Q的完整损害与纠正，nonmonotonic_cases保留全部指定方向的不单调评分案例。per_question.csv保留每题所有答案、Token IDs、Writer与预算。代表题按冻结SHA最小选择，不只展示成功。']
    for r in reps[:6]:
        text+=['',f'### {r["category"]}：{r["question_id"]} / {r["task"]}','',r['question'],'',f'Gold：{r["gold"]}。','','| 条件 | 完整回答 | 正确 | 新增位置 |','|---|---|---:|---|'];ans=json.loads(r['answers']);correct=json.loads(r['correct_per_condition']);writers=json.loads(r['writer_positions'])
        for c in NAMES:text.append(f'| {c} | {ans[c].replace("|","/")} | {correct[c]} | {sorted(set(writers[c])-set(writers[base_for(c)]))} |')
        text+=['', '完整Writer集合：`'+r['writer_positions']+'`。']
    text+=['','## 图表','','![Accuracy and actual R budget](accuracy_writer_budget.png)','','![Six task differences](task_accuracy_delta.png)','',
        '图1各点对应离散策略，无平滑或插值连线，误差条为点位95%CI；图2按各自S/Q基线显示六任务pp差，不能把两个不同基线的颜色直接解释成策略间差。另提供可编辑矢量SVG。','',
        '## 正确性、执行与资源','',
        f'两卡UUID：`{p["gpus"][0]["uuid"]}`、`{p["gpus"][1]["uuid"]}`，按Image SHA排序交替分片，各180题、720条新生成；每Worker独立模型、KV、样本文件、Checkpoint目录和文件锁，原子JSON/NPZ。启动前均重新满足无Compute进程、util=0、used<200MiB/free>22000MiB。两卡各自真实7B Smoke全部完成后CPU GO关卡才放行主采集。','',
        '每卡三道固定Smoke严格复现Native/No-op全部有效Logits，以及历史All/S/Q答案Token和首Token完整Logits向量torch.equal；独立KV、32MLP、原输入/权重、Hook正常及异常清理通过。另在每卡对四种冻结新增结构Writer进行实际Mask Smoke。1440条主生成全部逐层复核禁止概率0、非目标Mask不变、MLP32层执行、权重版本/输入Hash不变、同题投影视觉特征一致、Writer精确匹配、有效Logits逐步argmax与保存Token一致。Attention FP16行和仍使用历史0.002审计界限，没有改精度或放宽断言。','',
        f'历史复用1080条，新增1440条，共2520个条件标签，无NOT_GENERATED。异常标签数{len(val["anomalies"])}（详见validation.json）。双GPU采集墙钟{done["seconds"]:.1f}秒/{done["seconds"]/60:.2f}分钟，含加载、各自Smoke与正式生成，不含CPU冻结、统计和提交；不将该时间解释为推理优化加速。','',
        'GPU释放审计：'+json.dumps({k:val['gpu_release'][k] for k in ['all_own_children_exited','selected_devices_no_compute_processes','selected_uuids']},ensure_ascii=False)+'。只停止/退出本轮自有Worker，不影响其他实验；显存和进程以gpu_release.json的当时快照为准。','',
        'Git保存完整JSON检查点、Worker日志及checkpoint_index/smoke_raw_index；原始FP16首Token/有效Logits和逐层数值数组位于服务器资源目录，索引给出绝对路径、字节数与SHA256，不声称GitHub包含NPZ。历史复用的输入、Token组与生成JSON提交字节审计见history_byte_audit.json。执行命令：本包run freeze，提交冻结源码/协议，run driver；完成后CPU运行analyze。历史输出、main保持不变。本轮停止，不追加3000题、规则搜索、物理删除、Router、Kernel或跨模型实验。','']
    (P/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
