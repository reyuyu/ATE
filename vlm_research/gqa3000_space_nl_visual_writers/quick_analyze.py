"""CPU recomputation, paired bootstrap, case answers and quick-study figures."""
import collections,json,csv,subprocess,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .common import *
Q=OUT/'quick24'
def dump(n,x):(Q/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvout(n,rs):
    rs=list(rs)
    with (Q/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def main():
    p=read(Q/'protocol.json');plan=read(Q/'sample_plan.json');done=read(Q/'gpu_completion.json');assert done['passed'] and not (Q/'STOP.json').exists()
    from .quick import source_hashes
    assert p['sources']==source_hashes() and p['plan_sha256']==sha256(Q/'sample_plan.json')
    model_hashes={n:sha256(Path(n)) for n in p['model_files']};assert all(model_hashes[n]==v['sha256'] for n,v in p['model_files'].items())
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(Path(n))==h
    cases=plan['cases'];records={};rows=[];audits=0;anomalies=[];negative=[]
    for q,v in cases.items():
        d=v['design'];s=v['sample'];visual=set()
        for label in CONDITIONS:
            r=read(Q/'records'/q/(label+'.json'));g=r['generation'];w=d['writers'][label];assert r['writers']==w==g['writer_positions'] and g['correct']==int(g['prediction']==s['answer'])
            assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['audit']['passed']
            if r['reused']:assert sha256(ROOT/r['source']['path'])==r['source']['sha256']
            else:
                assert sha256(Path(r['raw']['path']))==r['raw']['sha256'];assert r['input_hashes']==d['input_hashes'] and r['MLP_calls']==[g['length']]*32 and r['weights_unchanged'] and r['hook_cleanup'];visual.add(r['projected_visual_sha256'])
                with np.load(r['raw']['path']) as z:
                    a=z['audit'];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
                    assert np.array_equal(z['first_logits'],z['valid_logits'][0]) and z['valid_logits'].shape[0]==g['length']
                    assert z['valid_logits'].argmax(-1).tolist()==g['token_ids'];audits+=1
            records[(q,label)]=r
            if not g['eos'] or g['empty'] or g['truncated'] or g['repeated_trigram']:anomalies.append(dict(question_id=q,condition=label,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeat=g['repeated_trigram']))
        assert len(visual)==1
        for label in CONDITIONS:
            g=records[(q,label)]['generation'];a=records[(q,'All-Writer')]['generation'];b=records[(q,'Semantic-k')]['generation'];w=d['writers'][label]
            row=dict(question_id=q,task=s['task'],cohort=v['cohort'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=label,answer=g['text'],prediction=g['prediction'],correct=g['correct'],token_ids=json.dumps(g['token_ids']),All_answer=a['text'],S_answer=b['text'],All_correct=a['correct'],S_correct=b['correct'],harm_vs_All=int(a['correct'] and not g['correct']),fix_vs_All=int(not a['correct'] and g['correct']),harm_vs_S=int(b['correct'] and not g['correct']),fix_vs_S=int(not b['correct'] and g['correct']),text_change_vs_S=int(g['text']!=b['text']),first_token_flip_vs_S=int(g['token_ids'][0]!=b['token_ids'][0]),k=len(w),R=len(d['R']),writer_ratio=len(w)/len(d['R']),writers=json.dumps(w),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],reused=records[(q,label)]['reused'])
            rows.append(row)
            if row['harm_vs_All'] or row['harm_vs_S']:negative.append(row)
    assert len(rows)==252 and audits==168
    csvout('per_question.csv',rows);csvout('diagnostic_cases.csv',[r for r in rows if r['cohort']=='diagnostic4'])
    if negative:csvout('negative_cases.csv',negative)
    summaries=[];pairs=[];tasks=[]
    for cohort,ids in [('quick24',plan['main_ids']),('diagnostic4',plan['diagnostic_ids'])]:
        images=sorted({cases[q]['sample']['image_sha256'] for q in ids});members=[[i for i,q in enumerate(ids) if cases[q]['sample']['image_sha256']==im] for im in images];rng=np.random.default_rng(20261104);ix=rng.integers(0,len(images),size=(20000,len(images)));counts=np.array([len(m) for m in members]);denom=counts[ix].sum(1)
        mat=np.array([[records[q,c]['generation']['correct'] for c in CONDITIONS] for q in ids]);ag=np.array([mat[m].sum(0) for m in members]);boots=ag[ix].sum(1)/denom[:,None]*100
        for j,label in enumerate(CONDITIONS):
            rr=[r for r in rows if r['cohort']==cohort and r['condition']==label];ci=np.percentile(boots[:,j],[2.5,97.5]);summaries.append(dict(cohort=cohort,condition=label,n=len(ids),correct=int(mat[:,j].sum()),accuracy_percent=float(mat[:,j].mean()*100),accuracy_CI_low=ci[0],accuracy_CI_high=ci[1],delta_vs_All_pp=float((mat[:,j]-mat[:,0]).mean()*100),harm_vs_All=sum(r['harm_vs_All'] for r in rr),fix_vs_All=sum(r['fix_vs_All'] for r in rr),harm_vs_S=sum(r['harm_vs_S'] for r in rr),fix_vs_S=sum(r['fix_vs_S'] for r in rr),answers_changed_vs_S=sum(r['text_change_vs_S'] for r in rr),first_token_flips_vs_S=sum(r['first_token_flip_vs_S'] for r in rr),mean_k=float(np.mean([r['k'] for r in rr])),mean_R_ratio=float(np.mean([r['writer_ratio'] for r in rr]))))
        comparisons=COMPARISONS+[[c,'All-Writer'] for c in CONDITIONS if c!='All-Writer']
        for left,right in comparisons:
            j=CONDITIONS.index(left);k=CONDITIONS.index(right);diff=boots[:,j]-boots[:,k];ci=np.percentile(diff,[2.5,97.5]);corr=np.percentile(diff,[.5,99.5]);pairs.append(dict(cohort=cohort,left=left,right=right,n=len(ids),delta_pp=float((mat[:,j]-mat[:,k]).mean()*100),CI95_low=ci[0],CI95_high=ci[1],primary_five_family=[left,right] in COMPARISONS,Bonferroni99_low=corr[0] if [left,right] in COMPARISONS else '',Bonferroni99_high=corr[1] if [left,right] in COMPARISONS else '',right_correct_left_wrong=int(((mat[:,k]==1)&(mat[:,j]==0)).sum()),right_wrong_left_correct=int(((mat[:,k]==0)&(mat[:,j]==1)).sum())))
        for task in TASKS:
            ii=[i for i,q in enumerate(ids) if cases[q]['sample']['task']==task]
            if not ii:continue
            for j,label in enumerate(CONDITIONS):tasks.append(dict(cohort=cohort,task=task,condition=label,n=len(ii),correct=int(mat[ii,j].sum()),accuracy_percent=float(mat[ii,j].mean()*100),delta_vs_All_pp=float((mat[ii,j]-mat[ii,0]).mean()*100)))
    csvout('accuracy_summary.csv',summaries);csvout('paired_comparisons.csv',pairs);csvout('task_accuracy.csv',tasks)
    plot(summaries,tasks)
    mainrows=[r for r in summaries if r['cohort']=='quick24'];s0=next(r for r in mainrows if r['condition']=='Semantic-k');sb=next(r for r in mainrows if r['condition']=='S+SPACE+NL');nd=next(r for r in mainrows if r['condition']=='Noun+Article');nb=next(r for r in mainrows if r['condition']=='Noun+Article+SPACE+NL')
    text=['# SPACE / NL快速验证：24题均衡样本与4题诊断','','**状态：QUICK24_COMPLETE；3000题正式实验未完成。** 本次按用户授权先做小规模快速验证，原3000题名单与接收集合未改动。','',
    '24题来自原pilot120，按固定SHA规则每任务4题，图像互不重复；4道用户指定历史失败题单列，不并入24题ACC。选择未使用本次输出，但原120题已有研究历史，因此并非独立确认样本。24题ACC每题相差4.167pp，不能外推3000题。','',
    '单卡GPU1，FP16/Eager/Transformers4.49.0，Greedy32与logits_to_keep=1。所有32层P−1和Decode禁读Image Key，只有原生Prefill R中选中位置可直接读图。SPACE=29871/位置581、NL=13/位置582；模板等预算对照是原Instruction中的Answer和the。原Prompt未改。','',
    f'共252个条件标签：审计复用84条All/S/N+D历史生成，新增168条；成功采集墙钟{done["seconds"]:.1f}秒（包含模型加载与三道Smoke，不含CPU准备/统计）。首次环境记录字段错误发生在任何forward之前，失败现场保存在父目录attempts/pre_collection_environment_failure；修复并重新冻结后才启动本次采集。','',
    '## 24题完整答案ACC','','| 条件 | 正确/24 | ACC | 相对All pp | S对→错 / 错→对 | 平均R预算 |','|---|---:|---:|---:|---:|---:|']
    for r in mainrows:text.append(f'| {r["condition"]} | {r["correct"]}/24 | {r["accuracy_percent"]:.3f}% | {r["delta_vs_All_pp"]:+.3f} | {r["harm_vs_S"]}/{r["fix_vs_S"]} | {r["mean_R_ratio"]*100:.2f}% |')
    text+=['','## 五组预设配对比较','','Image SHA整簇配对Bootstrap20000次，seed20261104；下列为普通探索性95%CI。另列五项家族Bonferroni99%区间，不以跨0作为非劣效证明。小样本为方向与正确性检查，不做确认性结论。','','| 左−右 | pp | 95%CI | 调整99%CI | 右对→左错 / 右错→左对 |','|---|---:|---|---|---:|']
    for r in pairs:
        if r['cohort']=='quick24' and r['primary_five_family']:text.append(f'| {r["left"]} − {r["right"]} | {r["delta_pp"]:+.3f} | [{r["CI95_low"]:+.3f}, {r["CI95_high"]:+.3f}] | [{r["Bonferroni99_low"]:+.3f}, {r["Bonferroni99_high"]:+.3f}] | {r["right_correct_left_wrong"]}/{r["right_wrong_left_correct"]} |')
    text+=['','## 四道历史诊断题（不并入24题）','','| ID / Gold | All | S | S+SPACE | S+NL | S+B | N+D | N+D+B | S+模板 | N+D+模板 |','|---|---|---|---|---|---|---|---|---|---|']
    for q in plan['diagnostic_ids']:
        s=cases[q]['sample'];text.append('| '+q+' / '+s['answer']+' | '+' | '.join(records[q,c]['generation']['text'] for c in CONDITIONS)+' |')
    text+=['','## 直接结论','',
    '本批24题没有观察到SPACE/NL带来的净ACC提升：S为21/24，而S+SPACE、S+NL和S+B均为20/24；S+B比S少1题（−4.167pp），同时低于等预算S+模板的21/24。N+D为21/24，N+D+B与N+D+模板均为20/24；后两者净ACC相同，但逐题各有1次相互损害/纠正，因此不是相同输出。置信区间和样本规模都不足以断言总体优劣。','',
    '4题诊断中，单加NL修复truck/train和Yes/Door两题；联合B只修复Yes/Door，truck/train在补SPACE后又回到Truck，说明单加NL的恢复不能直接组合。斑马黑白和原S为k=0的侧别题均未被B修复。4题的N+D基线已全部正确，补B没有新增修复。这里只说明特定案例的干预效果，不证明边界接收者总体有效。','',
    '主24题新增损害18509693：On which side is the fence?，Gold=left，S=Left，而SPACE、NL和联合B均变成Right；负面案例完整保留，没有用诊断题的成功掩盖主样本损害。','',
    '## 正确性与解释边界','',
    '三道固定真实7B Smoke通过：HF/No-op所有有效Logits严格torch.equal；All/S历史答案Token和首Token完整Logits向量严格一致；实际禁止概率0、非目标Mask不变、独立KV、32层MLP、Hook正常与异常清理检查通过。全部168条新生成逐层Mask与概率审计复核；每条件独立真实KV和前缀，原输入Hash与权重版本不变，同题投影视觉特征Hash一致。模型全部冻结文件、依赖实现及评分器SHA256重算通过。','',
    f'输出异常（包含复用标签计数）共{len(anomalies)}条，详见validation.json。全部答案、Token IDs、真实接收位置、首Token变化及正负案例见CSV；没有计算Gold NLL。','',
    f'24题上S+B相对S净变化{sb["correct"]-s0["correct"]:+d}题，N+D+B相对N+D净变化{nb["correct"]-nd["correct"]:+d}题。是否超出同预算模板，以预设配对表的直接比较为准；四个历史失败题的修复也必须结合模板结果解读。即使单题补位答对，也不证明该边界唯一存储视觉信息。','',
    '这是小样本探索结果，不能据此确认3000题提升或选择新词表。本次没有自动启动3000题矩阵，没有追加搜索或训练。原3000题设计与132题历史结果保持不变。','',
    '![Accuracy](quick_accuracy.png)','','![Budget](quick_budget.png)','','![Tasks](quick_tasks.png)']
    (Q/'report_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    dump('validation.json',dict(passed=True,new_raw_records_verified=audits,logical_records=252,model_file_SHA256_exact=True,dependency_scorer_SHA256_exact=True,all32_MLP=True,all_new_weights_inputs_hooks_exact=True,all_new_forbidden_probabilities_zero=True,all_non_target_masks_exact=True,projected_visual_hash_equal_within_question=True,all_generated_argmax_ids_rechecked=True,anomalies=anomalies,bootstrap_repetitions=20000,main_images=24,diagnostic_images=4,smoke=read(Q/'smoke_validation.json')))
    dump('completion.json',dict(status='QUICK24_COMPLETE',full3000_complete=False,main_questions=24,diagnostic_questions=4,new_generations=168,history_reuse=84,condition_labels=252,validation_passed=True,seconds=done['seconds'],frozen_execution_commit=done['code_commit'],new_joint_conditions_added=False,full3000_started=False))
    print(json.dumps(dict(main=[{k:r[k] for k in ['condition','correct','accuracy_percent']} for r in mainrows],diagnostic=[{k:r[k] for k in ['question_id','condition','answer','correct']} for r in rows if r['cohort']=='diagnostic4']),ensure_ascii=False),flush=True)
def plot(summary,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});rr=[r for r in summary if r['cohort']=='quick24'];names=[r['condition'] for r in rr];short=['All','S','S+SPACE','S+NL','S+B','N+D','N+D+B','S+Control','N+D+Control']
    with PdfPages(Q/'quick_figures.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(11,6));a=np.array([r['accuracy_percent'] for r in rr]);lo=np.array([r['accuracy_CI_low'] for r in rr]);hi=np.array([r['accuracy_CI_high'] for r in rr]);ax.bar(range(9),a,color=['#6b7280','#387a9f','#3c9a86','#3c9a86','#db892e','#6476a8','#db892e','#9f6ca6','#9f6ca6']);ax.errorbar(range(9),a,yerr=[a-lo,hi-a],fmt='none',ecolor='#333',capsize=3);ax.set_xticks(range(9),short,rotation=20);ax.set_ylim(0,105);ax.set_ylabel('Complete-answer accuracy (%)');ax.set_title('SPACE / NL quick pilot: 24 questions (4 per task)');fig.text(.5,.015,'Image-cluster bootstrap95% intervals; exploratory, historically studied sample. Four diagnostics excluded.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(Q/'quick_accuracy.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        groups=collections.defaultdict(list)
        for i,r in enumerate(rr):groups[(r['mean_R_ratio']*100,r['accuracy_percent'])].append(str(i+1))
        for i,((x,y),numbers) in enumerate(groups.items()):
            ax.scatter(x,y,s=65);ax.annotate(','.join(numbers),(x,y),xytext=(0,15+(i%4)*18),textcoords='offset points',ha='center',fontsize=9,arrowprops={'arrowstyle':'-','color':'#777','lw':.6})
        ax.text(1.02,.95,'Condition IDs\n\n'+'\n'.join(str(i+1)+': '+n for i,n in enumerate(short)),transform=ax.transAxes,va='top',fontsize=10)
        ax.set_xlim(0,106);ax.set_ylim(78,93);ax.set_xlabel('Mean accessible R receivers (%)');ax.set_ylabel('Complete-answer accuracy (%)');ax.set_title('Receiver budget and accuracy: quick24');ax.grid(alpha=.2);fig.tight_layout(rect=[0,0,.8,1]);fig.savefig(Q/'quick_budget.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        data=np.array([[next(r['delta_vs_All_pp'] for r in tasks if r['cohort']=='quick24' and r['task']==t and r['condition']==c) for c in CONDITIONS] for t in TASKS]);fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(data,cmap='RdBu',vmin=-100,vmax=100,aspect='auto');ax.set_xticks(range(9),short,rotation=25,ha='right');ax.set_yticks(range(6),TASKS)
        for i in range(6):
            for j in range(9):ax.text(j,i,f'{data[i,j]:+.0f}',ha='center',va='center',color='white' if abs(data[i,j])>60 else 'black')
        ax.set_title('Task delta vs All-Writer (pp): only4 questions per task');fig.colorbar(im,ax=ax,label='Accuracy difference (pp)');fig.tight_layout();fig.savefig(Q/'quick_tasks.png',dpi=180);pdf.savefig(fig);plt.close(fig)
if __name__=='__main__':main()
