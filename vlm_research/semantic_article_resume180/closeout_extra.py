"""CPU reporting supplements and final artifact index; no new inference."""
from .common import *
from .pipeline import load

def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))

def main():
    while not (P/'completion.json').exists():
        assert not (P/'closeout_failure.json').exists(),read(P/'closeout_failure.json')
        assert remaining()>180
        time.sleep(2)
    done=read(P/'completion.json');assert done['status']=='COMPLETE'
    # Preserve the first CPU closeout verbatim before publishing a reporting revision.
    archive=P/'reporting_pass1';archive.mkdir(exist_ok=True)
    if not (archive/'completion.json').exists():
        original=(P/'report_zh.md').read_text(encoding='utf-8').split('\n## 直接回答与选择器解释')[0]
        original=original.replace('尚未达到87%的点估计目标。','没有达到87%时如实作为未达成报告。').replace('已达到87%的点估计目标。','没有达到87%时如实作为未达成报告。')
        # A failed reporting attempt may have appended a newline before its supplement.
        want=next(x['sha256'] for x in rows(P/'artifact_index.csv') if x['path'].endswith('/report_zh.md'))
        variants=[original,original.rstrip()+'\n',original.rstrip()+'\n\n']
        original=next(s for s in variants if hashlib.sha256(s.encode()).hexdigest()==want)
        (archive/'report_zh.md').write_text(original,encoding='utf-8')
        for name in ['validation.json','artifact_index.csv','completion.json']:(P/name).rename(archive/name)
        atomic_json(P/'reporting_revision.json',dict(UTC=utc(),reason='Supplemental CPU report must not overwrite immutable first-pass validation/completion records; first pass preserved verbatim.',numerical_results_unchanged=True,first_report_sha256=want,first_completion_sha256=sha(archive/'completion.json')))
    winner=done['winner'];rr=rows(P/'per_question.csv');summ=rows(P/'accuracy_summary.csv');stages=rows(P/'stage_summary.csv');d,m=load(winner);changes=[]
    for r in rr:
        q=r['question_id'];sp=spec(q);w=m.select(sp,d['config']);S=sp['historical_writers']['Semantic-k'];added=set(w)-set(S);groups=[g for g in sp['groups'] if set(g['positions'])<=added];assert set(p for g in groups for p in g['positions'])==added
        changes.append(dict(question_id=q,task=r['task'],image_sha256=r['image_sha256'],question=sp['question'],S_k=len(S),new_k=len(w),added_k=len(added),R=len(sp['R']),S_R_ratio=len(S)/len(sp['R']),new_R_ratio=len(w)/len(sp['R']),added_groups=json.dumps([dict(words=g['words'],positions=g['positions'],pos=g.get('pos',[]),dep=g.get('dep',[]),question=g['question']) for g in groups],ensure_ascii=False),S_answer=r['S_answer'],new_answer=r['answer'],All_answer=r['All_answer'],S_correct=r['S_correct'],new_correct=r['correct']))
    writecsv(P/'token_changes.csv',changes)
    registry=[]
    for path in sorted((P/'candidates').glob('*/definition.json')):
        c=read(path);registry.append(dict(candidate_id=c['candidate_id'],config=json.dumps(c['config'],sort_keys=True),source_sha256=c['source_sha256'],definition_sha256=sha(path),created_UTC=c['created_utc'],all_evaluations=json.dumps([s for s in stages if s['candidate_id']==c['candidate_id']]),status='single_all3000_candidate' if c['candidate_id']==winner else 'not_expanded_to_all3000'))
    writecsv(P/'candidate_registry.csv',registry);atomic_json(P/'selection_rules.json',dict(candidate_id=winner,config=d['config'],source_sha256=d['source_sha256'],S_always_preserved=True,Prompt_unchanged=True,input_only=True,mean_R_limit=.5,strict_limit=True,reproduce_command="CUDA_VISIBLE_DEVICES='' PYTHONPATH=./vlm_research:/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research:/data/lk/ATE-worktrees/gqa360-structural-receiver-selection/vlm_research /data/lk/深度学习/.envs/ate-vlm-research/bin/python -m semantic_article_resume180.reproduce"))
    frontier=[]
    for x in summ:
        dom=any(int(y['correct'])>=int(x['correct']) and float(y['mean_R_ratio'])<=float(x['mean_R_ratio']) and (int(y['correct'])>int(x['correct']) or float(y['mean_R_ratio'])<float(x['mean_R_ratio'])) for y in summ)
        frontier.append(dict(**x,Pareto=not dom,scope='only policies actually fully evaluated on the same 3000 questions'))
    writecsv(P/'budget_accuracy_frontier.csv',frontier)
    dist={}
    for k in ['S_k','new_k','S_R_ratio','new_R_ratio','added_k']:
        a=np.array([float(x[k]) for x in changes]);dist[k]=dict(mean=float(a.mean()),min=float(a.min()),p25=float(np.percentile(a,25)),median=float(np.median(a)),p75=float(np.percentile(a,75)),p95=float(np.percentile(a,95)),max=float(a.max()))
    atomic_json(P/'budget_distribution.json',dist)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    task=rows(P/'task_accuracy.csv');names=[winner,'Semantic-k','S-assistant','Question-Only','All-Writer'];matrix=np.array([[float(next(x['ACC'] for x in task if x['condition']==n and x['task']==t))-float(next(x['ACC'] for x in task if x['condition']=='All-Writer' and x['task']==t)) for t in TASKS] for n in names]);fig,ax=plt.subplots(figsize=(11,4));lim=max(1,float(np.abs(matrix).max()));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim);ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(TASKS)),TASKS)
    for i in range(len(names)):
        for j in range(len(TASKS)):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>lim*.6 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy difference from All-Writer (pp)');ax.set_title('Six GQA tasks, 500 questions each');fig.tight_layout();fig.savefig(P/'task_delta_accuracy.png',dpi=220);fig.savefig(P/'task_delta_accuracy.pdf');plt.close(fig)
    win=next(x for x in summ if x['condition']==winner);sa=next(x for x in summ if x['condition']=='S-assistant');q=next(x for x in summ if x['condition']=='Question-Only');allw=next(x for x in summ if x['condition']=='All-Writer');best=max((x for x in summ if float(x['mean_R_ratio'])<.5),key=lambda x:int(x['correct']));net=int(win['correct'])-int(sa['correct']);qnet=int(win['correct'])-int(q['correct'])
    path=P/'report_zh.md';text=(archive/'report_zh.md').read_text(encoding='utf-8');text=text.replace('没有达到87%时如实作为未达成报告。',f"{'已达到87%的点估计目标。' if float(win['ACC'])>=87 else '尚未达到87%的点估计目标。'}")
    text+='\n## 直接回答与选择器解释\n\n'
    text+=f"1. 相较上轮S＋ASSISTANT，新方案净{net:+d}题；相较Question-Only净{qnet:+d}题。当前本报告所有完整3000题实测、平均R<50%的方法中，最高点估计为 **{best['condition']}：{best['correct']}/3000，{float(best['ACC']):.3f}%，R={100*float(best['mean_R_ratio']):.3f}%**。没有用180/360题高分替代3000题结论。\n"
    text+='新方案在ACC和R预算两个点估计上均被Question-Only支配。相比S＋ASSISTANT，继续补名词和冠词只增加3题，同时平均R预算从39.408%增至46.336%；其配对区间跨0，不能认为获得可靠改进。若坚持从S出发，S＋ASSISTANT仍是本报告中更节省预算的实测选择。\n'
    text+='2. 本轮新方案以完整词组保护原S，额外开放原生ASSISTANT的4个Subword，以及Question中固定POS的NOUN/PROPN和dep=det的a/an/the。没有修改STOP词表、Prompt或模型。token_changes.csv逐题列出新增词组、POS、依存、位置、预算和真实答案；没有根据当前题Gold选择接收者。\n'
    text+='3. DEV180中，S＋ASSISTANT为167/180，等新增4 Token的S＋instruction4为161/180；这是同预算下位置有作用的小样本证据，但不足以推出所有模板Token或冠词的普遍必要性。加入名词/冠词与S、S＋ASSISTANT的比较又改变了预算；不能将其解释为纯词类效应。\n'
    text+='4. 原预算≤35%的主目标已由后续授权放宽为平均R<50%。本轮不重新宣称解决35%目标；≤30%、≤35%、≤42%的历史结果和失败策略见初轮a1c9d81与续轮1df71f4报告。本次不再训练风险模型或新增Question剪枝。\n'
    text+='5. 所有3000题此前已查看，不能判断本轮改进能否迁移到未参与研究的问题，也不能再把CHECK600称为盲测Holdout。结果支持接收位置对功能有影响，但不证明获得了可泛化最优策略。若后续另获授权，最小有价值的下一步是固定当前单一规则后在未研究图片上评估；本轮按要求不新增数据、不执行该建议。\n'
    path.write_text(text,encoding='utf-8')
    # Complete scientific output; external receipt records publication and final elapsed time.
    done.pop('administrative_push_pending',None);done['publication_audit_receipt']=str(PRIVATE/'delivery_receipt.json');done['elapsed_minutes']=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60
    validation=read(archive/'validation.json');validation['reporting_source_sha256']=sha(__file__);validation['reproduce_source_sha256']=sha(Path(__file__).with_name('reproduce.py'));validation['reporting_first_pass_archive']=str(archive);atomic_json(P/'validation.json',validation)
    index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f not in [P/'artifact_index.csv',P/'completion.json'] and f.suffix!='.lock'];writecsv(P/'artifact_index.csv',index);done['artifact_index_sha256']=sha(P/'artifact_index.csv');done['validation_sha256']=sha(P/'validation.json');atomic_json(P/'completion.json',done)
    print(json.dumps(dict(reporting_complete=True,winner=win,best_observed_under_half=best,net_vs_prior_S_assistant=net,net_vs_question_only=qnet),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
