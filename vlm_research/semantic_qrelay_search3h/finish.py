"""Wait for the single-GPU search, recompute ACC/statistics, and release a reviewable result."""
import collections,shutil,traceback
from .common import *
from .cpu_cache import enable
from .pipeline import load,audit_reuse
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
enable()

def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))

def main():
    while not (P/'gpu_release.json').exists():
        assert not (P/'STOP.json').exists(),read(P/'STOP.json');assert remaining()>300;time.sleep(2)
    assert (P/'PIPELINE_COMPLETE.json').exists() and read(P/'gpu_release.json')['all_own_children_exited']
    freeze=read(P/'winner_freeze.json');winner=freeze['candidate_id'];rr=rows(P/'jobs/all3000/results.csv');assert len(rr)==3000 and len({r['question_id'] for r in rr})==3000
    h=History();d,m=load('S-assistant');source=[];budgets=collections.defaultdict(list)
    for r in rr:
        q=r['question_id'];sp=spec(q);w=json.loads(r['writers']);rec=score_record(q,winner,w,h);audit_reuse(rec);g=rec['generation'];r['correct']=int(r['correct']);r['R_ratio']=float(r['R_ratio']);assert r['answer']==g['text'] and json.loads(r['token_ids'])==g['token_ids'] and r['correct']==int(gqa_prediction(g['text'])==gold(q))
        assert set(sp['historical_writers']['Semantic-k'])<=set(w)<=set(sp['R']);assert g['eos'] and not any(g[k] for k in ['empty','truncated','repeated_trigram'])
        sw=m.select(sp,d['config']);sa=score_record(q,'S-assistant',sw,h);audit_reuse(sa);sg=sa['generation'];r.update(SA_correct=int(gqa_prediction(sg['text'])==gold(q)),SA_answer=sg['text'],SA_first_flip=int(g['token_ids'][0]!=sg['token_ids'][0]));budgets['SA'].append(len(sw)/len(sp['R']))
        for label,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-first-Q80','Q80'),('Semantic-k','S')]:
            b=h.get(q,label);assert int(r[short+'_correct'])==b['generation']['correct'] and r[short+'_answer']==b['generation']['text'];r[short+'_correct']=int(r[short+'_correct']);budgets[short].append(len(b['writers'])/len(sp['R']))
        for short in ['All','Q100','Q80','S','SA']:
            r[short+'_harm']=int(r[short+'_correct'] and not r['correct']);r[short+'_fix']=int(not r[short+'_correct'] and r['correct']);r[short+'_first_flip']=int(r[short+'_first_flip'])
        path=rec.get('source',{}).get('path',str(result_path(q,w)));source.append(dict(question_id=q,path=path,sha256=sha(path),reused=rec.get('reused',False),source_commit=rec.get('source',{}).get('commit','current_run')))
    h.close();atomic_json(P/'historical_audit_index.json',h.files);writecsv(P/'per_question.csv',rr);writecsv(P/'winner_provenance.csv',source)
    records=[]
    for path in sorted((P/'checkpoints').glob('*/*.json')):
        rec=read(path);g=rec['generation'];assert sha(rec['raw']['path'])==rec['raw']['sha256'] and rec['input_hashes']==spec(rec['question_id'])['input_hashes'];assert rec['weights_unchanged'] and rec['hook_cleanup'] and rec['MLP_calls']==[g['length']]*32 and g['audit']['passed'] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
        with np.load(rec['raw']['path']) as z:assert (z['audit'][:,0]==0).all() and (z['audit'][:,4:]==1).all() and z['token_ids'].tolist()==g['token_ids']
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),raw_path=rec['raw']['path'],raw_sha256=rec['raw']['sha256'],question_id=rec['question_id'],seconds=rec['seconds'],cuda_event_ms=rec['cuda_event_ms']))
    writecsv(P/'checkpoint_index.csv',records);images=[r['image_sha256'] for r in rr];summary=[];task=[];paired=[];names={'All':'All-Writer','Q100':'Question-Only','Q80':'Semantic-first-Q80','S':'Semantic-k','SA':'S-assistant'}
    for label,short in [(winner,None)]+[(v,k) for k,v in names.items()]:
        values=[r['correct' if short is None else short+'_correct'] for r in rr];ratio=[r['R_ratio'] for r in rr] if short is None else budgets[short];lo,hi=bootstrap(values,images);summary.append(dict(condition=label,n=3000,images=len(set(images)),correct=sum(values),ACC=100*np.mean(values),mean_R_ratio=float(np.mean(ratio)),CI95_low=lo,CI95_high=hi))
        for t in TASKS:
            ix=[i for i,r in enumerate(rr) if r['task']==t];assert len(ix)==500;task.append(dict(condition=label,task=t,n=500,correct=sum(values[i] for i in ix),ACC=100*np.mean([values[i] for i in ix])))
    for short,label in names.items():
        diff=np.array([r['correct']-r[short+'_correct'] for r in rr]);lo,hi=bootstrap(diff,images);paired.append(dict(candidate=winner,baseline=label,delta_pp=float(100*diff.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(diff==-1)),fix=int(sum(diff==1)),answer_changes=sum(r['answer']!=r[short+'_answer'] for r in rr),first_token_flips=sum(r[short+'_first_flip'] for r in rr),cluster_signflip_p=cluster_p(rr,short),Holm_p=1.,repetitions=20000,seed=20261104))
    prev=0.
    for i,x in enumerate(sorted(paired,key=lambda x:x['cluster_signflip_p'])):prev=max(prev,min(1.,(5-i)*x['cluster_signflip_p']));x['Holm_p']=prev
    writecsv(P/'accuracy_summary.csv',summary);writecsv(P/'paired_comparisons.csv',paired);writecsv(P/'task_accuracy.csv',task);writecsv(P/'harm_cases.csv',[r for r in rr if any(r[s+'_harm'] for s in names)],list(rr[0]));writecsv(P/'recovered_cases.csv',[r for r in rr if any(r[s+'_fix'] for s in names)],list(rr[0]))
    stage=[];timeline=[]
    for name in [f.parent.name for f in sorted((P/'jobs').glob('*/manifest.json'),key=lambda f:read(f)['monotonic'])]:
        D=P/'jobs'/name;stage.extend([dict(stage=name,**x) for x in rows(D/'summary.csv')]);man=read(D/'manifest.json');wd=read(D/'worker0_done.json');assert wd['requested']==len(man['tasks']);timeline.append(dict(stage=name,new_generations=man['new_generations_requested'],logical_requests=man['logical_requests'],inference_seconds=wd['inference_seconds'],cuda_event_ms=wd['cuda_event_ms'],freeze_UTC=man['UTC'],completed_UTC=wd['utc']))
    assert sum(x['new_generations'] for x in timeline)<=len(records);writecsv(P/'stage_summary.csv',stage);writecsv(P/'experiment_timeline.csv',timeline)
    candidates=[]
    for f in sorted((P/'candidates').glob('*/definition.json')):
        c=read(f);c['config']=json.dumps(c['config'],sort_keys=True);c['evaluations']=json.dumps([x for x in stage if x['candidate_id']==c['candidate_id']]);candidates.append(c)
    writecsv(P/'candidate_registry.csv',candidates)
    frontier=[dict(x,Pareto=not any(y['correct']>=x['correct'] and y['mean_R_ratio']<=x['mean_R_ratio'] and (y['correct']>x['correct'] or y['mean_R_ratio']<x['mean_R_ratio']) for y in summary)) for x in summary];writecsv(P/'budget_accuracy_frontier.csv',frontier)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});fig,ax=plt.subplots(figsize=(10,5));order=sorted(summary,key=lambda x:x['ACC']);ax.barh(range(len(order)),[x['ACC'] for x in order]);ax.set_yticks(range(len(order)),[f"{x['condition']} ({x['correct']}/3000)" for x in order]);ax.set_xlim(80,89);ax.set_xlabel('Complete-answer accuracy (%)');ax.set_title('Single-GPU S-preserving search: same historical GQA3000');fig.tight_layout();fig.savefig(P/'accuracy_comparison.png',dpi=220);fig.savefig(P/'accuracy_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for i,x in enumerate(summary):ax.scatter(100*x['mean_R_ratio'],x['ACC']);ax.annotate(x['condition'],(100*x['mean_R_ratio'],x['ACC']),xytext=(4,10 if i%2==0 else -13),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean open R receivers (%)');ax.set_ylabel('Complete-answer accuracy (%)');ax.set_xlim(20,110);ax.set_ylim(83,89);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'budget_accuracy.png',dpi=220);fig.savefig(P/'budget_accuracy.pdf');plt.close(fig)
    smoke=read(P/'worker0/smoke.json');assert smoke['passed'];assert read(P/'worker0/completion.json')['passed'];verify_sources()
    win=summary[0];best=max((x for x in summary if x['mean_R_ratio']<.5),key=lambda x:x['correct']);elapsed=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60
    lines=['# 单GPU三小时：基于原S的接收者搜索','',f"冻结后3000题方案 **{winner}**：**{win['correct']}/3000，{win['ACC']:.3f}%**，平均R预算 **{100*win['mean_R_ratio']:.3f}%**。本轮所有完整3000题实测方法中，预算<50%的最高点估计为{best['condition']}（{best['ACC']:.3f}%）。{'达到' if win['ACC']>=87 else '未达到'}87%的点估计目标。",'','## 冻结与筛选','','原S逐题保留；仅改变原生Prefill R Query读取Image Key的权限。ASSISTANT为原Prompt完整4 Subword词组，不改Prompt。全部32层P−1及真实Decode禁读Image Key；FP16/Eager、Transformers4.49.0、原模型revision、Greedy32、logits_to_keep=1保持原样。只有物理GPU1加载一份模型，其他卡和进程不干预。','','预先冻结的候选包括S基线、S＋ASSISTANT、上轮名词冠词增强、46%/49%目标预算下的整词补位排序及按Question结构选择补齐。所有候选先只读CPU核对完整3000题的平均预算，超50%者保留排除记录但不运行GPU。实际完整集合与预算见CPU_candidate_budget.csv。依次60→180→360→600逐级筛选，最后冻结一个全局规则后扩展原3000题；排名按正确数、较低预算、固定ID决定，不逐题挑答案。','','最终配置：`'+json.dumps(freeze['config'],sort_keys=True)+'`。逐题接收位置见winner_freeze.json；源码与SHA见candidates/。cap仅为完整词组可行目标，不保证每题恰好49%或46%。完整预算定义为逐题k/R再平均。','','|阶段|候选|正确/总数|平均R|','|---|---|---|---|']
    for x in stage:lines.append(f"|{x['stage']}|{x['candidate_id']}|{x['correct']}/{x['n']}|{100*float(x['mean_R_ratio']):.3f}%|")
    lines+=['','## 完整3000题ACC','','|条件|正确/3000|ACC|平均R|','|---|---|---|---|']
    for x in summary:lines.append(f"|{x['condition']}|{x['correct']}|{x['ACC']:.3f}%|{100*x['mean_R_ratio']:.3f}%|")
    lines+=['','|新方案−对照|ΔACC pp|95%CI pp|原对→错|原错→对|Holm p|','|---|---|---|---|---|---|']
    for x in paired:lines.append(f"|{x['baseline']}|{x['delta_pp']:+.3f}|[{x['CI95_low_pp']:+.3f}, {x['CI95_high_pp']:+.3f}]|{x['harm']}|{x['fix']}|{x['Holm_p']:.5f}|")
    lines+=['','3000题包含重复图片，以Image SHA256整簇配对Bootstrap 20,000次，seed20261104。p值用图片簇配对条件符号交换、20,000次并对最终五个比较作Holm校正。CI为普通探索性区间，未校正自适应搜索选择；CI跨0不是非劣效证明。所有3000题的历史结果已看过，600题不是新盲测，不能声称独立外部确认。','','## 审计、负面结果与边界','',f"五项CPU测试、单卡三道7B Smoke通过；Native/No-op全部有效Logits逐位torch.equal，历史S/Q80/Q100答案与首向量严格复现。新增{len(records)}次唯一生成，全部逐项复核Mask实际禁止概率为0、非目标Mask不变、32层MLP、输入/权重/Hook及独立KV。所有原始Checkpoint、Token IDs、负面案例和来源SHA保留。compact NPZ只在服务器资源目录，Git保存路径及SHA，不宣称GitHub含数组。模型在正式矩阵结束后自动退出。",'',f"从首次任务时间{read(P/'clock.json')['start_time']}计时，CPU分析结束约{elapsed:.1f}分钟；包含推送的最终耗时记录在资源目录delivery_receipt.json。硬限3小时，单卡GPU边界160分钟，预留交付时间，不重置时钟。",'','完整问题、真实答案及全部对照见per_question.csv，所有损害见harm_cases.csv。新增预算与位置变化混合，不能将相对S的增益全部归因于某一词类。Hard Mask只是功能性稀疏，不是物理加速。All-Writer也不是Native Full。新策略如未在ACC和预算上超过已有基线，应保留基线，不以小样本高分掩盖完整3000题负面结果。','','## 停止','','完成一个冻结方案的3000题后停止，不据其结果继续改规则，不新增数据、训练VLM或开发Kernel。','', '![Accuracy](accuracy_comparison.png)','', '![Budget and accuracy](budget_accuracy.png)']
    (P/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(P/'validation.json',dict(passed=True,UTC=utc(),complete_questions=3000,unique_images=len(set(images)),new_generations=len(records),CPU_tests=5,one_GPU_three_smokes=True,Native_noop_strict_equal=True,original_S_replay_exact=True,all_new_raw_hashes_verified=True,actual_probability_zero_and_nontarget_unchanged=True,all32_MLP_and_KV_verified=True,all_S_preserved=True,all3000_ACC_recomputed=True,bootstrap=20000,seed=20261104,all_data_preseen=True,source_manifest={str(f.relative_to(ROOT.parent)):sha(f) for f in Path(__file__).parent.glob('*.py')},gpu_release=read(P/'gpu_release.json')))
    atomic_json(P/'completion.json',dict(status='COMPLETE',UTC=utc(),winner=winner,correct=win['correct'],ACC=win['ACC'],mean_R_ratio=win['mean_R_ratio'],new_unique_generations=len(records),elapsed_minutes=elapsed,within_three_hours=remaining()>0))
    print(json.dumps(dict(summary=summary,paired=paired,elapsed_minutes=elapsed)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:atomic_json(P/'analysis_failure.json',dict(UTC=utc(),traceback=traceback.format_exc()));raise




