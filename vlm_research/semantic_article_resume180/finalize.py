"""CPU-only closeout of the fixed staged continuation; never schedules GPU work."""
import collections,math,traceback
from .common import *
from .cpu_cache import enable
from .pipeline import load,audit_reuse
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
from semantic_visual_writers.groups import intact
enable()

def rows(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8')))

def flag(x):return str(x).lower() in ['true','1']

def wait():
    while not (P/'gpu_release.json').exists():
        if (P/'STOP.json').exists():raise RuntimeError(read(P/'STOP.json'))
        assert remaining()>300,'Delivery reserve reached; no further analysis admission'
        time.sleep(2)
    assert (P/'PIPELINE_COMPLETE.json').exists() and not (P/'STOP.json').exists()
    release=read(P/'gpu_release.json');assert release['all_own_children_exited']

def main():
    wait();started=time.monotonic();winner=read(P/'winner_freeze.json')['candidate_id'];r=rows(P/'jobs/all3000/results.csv');assert len(r)==3000 and len({x['question_id'] for x in r})==3000
    h=History();d,m=load('S-assistant');provenance=[];allrows=[];baseline_values={s:[] for s in ['All','Q100','Q80','S','SA']};baseline_writers={s:[] for s in baseline_values};input_audit=[]
    for rr in r:
        q=rr['question_id'];sp=spec(q);w=json.loads(rr['writers']);rec=score_record(q,winner,w,h);audit_reuse(rec);g=rec['generation'];assert int(rr['correct'])==int(g['prediction']==gold(q)) and rr['answer']==g['text'] and json.loads(rr['token_ids'])==g['token_ids']
        assert sorted(set(w))==w and set(sp['historical_writers']['Semantic-k'])<=set(w)<=set(sp['R']) and intact(w,sp['groups'])
        rr['correct']=int(rr['correct']);rr['k']=int(rr['k']);rr['R']=int(rr['R']);rr['R_ratio']=float(rr['R_ratio']);assert len(w)==rr['k'] and len(sp['R'])==rr['R']
        sources={winner:rec}
        for name,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-first-Q80','Q80'),('Semantic-k','S')]:
            br=h.get(q,name);bg=br['generation'];assert int(rr[short+'_correct'])==bg['correct'] and rr[short+'_answer']==bg['text'];rr[short+'_correct']=int(rr[short+'_correct']);sources[name]=br
            baseline_values[short].append(bg['correct']);baseline_writers[short].append(len(br['writers'])/len(sp['R']))
        sw=m.select(sp,d['config']);br=score_record(q,'S-assistant',sw,h);audit_reuse(br);bg=br['generation'];assert bg['prediction']==gqa_prediction(bg['text']);rr.update(SA_correct=int(bg['prediction']==gold(q)),SA_answer=bg['text'],SA_first_flip=int(g['token_ids'][0]!=bg['token_ids'][0]));sources['S-assistant']=br
        baseline_values['SA'].append(rr['SA_correct']);baseline_writers['SA'].append(len(sw)/len(sp['R']))
        for short in baseline_values:
            rr[short+'_harm']=int(rr[short+'_correct'] and not rr['correct']);rr[short+'_fix']=int(not rr[short+'_correct'] and rr['correct']);rr[short+'_first_flip']=int(rr[short+'_first_flip'])
        for name,sr in sources.items():
            source=sr.get('source',{});path=source.get('path',str(result_path(q,sr['writers'])));raw=sr.get('raw',read(path).get('raw',{}));provenance.append(dict(question_id=q,condition=name,path=path,sha256=sha(path),source_commit=source.get('commit','current_run'),raw_path=raw.get('path',''),raw_sha256=raw.get('sha256',''),reused=sr.get('reused',False)))
        input_audit.append(dict(question_id=q,input_hashes_sha256=digest(sp['input_hashes']),original_sample_sha256=sp['original_sample_sha256'],question_unchanged=True,S_subset=True,whole_groups=True,k=len(w),R=len(sp['R']),writers=rr['writers']))
        allrows.append(rr)
    h.close();atomic_json(P/'final_historical_audit_index.json',h.files);writecsv(P/'provenance.csv',provenance);writecsv(P/'token_selection_audit.csv',input_audit);writecsv(P/'per_question.csv',allrows)
    assert np.mean([x['R_ratio'] for x in allrows])<.5
    # Compact original arrays remain on the server; Git contains their hashes, not arrays.
    checkpoints=[];feature_hashes={};seen=set()
    for path in sorted((P/'checkpoints').glob('*/*.json')):
        rec=read(path);q=rec['question_id'];g=rec['generation'];assert rec['input_hashes']==spec(q)['input_hashes'] and sha(rec['raw']['path'])==rec['raw']['sha256'];assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
        assert rec['weights_unchanged'] and rec['hook_cleanup'] and rec['MLP_calls']==[g['length']]*32 and g['prediction']==gqa_prediction(g['text'])
        assert rec['worker_source_sha256']==sha(Path(__file__).with_name('worker.py'))
        if q in feature_hashes:assert feature_hashes[q]==rec['projected_visual_sha256']
        feature_hashes[q]=rec['projected_visual_sha256']
        with np.load(rec['raw']['path']) as z:
            a=z['audit'];assert (a[:,0]==0).all() and (a[:,4:]==1).all();assert z['token_ids'].tolist()==g['token_ids'];assert json.loads(str(z['record_metadata']))['first_logits_sha256']==rec['first_logits_sha256']
        assert not any([g['empty'],g['truncated'],g['repeated_trigram']]) and g['eos'];key=(q,canonical(rec['writers']));assert key not in seen;seen.add(key)
        checkpoints.append(dict(question_id=q,writers_sha256=digest(rec['writers']),path=str(path),sha256=sha(path),raw_path=rec['raw']['path'],raw_sha256=rec['raw']['sha256'],worker=rec['worker'],job_id=rec['job_id'],seconds=rec['seconds'],cuda_event_ms=rec['cuda_event_ms'],eos=g['eos']))
    writecsv(P/'checkpoint_index.csv',checkpoints)
    # Whole images, not individual questions, are the resampling units.
    imgs=[x['image_sha256'] for x in allrows];yy=np.array([x['correct'] for x in allrows]);summary=[];paired=[];tasks=[]
    labels={'All':'All-Writer','Q100':'Question-Only','Q80':'Semantic-first-Q80','S':'Semantic-k','SA':'S-assistant'}
    for name,values,budget in [(winner,yy,[x['R_ratio'] for x in allrows])]+[(labels[s],baseline_values[s],baseline_writers[s]) for s in labels]:
        lo,hi=bootstrap(values,imgs);summary.append(dict(condition=name,n=3000,images=len(set(imgs)),correct=int(sum(values)),ACC=float(100*np.mean(values)),CI95_low=lo,CI95_high=hi,mean_R_ratio=float(np.mean(budget))))
        for task in TASKS:
            ix=[i for i,x in enumerate(allrows) if x['task']==task];assert len(ix)==500;tasks.append(dict(condition=name,task=task,n=len(ix),correct=int(sum(values[i] for i in ix)),ACC=float(100*np.mean([values[i] for i in ix])),mean_R_ratio=float(np.mean([budget[i] for i in ix]))))
    for short,label in labels.items():
        diff=yy-np.array(baseline_values[short]);lo,hi=bootstrap(diff,imgs);paired.append(dict(candidate=winner,baseline=label,n=3000,images=len(set(imgs)),delta_pp=float(100*diff.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(diff==-1)),fix=int(sum(diff==1)),answer_changes=sum(x['answer']!=x[short+'_answer'] for x in allrows),first_token_flips=sum(x[short+'_first_flip'] for x in allrows),cluster_signflip_p=cluster_p(allrows,short),Holm_p=1.,bootstrap=20000,seed=20261104,interval_scope='pointwise exploratory; adaptive selection not corrected'))
    prev=0.
    for i,p in enumerate(sorted(paired,key=lambda x:x['cluster_signflip_p'])):prev=max(prev,min(1.,(5-i)*p['cluster_signflip_p']));p['Holm_p']=prev
    writecsv(P/'accuracy_summary.csv',summary);writecsv(P/'paired_comparisons.csv',paired);writecsv(P/'task_accuracy.csv',tasks)
    fields=list(allrows[0]);writecsv(P/'harm_cases.csv',[x for x in allrows if any(x[s+'_harm'] for s in labels)],fields);writecsv(P/'recovered_cases.csv',[x for x in allrows if any(x[s+'_fix'] for s in labels)],fields)
    stage=[];stage_paired=[];timing=[]
    for stage_name in ['s180','s360','check600','all3000']:
        D=P/'jobs'/stage_name;sr=rows(D/'results.csv');ss=rows(D/'summary.csv');stage.extend([dict(stage=stage_name,**x) for x in ss]);manifest=read(D/'manifest.json');wr=[read(D/f'worker{i}_done.json') for i in range(2)];assert sum(x['new_generations'] for x in wr)==manifest['new_generations_requested'];assert sum(x['requested'] for x in wr)==len(manifest['tasks'])
        # Verify exclusive deterministic assignment for every image and every mask.
        for t in manifest['tasks']:assert t['worker']==int(digest(['worker-v1',sample(t['question_id'])['image_sha256']]),16)%2;assert result_path(t['question_id'],t['writers']).exists()
        timing.append(dict(stage=stage_name,logical_requests=manifest['logical_requests'],new_generations=manifest['new_generations_requested'],frozen_UTC=manifest['UTC'],summary_UTC=read(D/'analysis_completion.json')['UTC'],worker_inference_seconds=sum(x['inference_seconds'] for x in wr),cuda_event_ms=sum(x['cuda_event_ms'] for x in wr)))
        for cid in sorted({x['candidate_id'] for x in sr}):
            rr=[x for x in sr if x['candidate_id']==cid]
            for short in ['All','Q100','S']:
                dif=np.array([int(x['correct'])-int(x[short+'_correct']) for x in rr]);lo,hi=bootstrap(dif,[x['image_sha256'] for x in rr]);stage_paired.append(dict(stage=stage_name,candidate=cid,baseline=labels[short],n=len(rr),delta_pp=float(100*dif.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(dif==-1)),fix=int(sum(dif==1)),bootstrap=20000,scope='exploratory adaptive stage'))
    writecsv(P/'stage_summary.csv',stage);writecsv(P/'stage_paired_comparisons.csv',stage_paired);writecsv(P/'experiment_timeline.csv',timing)
    assert sum(x['new_generations'] for x in timing)==len(checkpoints)
    for i in range(2):assert read(P/f'worker{i}/smoke.json')['passed'] and read(P/f'worker{i}/completion.json')['passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    order=sorted(summary,key=lambda x:x['ACC']);fig,ax=plt.subplots(figsize=(10,4.8));ys=np.arange(len(order));ax.barh(ys,[x['ACC'] for x in order],color=['#228070' if x['condition']==winner else '#809bb7' for x in order]);ax.set_yticks(ys,[f"{x['condition']} ({x['correct']}/3000)" for x in order]);ax.set_xlim(80,89);ax.set_xlabel('Complete-answer GQA accuracy (%)');ax.set_title('GQA3000: exploratory continuation, same historical images');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(P/'accuracy_comparison.png',dpi=220);fig.savefig(P/'accuracy_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));
    for j,x in enumerate(summary):ax.scatter(100*x['mean_R_ratio'],x['ACC'],s=60);ax.annotate(x['condition'],(100*x['mean_R_ratio'],x['ACC']),xytext=(5,(-12 if j%2 else 9)),textcoords='offset points',fontsize=8)
    ax.axvline(50,color='#999',ls='--',lw=1);ax.set_xlim(20,110);ax.set_ylim(83,88);ax.set_xlabel('Mean allowed R-to-image receiver ratio (%)');ax.set_ylabel('Complete-answer accuracy (%)');ax.set_title('Observed policies; access budget is not runtime savings');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'budget_accuracy.png',dpi=220);fig.savefig(P/'budget_accuracy.pdf');plt.close(fig)
    val=dict(passed=True,UTC=utc(),all3000_complete=True,question_count=3000,unique_images=len(set(imgs)),S_preserved_every_question=True,whole_groups=True,mean_R_under_half=True,new_unique_generations=len(checkpoints),all_new_checkpoints_hashed=True,compact_NPZ_hashed=True,actual_forbidden_probability_zero=True,non_target_mask_unchanged=True,all32_MLP_verified=True,weight_input_hook_audits=True,all_EOS_no_anomalies=True,independent_image_shards=True,CPU_tests=5,two_GPU_three_case_smokes=True,Native_noop_all_valid_logits_torch_equal=True,original_S_replay_exact=True,source_files={str(f.relative_to(ROOT.parent)):sha(f) for f in Path(__file__).parent.glob('*.py')},shared_source_verification=verify_sources()['shared_sources'],bootstrap_repetitions=20000,seed=20261104,Holm_family=5,all_data_previously_seen=True,no_confirmation_claim=True,resource_release=read(P/'gpu_release.json'),analysis_seconds=time.monotonic()-started)
    atomic_json(P/'validation.json',val)
    win=summary[0];old=next(x for x in summary if x['condition']=='S-assistant');best=next(x for x in summary if x['condition']=='Question-Only');elapsed=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60
    lines=[f'# Semantic 接收者增强：三小时续跑正式记录', '',f"本轮固定胜出方案为 **{winner}**：完整3000题 **{win['correct']}/3000，{win['ACC']:.3f}%**，平均开放R接收位置 **{100*win['mean_R_ratio']:.3f}%**。原S全部保留。相较上轮S＋ASSISTANT的{old['correct']}/3000，净变化为{win['correct']-old['correct']:+d}题；相较Question-Only的{best['correct']}/3000，净变化为{win['correct']-best['correct']:+d}题。没有达到87%时如实作为未达成报告。",'', '## 冻结策略与选择过程','',f"采用原生完整ASSISTANT词组、固定POS名词与依存冠词的输入规则增强原S；具体最终配置为 `{json.dumps(read(P/'winner_freeze.json')['config'],ensure_ascii=False,sort_keys=True)}`。既有Prompt、图像、分词和评分不变，所有32层P−1及真实Decode仍禁止直接读Image Key，只有所选R Query开放读图。无o_proj消融，无训练，无KV剪枝。新增位置增加了预算，不能将相对S的收益全部归因于选择质量。",'', '预先冻结的8个方案先比较DEV180，再选3个比较DEV360、2个比较CHECK600，按正确数优先、预算和ID打破平局，固定一个全局方案后补齐3000题。完整冻结清单、输入SHA、Source SHA、全部阶段负面结果均保留；没有逐题按答案选条件。180题中的等新增预算Instruction4和Question额外4组控制也完整保留。', '', '|阶段|方案|正确/总数|ACC|平均R比例|', '|---|---|---|---|---|']
    for x in stage:lines.append(f"|{x['stage']}|{x['candidate_id']}|{x['correct']}/{x['n']}|{float(x['ACC']):.3f}%|{100*float(x['mean_R_ratio']):.3f}%|")
    lines+=['','## 同3000题配对结果','','|对照|正确/3000|ACC|平均R比例|','|---|---|---|---|']
    for x in summary:lines.append(f"|{x['condition']}|{x['correct']}|{x['ACC']:.3f}%|{100*x['mean_R_ratio']:.3f}%|")
    lines+=['','|新方案−对照|ΔACC pp|普通95%CI pp|原对→错|原错→对|Holm p|','|---|---|---|---|---|---|']
    for x in paired:lines.append(f"|{x['baseline']}|{x['delta_pp']:+.3f}|[{x['CI95_low_pp']:+.3f}, {x['CI95_high_pp']:+.3f}]|{x['harm']}|{x['fix']}|{x['Holm_p']:.5f}|")
    lines+=['','按Image SHA256整簇配对Bootstrap 20,000次，seed=20261104；随机化检验也以完整图片簇交换条件。五个最终对照的p值作Holm校正，表中CI为普通点位区间。该校正不覆盖先前及本轮自适应策略筛选，不能据此宣称新的确认性发现、非劣效或外部泛化。', '', '## 任务、失败与解释','','所有六任务每类500题见task_accuracy.csv。harm_cases.csv保留相对任一对照的全部损害，recovered_cases.csv保留全部纠正；per_question.csv保留Gold、真实完整答案、Token IDs、所有配对基线及首Token翻转。全体3000题有2409张不同图片，不能按3000个独立图片处理。', '', '这些结果仅衡量功能性视觉写入稀疏。较高ACC不能证明新增位置唯一存储视觉信息；平均R比例不是FLOPs、显存或延迟收益。原S与S增强的预算不同，无法隔离位置作用与预算作用。All-Writer也是答案阶段禁读图的基线，与原生Full不同。', '', '## 审计与时间','','CPU五项测试、两卡各三道真实7B Smoke通过。原生与No-op全部有效logits逐位torch.equal，原S/Q80/Q100的历史答案和首向量严格复现；每次新生成均验证实际禁止概率为0、非目标Mask不变、32层MLP、真实独立Cache、输入/权重/Hook和EOS。原生首向量校验后不保留大Logits归档。', '',f"本轮新增唯一生成 **{len(checkpoints)}** 次；每个条件标签、复用来源和SHA均记录。此前60题记录、初轮结果及原历史基线只读按SHA复用，不重新生成。所有16/32位和模型协议保持冻结；完整原始compact NPZ位于服务器资源目录，Git仅保存路径和SHA，不能宣称GitHub包含这些数组。",'',f"启动时间（UTC）：{read(P/'clock.json')['start_time']}。分析完成时已用约{elapsed:.1f}分钟；最终交付耗时另由服务器资源目录delivery_receipt.json记录，含提交和推送。本轮模型自动随流水线结束立即退出，无等待人工决策的模型常驻。资源释放详情见gpu_release.json。", '', '## 范围与停止','','全部原3000题历史结果已看过。DEV、CHECK名称只表示开发分片，600题并非新的盲测或外部确认集。本轮不新增数据，不回改历史结果，不修改main，不追加词表搜索或GPU条件。最终3000题结果完成后停止。']
    (P/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Index every result file except the index itself and root completion, including worker completion files.
    index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(P.rglob('*')) if f.is_file() and f not in [P/'artifact_index.csv',P/'completion.json'] and f.suffix!='.lock']
    writecsv(P/'artifact_index.csv',index)
    atomic_json(P/'completion.json',dict(status='COMPLETE',UTC=utc(),all3000_complete=True,validation_sha256=sha(P/'validation.json'),artifact_index_sha256=sha(P/'artifact_index.csv'),new_unique_generations=len(checkpoints),winner=winner,correct=win['correct'],ACC=win['ACC'],mean_R_ratio=win['mean_R_ratio'],within_three_hours=remaining()>0,elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60,administrative_push_pending=True))
    print(json.dumps(dict(COMPLETE=True,winner=winner,summary=summary,paired=paired,elapsed_minutes=elapsed),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(P/'closeout_failure.json',dict(UTC=utc(),traceback=traceback.format_exc()));raise
