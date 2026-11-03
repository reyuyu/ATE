"""Complete-answer accuracy and descriptive token/budget attribution; CPU only."""
import csv,json
from .run import *

def bootstrap(ids,cases,records):
    images=sorted({cases[q]['sample']['image_sha256'] for q in ids});members=[[i for i,q in enumerate(ids) if cases[q]['sample']['image_sha256']==im] for im in images]
    mat=np.array([[records[q,c]['generation']['correct'] for c in NAMES] for q in ids],dtype=np.int64);sums=np.array([mat[m].sum(0) for m in members]);sizes=np.array([len(m) for m in members]);ix=np.random.default_rng(20261104).integers(0,len(images),size=(20000,len(images)));boots=np.empty((20000,4))
    for a in range(0,20000,500):j=ix[a:a+500];boots[a:a+500]=100*sums[j].sum(1)/sizes[j].sum(1)[:,None]
    return mat,boots

def main():
    p=verify();plan=read(P/'sample_plan.json');cases=plan['cases'];done=read(P/'gpu_completion.json');assert done['passed']
    for n,info in p['model_files'].items():assert sha256(n)==info['sha256']
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h
    original_manifest=BASE/'results/gqa360_structural_receiver_selection/sample_manifest.csv';assert sha256(original_manifest)==p['manifest_sha256']
    tokens={r['question_id']:r for r in csv.DictReader((P/'token_changes.csv').open())};records={};rows=[];index=[];anomalies=[];checked=0;reuses=0
    for q,v in cases.items():
        s=v['sample'];gs={}
        for c in NAMES:
            path=P/f'worker{v["worker"]}'/'records'/q/(c+'.json');assert path.exists(),'NOT_GENERATED: '+str(path);r=read(path);g=r['generation'];w=v['writers'][c]
            assert w==r['writers']==g['writer_positions'] and g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer']) and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
            if r['reused']:
                assert sha256(r['source']['path'])==r['source']['sha256'] and read(r['source']['path'])['generation']==g and sha256(r['historical_raw']['path'])==r['historical_raw']['sha256'];reuses+=1
                if c==NEW:assert not v['changed'] and g==read(P/f'worker{v["worker"]}'/'records'/q/'Semantic-k.json')['generation']
            else:
                assert c==NEW and v['changed'] and r['plan_sha256']==p['plan_sha256'] and r['input_hashes']==v['design']['input_hashes'] and r['projected_visual_sha256']==v['reference_visual_sha256'] and r['weights_unchanged'] and r['hook_cleanup'] and r['MLP_calls']==[g['length']]*32
                assert len(r['first_logits_sha256'])==64 and sha256(r['raw']['path'])==r['raw']['sha256']
                with np.load(r['raw']['path']) as z:
                    assert set(z.files)=={'audit','token_ids','record_metadata'}
                    a=z['audit'];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all() and len(a)==32*g['length']
                    assert z['token_ids'].tolist()==g['token_ids'];m=json.loads(str(z['record_metadata']));assert m=={k:r[k] for k in m}
                checked+=1;cp=P/f'worker{v["worker"]}'/'checkpoints'/q/(digest(w)+'.json');index.append(dict(question_id=q,worker=v['worker'],checkpoint=str(cp.relative_to(P)),checkpoint_sha256=sha256(cp),first_logits_sha256=r['first_logits_sha256'],compact_audit_path=r['raw']['path'],compact_audit_sha256=r['raw']['sha256'],compact_audit_bytes=r['raw']['bytes']))
            records[q,c]=r;gs[c]=g
            if not g['eos'] or g['empty'] or g['truncated'] or g['repeated_trigram']:anomalies.append(dict(question_id=q,condition=c,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeat=g['repeated_trigram']))
        old=gs['Semantic-k'];new=gs[NEW];tc=tokens[q]
        rows.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],old_answer=old['text'],new_answer=new['text'],old_correct=old['correct'],new_correct=new['correct'],All_answer=gs['All-Writer']['text'],All_correct=gs['All-Writer']['correct'],Q_answer=gs['Question-Only']['text'],Q_correct=gs['Question-Only']['correct'],old_token_ids=json.dumps(old['token_ids']),new_token_ids=json.dumps(new['token_ids']),writer_changed=v['changed'],has_to=v['has_to'],has_can=v['has_can'],can_POS=json.dumps(v['can_POS']),old_k=v['old_k'],new_k=v['new_k'],delta_k=v['new_k']-v['old_k'],R=len(v['design']['R']),old_R_ratio=v['old_k']/len(v['design']['R']),new_R_ratio=v['new_k']/len(v['design']['R']),old_writers=json.dumps(v['writers']['Semantic-k']),new_writers=json.dumps(v['writers'][NEW]),to_removed=tc['to_removed'],can_added=tc['can_added'],other_added=tc['other_added'],other_removed=tc['other_removed'],other_positions_changed=v['other_positions_changed'],budget_changed=v['budget_changed'],harm_vs_S=int(old['correct'] and not new['correct']),fix_vs_S=int(not old['correct'] and new['correct']),answer_change=int(old['text']!=new['text']),first_token_flip=int(old['token_ids'][0]!=new['token_ids'][0]),reused=records[q,NEW]['reused'],status='REUSED_EXACT_WRITERS' if records[q,NEW]['reused'] else 'GENERATED_VERIFIED',eos=new['eos'],empty=new['empty'],truncated=new['truncated'],repeated_trigram=new['repeated_trigram']))
    assert checked==p['new_generations']==94 and reuses==p['history_reuse']==1346 and len(rows)==360
    assert sum(r['old_correct'] for r in rows)==319 and sum(r['All_correct'] for r in rows)==327 and sum(r['Q_correct'] for r in rows)==324
    # Descriptive grammar audit only: never used by STOP selection or inference.
    can_context={'12642596':'garbage can','17484369':'soda can','18699253':'garbage can'}
    assert {r['question_id'] for r in rows if r['has_can']}==set(can_context)
    for r in rows:
        r['can_context']=can_context.get(r['question_id'],'')
        r['can_descriptive_use']='NOUN (container); frozen parser says AUX' if r['has_can'] else ''
        r['can_POS_disagreement']=bool(r['has_can'])
    csvout(P/'per_question.csv',rows);csvout(P/'checkpoint_index.csv',index);fields=list(rows[0])
    for name,subset in [('harm_cases.csv',[r for r in rows if r['harm_vs_S']]),('recovered_cases.csv',[r for r in rows if r['fix_vs_S']]),('to_cases.csv',[r for r in rows if r['has_to']]),('can_cases.csv',[r for r in rows if r['has_can']]),('budget_mediation_cases.csv',[r for r in rows if r['other_positions_changed']])]:csvout(P/name,subset,fields)
    groups=[('all360',list(cases)),('writers_changed',[q for q,v in cases.items() if v['changed']]),('writers_unchanged',[q for q,v in cases.items() if not v['changed']]),('to_related',[q for q,v in cases.items() if v['has_to']]),('to_only',[q for q,v in cases.items() if v['has_to'] and not v['has_can']]),('can_related',[q for q,v in cases.items() if v['has_can']]),('can_only',[q for q,v in cases.items() if v['has_can'] and not v['has_to']]),('to_and_can',[q for q,v in cases.items() if v['has_can'] and v['has_to']]),('other_selection_changes',[q for q,v in cases.items() if v['other_positions_changed']])]
    for label in ['NOUN','AUX','VERB']:
        ids=[q for q,v in cases.items() if any(label in w['POS'] for w in v['can_POS'])]
        if ids:groups.append(('can_POS_'+label,ids))
    groups += [(t,[q for q,v in cases.items() if v['sample']['task']==t]) for t in TASKS]
    summary=[];pairs=[];tasks=[];mechanisms=[]
    for group,ids in groups:
        if not ids:continue
        mat,boots=bootstrap(ids,cases,records);rr=[r for r in rows if r['question_id'] in ids]
        for j,c in enumerate(NAMES):
            ci=np.percentile(boots[:,j],[2.5,97.5]);r=dict(group=group,condition=c,n=len(ids),correct=int(mat[:,j].sum()),accuracy_percent=float(100*mat[:,j].mean()),CI95_low=ci[0],CI95_high=ci[1],mean_k=float(np.mean([len(cases[q]['writers'][c]) for q in ids])),mean_R_ratio=float(np.mean([len(cases[q]['writers'][c])/len(cases[q]['design']['R']) for q in ids])),reused=sum(records[q,c]['reused'] for q in ids))
            (tasks if group in TASKS else summary).append(r)
        for c in ['Semantic-k','All-Writer','Question-Only'] if group=='all360' else ['Semantic-k']:
            k=NAMES.index(c);diff=boots[:,-1]-boots[:,k];ci=np.percentile(diff,[2.5,97.5]);pairs.append(dict(group=group,left=NEW,right=c,n=len(ids),delta_pp=float(100*(mat[:,-1]-mat[:,k]).mean()),CI95_low=ci[0],CI95_high=ci[1],primary_comparison=group=='all360' and c=='Semantic-k',right_correct_left_wrong=int(((mat[:,k]==1)&(mat[:,-1]==0)).sum()),right_wrong_left_correct=int(((mat[:,k]==0)&(mat[:,-1]==1)).sum()),answers_changed=sum(records[q,c]['generation']['text']!=records[q,NEW]['generation']['text'] for q in ids),first_token_flips=sum(records[q,c]['generation']['token_ids'][0]!=records[q,NEW]['generation']['token_ids'][0] for q in ids)))
        mechanisms.append(dict(group=group,n=len(ids),old_correct=sum(r['old_correct'] for r in rr),new_correct=sum(r['new_correct'] for r in rr),harm=sum(r['harm_vs_S'] for r in rr),fix=sum(r['fix_vs_S'] for r in rr),mean_old_k=float(np.mean([r['old_k'] for r in rr])),mean_new_k=float(np.mean([r['new_k'] for r in rr])),mean_old_R_ratio=float(np.mean([r['old_R_ratio'] for r in rr])),mean_new_R_ratio=float(np.mean([r['new_R_ratio'] for r in rr])),budget_changed=sum(r['budget_changed'] for r in rr),other_positions_changed=sum(r['other_positions_changed'] for r in rr)))
    csvout(P/'accuracy_summary.csv',summary);csvout(P/'task_accuracy.csv',tasks);csvout(P/'paired_comparisons.csv',pairs);csvout(P/'subgroup_effects.csv',mechanisms)
    smokes=[]
    for i in range(2):
        W=P/f'worker{i}';assert read(W/'smoke_validation.json')['passed'] and read(W/'structural_smoke_validation.json')['passed'];r=read(W/'structural_smoke_validation.json')['raw'];assert sha256(r['path'])==r['sha256']
        with np.load(r['path']) as z:assert set(z.files)=={'audit','token_ids'}
        smokes.append(dict(worker=i,audit_path=r['path'],audit_sha256=r['sha256'],bytes=r['bytes']))
    csvout(P/'smoke_audit_index.csv',smokes);release=read(P/'gpu_release.json');assert release['all_own_children_exited']
    validation=dict(passed=True,questions=360,new_generations_checked=94,new_label_reused=266,baseline_reused=1080,logical_records=1440,all_conditions_present=True,missing_outputs=[],exact_parent360_manifest_SHA256=True,all360_original_natural_S_reconstructed_exactly=True,only_to_can_STOP_membership_changed=True,all_grouping_regex_order_budget_algorithms_byte_identical=True,CPU_tests=p['CPU_tests'],all_changed_writer_sets_and_actual_budgets_verified=True,all_compact_raw_mask_audits_zero_forbidden_and_exact_non_targets=True,all_weights_inputs_hooks_MLP_exact=True,projected_features_equal_same_question_historical=True,model_dependency_scorer_SHA256_exact=True,raw_logit_vectors_archived=False,compact_main_audit_bytes=sum(r['compact_audit_bytes'] for r in index),first_logit_sha_only=True,all_changed_token_ID_archives_match_checkpoints=True,smoke_on_each_GPU_native_noop_all_valid_logits_torch_equal=True,original_S_GPU_replay_exact=True,can_POS_from_existing_frozen_parse=True,anomalies=anomalies,analysis_source_sha256=sha256(__file__),release=release,utc=stamp());atomic_json(P/'validation.json',validation)
    atomic_json(P/'analysis_bundle.json',dict(summary=summary,pairs=pairs,mechanisms=mechanisms,tasks=tasks))
    report(p,plan,done,rows,summary,pairs,mechanisms,validation)
    atomic_json(P/'completion.json',dict(status='SEMANTIC_STOP_TO_KEEP_CAN_360_COMPLETE',questions=360,new_generations=94,new_label_reuse=266,baseline_history_reuse=1080,logical_labels=1440,validation_passed=True,seconds=done['seconds'],execution_commits=[read(P/f'worker{i}_completion.json')['code_commit'] for i in range(2)],full3000_started=False,additional_STOP_search_performed=False,utc=stamp()));(P/'.gitignore').write_text('*.lock\n',encoding='utf-8')
    print(json.dumps(dict(overall=[r for r in summary if r['group']=='all360'],primary=next(r for r in pairs if r['primary_comparison']),subgroups=mechanisms,anomalies=anomalies),ensure_ascii=False),flush=True)

def report(p,plan,done,rows,summary,pairs,mechanisms,val):
    by={(r['group'],r['condition']):r for r in summary};me={r['group']:r for r in mechanisms};new=by['all360',NEW];old=by['all360','Semantic-k'];pair=next(r for r in pairs if r['primary_comparison']);delta=pair['delta_pp'];budgetdelta=100*(new['mean_R_ratio']-old['mean_R_ratio']);text=[
        '# Semantic-k STOP两项微调：GQA360 Pilot','',f'**新条件{new["correct"]}/360，Accuracy={new["accuracy_percent"]:.3f}%；原S为319/360（88.611%）。** 相对S净{new["correct"]-319:+d}题，ΔACC={delta:+.3f}pp，图片簇配对95%CI [{pair["CI95_low"]:+.3f}, {pair["CI95_high"]:+.3f}]pp。纠正{pair["right_wrong_left_correct"]}题、损害{pair["right_correct_left_wrong"]}题。','',
        '## 两项规则与样本冻结','',
        '只把to加入STOP、can移出STOP；在独立groups_variant.py中保留原groups.py全部其他字节。16项CPU测试包含原12项及4项新增：STOP差集精确、其余函数与正则/种子逐字相同、合成完整词组可行性导致k=4→0及0→4。所有360题均重新运行原token_groups/select，完整natural输出与历史冻结记录完全一致；新版本随后按原算法自然得到新k，不做事后等预算修正。','',
        '样本逐字复用commit 74afff563d255f23b0859f19def2de64accff109的同一360题，六任务各60、360张唯一图片。原Prompt IDs、词组结构、正则词边界、词组顺序、图像与输入Hash均不变；to/can保留在Prompt中，唯一改变是Prefill R直接读Image Key的权限。没有正确性筛选或重新抽样。','',
        f'94题Writer改变、266题不变；266题均不含to/can，审计后精确复用原S。含to的92题、含can的3题，交集1题。94题全部发生k变化，2题还改变了其他词的最终选择。原平均k={old["mean_k"]:.6f}，新平均k={new["mean_k"]:.6f}；平均R预算{100*old["mean_R_ratio"]:.3f}%→{100*new["mean_R_ratio"]:.3f}%（{budgetdelta:+.3f}pp）。这是词表与预算共同改变的干预，不是固定k下to/can的独立位置贡献。','',
        '## 主要统计与六任务','',
        '主比较仅为新条件−原S，按Image SHA整簇配对Bootstrap20000次、seed20261104，Question等权，报告探索性95%CI；All/Q和子组为辅助描述。360题为内部Pilot，历史图片排除与任务均衡约束仍存在。Changed子集按输入权限变化定义，不按答案筛选；其效果不能外推为总体样本效果。','',
        '| 条件 | 正确/360 | ACC | 平均k | 平均k/R |','|---|---:|---:|---:|---:|']
    for c in NAMES:
        r=by['all360',c];text.append(f'| {c} | {r["correct"]}/360 | {r["accuracy_percent"]:.3f}% | {r["mean_k"]:.3f} | {100*r["mean_R_ratio"]:.3f}% |')
    text+=['','| 子集 | n | 原S正确 | 新正确 | 纠正 / 损害 |','|---|---:|---:|---:|---:|']
    for group in ['writers_changed','writers_unchanged','to_related','to_only','can_related','can_only','to_and_can','other_selection_changes']:
        if group in me:
            r=me[group];text.append(f'| {group} | {r["n"]} | {r["old_correct"]} | {r["new_correct"]} | {r["fix"]}/{r["harm"]} |')
    text+=['','子集可重叠，尤其to_related和can_related共享1题；不能把两组净变化相加。六任务所有条件ACC见task_accuracy.csv；各任务新−S及Changed子集的配对CI见paired_comparisons.csv。','',
        '## to规则与完整损害案例','',
        f'to相关92题共纠正{me["to_related"]["fix"]}、损害{me["to_related"]["harm"]}；不含can的91题共纠正{me["to_only"]["fix"]}、损害{me["to_only"]["harm"]}。这些是对选择规则整体干预的结果，同时包含k与可行预算变化，不能解释为单个to位置的单位因果贡献。全部to题、实际移除位置、答案和正负评分见to_cases.csv；token_changes.csv另列to/can实际位置、候选预算、最终新增/移除位置和其他词连带变化。','',
        '| 损害ID | 任务 | 完整问题 | Gold | 原S | 新条件 | old k→new k |','|---|---|---|---|---|---|']
    for r in rows:
        if r['harm_vs_S']:text.append(f'| {r["question_id"]} | {r["task"]} | {r["question"].replace("|","/")} | {r["gold"]} | {r["old_answer"]} | {r["new_answer"]} | {r["old_k"]}→{r["new_k"]} |')
    if not any(r['harm_vs_S'] for r in rows):text.append('| 本批无损害题 | | | | | | |')
    text+=['','## can：全部实际例子','',
        '词性来自既有冻结spaCy解析，仅作描述，不安装资源、不按词性重新决定can权限，也不人工依答案修正规则。下表包含全部3题，因此不会只展示修复成功例。','',
        '| ID | 完整问题 | 既有POS/依存 | Gold | 原S | 新条件 | k变化 | 实际can新增位置 |','|---|---|---|---|---|---|---|']
    for r in rows:
        if r['has_can']:
            desc='; '.join('/'.join(w['POS'])+' '+','.join(w['dep'])+' →'+','.join(w['head']) for w in json.loads(r['can_POS']));text.append(f'| {r["question_id"]} | {r["question"].replace("|","/")} | {desc} | {r["gold"]} | {r["old_answer"]} | {r["new_answer"]} | {r["old_k"]}→{r["new_k"]} | {r["can_added"]} |')
    seen={tag for r in rows for w in json.loads(r['can_POS']) for tag in w['POS']}
    text+=['','**解析歧义审计：上述3题的can实际均指容器名词（garbage can / soda can），既有解析器却全部标为AUX，这是解析标注错误，不能把这3题解释为助动词对照。** 该人工语法说明只用于描述，未修改STOP、Token集合、Prompt或任何生成。3道名词can题答案全部不变；本360题没有实际助动词can例子，不能评价助动词开放的效果，也未追加样本。can_cases.csv保留原POS及描述性用法两列。']
    text+=['','## 预算连带变化：不能单独归因','',
        '全部94个Changed题的新旧k均不同；word_group_changes.csv保存每个改变词组的原/新core和选中标记。下列2题在to/can以外还有接收位置改变，完整答案与所有新增/移除位置均列出。','']
    for r in rows:
        if r['other_positions_changed']:text += [f'- **{r["question_id"]}**：{r["question"]} Gold={r["gold"]}；S={r["old_answer"]}（{r["old_correct"]}），新={r["new_answer"]}（{r["new_correct"]}）；k {r["old_k"]}→{r["new_k"]}；其他新增={r["other_added"]}，其他移除={r["other_removed"]}。']
    text+=['','## 是否值得扩展到3000题','']
    if new['correct']<=319:text+=['本批没有净Accuracy收益，应报告负面/零净收益结果，不建议据此直接扩展到3000题。预算变小或个别案例修复也不能替代完整答案净质量评估。本轮未追加任何STOP搜索、单独to/can新条件或大规模生成。']
    else:text += ['本批出现净Accuracy收益，但这是词表和预算共同改变的内部Pilot结果；只有3道can题，不能确认某个词的普遍作用。可以把本次完整规则作为后续独立图像验证候选，是否扩展3000题需单独授权。本轮没有追加生成或搜索其他词表。']
    text+=['','## 执行与可验证交付','',
        f'独立分支codex/gqa360-semantic-stop-to-keep-can。sample_manifest SHA256=`{p["manifest_sha256"]}`；sample_plan SHA256=`{p["plan_sha256"]}`。GPU冻结执行commit `{read(P/"worker0_completion.json")["code_commit"]}`。模型revision `{p["model_revision"]}`；FP16/Eager/Transformers4.49.0，Greedy32，logits_to_keep=1；P−1/真实Decode在32层禁读Image Key，MLP/Residual/QKV/完整KV保持原生，每条件独立真实Cache。','',
        f'使用UUID `{p["gpus"][0]["uuid"]}` 与 `{p["gpus"][1]["uuid"]}`，各47条实际新增，分片按图像SHA且科学题互不重叠。每卡三道固定真实Smoke通过Native/No-op全部有效Logits严格torch.equal、原All/S答案Token和首Token完整向量严格复现、独立KV、Hook正常/异常清理与权重/输入不变；再对自身一个真正Changed的新Writer输入执行实际Mask与32层MLP Smoke，两卡全部通过后才放行正式采集。','',
        f'94条新增、266条新条件精确复用，另有1080条All/S/Q复用，共1440个标签、360题完整覆盖，无缺失。GPU采集墙钟{done["seconds"]:.1f}秒（含加载/Smoke/正式处理，不含CPU准备和统计）。输出异常标签数{len(val["anomalies"])}。全部原始JSON答案/Token、紧凑NPZ审计和SHA索引已保存。新主审计NPZ总计{val["compact_main_audit_bytes"]}字节，只含Mask审计、Token IDs和元数据；Logits仅保存哈希，未归档首Token或逐步完整向量。Smoke严格比较的完整向量仅在内存使用。不能据哈希声称可从本包重算完整Logit分布。','',
        '磁盘/data不足以做完整检出，本轮从一开始就在空间充足的/data1创建稀疏独立Worktree并冻结磁盘预检；共享历史源码只读、按SHA验证。首次CPU冻结后仅把Smoke输入改为每卡真正Changed题，并重新CPU冻结；两次360题Writer计划及token_changes哈希完全相同，旧CPU冻结保存在attempts，最终冻结前无GPU启动。','',
        'GPU释放：'+json.dumps({k:val['release'][k] for k in ['all_own_children_exited','selected_devices_no_compute_processes','selected_uuids']},ensure_ascii=False)+'。模型/依赖/评分器完整SHA复核通过，全部新主生成Mask禁止概率0、非目标精确，权重版本/输入/同题投影视觉特征保持不变，32MLP和独立KV/Hook检查通过。原STOP选择器、历史结果及main未修改；不声明加速或物理剪枝。执行顺序：run freeze→提交冻结→run driver→CPU analyze。本轮停止。','']
    (P/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
