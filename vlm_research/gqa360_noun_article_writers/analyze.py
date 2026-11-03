"""One new score, unchanged360, full checkpoint audit and paired comparisons."""
import csv,json
from .run import *

def main():
    p=verify();plan=read(P/'sample_plan.json');cases=plan['cases'];done=read(P/'gpu_completion.json');assert done['passed']
    for n,info in p['model_files'].items():assert sha256(n)==info['sha256']
    for n,h in {**p['transformers_sources'],**p['scorer_sources']}.items():assert sha256(n)==h
    rows=[];index=[];records={};anomalies=[];checked=0
    for q,v in cases.items():
        s=v['sample'];gg={}
        for c in NAMES:
            r=read(P/f'worker{v["worker"]}'/'records'/q/(c+'.json'));g=r['generation'];w=v['writers'][c]
            assert r['writers']==w==g['writer_positions'] and g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer']) and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and g['audit']['passed']
            if r['reused']:assert sha256(r['source']['path'])==r['source']['sha256'] and read(r['source']['path'])['generation']==g and sha256(r['historical_raw']['path'])==r['historical_raw']['sha256']
            else:
                assert c=='Noun+Article' and r['plan_sha256']==p['plan_sha256'] and r['input_hashes']==v['design']['input_hashes'] and r['projected_visual_sha256']==v['reference_visual_sha256'] and r['weights_unchanged'] and r['hook_cleanup'] and r['MLP_calls']==[g['length']]*32 and sha256(r['raw']['path'])==r['raw']['sha256']
                with np.load(r['raw']['path']) as z:
                    a=z['audit'];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
                    assert np.array_equal(z['first_logits'],z['valid_logits'][0]) and z['valid_logits'].argmax(-1).tolist()==g['token_ids'];m=json.loads(str(z['record_metadata']));assert m=={k:r[k] for k in m}
                checked+=1;cp=P/f'worker{v["worker"]}'/'checkpoints'/q/(digest(w)+'.json');index.append(dict(question_id=q,worker=v['worker'],checkpoint=str(cp.relative_to(P)),checkpoint_sha256=sha256(cp),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256'],raw_bytes=r['raw']['bytes']))
            gg[c]=g;records[q,c]=r
            if not g['eos'] or g['empty'] or g['truncated'] or g['repeated_trigram']:anomalies.append(dict(question_id=q,condition=c,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeat=g['repeated_trigram']))
        nd=gg['Noun+Article'];ss=gg['Semantic-k']
        rows.append(dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],ND_answer=nd['text'],ND_correct=nd['correct'],ND_token_ids=json.dumps(nd['token_ids']),All_answer=gg['All-Writer']['text'],S_answer=ss['text'],Q_answer=gg['Question-Only']['text'],SB_answer=gg['S+SPACE+NL']['text'],correct_per_condition=json.dumps({c:g['correct'] for c,g in gg.items()}),N=json.dumps(v['N']),D=json.dumps(v['D']),ND_writers=json.dumps(v['writers']['Noun+Article']),k=len(v['writers']['Noun+Article']),R=len(v['design']['R']),k_R=len(v['writers']['Noun+Article'])/len(v['design']['R']),harm_vs_S=int(ss['correct'] and not nd['correct']),fix_vs_S=int(not ss['correct'] and nd['correct']),answer_changed_vs_S=int(nd['text']!=ss['text']),first_token_flip_vs_S=int(nd['token_ids'][0]!=ss['token_ids'][0]),reused=records[q,'Noun+Article']['reused'],eos=nd['eos'],empty=nd['empty'],truncated=nd['truncated'],repeated_trigram=nd['repeated_trigram']))
    assert checked==350 and len(rows)==360 and len({r['image_sha256'] for r in rows})==360
    csvout(P/'per_question.csv',rows);csvout(P/'checkpoint_index.csv',index);fields=list(rows[0]);csvout(P/'harm_cases.csv',[r for r in rows if r['harm_vs_S']],fields);csvout(P/'recovered_cases.csv',[r for r in rows if r['fix_vs_S']],fields)
    summaries=[];pairs=[];tasks=[]
    for group,ids in [('all360',list(cases))]+[(t,[q for q,v in cases.items() if v['sample']['task']==t]) for t in TASKS]:
        images=sorted({cases[q]['sample']['image_sha256'] for q in ids});members=[[i for i,q in enumerate(ids) if cases[q]['sample']['image_sha256']==im] for im in images]
        mat=np.array([[records[q,c]['generation']['correct'] for c in NAMES] for q in ids]);sums=np.array([mat[m].sum(0) for m in members]);sizes=np.array([len(m) for m in members]);ix=np.random.default_rng(20261104).integers(0,len(images),size=(20000,len(images)));boots=np.empty((20000,len(NAMES)))
        for a in range(0,20000,500):j=ix[a:a+500];boots[a:a+500]=100*sums[j].sum(1)/sizes[j].sum(1)[:,None]
        for j,c in enumerate(NAMES):
            ci=np.percentile(boots[:,j],[2.5,97.5]);w=[cases[q]['writers'][c] for q in ids];r=dict(group=group,condition=c,n=len(ids),correct=int(mat[:,j].sum()),accuracy_percent=float(100*mat[:,j].mean()),CI95_low=ci[0],CI95_high=ci[1],mean_k=float(np.mean([len(x) for x in w])),mean_R_ratio=float(np.mean([len(cases[q]['writers'][c])/len(cases[q]['design']['R']) for q in ids])),reused=sum(records[q,c]['reused'] for q in ids))
            (summaries if group=='all360' else tasks).append(r)
        for j,c in enumerate(NAMES[:-1]):
            diff=boots[:,-1]-boots[:,j];ci=np.percentile(diff,[2.5,97.5]);pairs.append(dict(group=group,left='Noun+Article',right=c,n=len(ids),delta_pp=float(100*(mat[:,-1]-mat[:,j]).mean()),CI95_low=ci[0],CI95_high=ci[1],right_correct_left_wrong=int(((mat[:,j]==1)&(mat[:,-1]==0)).sum()),right_wrong_left_correct=int(((mat[:,j]==0)&(mat[:,-1]==1)).sum()),answers_changed=sum(records[q,c]['generation']['text']!=records[q,'Noun+Article']['generation']['text'] for q in ids),first_token_flips=sum(records[q,c]['generation']['token_ids'][0]!=records[q,'Noun+Article']['generation']['token_ids'][0] for q in ids)))
    csvout(P/'accuracy_summary.csv',summaries);csvout(P/'paired_comparisons.csv',pairs);csvout(P/'task_accuracy.csv',tasks)
    release=read(P/'gpu_release.json');assert release['all_own_children_exited'];smokes=[]
    for i in range(2):
        W=P/f'worker{i}';assert read(W/'smoke_validation.json')['passed'] and read(W/'structural_smoke_validation.json')['passed']
        for f in sorted((W/'smoke_cases').glob('*.json')):
            r=read(f)['raw'];assert sha256(r['path'])==r['sha256'];smokes.append(dict(worker=i,raw_path=r['path'],raw_sha256=r['sha256']))
        r=read(W/'structural_smoke_validation.json')['raw'];assert sha256(r['path'])==r['sha256'];smokes.append(dict(worker=i,raw_path=r['path'],raw_sha256=r['sha256']))
    csvout(P/'smoke_raw_index.csv',smokes)
    validation=dict(passed=True,questions=360,unchanged_parent_manifest_SHA256_exact=True,new_raw_records_verified=350,ND_history_reuse=10,total_history_reuse=1450,all_conditions_present=True,logical_records=1800,all_model_dependency_scorer_SHA256_exact=True,all_new_raw_argmax_ids_exact=True,all_forbidden_probabilities_zero=True,all_non_target_masks_exact=True,all_32_MLP=True,weights_inputs_hooks_exact=True,visual_features_equal_historical_same_question=True,smoke_both_GPUs_passed=True,anomalies=anomalies,analysis_source_sha256=sha256(__file__),bootstrap_repetitions=20000,bootstrap_seed=20261104,release=release,utc=stamp());atomic_json(P/'validation.json',validation)
    nd=summaries[-1];sd=next(r for r in pairs if r['group']=='all360' and r['right']=='Semantic-k');ndreuse=nd['reused']
    report=['# GQA360补测：Noun+Article（N∪D）','',f'**名词＋冠词：{nd["correct"]}/360，Accuracy={nd["accuracy_percent"]:.3f}%。** 平均Writer {nd["mean_k"]:.3f}个，平均R预算{100*nd["mean_R_ratio"]:.2f}%。相对S为{sd["delta_pp"]:+.3f}pp，纠正{sd["right_wrong_left_correct"]}题、损害{sd["right_correct_left_wrong"]}题，配对95%CI [{sd["CI95_low"]:+.3f}, {sd["CI95_high"]:+.3f}]pp。','',
        '样本逐字复用Structural Receiver的同一批360题、六任务各60、360张唯一图片，sample_manifest.csv SHA256与父实验完全一致，不重采样、不按输出选择。只新增Noun+Article = 冻结NOUN/PROPN与依存修饰冠词a/an/the的N∪D；不加Semantic或SPACE/NL，不重跑解析器或改词性规则。原固定版本解析器信息及N、D逐题位置均保存在sample_plan.json，历史字节绑定见history_byte_audit.json。',
        '',f'本批有{sum(bool(v["D"]) for v in cases.values())}道含冻结D位置；其余题N∪D自然等于N。全部360题保留，不能把本轮组合表现解释为冠词单独的贡献。','',
        '严格沿用LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager/Transformers4.49.0、原Prompt/图像/评分器、Greedy32与logits_to_keep=1。所有32层P−1/真实Decode禁读Image Key，仅所选R可直接读图；原生QKV/MLP/Residual/完整KV保持不变，每次独立真实KV。没有物理跳过或加速实验。','',
        f'350条新增ND生成，10条ND历史记录因同题Writer集合、输入、协议完全相同而审计复用；另复用1440条All/S/Q/S+B对照。ND历史相同集合可能具有其他原标签，以实际位置为准；全部标签与SHA见计划。双GPU采集含加载/三道Smoke/ND Smoke墙钟{done["seconds"]:.1f}秒。两卡各自HF/No-op全部有效Logits严格torch.equal、历史All/S Token与首Token向量严格复现、实际Mask/独立Cache/32MLP/权重/输入/Hook清理检查通过。350条新增首Token及有效Logits逐步argmax、禁止概率0与非目标Mask精确、投影视觉Hash等于同题历史S+B记录，均复核通过。输出异常{len(anomalies)}条。','',
        '为避免/data空间不足，独立分支使用/data1稀疏Worktree，只读引用原工作区的冻结源码与历史文件。最初完整检出失败发生在任何GPU操作之前，未覆盖或清理其他人的文件。新NPZ原始数值保存在/data1/likai/ATE-resources/experiments/gqa360_noun_article_writers，索引保存SHA256与路径，Git保存JSON检查点及日志；不声称GitHub包含NPZ。','',
        '## 同一360题得分','','| 条件 | 正确/360 | ACC | 平均R预算 | 来源 |','|---|---:|---:|---:|---|']
    for r in summaries:report.append(f'| {r["condition"]} | {r["correct"]}/360 | {r["accuracy_percent"]:.3f}% | {100*r["mean_R_ratio"]:.2f}% | {"350新增+10精确复用" if r["condition"]=="Noun+Article" else "历史复用"} |')
    report+=['','## 配对比较','','图片SHA整簇配对Bootstrap20000次，seed20261104，Question等权。下列95%区间为此次补测的探索性、未作多重校正；不作为确认性检验或非劣效证明。六任务完整结果见task_accuracy.csv。Native Full本轮未执行，All-Writer保留答案阶段禁读图协议，不能相互替代。','','| ND−对照 | pp | 95%CI | 对照正确→ND错 / 对照错误→ND对 |','|---|---:|---|---:|']
    for r in pairs:
        if r['group']=='all360':report.append(f'| {r["right"]} | {r["delta_pp"]:+.3f} | [{r["CI95_low"]:+.3f}, {r["CI95_high"]:+.3f}] | {r["right_correct_left_wrong"]}/{r["right_wrong_left_correct"]} |')
    report+=['','本次只回答N∪D组合在同一360题上的功能表现，不证明名词或冠词各自的独立因果贡献，也不能外推3000题总体ACC。原360题有任务均衡与历史图片排除约束，属于内部探索样本。全部原始答案、Token、N/D/ND位置及正负翻转已保留。GPU释放状态见gpu_release.json；main及历史Structural结果未改。本轮到此停止。',''];(P/'report_zh.md').write_text('\n'.join(report),encoding='utf-8')
    atomic_json(P/'completion.json',dict(status='NOUN_ARTICLE_360_COMPLETE',questions=360,new_generations=350,ND_history_reuse=10,total_history_reuse=1450,logical_labels=1800,validation_passed=True,seconds=done['seconds'],code_commits=[read(P/f'worker{i}_completion.json')['code_commit'] for i in range(2)],utc=stamp()))
    (P/'.gitignore').write_text('*.lock\n',encoding='utf-8');print(json.dumps(dict(ND=nd,pairs=[r for r in pairs if r['group']=='all360'],anomalies=anomalies)),flush=True)

if __name__=='__main__':main()
