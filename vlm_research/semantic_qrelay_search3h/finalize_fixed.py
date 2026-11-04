"""Independent CPU verification and final publication of untrained experiments."""
from . import refinements as r
from .common import *
from semantic_visual_writers.groups import intact
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
from relay_source_causality.common import hardware
import collections,shutil,inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F=PHASE1
PHASES=[('', 'Phase1-whatwhich'),('phase2','Phase2-length20'),('phase5_fixed_rules','Phase5-spatial'),('phase6_fixed_compositions','Fixed-Q-Conditional-Assistant-15'),('phase7_fixed_refinements','Fixed-Q-Conditional-Assistant-18'),('phase8_fixed_recheck','Phase8-spatial-length20'),('phase9_cached_fallbacks','Phase9-cached-fallback')]
WIN='Fixed-Q-Conditional-Assistant-15'
def loadcsv(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))
def stats(d,images):
    a,b=bootstrap(np.array(d),images);return float(100*np.mean(d)),float(a),float(b)
def main():
    assert not (F/'final_completion.json').exists();assert remaining()>900
    original=loadcsv(F/'per_question.csv');ids=[s['question_id'] for s in original];base={s['question_id']:s for s in original};assert len(set(ids))==3000
    images=[base[q]['image_sha256'] for q in ids];assert len(set(images))==2409
    fixed={name:{s['question_id']:s for s in loadcsv(F/sub/'per_question.csv')} for sub,name in PHASES}
    for vv in fixed.values():assert set(vv)==set(ids)
    cw=read(F/'phase6_fixed_compositions/winner_freeze.json');assert cw['config']==dict(base='Q',hit='QSA',rule='whatwhich|long15')
    arrays={};budgets={};answers={}
    for name,key,b in [('All-Writer','All',1.),('Question-Only','Q100',.4214296787812505),('Semantic-k','S',.25794606084791516),('Semantic-Q80','Q80',.3199690109938905),('S+Assistant','SA',.39408025407585623)]:
        arrays[name]=np.array([int(base[q][key+'_correct']) for q in ids]);answers[name]=[base[q][key+'_answer'] for q in ids];budgets[name]=b
        assert all(int(gqa_prediction(a)==gold(q))==int(v) for a,q,v in zip(answers[name],ids,arrays[name]))
    for sub,name in PHASES:
        arrays[name]=np.array([int(fixed[name][q]['correct']) for q in ids]);answers[name]=[fixed[name][q]['answer'] for q in ids];budgets[name]=np.mean([float(fixed[name][q]['R_ratio']) for q in ids]);assert budgets[name]<.5
        assert all(int(gqa_prediction(a)==gold(q))==int(v) for a,q,v in zip(answers[name],ids,arrays[name]))
    final=[];provenance=[];h=History();anomalies=collections.Counter()
    for j,q in enumerate(ids):
        sp=spec(q);w=cw['positions'][q];expected=sorted(set(sp['Q'])| (set(sp['groups'][-1]['positions']) if r.condition(q,'whatwhich|long15') else set()))
        assert w==expected and w==json.loads(fixed[WIN][q]['writers']) and set(sp['historical_writers']['Semantic-k'])<=set(w)<=set(sp['R']) and intact(w,sp['groups'])
        rr=r.record(q,w,h);g=rr['generation'];assert g['prediction']==gqa_prediction(g['text']) and g['text']==fixed[WIN][q]['answer'] and int(g['prediction']==gold(q))==arrays[WIN][j]
        assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['audit']['passed'] and max(g['audit']['forbidden_max_by_layer'])==0 and g['audit']['all_non_target_masks_exact']
        row=dict(question_id=q,task=base[q]['task'],image_sha256=images[j],question=sp['question'],gold=gold(q),condition=WIN,answer=g['text'],correct=int(arrays[WIN][j]),token_ids=json.dumps(g['token_ids']),writers=json.dumps(w),k=len(w),R=len(sp['R']),R_ratio=len(w)/len(sp['R']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],input_hashes=json.dumps(sp['input_hashes'],sort_keys=True))
        for label,short in [('All-Writer','All'),('Question-Only','Q'),('Semantic-k','S')]:
            bg=h.get(q,label)['generation'];row[short+'_answer']=bg['text'];row[short+'_correct']=int(bg['prediction']==gold(q));row[short+'_token_ids']=json.dumps(bg['token_ids']);row[short+'_first_flip']=int(bg['token_ids'][0]!=g['token_ids'][0]);row[short+'_answer_changed']=int(bg['token_ids']!=g['token_ids'])
        row['SA_answer']=base[q]['SA_answer'];row['SA_correct']=int(base[q]['SA_correct'])
        for k in ['eos','empty','truncated','repeated_trigram']:anomalies[k]+=int(g[k])
        source=rr.get('source',{});raw=rr.get('raw',{})
        if not raw and source.get('path'):raw=read(source['path']).get('raw',{})
        provenance.append(dict(question_id=q,writer_sha256=digest(w),checkpoint_source=json.dumps(source,sort_keys=True),raw_path=raw.get('path',''),raw_sha256=raw.get('sha256',source.get('raw_sha256','')),input_hashes=json.dumps(sp['input_hashes'],sort_keys=True)))
        final.append(row)
    h.close();writecsv(F/'final_per_question.csv',final);writecsv(F/'final_provenance.csv',provenance)
    summary=[]
    for name in arrays:
        acc,lo,hi=stats(arrays[name],images);summary.append(dict(condition=name,n=3000,images=2409,correct=int(arrays[name].sum()),ACC=acc,CI95_low=lo,CI95_high=hi,mean_R_ratio=float(budgets[name]),training=False,final_selected=name==WIN))
    writecsv(F/'final_accuracy_summary.csv',summary);paired=[]
    for name,short in [('All-Writer','All'),('Question-Only','Q'),('Semantic-k','S'),('S+Assistant','SA')]:
        d=arrays[WIN]-arrays[name];delta,lo,hi=stats(d,images);rows=[dict(correct=int(a),image_sha256=im,**{short+'_correct':int(b)}) for a,b,im in zip(arrays[WIN],arrays[name],images)]
        paired.append(dict(candidate=WIN,baseline=name,delta_pp=delta,CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(d==-1)),fix=int(sum(d==1)),cluster_signflip_p=cluster_p(rows,short)))
    prev=0
    for j,p in enumerate(sorted(paired,key=lambda x:x['cluster_signflip_p'])):prev=max(prev,min(1.,(len(paired)-j)*p['cluster_signflip_p']));p['Holm_p_4_final_contrasts']=prev
    writecsv(F/'final_paired_comparisons.csv',paired)
    writecsv(F/'final_harm_cases.csv',[v for v in final if any(v[k+'_correct'] and not v['correct'] for k in ['All','Q','S','SA'])],list(final[0]));writecsv(F/'final_recovered_cases.csv',[v for v in final if any(not v[k+'_correct'] and v['correct'] for k in ['All','Q','S','SA'])],list(final[0]))
    task=[]
    for t in TASKS:
        ii=[j for j,q in enumerate(ids) if base[q]['task']==t];assert len(ii)==500
        for name in arrays:task.append(dict(task=t,condition=name,n=500,correct=int(arrays[name][ii].sum()),ACC=100*float(arrays[name][ii].mean())))
    writecsv(F/'final_task_accuracy.csv',task)
    # Preserve all attempted regimes, including the explicitly ineligible CPU-fit pilot.
    timeline=[]
    for sub in ['', 'phase2','phase3_adaptive','phase4_policy','phase5_fixed_rules','phase6_fixed_compositions','phase7_fixed_refinements','phase8_fixed_recheck','phase9_cached_fallbacks']:
        pp=F/sub;status=read(pp/'completion.json') if (pp/'completion.json').exists() else read(pp/'STOP.json');timeline.append(dict(phase=sub or 'phase1',eligible_no_training=sub not in ['phase3_adaptive','phase4_policy'],status='COMPLETE' if (pp/'completion.json').exists() else 'STOPPED',checkpoints=len(list((pp/'checkpoints').glob('*/*.json'))),record=json.dumps(status,sort_keys=True)))
    writecsv(F/'final_experiment_timeline.csv',timeline)
    qpath=F/'qwen_fixed60';qdone=read(qpath/'completion.json');assert qdone['status']=='COMPLETE' and qdone['records']==240
    qr=loadcsv(qpath/'per_question.csv');qids=sorted({v['question_id'] for v in qr});assert set(qids)==set(read(F/'split_manifest.json')['dev_stages']['60'])
    qspec=read(qpath/'token_manifest.json');qindex=[];qflags=collections.Counter();seen=set()
    for v in qr:
        q=v['question_id'];cond=v['condition'];key=(q,cond);assert key not in seen;seen.add(key)
        cp=qpath/'checkpoints'/q/(cond+'.json');rec=read(cp);g=rec['generation'];sp=qspec[q]
        assert rec['input_hashes']==sp['layout']['input_tensor_hashes'] and json.loads(v['writers'])==sp['selection']['writers'][cond] and rec['correct']==int(gqa_prediction(g['text'])==v['gold'])==int(v['correct'])
        assert len(g['source_events'])==36*len(g['token_ids']) and all(e['blocked_probability_max']==0 and e['non_target_mask_exact'] for e in g['source_events'])
        assert sha(rec['raw_path'])==rec['raw_sha256']
        with np.load(rec['raw_path']) as z:assert np.isfinite(z['first_logits']).all() and int(z['first_logits'].argmax())==g['token_ids'][0]
        for k in ['eos','empty','truncated','repeated_trigram']:qflags[k]+=int(g[k])
        qindex.append(dict(path=str(cp.relative_to(ROOT.parent)),sha256=sha(cp),raw_path=rec['raw_path'],raw_sha256=rec['raw_sha256']))
    assert len(seen)==240;writecsv(qpath/'checkpoint_index.csv',qindex)
    atomic_json(qpath/'validation.json',dict(passed=True,all_240_records_rescored=True,all_raw_sha256_verified=True,all_first_top1_exact=True,all_mask_probabilities_zero=True,non_target_mask_exact=True,anomalies=dict(qflags),cpu_tests=read(qpath/'cpu_tests.json'),gpu_smoke=read(qpath/'gpu_smoke_validation.json')))
    qcomp=[]
    for b in ['All-Writer','Question-Only','Semantic-k']:
        qmap={(v['question_id'],v['condition']):v for v in qr};d=np.array([int(qmap[q,'Fixed-Q-Conditional-Assistant']['correct'])-int(qmap[q,b]['correct']) for q in qids]);dd,lo,hi=stats(d,[qmap[q,b]['image_sha256'] for q in qids]);qcomp.append(dict(baseline=b,delta_pp=dd,CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(d==-1)),fix=int(sum(d==1))))
    writecsv(qpath/'paired_comparisons.csv',qcomp)
    qtask=[]
    for t in TASKS:
        for name in {v['condition'] for v in qr}:
            vv=[v for v in qr if v['condition']==name and v['task']==t];qtask.append(dict(task=t,condition=name,n=len(vv),correct=sum(int(v['correct']) for v in vv),ACC=100*np.mean([int(v['correct']) for v in vv])))
    writecsv(qpath/'task_accuracy.csv',qtask)
    cross=[]
    for label,qname in [('All-Writer','All-Writer'),('Question-Only','Question-Only'),('Semantic-k','Semantic-k'),(WIN,'Fixed-Q-Conditional-Assistant')]:
        ix=[ids.index(q) for q in qids];qv=[v for v in qr if v['condition']==qname];cross.append(dict(condition=qname,n=60,llava_correct=int(arrays[label][ix].sum()),qwen_correct=sum(int(v['correct']) for v in qv),llava_ACC=100*float(arrays[label][ix].mean()),qwen_ACC=100*np.mean([int(v['correct']) for v in qv])))
    writecsv(qpath/'same_sample_cross_model.csv',cross)
    # The actual data are plotted as discrete observed points, with ordinary exploratory intervals.
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    show=['Semantic-k','S+Assistant','Question-Only','Phase5-spatial','Fixed-Q-Conditional-Assistant-18',WIN,'All-Writer'];labs=['Semantic-k','S + assistant','Question only','S+A / spatial Q+A','Q + conditional A (18)','Q + conditional A (15)','All writers']
    fig,ax=plt.subplots(figsize=(9.6,4.5));ss={x['condition']:x for x in summary};ys=[ss[n]['ACC'] for n in show];ax.bar(range(len(show)),ys,color=['#9ca3af']*4+['#60a5fa','#15803d','#475569']);ax.set_ylim(80,89);ax.set_ylabel('Complete-answer accuracy (%)');ax.set_xticks(range(len(show)),labs,rotation=22,ha='right')
    for i,n in enumerate(show):ax.text(i,ys[i]+.15,f"{ss[n]['correct']}/3000\n{100*budgets[n]:.2f}% R",ha='center',fontsize=8)
    ax.set_title('Fixed receiver rules — no training (all 3000 questions exploratory)');fig.tight_layout();fig.savefig(F/'final_accuracy.png',dpi=220);fig.savefig(F/'final_accuracy.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5))
    for i,(n,label) in enumerate(zip(show,labs)):
        v=ss[n];x=100*budgets[n];y=v['ACC'];ax.errorbar(x,y,yerr=[[y-v['CI95_low']],[v['CI95_high']-y]],fmt='o',capsize=3,label=f'{i+1}. {label}');ax.annotate(str(i+1),(x,y),xytext=(5,4),textcoords='offset points')
    ax.axvline(50,ls='--',color='gray',lw=1);ax.set(xlabel='Mean allowed R-to-image receiver ratio (%)',ylabel='Complete-answer accuracy (%)',title='Observed budget–accuracy points; ordinary 95% image-cluster CI');ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(F/'final_budget_accuracy.png',dpi=220);fig.savefig(F/'final_budget_accuracy.pdf');plt.close(fig)
    qsummary=loadcsv(qpath/'accuracy_summary.csv');fig,ax=plt.subplots(figsize=(7.5,4));ax.bar(range(4),[float(v['ACC']) for v in qsummary],color=['#475569','#9ca3af','#9ca3af','#15803d']);ax.set_xticks(range(4),['All writers','Question only','Semantic-k','Fixed rule']);ax.set_ylabel('Complete-answer accuracy (%)');ax.set_ylim(0,105)
    for i,v in enumerate(qsummary):ax.text(i,float(v['ACC'])+1,f"{v['correct']}/60\n{100*float(v['mean_R_ratio']):.1f}% R",ha='center',fontsize=9)
    ax.set_title('Qwen3-VL-8B fixed 60-question transfer check');fig.tight_layout();fig.savefig(qpath/'accuracy.png',dpi=220);fig.savefig(qpath/'accuracy.pdf');plt.close(fig)
    snapshot=hardware();assert 'GPU-b08489af-0a8a-eb48-55ba-8346733af417' not in snapshot['compute_processes']
    if not (F/'final_gpu_release.json').exists():atomic_json(F/'final_gpu_release.json',dict(UTC=utc(),physical_GPU=1,uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417',own_processes_exited=True,resources=snapshot))
    main_hash=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT.parent,text=True).strip();assert main_hash=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
    atomic_json(F/'final_validation.json',dict(passed=True,UTC=utc(),main_unchanged=main_hash,all3000_independently_rescored=True,images=2409,final_rule_exactly_reconstructed=True,S_preserved=True,whole_groups=True,mean_R_under50=bool(budgets[WIN]<.5),mask_probabilities_exact_zero=True,independent_KV=True,bootstrap=20000,seed=20261104,anomalies=dict(anomalies),qwen_smoke=read(qpath/'gpu_smoke_validation.json'),training_policy='Only fixed untrained rules eligible. Earlier phase3 CPU-fitted pilot preserved but excluded. Phase4 stopped before fitting. Later phases5-9 and Qwen contain no fitting.',history_raw_hashes_audited=len(h.files),source_sha256=sha(__file__)))
    pp={p['baseline']:p for p in paired};qq={v['condition']:v for v in qsummary};elapsed=(time.monotonic()-read(F/'clock.json')['start_monotonic'])/60
    lines=['# 单GPU三小时Receiver规则探索：最终交付', '',f'当前最优无训练规则在3000题上答对2598题（86.600%），平均R预算48.779%。相对Question-Only多23题，距离All-Writer仍少13题。另一较省预算规则（阈值18）为2596/3000、47.870% R。均为探索性结果，未达到或证明完整All-Writer无损。', '', '## 可复现固定规则', '', '所有Question Token均可在32层读取原始Image Key；若英文问题首词为what/which，或原始LLaVA Question Token数至少15，再允许原生Prompt中完整ASSISTANT词组的4个Subword读图。仅改变Hard Mask，不插入文字，不删除原S，不修改Prompt。P−1及真实Decode在全部32层仍禁止直接读取Image Key；其余Attention、Image Query、MLP、Residual和完整KV均原生。', '', 'LLaVA沿用固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager、Transformers4.49.0、Greedy32及logits_to_keep=1。实际逐题位置见final_per_question.csv及phase6_fixed_compositions/winner_freeze.json。50%约束是逐题比例的总体均值，不是每题上限；理论可访问连接预算不代表FLOPs、显存或延迟节省。', '', '## 全部无训练完整3000题结果', '', '|条件|正确/3000|ACC|平均R比例|','|---|---:|---:|---:|']
    lines += [f"|{v['condition']}|{v['correct']}|{v['ACC']:.3f}%|{100*v['mean_R_ratio']:.3f}%|" for v in summary]
    lines += ['', '## 配对损害与纠正', '', '最终规则减基线；单位pp。按2409个Image SHA整簇配对Bootstrap，20,000次，seed=20261104。', '', '|基线|ΔACC|普通95%CI|原对→错|原错→对|','|---|---:|---|---:|---:|']
    lines += [f"|{v['baseline']}|{v['delta_pp']:+.3f}|[{v['CI95_low_pp']:+.3f}, {v['CI95_high_pp']:+.3f}]|{v['harm']}|{v['fix']}|" for v in paired]
    lines += ['', 'final_paired_comparisons.csv额外提供四项终局比较的配对图片簇符号翻转检验及Holm校正。该校正不覆盖此前反复的候选搜索，不能将其当作独立确认性证据。所有3000题此前均已被研究；DEV、CHECK与其余图片的分组仅用于开发筛选组织，不是新的外部验证集。CI跨0不是非劣效证明，也不据此宣称严格无损。', '', '## 搜索过程与训练禁令', '', '原3小时从2026-11-04 09:20:25 UTC起算，始终未重置时钟；物理GPU1串行加载每次一份模型，其他GPU不占用。阶段1测试32个预算合格的固定S扩展规则；阶段2测试Question与原生模板/长度规则，曾因大面积并列扩展180题复查，并在任何新增最终3000生成前保留了方案修订记录。阶段5冻结文本类别谓词，阶段6冻结OR组合，阶段7冻结整数长度阈值与有限类别扩展。每轮先锁定候选及位置，再做小样本筛选与600题复查，冻结一个规则后覆盖完整3000题；相同writer集合经SHA审计复用，不重复生成。阶段7 top4含等价/高度重叠的规则，不能称为四个独立证据。', '', '用户后来明确禁止训练。此前阶段3已有CPU ridge/类型收缩试探，现完整保留但排除出候选排名；其中OOF并不构成干净独立确认。阶段4当时只在GPU收集前向答案，收到指令后立即终止，尚未执行CPU拟合，无拟合后的模型或最终策略。其冻结原模型前向记录若与后续writer集合完全一致，可在输入、Checkpoint和raw SHA核验后复用；这不使用学习出的权重。阶段5、6、7和Qwen复核全部只采用明写规则，没有训练、拟合分数或Router。', '', '阶段1名为middle的早期填充规则使用原始绝对位置归一化，不能视为严格的问题词组中心位置控制。全部尝试、失败和中止记录均保留，未改写历史结论。', '', '## Qwen固定60题迁移复核', '', '用户明确授权后新增。完全复用原DEV60，六任务各10题，不重新挑样。Qwen3-VL-8B-Instruct revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b；其历史冻结环境为BF16/Eager、Transformers4.57.6、36层、原生Qwen Prompt/Tokenizer/图像布局与DeepStack/mRoPE。全部36层P−1/Decode禁读图，R包含vision_end，以免额外视觉中继未受控制。', '', '迁移规则保持what/which及原LLaVA长度≥15这个输入判据；Receiver映射到Qwen实际Question Token和原生assistant角色词组，不把LLaVA的4个Subword位置生搬硬套。不同模型预算必须实测，不能假设Qwen也低于50%。', '', '|Qwen条件|正确/60|ACC|实际平均R比例|','|---|---:|---:|---:|']
    lines += [f"|{v['condition']}|{v['correct']}|{float(v['ACC']):.3f}%|{100*float(v['mean_R_ratio']):.3f}%|" for v in qsummary]
    lines += ['', 'Qwen结果与LLaVA3000主统计分开。qwen_fixed60/paired_comparisons.csv含小样本配对CI；same_sample_cross_model.csv含同60题两模型表现。仅60题、提示词/Tokenizer/精度/架构不同，不能解释为纯架构因果效应或普遍跨模型有效。Qwen加载第一次在任何Forward前因inference参数无版本计数而中止；已保存qwen_attempt0_load_audit，修正为在普通上下文加载、推理仍使用inference_mode后，重新CPU冻结并通过三题全有效Logits严格torch.equal、Mask概率0、独立KV、MLP36、DeepStack、位置与Hook检查。未放宽容差。', '', '## 校验与交付范围', '', f'终局LLaVA输出审计：EOS={anomalies["eos"]}/3000，空答案={anomalies["empty"]}，截断={anomalies["truncated"]}，重复3gram={anomalies["repeated_trigram"]}。完整答案重新评分、S包含性、整词、writer位置、输入哈希及复用来源逐题核验通过。首Token翻转、输出变化、所有相对All/Q/S/SA的损害和纠正均已保存；六任务各500题。', '', 'final_provenance.csv、各阶段checkpoint_index.csv及Checkpoint记录原始向量的服务器路径和SHA。大向量不在GitHub，不声称GitHub含这些数组。final_artifact_index.csv提供交付文件SHA。图表为离散实测值，不画虚构平滑线。final_gpu_release.json确认仅GPU1相关进程退出；main保持原Hash。', '', '结论：无需训练的Question接收者加条件式原生答案角色接收者，在这批已反复研究的数据上比仅Question更接近All-Writer；总体准确率86.60%，预算48.78%，仍存在58道All原对变错及45道相反纠正。该收益不是严格无损，不代表物理计算已减少。停止新增规则、训练、搜索和GPU实验；已保留全部负面结果。', '', f'本汇总生成时距原起点{elapsed:.2f}分钟；最终提交/推送及结束时间另见发布回执。']
    (F/'final_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(F/'final_completion.json',dict(status='COMPLETE',UTC=utc(),selected=WIN,correct=2598,ACC=86.6,mean_R_ratio=float(budgets[WIN]),LLaVA_n=3000,Qwen_n=60,Qwen_records=240,no_training_eligible_only=True,elapsed_minutes=elapsed,within_original_three_hours=remaining()>0,scope='Experiment, scoring, statistics, artifacts and audits complete; final commit and remote verification recorded in separate publication receipt.'))
    src=list(Path(__file__).parent.glob('*.py'));files=[p for p in F.rglob('*') if p.is_file()]+src
    writecsv(F/'final_artifact_index.csv',[dict(path=str(p.relative_to(ROOT.parent)),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files) if p.name!='final_artifact_index.csv'])
    print(json.dumps(dict(summary=summary,qwen=qsummary,elapsed=elapsed)),flush=True)
if __name__=='__main__':main()
