"""CPU-only integrated report and publication for both exploratory phases."""
from .common import *
from .pipeline import load
from gqa5h_receiver_search.analyze import bootstrap
from relay_source_causality.common import hardware

def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8')))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT.parent,text=True).strip()

def prepare_delivery():
    assert read(P/'completion.json')['status']=='COMPLETE' and read(P/'validation.json')['passed'];assert read(P/'gpu_release.json')['all_own_children_exited'] and remaining()>180
    parent=PHASE1;first=read(parent/'completion.json');second=read(P/'completion.json');a=rows(parent/'per_question.csv');b=rows(P/'per_question.csv');am={r['question_id']:r for r in a};assert len(am)==len(b)==3000
    for base,c in [(parent,first),(P,second)]:
        assert sum(int(x['new_generations']) for x in rows(base/'experiment_timeline.csv'))==c['new_unique_generations']==len(list((base/'checkpoints').glob('*/*.json')))
    d,m=load(second['winner']);changes=[]
    for r in b:
        q=r['question_id'];sp=spec(q);w=m.select(sp,d['config']);assert w==json.loads(r['writers']);assert int(r['correct'])==int(gqa_prediction(r['answer'])==gold(q));assert set(sp['historical_writers']['Semantic-k'])<=set(w)<=set(sp['R'])
        changes.append(dict(question_id=q,task=r['task'],question=sp['question'],k=len(w),R=len(sp['R']),R_ratio=len(w)/len(sp['R']),S_k=len(sp['historical_writers']['Semantic-k']),writers=json.dumps(w),groups=json.dumps([dict(words=g['words'],decoded=g.get('decoded',''),positions=g['positions']) for g in sp['groups'] if set(g['positions'])<=set(w)],ensure_ascii=False),answer=r['answer'],correct=r['correct'],phase1_answer=am[q]['answer'],phase1_correct=am[q]['correct']))
    writecsv(P/'token_selection_audit.csv',changes)
    atomic_json(P/'budget_distribution.json',{key:dict(mean=float(np.mean([x[key] for x in changes])),min=float(min(x[key] for x in changes)),median=float(np.median([x[key] for x in changes])),p95=float(np.percentile([x[key] for x in changes],95)),max=float(max(x[key] for x in changes))) for key in ['k','R','R_ratio','S_k']})
    diff=np.array([int(r['correct'])-int(am[r['question_id']]['correct']) for r in b]);lo,hi=bootstrap(diff,[r['image_sha256'] for r in b]);between=dict(comparison='phase2 minus phase1',delta_pp=float(100*diff.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(diff==-1)),fix=int(sum(diff==1)),scope='additional post-hoc exploratory contrast; not in five-comparison Holm family');atomic_json(P/'versus_phase1.json',between)
    summary=rows(P/'accuracy_summary.csv');old=rows(parent/'accuracy_summary.csv')[0];combined=[dict(x) for x in summary]+[old];writecsv(parent/'final_accuracy_summary.csv',combined)
    paired=rows(P/'paired_comparisons.csv');winner=summary[0];best=max((x for x in combined if float(x['mean_R_ratio'])<.5),key=lambda x:int(x['correct']));stages=[];timeline=[]
    for base,label in [(parent,'phase1'),(P,'phase2')]:
        stages.extend([dict(phase=label,**r) for r in rows(base/'stage_summary.csv')]);timeline.extend([dict(phase=label,**r) for r in rows(base/'experiment_timeline.csv')])
    writecsv(parent/'final_stage_summary.csv',stages);writecsv(parent/'final_experiment_timeline.csv',timeline)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});fig,ax=plt.subplots(figsize=(11,5));order=sorted(combined,key=lambda x:float(x['ACC']));ax.barh(range(len(order)),[float(x['ACC']) for x in order],color=['#267c70' if x['condition']==second['winner'] else '#8aa2bc' for x in order]);ax.set_yticks(range(len(order)),[f"{x['condition']} ({x['correct']}/3000)" for x in order]);ax.set_xlim(80,89);ax.set_xlabel('Complete-answer GQA accuracy (%)');ax.set_title('S-preserving exploration: original GQA3000, descriptive results');fig.tight_layout();fig.savefig(parent/'final_accuracy.png',dpi=220);fig.savefig(parent/'final_accuracy.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5));labels=[]
    for x in combined:ax.scatter(100*float(x['mean_R_ratio']),float(x['ACC']),s=45);labels.append(x['condition'])
    # Legend numbers avoid dense label overlap around 40-50% R.
    for i,x in enumerate(combined):ax.annotate(str(i+1),(100*float(x['mean_R_ratio']),float(x['ACC'])),xytext=(4,5),textcoords='offset points')
    ax.legend([f'{i+1}. {n}' for i,n in enumerate(labels)],loc='lower right',fontsize=8);ax.set_xlabel('Mean open R receiver ratio (%)');ax.set_ylabel('Complete-answer accuracy (%)');ax.set_xlim(20,110);ax.set_ylim(83,89);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(parent/'final_budget_accuracy.png',dpi=220);fig.savefig(parent/'final_budget_accuracy.pdf');plt.close(fig)
    lines=['# 单GPU三小时续研：基于原Semantic集合的视觉接收者搜索','',f"第二轮小样本筛选后冻结的方案为 **{second['winner']}**：**{second['correct']}/3000，{second['ACC']:.3f}%**，平均R预算 **{100*second['mean_R_ratio']:.3f}%**。本次所有完整3000题实测策略中，预算<50%的最高点估计为 **{best['condition']}，{best['correct']}/3000，{float(best['ACC']):.3f}%，R={100*float(best['mean_R_ratio']):.3f}%**。{'已达到' if second['ACC']>=87 else '仍未达到'}87%的点估计目标。",'','## 两轮搜索与冻结边界','','从原09:20:25 UTC启动时间起计同一3小时，不重置时钟；只使用物理GPU1。第一轮32个合格候选围绕S＋ASSISTANT补问题位置，依次60→180→360→600筛选后扩展得到2570/3000（85.667%，R44.852%），完整冻结于03832c68f7bb09adea0af55b82014344806131b4。原数据和这些答案均属于已看过的探索池，没有新的盲测。','','第一轮未超过Question-Only，因此在剩余同一预算中继续比较低成本模板接收组。第二轮35个候选全部保留原S，包括全部Question开放后追加原生指令组，以及按原输入文本结构使用Question与S＋ASSISTANT的混合。所有规则和Token集合在各阶段前冻结；不能按某题Gold或输出正确性挑条件。每轮只将小样本筛选后的一条全局策略扩展3000题；本轮总共两个完整条件，不把它们包装成一次确认性测试。','','第二轮最终配置：`'+json.dumps(read(P/'winner_freeze.json')['config'],sort_keys=True)+'`。完整位置见phase2/winner_freeze.json，词组、真实解码和预算见phase2/token_selection_audit.csv。Question包含其原生标点。模板Token取自原Prompt原生完整组，未插入、删除或改写任何文本。其余生成、评分、FP16/Eager、Transformers4.49.0、模型revision、Greedy32、logits_to_keep=1与历史一致。全部32层P−1与真实Decode禁读原Image Key；只有选定R Query允许读图。','','## 同3000题完整ACC','','|条件|正确/3000|ACC|平均R|','|---|---|---|---|']
    for x in combined:lines.append(f"|{x['condition']}|{x['correct']}|{float(x['ACC']):.3f}%|{100*float(x['mean_R_ratio']):.3f}%|")
    lines+=['','|最终方案−对照|ΔACC pp|95%CI pp|原对→错|原错→对|Holm p|','|---|---|---|---|---|---|']
    for x in paired:lines.append(f"|{x['baseline']}|{float(x['delta_pp']):+.3f}|[{float(x['CI95_low_pp']):+.3f}, {float(x['CI95_high_pp']):+.3f}]|{x['harm']}|{x['fix']}|{float(x['Holm_p']):.5f}|")
    lines+=['',f"第二轮相对第一轮净变化{int(diff.sum()):+d}题，Δ={between['delta_pp']:+.3f}pp，探索性95%CI [{lo:+.3f}, {hi:+.3f}]pp，损害{between['harm']}题、纠正{between['fix']}题。两轮实际预算不同，不能把差异全部解释为位置选择的净效果。",'','Image SHA256整簇配对Bootstrap 20,000次，seed20261104；3000题有2409张不同图片。最终五个比较的图片簇条件交换检验作Holm校正，CI为普通点位探索性区间。该校正不覆盖自适应搜索过程，两轮都不是新独立确认集；CI跨0不代表非劣效。','','## 小样本全部结果','','|轮次|阶段|候选|正确/总数|平均R|','|---|---|---|---|---|']
    for x in stages:lines.append(f"|{x['phase']}|{x['stage']}|{x['candidate_id']}|{x['correct']}/{x['n']}|{100*float(x['mean_R_ratio']):.3f}%|")
    lines+=['','## 审计与停止','','原S逐题保留，未训练或修改7B模型。每次模型启动均通过原生/No-op全部有效logits严格torch.equal、历史S/Q80/Q100精确复现、Mask实际禁止概率0及非目标连接不变、独立KV、32层MLP、权重/输入/Hook审计。第一轮复用路径多一级的CPU审计问题已保留现场并修复，未放宽数值检查，时钟未重置。继承日志中的BOTH_GPU标签只是文本，实际protocol只有GPU1，未使用第二张卡。','','全部负面案例、逐题真实答案、Token IDs、任务表现、原始Checkpoint和SHA见各轮目录。compact NPZ位于服务器资源目录，Git只保存路径及SHA。没有物理跳过Attention、KV剪枝或已证明的加速。平均R比例不是FLOPs/延迟收益。所有模型在生成完成后自动退出；发布前再次检查释放。','','第一轮middle排序公式使用原始绝对位置而非去掉图像前缀后的Question相对位置，不能作为严格的中心位置实验解释；代码与负面结果完整保留。第二轮并未继承这一排序。','','最终结果只说明本批探索池中的点估计和配对变化，不能宣布找到全局最优接收者或证明外部泛化。若最终策略仍被Question-Only同时在预算与ACC支配，应继续保留Question-Only作为已有强基线。完成本轮交付后停止，不自动新增数据或继续GPU搜索。','','![Final accuracy](final_accuracy.png)','','![Budget and accuracy](final_budget_accuracy.png)']
    lines+=['','第二轮60题大量并列，原先仅按较低预算淘汰会使多数模板方案止步于60题，因此在最终3000题尚未生成任何新记录（0条）的情况下修订覆盖：所有最高分并列者进入180b，再选8个到360b、3个到600b、1个到最终3000。原180/360/600路径及未使用的提前冻结记录保留于attempt0_audit和jobs中；最终生成方案只由修订后路径决定。此为公开的探索性计划修订，不冒称预注册确认。']
    (parent/'final_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Correct the inherited phase-one candidate description in the new, uncommitted phase-two report.
    text=(P/'report_zh.md').read_text(encoding='utf-8');text=text.replace('预先冻结的候选包括S基线、S＋ASSISTANT、上轮名词冠词增强、46%/49%目标预算下的整词补位排序及按Question结构选择补齐。','本轮预先冻结的候选为原S、S＋ASSISTANT、Question-Only、Question追加原生模板完整组，以及输入结构决定的混合策略；详见CPU_candidate_budget.csv。');(P/'report_zh.md').write_text(text,encoding='utf-8')
    atomic_json(parent/'final_completion.json',dict(status='COMPLETE',UTC=utc(),phase1_completion_sha256=sha(parent/'completion.json'),phase2_completion_sha256=sha(P/'completion.json'),phase2_validation_sha256=sha(P/'validation.json'),same_start_monotonic=read(P/'clock.json')['start_monotonic'],final_winner=second['winner'],final_correct=second['correct'],final_ACC=second['ACC'],final_mean_R=second['mean_R_ratio'],new_generations_total=first['new_unique_generations']+second['new_unique_generations'],wall_elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60,within_three_hours=remaining()>0))
    print(json.dumps(dict(report_ready=True,second=second,best=best,between=between)),flush=True)

def publish():
    branch='codex/semantic-single-gpu-search3h';assert git('branch','--show-current')==branch and git('rev-parse','HEAD')=='03832c68f7bb09adea0af55b82014344806131b4';assert remaining()>180
    assert read(PHASE1/'final_completion.json')['status']=='COMPLETE';mainref=git('rev-parse','main');assert mainref=='7970be981e3462560cb66d83e1014eaa1b1ebdae';state=hardware();assert 'GPU-b08489af-0a8a-eb48-55ba-8346733af417' not in state['compute_processes']
    lock=P/'worker0/worker.lock'
    if lock.exists():lock.rename(PRIVATE/'retired_worker.lock')
    atomic_json(PHASE1/'final_delivery_validation.json',dict(UTC=utc(),passed=True,one_GPU_only=True,all_own_workers_exited=True,current_resources=state,figures_reviewed=True,source_manifest={str(f.relative_to(ROOT.parent)):sha(f) for f in Path(__file__).parent.glob('*.py')}))
    files=[f for f in PHASE1.rglob('*') if f.is_file() and f!=PHASE1/'final_artifact_index.csv']+[f for f in Path(__file__).parent.glob('*.py')];index=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(files)];writecsv(PHASE1/'final_artifact_index.csv',index)
    paths=[str(f.relative_to(ROOT.parent)) for f in files]+[str((PHASE1/'final_artifact_index.csv').relative_to(ROOT.parent))];assert all(x.startswith(('vlm_research/results/semantic_single_gpu_search3h/','vlm_research/semantic_qrelay_search3h/')) for x in paths)
    arg=PRIVATE/'git_paths.nul';arg.write_bytes(b'\0'.join(x.encode() for x in paths)+b'\0');subprocess.run(['git','add','-f','--pathspec-from-file='+str(arg),'--pathspec-file-nul'],cwd=ROOT.parent,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=ROOT.parent,check=True)
    subprocess.run(['git','-c','gc.auto=0','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-qm','Complete second small-sample receiver search and final GQA3000 comparison'],cwd=ROOT.parent,check=True);head=git('rev-parse','HEAD');print('COMMITTED',head,flush=True);subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=ROOT.parent,check=True);assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==head and git('rev-parse','main')==mainref and not git('status','--porcelain')
    elapsed=time.monotonic()-read(P/'clock.json')['start_monotonic'];atomic_json(PRIVATE/'delivery_receipt.json',dict(UTC=utc(),commit=head,push_verified=True,working_tree_clean=True,main_unchanged=mainref,elapsed_seconds=elapsed,within_three_hours=elapsed<10800,report=f'https://github.com/reyuyu/ATE/blob/{head}/vlm_research/results/semantic_single_gpu_search3h/final_report_zh.md'));print(read(PRIVATE/'delivery_receipt.json'),flush=True)

if __name__=='__main__':
    import sys
    prepare_delivery() if len(sys.argv)==1 else publish()
