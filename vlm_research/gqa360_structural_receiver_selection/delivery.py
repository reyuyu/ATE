"""Independent selection/count reconciliation and delivery file hashes."""
import csv,json,hashlib
from pathlib import Path
from .run import P,NAMES,PAIRS,TASKS,read,sha256,atomic_json,csvout,stamp

def main():
    p=read(P/'protocol.json');plan=read(P/'sample_plan.json');val=read(P/'validation.json');done=read(P/'completion.json');assert val['passed'] and done['status']=='GQA360_STRUCTURAL_RECEIVER_COMPLETE'
    candidate=list(csv.DictReader((P/'candidate_audit.csv').open()));assert len(candidate)==3000
    selected=[];used=set()
    for task in TASKS:
        pool=[r for r in candidate if r['task']==task and r['token_feasible']=='True' and r['hard_excluded_image']=='False'];n=0
        for r in sorted(pool,key=lambda r:(int(r['preference_tier']),r['selection_sha256'])):
            expected=hashlib.sha256(f'structural-receiver-gqa360-v1|{task}|{r["question_id"]}|{r["image_sha256"]}'.encode()).hexdigest();assert r['selection_sha256']==expected
            if n==60:break
            if r['image_sha256'] in used:continue
            selected.append(r['question_id']);used.add(r['image_sha256']);n+=1
        assert n==60
    assert set(selected)==set(plan['cases']) and len(used)==360
    for v in plan['cases'].values():assert sha256(v['sample']['image_path'])==v['sample']['image_sha256']
    rows=list(csv.DictReader((P/'per_question.csv').open()));scores={r['question_id']:json.loads(r['correct_per_condition']) for r in rows};summary=list(csv.DictReader((P/'accuracy_summary.csv').open()));pairs=list(csv.DictReader((P/'paired_comparisons.csv').open()))
    assert set(scores)==set(selected)
    for r in summary:assert int(r['correct'])==sum(s[r['condition']] for s in scores.values())
    for r in pairs:
        ids=list(scores) if r['group']=='all360' else [q for q,v in plan['cases'].items() if v['sample']['task']==r['group']]
        harm=sum(scores[q][r['right']] and not scores[q][r['left']] for q in ids);fix=sum(not scores[q][r['right']] and scores[q][r['left']] for q in ids)
        assert harm==int(r['right_correct_left_wrong']) and fix==int(r['right_wrong_left_correct']) and abs(float(r['delta_pp'])-100*(fix-harm)/len(ids))<1e-10
    representatives=list(csv.DictReader((P/'representative_cases.csv').open()))
    for r in representatives:
        pool=[v for v in rows if v[r['category']]=='True'];assert r['question_id']==min(pool,key=lambda v:(v['selection_sha256'],v['question_id']))['question_id']
    # Report the observed negative result explicitly, rather than offer a generic candidate list.
    bundle=read(P/'analysis_bundle.json');lookup={r['condition']:r for r in bundle['summary']};assert bundle['pareto']==NAMES[:3]
    old=(P/'report_zh.md').read_text(encoding='utf-8');text=old
    line5=next(l for l in text.splitlines() if l.startswith('5. 本批点估计'))
    text=text.replace(line5,'5. 最好的预算取舍仍来自原有Question-Only：平均R预算42.20%时324/360（90.00%），与约49.00%预算的Q+Post相同，且高于同预算Q+B的323题。更低预算的原Semantic-k为25.95%、319/360；两种S+2策略均用32.75%预算但答对更少。所有四个新增条件在本批点估计上均被其原S或Q基线支配；非支配前沿仅为S、Q、All-Writer。这不是总体最优或非劣效证明。')
    line6=next(l for l in text.splitlines() if l.startswith('6. Question-conditioned'))
    text=text.replace(line6,'6. 没有足够证据支持Question-conditioned Receiver优于前置边界：Q+Post比Q+B多1题（+0.278pp），S+Tail比S+B多2题（+0.556pp），但两项普通95%及调整99%区间均跨0，且它们都没有超过各自原Q/S基线。后置位置在因果时序上可以读取完整问题，并不保证开放直接读图权限有净收益；Q+Post还混入词法Answer，不能将差异纯归因于边界。')
    line7=next(l for l in text.splitlines() if l.startswith('7. 可将样本前沿'))
    text=text.replace(line7,'7. 本批没有出现值得优先带入大规模验证的新结构策略，不建议仅因历史失败题恢复而扩展S+B。若之后获得独立授权，更合理的固定验证对象是Question-Only这一较好的现有预算折中，而不是继续搜索边界或尾部位置。任何更大规模、图像不重叠验证都未在本轮执行。')
    text=text.replace('全部3000候选及排除/未入选理由见candidate_audit.csv。','全部3000候选及排除/未入选理由见candidate_audit.csv。历史失败Probe图片的排除及任务均衡也会改变样本分布，因此本360题较高的历史基线ACC不能直接外推3000题。')
    text=text.replace('完成后CPU运行analyze。','完成后CPU运行analyze及delivery。')
    (P/'report_zh.md').write_text(text,encoding='utf-8');(P/'.gitignore').write_text('*.lock\n',encoding='utf-8')
    assert sha256(Path(__file__).with_name('analyze.py'))==val['analysis_source_sha256']
    audit=dict(passed=True,independent_selection_from_candidate_table_exact=True,all_selected_image_file_SHA256_exact=True,all_paired_discordances_and_accuracy_reconciled=True,all_representatives_follow_frozen_SHA_order=True,analysis_source_sha256=sha256(Path(__file__).with_name('analyze.py')),delivery_source_sha256=sha256(__file__),report_sha256=sha256(P/'report_zh.md'),raw_NPZ_arrays_in_git=False,utc=stamp())
    atomic_json(P/'delivery_audit.json',audit)
    files=[dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha256(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.suffix!='.lock' and f.name!='artifact_index.csv']
    csvout(P/'artifact_index.csv',files)
    print(json.dumps(dict(delivery=audit,artifact_files=len(files))),flush=True)

if __name__=='__main__':main()
