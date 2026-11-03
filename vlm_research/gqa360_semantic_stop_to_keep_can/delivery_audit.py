"""Independent CPU cross-check of CSVs, frozen plans, checkpoints and delivery scope."""
import collections,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/gqa360_semantic_stop_to_keep_can'
NEW='Semantic-STOP-to-KEEP-can'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(n):return list(csv.DictReader((P/n).open(encoding='utf-8')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    qs=rows('per_question.csv');tc={r['question_id']:r for r in rows('token_changes.csv')};plan=read(P/'sample_plan.json');proto=read(P/'protocol.json')
    assert len(qs)==len(tc)==len(plan['cases'])==360
    assert len({r['question_id'] for r in qs})==len({r['image_sha256'] for r in qs})==360
    assert set(collections.Counter(r['task'] for r in qs).values())=={60}
    assert sha(P/'sample_plan.json')==proto['plan_sha256'] and sha(P/'sample_manifest.csv')==proto['manifest_sha256']
    for r in qs:
        q=r['question_id'];t=tc[q];v=plan['cases'][q]
        for name in ['old_k','new_k','R','old_writers','new_writers','other_added','other_removed','can_added','to_removed']:
            assert r[name]==t[name],(q,name)
        old=json.loads(r['old_writers']);new=json.loads(r['new_writers'])
        assert old==v['writers']['Semantic-k'] and new==v['writers'][NEW]
        assert len(old)==int(r['old_k']) and len(new)==int(r['new_k'])
        assert float(r['old_R_ratio'])==len(old)/int(r['R']) and float(r['new_R_ratio'])==len(new)/int(r['R'])
        assert int(r['harm_vs_S'])==int(r['old_correct']=='1' and r['new_correct']=='0')
        assert int(r['fix_vs_S'])==int(r['old_correct']=='0' and r['new_correct']=='1')
    for r in rows('checkpoint_index.csv'):
        assert sha(P/r['checkpoint'])==r['checkpoint_sha256'] and sha(Path(r['compact_audit_path']))==r['compact_audit_sha256']
    for name,field in [('harm_cases.csv','harm_vs_S'),('recovered_cases.csv','fix_vs_S')]:
        assert {r['question_id'] for r in rows(name)}=={r['question_id'] for r in qs if r[field]=='1'}
    totals={c:sum(int(r[k]) for r in qs) for c,k in [('All-Writer','All_correct'),('Semantic-k','old_correct'),('Question-Only','Q_correct'),(NEW,'new_correct')]}
    assert totals=={'All-Writer':327,'Semantic-k':319,'Question-Only':324,NEW:316}
    for r in rows('accuracy_summary.csv'):
        if r['group']=='all360':assert int(r['correct'])==totals[r['condition']]
    for r in rows('task_accuracy.csv'):
        k={'All-Writer':'All_correct','Semantic-k':'old_correct','Question-Only':'Q_correct',NEW:'new_correct'}[r['condition']]
        subset=[q for q in qs if q['task']==r['group']];assert len(subset)==int(r['n'])==60 and sum(int(q[k]) for q in subset)==int(r['correct'])
    # Independent scalar bootstrap from question rows, same unique-image ordering.
    ordered=sorted(qs,key=lambda r:r['image_sha256']);delta=np.array([int(r['new_correct'])-int(r['old_correct']) for r in ordered]);rng=np.random.default_rng(20261104);boots=[]
    for _ in range(40):boots.extend((100*delta[rng.integers(0,360,size=(500,360))].mean(axis=1)).tolist())
    ci=np.percentile(boots,[2.5,97.5]);pair=next(r for r in rows('paired_comparisons.csv') if r['primary_comparison']=='True');assert np.allclose(ci,[float(pair['CI95_low']),float(pair['CI95_high'])],rtol=0,atol=1e-12)
    assert sum(r['writer_changed']=='True' for r in qs)==94 and sum(r['reused']=='True' for r in qs)==266
    can=rows('can_cases.csv');assert len(can)==3 and all(r['can_POS_disagreement']=='True' and r['can_descriptive_use'].startswith('NOUN') for r in can)
    for i in range(2):
        cp=read(P/f'worker{i}/completion.json');assert cp['passed']
    val=read(P/'validation.json');rel=read(P/'gpu_release.json');assert val['passed'] and not val['anomalies'] and rel['all_own_children_exited'] and rel['selected_devices_no_compute_processes']
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT.parent,text=True).strip();assert branch=='codex/gqa360-semantic-stop-to-keep-can'
    diff=subprocess.check_output(['git','diff','--name-only',proto['base_commit'],'HEAD'],cwd=ROOT.parent,text=True).splitlines();prefixes=('vlm_research/gqa360_semantic_stop_to_keep_can/','vlm_research/results/gqa360_semantic_stop_to_keep_can/')
    assert all(n.startswith(prefixes) for n in diff)
    result=dict(passed=True,questions=360,unique_images=360,task_sizes=collections.Counter(r['task'] for r in qs),correct=totals,generated=94,reused_new=266,harm=4,recovered=1,bootstrap_repetitions=20000,independently_recomputed_CI95_pp=ci.tolist(),all_csv_plan_budget_checkpoint_hashes_consistent=True,frozen_plan_unchanged=True,all_changes_in_new_package_and_results=True,parser_disagreement_documented=True,GPU_released=True,audit_source_sha256=sha(Path(__file__)))
    (P/'delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=P/'report_zh.md';text=report.read_text(encoding='utf-8');marker='## 交付复核与存储说明'
    if marker not in text:
        text+='\n'+marker+'\n\n独立CPU交付复核从逐题CSV重新计数，并独立重算20,000次配对Bootstrap，与主统计一致；所有Checkpoint、Token预算、任务汇总与冻结计划一致，详见delivery_audit.json。\n\nCSV、JSON答案及原始Checkpoint提交Git；紧凑NPZ只保存在服务器`/data1/likai/ATE-resources/experiments/gqa360_semantic_stop_to_keep_can/`，路径及SHA见checkpoint_index.csv与smoke_audit_index.csv，不声称GitHub包含这些NPZ。\n\n初版CPU统计中的can解释曾机械沿用错误的AUX标注，现已改为明确区分原解析和实际容器名词用法；初版派生文件保存在attempts/analysis_before_can_POS_audit。该修订只更正描述，所有Token计划、GPU输出、Accuracy及预算统计均未变，也没有追加GPU运行。\n'
        report.write_text(text,encoding='utf-8')
    files=[dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.suffix!='.lock' and f.name!='artifact_index.csv']
    with (P/'artifact_index.csv').open('w',encoding='utf-8',newline='') as out:
        w=csv.DictWriter(out,fieldnames=['path','bytes','sha256'],lineterminator='\n');w.writeheader();w.writerows(files)
    result['indexed_artifacts']=len(files);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
