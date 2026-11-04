"""Report both equally weighted per-question and pooled token-count budgets."""
from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parents[1]/'results/semantic_single_gpu_search3h'
base=list(csv.DictReader((P/'final_per_question.csv').open()));R={v['question_id']:int(v['R']) for v in base};total=sum(R.values())
rows=[]
for sub,label in [('phase6_fixed_compositions','Fixed-Q-Conditional-Assistant-15'),('phase7_fixed_refinements','Fixed-Q-Conditional-Assistant-18')]:
    vv=list(csv.DictReader((P/sub/'per_question.csv').open()));k={v['question_id']:len(json.loads(v['writers'])) for v in vv};assert set(k)==set(R)
    mean=sum(k[q]/R[q] for q in R)/3000;pooled=sum(k.values())/total
    rows.append(dict(condition=label,correct=sum(int(v['correct']) for v in vv),mean_per_question_k_over_R=mean,pooled_sum_k_over_sum_R=pooled,total_k=sum(k.values()),total_R=total,mean_under50=mean<.5,pooled_under50=pooled<.5))
with (P/'final_budget_definitions.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=rows[1];report=P/'final_report_zh.md';text=report.read_text();assert '若要求合并后的连接预算也低于50%' not in text
report.write_text(text+f"\n若要求合并后的连接预算也低于50%，阈值18规则符合：逐题平均{100*r['mean_per_question_k_over_R']:.3f}%，合并总k/总R为{100*r['pooled_sum_k_over_sum_R']:.3f}%，正确2596/3000（86.533%）。阈值15是原先逐题平均预算口径下的已测最优；不将其50.066%的合并连接预算写成低于50%。两种口径的完整分子、分母见final_budget_definitions.csv。\n",encoding='utf-8')
WT=P.parents[2];files=[p for p in P.rglob('*') if p.is_file()]+list(Path(__file__).parent.glob('*.py'))
with (P/'final_artifact_index.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','size','sha256']);w.writeheader()
    for p in sorted(files):
        if p.name!='final_artifact_index.csv':w.writerow(dict(path=str(p.relative_to(WT)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(rows))
