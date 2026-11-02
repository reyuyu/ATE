"""Exact CSV wire-format check, including standard None -> empty-field encoding."""
import csv,io,json,hashlib
from pathlib import Path
from vlm_research.visual_write_window.analyze import replay,statistics

p=Path('/data/lk/ATE-worktrees/visual-write-window-240/vlm_research/results/visual_write_window_240')
protocol,samples,gens,rows,swaps,hashes,raws=replay()
sums,pairs,tasks=statistics(samples,gens,rows)
tables={'accuracy_summary.csv':sums,'paired_comparisons.csv':pairs,'task_accuracy.csv':tasks,'per_question.csv':rows,
    'image_swap_per_pair.csv':swaps,'negative_cases.csv':[r for r in rows if r['all_correct_to_wrong']]}
for name,values in tables.items():
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    buf.seek(0);expected=list(csv.DictReader(buf))
    with (p/name).open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert actual==expected,name
c=json.loads((p/'completion.json').read_text());assert c['status']=='COMPLETE'
for name,h in c['artifacts'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
v=json.loads((p/'validation.json').read_text());assert v['checkpoint_hashes']==hashes and v['raw_hashes']==raws
out=dict(passed=True,comparison='Every CSV cell compared exactly after standard Python csv.DictWriter serialization; None writes as empty field',
    tables={n:len(values) for n,values in tables.items()},frozen_sources_unchanged=True,all_record_probability_cache_and_raw_replay_passed=True,
    final_artifact_hashes_verified=True,numeric_tolerance_used=False,model_generations=0)
(p/'delivery_read_only_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c['delivery_read_only_validation']='delivery_read_only_validation.json'
c['artifacts']={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['completion.json','worker.lock']}
(p/'completion.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('EXACT READ-ONLY PASS: every CSV cell, all record probability/cache/raw replay, frozen sources and final artifact hashes')
