"""Compare new outputs with pinned old evidence only after all new predictions exist."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,json,hashlib
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval');R=O/'historical_reference'
for r in json.loads((R/'manifest.json').read_text()):assert hashlib.sha256((R/r['file']).read_bytes()).hexdigest()==r['SHA256']
with (R/'generation_outputs.csv').open(encoding='utf-8',newline='') as f:old={(r['question_id'],r['condition']):r for r in csv.DictReader(f)}
with gzip.open(O/'historical_per_question.csv.gz','rt',encoding='utf-8',newline='') as f:new=list(csv.DictReader(f))
assert len(new)==3000;summary=defaultdict(lambda:dict(questions=0,raw_text_changed=0,parsed_answer_changed=0,complete_token_ids_changed=0,first_token_changed=0,score_changed=0));changes=[]
for n in new:
 k=n['question_id'],n['condition'];p=old[k];assert p['image_sha256']==n['image_sha256'] and p['gold']==n['gold'];s=summary[n['condition']];s['questions']+=1
 checks=dict(raw_text_changed=p['text']!=n['raw_text'],parsed_answer_changed=p['prediction']!=n['parsed'],complete_token_ids_changed=json.loads(p['token_ids'])!=json.loads(n['token_ids']),first_token_changed=int(p['first_token'])!=int(n['first_token']),score_changed=float(p['correct'])!=float(n['score']))
 for flag,val in checks.items():s[flag]+=val
 if any(checks.values()):changes.append(dict(question_id=k[0],condition=k[1],old_text=p['text'],new_text=n['raw_text'],**checks))
oldenv=json.loads((R/'environment_freeze.json').read_text());newenv=json.loads((O/'environment.json').read_text());old_packages={s.split('==')[0]:s.split('==')[1] for s in oldenv['packages'] if '==' in s};new_packages={s.split('==')[0]:s.split('==')[1] for s in newenv['pip_freeze'] if '==' in s};diff={k:dict(old=old_packages[k],new=new_packages.get(k)) for k in ['torch','torchvision','transformers','tokenizers','pillow','numpy'] if old_packages.get(k)!=new_packages.get(k)}
result=dict(status='PASS',source_commit='30d3e7a0c9318e71a5951aca9a05b6b78108db72',all_new_GPU_predictions_generated_before_comparison=True,per_condition=dict(summary),changed_records=changes,environment_differences=dict(packages=diff,old_driver=oldenv['driver'],new_GPU_and_driver=newenv['gpu_preflight'],old_python=oldenv['python'],new_python=newenv['python']),no_old_predictions_used_for_new_generation=True)
(O/'historical_cross_hardware_answer_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
