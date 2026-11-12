from pathlib import Path
import json,csv,collections
O=Path(__file__).resolve().parent.parent
records={};rows=[]
for group in ['LLaVA','Qwen','InternVL']:
 for session in ['session1','session2']:
  p=(O/session if group=='LLaVA' else O/group/session)/'requests.jsonl'
  if not p.exists():continue
  with p.open() as f:
   for line in f:
    r=json.loads(line)
    if r['category']=='formal':records[(group,session,r['method'],r['key'])]=r
for group,session,method,key in records:
 if session!='session1' or (group,'session2',method,key) not in records:continue
 a=records[(group,'session1',method,key)];b=records[(group,'session2',method,key)]
 assert a['input_hashes']==b['input_hashes']
 rows.append(dict(backbone=group,method=method,key=key,input_hashes_equal=True,token_ids_equal=a['generation']['token_ids']==b['generation']['token_ids'],first_token_equal=a['generation']['token_ids'][0]==b['generation']['token_ids'][0]))
with (O/'session_token_consistency.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['backbone']);w.writeheader();w.writerows(rows)
summary=[]
for group,method in sorted({(r['backbone'],r['method']) for r in rows}):
 rr=[r for r in rows if (r['backbone'],r['method'])==(group,method)]
 summary.append(dict(backbone=group,method=method,paired_inputs=len(rr),full_tokens_equal=sum(r['token_ids_equal'] for r in rr),first_token_equal=sum(r['first_token_equal'] for r in rr)))
(O/'session_token_consistency.json').write_text(json.dumps(dict(status='AUDITED',conditions=summary),indent=2))
print(json.dumps(summary))
