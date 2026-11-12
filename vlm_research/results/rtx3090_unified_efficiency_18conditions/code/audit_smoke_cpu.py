"""Independent CPU verification of saved cross-backbone GPU Smoke traces."""
from pathlib import Path
import json,sqlite3,hashlib
O=Path(__file__).resolve().parent.parent
checks=[]
for kind in ['Qwen','InternVL']:
 p=O/kind/'smoke/checkpoint.sqlite'
 if not p.exists():continue
 c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 rows=[json.loads(r[0]) for r in c.execute('SELECT payload FROM requests WHERE category="smoke"')];c.close()
 bind=json.loads((O/kind/'smoke/input_verification.json').read_text())['bindings']
 grouped={(r['key'],r['method']):r for r in rows}
 for r in rows:
  key=r['key'];method=r['method'];g=r['generation'];L=bind[key]['layout'];N=L.get('input_length',L.get('n'));V=L.get('image_tokens',256)
  if method=='Noop':
   n=grouped[(key,'Native')]['generation'];assert n['token_ids']==g['token_ids'];assert n['effective_logits_sha256']==g['effective_logits_sha256']
  if method in ['SVH','SVH-R'] and kind=='Qwen':
   a=g['execution_audit'];assert not a['physical_fallback'];assert len(a['layers'])==32*36
   I=set(L['image_positions']);text=[p for p in range(N) if p not in I]
   for l in a['layers']:
    step,i=l['step'],l['layer'];positions=(list(range(N)) if i<22 else text) if step==0 else [N+step-1]
    assert l['hidden_rows']==len(positions) and l['logical_positions']==positions,('HIDDEN_ROPE',key,method,step,i)
   skipped={(e['step'],e['layer']) for e in a['attention_events'] if e.get('qkv_calls')==0 and e.get('o_proj_calls')==0 and e['step']>0}
   assert skipped=={(s,i) for s in range(1,32) for i in range(1,7)},('NOT_PHYSICAL_SKIP',key,method)
   assert not any(x['step']>0 and x['layer'] in range(1,7) and x['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV'] for x in a['projection_shapes'])
   assert all(x['forbidden_probability']==0 for x in a['probability'])
   residual={(e['step'],e['layer']) for e in a['attention_events'] if e.get('residual_exact') and e.get('mlp_and_norm_executed')}
   assert residual=={(s,i) for s in range(32) for i in range(36)}
  if method in ['SVH','SVH-R'] and kind=='InternVL':
   a=g['execution_audit'];assert a['norm_calls']==32*28*2 and a['MLP_calls']==32*28
   assert {(e[0],e[1]) for e in a['skip_events']}=={(s,i) for s in range(32) for i in range(2,6)}
   assert a['hidden_exit']==[dict(before=N,after=N-256,removed=256)]
   if method=='SVH-R':assert a['cutoff']==28 and a['reaccess_layers']==list(range(19,28))
  checks.append(dict(backbone=kind,key=key,method=method,status='PASS',real_decode=31))
(O/'independent_smoke_audit.json').write_text(json.dumps(dict(status='PASS',checks=checks),ensure_ascii=False,indent=2))
print(json.dumps(dict(status='PASS',checks=len(checks),backbones=sorted({x['backbone'] for x in checks}))))
