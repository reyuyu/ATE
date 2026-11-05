"""Offline CPU verification using only arrays and JSON in this result directory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib
import torch,numpy as np
torch.set_num_threads(1)
O=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
z=np.load(O/'arrays/effective_logits.npz')
assert hashlib.sha256((O/'arrays/effective_logits.npz').read_bytes()).hexdigest()==read(O/'arrays/effective_logits.json')['sha256']
for run in ['shard1','repair_shard0']:assert np.array_equal(z[run+'__N'],z[run+'__Noop'])
rows=read(O/'shard1/factor_isolation.json')+read(O/'repair_shard0/repair_validation.json')['comparisons']
for i,r in enumerate(rows):
 run='shard1' if i<8 else 'repair_shard0';a=torch.from_numpy(z[run+'__'+r['reference']][r['step']]);b=torch.from_numpy(z[run+'__'+r['candidate']][r['step']]);d=a-b
 assert d.abs().max().item()==r['max_abs'] and d.abs().mean().item()==r['mean_abs'] and d.square().mean().sqrt().item()==r['rmse']
 assert bool(a.argmax()==b.argmax())==r['top1_equal']
e=np.load(O/'arrays/first_divergence_evidence.npz');meta=read(O/'arrays/first_divergence_evidence.json');assert hashlib.sha256((O/'arrays/first_divergence_evidence.npz').read_bytes()).hexdigest()==meta['sha256']
def t(key):
 a=torch.from_numpy(e[key].copy());d=next(x for x in meta['arrays'] if x['key']==key);return a.view(torch.bfloat16) if d['dtype']=='torch.bfloat16' else a
for prefix,expected in [('L1_F_B','attention_AV_output'),('L22_W_P','k_proj'),('repaired_L22_F_C','k_proj')]:
 stages=[x['key'].split('__')[1] for x in meta['arrays'] if x['key'].startswith(prefix+'__') and x['key'].endswith('__reference')]
 first=next(s for s in stages if not torch.equal(t(prefix+'__'+s+'__reference'),t(prefix+'__'+s+'__candidate')));assert first==expected;print(prefix,first)
print('PASS: 12 paired rows exactly reproduced; Native/No-op strict; saved first-divergence tensors verified. Full C remains FAILED.')
