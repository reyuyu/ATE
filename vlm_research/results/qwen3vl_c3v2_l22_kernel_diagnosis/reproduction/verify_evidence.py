"""CPU audit of the checked-in evidence. Requires NumPy only; never opens a GPU."""
import pathlib,json,csv,hashlib
import numpy as np
O=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
index=json.loads((O/'artifact_index.json').read_text())
for r in index['files']:assert sha(O/r['path'])==r['sha256'],r['path']
a=np.load(O/'arrays/operator_excerpt_bf16_bits.npz')
for name in ['k_proj','v_proj','gate_proj','up_proj','down_proj']:assert np.array_equal(a[name+'__reference'],a[name+'__Lt_dense_heuristic_0'])
for r in json.loads((O/'arrays/operator_excerpt_index.json').read_text())['arrays'].items():assert hashlib.sha256(a[r[0]].tobytes()).hexdigest()==r[1]['sha256']
v=np.load(O/'arrays/full_valid_logits.npz');rows=list(csv.DictReader((O/'logits_comparison.csv').open()))
for left,right,comparison in [('F','C_final','F__C_L35'),('F','B','F__B'),('W22-functional','P_final','W22-functional__P_final')]:
 for step in [0,1]:
  x=v[f'rank0__{left}__step{step}'];y=v[f'rank0__{right}__step{step}'];row=next(r for r in rows if r['rank']=='0' and r['phase']=='final_fixed_L22_L35' and r['comparison']==comparison and r['step']==str(step));d=x-y
  # Torch/NumPy mean reductions may differ in their final float32 bits.
  # Check published aggregate within float32 reduction error; exact-vector checks remain bitwise.
  assert abs(float(np.abs(d).mean(dtype=np.float64))-float(row['mean_abs']))<1e-7
  assert float(np.abs(d).max())==float(row['max_abs'])
  assert np.argmax(x)==np.argmax(y)
for key in v.files:
 if key.startswith('rank0__'):assert np.array_equal(v[key],v[key.replace('rank0__','rank1__')])
assert json.loads((O/'completion.json').read_text())['status']=='PARTIAL_REPAIR'
print('PASS: artifact SHA, BF16 excerpts, complete valid-logit metrics and cross-GPU strict equality')
