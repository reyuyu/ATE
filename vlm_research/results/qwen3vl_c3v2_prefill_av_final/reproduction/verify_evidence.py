"""Pure CPU check of the delivered full-vocabulary evidence; NumPy only."""
import pathlib,json,csv,hashlib
import numpy as np
O=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(2**20),b''):h.update(c)
 return h.hexdigest()
for r in json.loads((O/'artifact_index.json').read_text())['files']:assert sha(O/r['path'])==r['sha256'],r['path']
z=np.load(O/'arrays/full_valid_logits_bf16_bits.npz');index=json.loads((O/'arrays/logits_index.json').read_text())
for k in z.files:assert hashlib.sha256(z[k].tobytes()).hexdigest()==index['arrays'][k]['sha256']
def get(s,c,t):return (z[s+'__'+c+'__'+str(t)].view(np.uint16).astype(np.uint32)<<16).view(np.float32)
rows=list(csv.DictReader((O/'logits_comparison.csv').open()));mapping={'F__B':('F','B'),'F__C_before':('F','C_before'),'F__C_after':('F','C_after'),'W22-functional__P_after':('W22-functional','P_after')}
for r in rows:
 if r['rank']!='0':continue
 s=r['sample'];a,b=mapping[r['comparison']];step=int(r['step']);x,y=get(s,a,step),get(s,b,step);d=np.abs(x-y);assert float(d.max())==float(r['max_abs']);assert abs(float(d.mean(dtype=np.float64))-float(r['mean_abs']))<1e-7
 # The float64 mean check only reconciles CPU aggregate-reduction bits, never model tolerances.
 assert np.argmax(x)==np.argmax(y)
for t in [0,1]:assert np.array_equal(get('GQA__06905471','F',t),get('GQA__06905471','C_after',t))
for sid in ['GQA__06905471','GQA__02497673']:
 for t in [0,1]:assert np.array_equal(get(sid,'Native',t),get(sid,'Noop',t))
assert json.loads((O/'completion.json').read_text())['status']=='PARTIAL_REPAIR'
print('PASS: all delivery SHA, full-vocabulary metrics, original F/C strict equality and Native/Noop evidence')
