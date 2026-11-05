import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib
import numpy as np,torch
torch.set_num_threads(1)
O=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
z=np.load(O/'arrays/effective_logits.npz');assert hashlib.sha256((O/'arrays/effective_logits.npz').read_bytes()).hexdigest()==read(O/'arrays/effective_logits.json')['sha256']
for run in ['phase1_rank0','phase2_rank0','phase2_rank1']:assert np.array_equal(z[run+'__N'],z[run+'__Noop'])
for c in ['N','Noop','F','decode-zero-key__B','decode-zero-key__C']:assert np.array_equal(z['phase2_rank0__'+c],z['phase2_rank1__'+c])
rows=read(O/'logits_comparison_full.json');assert len(rows)==26
for r in rows:
 a=torch.from_numpy(z['phase1_rank0__'+r['reference']][r['step']]);key=r['candidate'] if r['run']=='phase1_rank0' else r['variant']+'__'+r['candidate'];b=torch.from_numpy(z[r['run']+'__'+key][r['step']]);d=a-b
 assert d.abs().max().item()==r['max_abs'] and d.abs().mean().item()==r['mean_abs'] and d.square().mean().sqrt().item()==r['rmse']
 assert bool(a.argmax()==b.argmax())==r['top1_equal']
for step in [0,1]:assert np.array_equal(z['phase1_rank0__F'][step],z['phase2_rank0__decode-zero-key__B'][step])
e=np.load(O/'arrays/first_difference_tensors.npz');meta=read(O/'arrays/first_difference_tensors.json');assert hashlib.sha256((O/'arrays/first_difference_tensors.npz').read_bytes()).hexdigest()==meta['sha256']
def t(key):
 a=torch.from_numpy(e[key].copy());dt=next(r['dtype'] for r in meta['arrays'] if r['key']==key);return a.view(torch.bfloat16) if dt=='torch.bfloat16' else a
def pair(prefix,stage):return [t(prefix+'__'+stage+'__'+s) for s in ['reference','candidate']]
for stage in ['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k','scores_before_mask','scores_after_mask','attention_probabilities']:
 a,b=pair('before_decode_F_B','S1_L12_'+stage);assert torch.equal(a,b)
a,b=pair('before_decode_F_B','S1_L12_attention_AV_output');assert int((a!=b).sum())==1
for stage in ['layer_input','q_proj','k_proj','v_proj','scores_after_mask','attention_probabilities','attention_AV_output','layer_output']:
 a,b=pair('after_decode_F_B','S1_L12_'+stage);assert torch.equal(a,b)
for stage in ['layer_input','input_norm','q_proj']:
 a,b=pair('remaining_L22_F_C','S0_L22_'+stage);assert torch.equal(a,b)
a,b=pair('remaining_L22_F_C','S0_L22_k_proj');assert not torch.equal(a,b)
print('PASS: 26 paired rows exact; Native/No-op and cross-GPU equality; L12 AV first difference and local repair verified; L22 K remains different. PARTIAL_REPAIR.')
