"""CPU scope tests use native causal masks and the frozen historical primitive."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from .engine import blocked_edges,apply_mask,blocked_layer
from .common import *
from gqa3000_token_visual_writing.engine import cpu_mask
def main():
 torch.set_num_threads(1);cases=[]
 for n in [583,611,649]:
  layout=dict(prompt_length=n,I=list(range(5,581)),R=list(range(581,n-1)))
  for past in [0,n,n+6]:
   q=1 if past else n;length=past+1 if past else n
   native=torch.zeros(1,1,q,length,dtype=torch.float16)
   if not past:native[0,0].masked_fill_(torch.ones(q,length,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
   for c in CONDITIONS:
    for l in range(32):
     changed,mask=apply_mask(native,layout,c,l,past)
     active=blocked_layer(c,l,bool(past));assert int(mask.sum())==576*active
     assert torch.equal(changed[0,0][~mask],native[0,0][~mask])
     assert torch.isneginf(changed[0,0][mask]).all()
     probs=changed[0,0].float().softmax(-1);assert (probs[mask]==0).all()
     assert torch.equal(apply_mask(native,layout,c,l,past,True)[0],native)
     if not active:assert changed is native
     if c=='B':assert torch.equal(changed,cpu_mask(layout,layout['R'],past)[1])
     if not past:assert not mask[:-1].any()
     cases.append(dict(prompt_length=n,past=past,condition=c,layer=l,blocked=int(mask.sum())))
 for k,expect in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:assert abs(score('White',['white']*k+['black']*(10-k))-expect)<1e-12
 assert not torch.cuda.is_initialized()
 write(OUT/'CPU_test_suite.json',dict(passed=True,UTC=utc(),scope_cases=len(cases),native_noop_exact=True,forbidden_probabilities_zero=True,old_B_primitive_equal=True,dynamic_positions=True,scorer_fixtures=True,no_GPU=True))
 print('CPU_TEST_PASS',len(cases),flush=True)
if __name__=='__main__':main()
