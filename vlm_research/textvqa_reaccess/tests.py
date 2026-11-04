import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from .engine import apply_mask,blocked_layer
import torch
from textvqa_layer_recovery.engine import apply_mask as oldmask
torch.set_num_threads(1)
def main():
 count=0
 for n in [583,627]:
  layout=dict(prompt_length=n,I=list(range(5,581)),R=list(range(581,n-1)))
  for past in [0,n,n+7]:
   q=1 if past else n;length=past+1 if past else n;native=torch.zeros(1,1,q,length,dtype=torch.float16)
   if not past:native[0,0].masked_fill_(torch.ones(q,length,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
   for c in CONDITIONS:
    for l in range(32):
     new,edges=apply_mask(native,layout,c,l,past);assert int(edges.sum())==576*blocked_layer(c,l,bool(past));assert torch.equal(new[0,0][~edges],native[0,0][~edges]);assert torch.isneginf(new[0,0][edges]).all() and (new[0,0].float().softmax(-1)[edges]==0).all()
     assert torch.equal(apply_mask(native,layout,c,l,past,True)[0],native)
     if not past:assert not edges[:-1].any()
     if c in 'ABCDE':assert torch.equal(new,oldmask(native,layout,c,l,past)[0])
     if not blocked_layer(c,l,bool(past)):assert new is native
     count+=1
 d=DESIGN
 for a,b in zip(KS,KS[1:]):assert set(d['nested_sets'][str(a)])<=set(d['nested_sets'][str(b)])
 for c,v in d['conditions'].items():assert v['Pminus1']==sorted(set(v['Pminus1'])) and v['Decode']==sorted(set(v['Decode']))
 for n,expect in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:assert abs(score('White',['white']*n+['black']*(10-n))-expect)<1e-12
 write(OUT/'CPU_test_suite.json',dict(passed=True,UTC=utc(),scope_cases=count,all_historical_A_E_masks_exact=True,forbidden_probability_zero=True,dynamic_positions=True,nested_layers=True,scorer_fixtures=True,no_CUDA=not torch.cuda.is_initialized()))
 print('CPU_TEST_PASS',count,DESIGN['nested_sets'],flush=True)
if __name__=='__main__':main()
