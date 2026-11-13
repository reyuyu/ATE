import sys,json,hashlib
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path[:0]=[str(O),str(L),str(L/'sources/qwen')]
import torch
from physical_engine import causal_bias
from run_common import atomic,sha
torch.set_num_threads(1)
cases=[]
for n,lo,hi in [(19,3,10),(63,7,40),(91,9,72)]:
 image=torch.arange(lo,hi);text=torch.tensor([i for i in range(n) if not lo<=i<hi])
 base=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(1)
 early=base.clone();early[...,-1,image]=-torch.inf
 late=early.clone();qi=torch.arange(hi,n-1);late[0,0,qi[:,None],image[None,:]]=-torch.inf
 for c in ['C','F']:
  actual=causal_bias(n,torch.bfloat16,'cpu');actual[...,-1,image]=-torch.inf
  for layer in range(36):
   if layer==22:
    if c=='C':actual=causal_bias(len(text),torch.bfloat16,'cpu')
    else:actual[0,0,qi[:,None],image[None,:]]=-torch.inf
   expected=late if layer>=22 else early
   if c=='C' and layer>=22:expected=expected.index_select(-2,text).index_select(-1,text)
   assert torch.equal(actual,expected),(n,c,layer)
   assert actual.stride(-2)%8==0
   if c=='C' and layer in range(1,7):assert torch.equal(actual[...,:-1,:-1],expected[...,:-1,:-1])
  cases.append(dict(n=n,condition=c,layers=36,all_additive_bias_values_exact=True))
assert not torch.cuda.is_initialized()
atomic(O/'physical_cpu_tests.json',dict(status='PASS',cases=cases,CUDA_initialized=False,sources={f:sha(O/f) for f in ['physical_engine.py','physical_cpu_test.py']}))
print(json.dumps(dict(status='PASS',cases=len(cases),layer_comparisons=len(cases)*36,CUDA_initialized=False)))
