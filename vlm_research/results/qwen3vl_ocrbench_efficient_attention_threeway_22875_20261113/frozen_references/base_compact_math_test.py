"""Independent CPU FP64 proof on actual compact/dense additive masks."""
import sys,json
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path[:0]=[str(O),str(L),str(L/'sources/qwen')]
import torch
from physical_engine import causal_bias
from run_common import atomic
torch.set_num_threads(1);torch.manual_seed(113)
out=[]
for n,lo,hi in [(19,3,10),(63,7,40),(91,9,72)]:
    I=torch.arange(lo,hi);T=torch.tensor([p for p in range(n) if not lo<=p<hi])
    q=torch.randn(1,4,n,8,dtype=torch.float64);k=torch.randn_like(q);v=torch.randn_like(q)
    early=causal_bias(n,torch.float64,'cpu');early[...,-1,I]=-torch.inf
    late=early.clone();R=torch.arange(hi,n-1);late[0,0,R[:,None],I[None,:]]=-torch.inf
    def att(q,k,v,m):return ((q@k.transpose(-2,-1)/8**.5)+m).softmax(-1)@v
    # Skip P attention while preserving every prefix query and key it can read.
    f=att(q,k,v,early);f[...,-1,:]=0
    c=torch.cat([att(q[...,:-1,:],k[...,:-1,:],v[...,:-1,:],early[...,:-1,:-1]),torch.zeros_like(q[...,-1:,:])],dim=-2)
    torch.testing.assert_close(c,f,atol=1e-12,rtol=1e-12)
    bypass=float((c-f).abs().max())
    f=att(q,k,v,late).index_select(-2,T)
    c=att(q.index_select(-2,T),k.index_select(-2,T),v.index_select(-2,T),causal_bias(len(T),torch.float64,'cpu'))
    torch.testing.assert_close(c,f,atol=1e-12,rtol=1e-12)
    compact=float((c-f).abs().max())
    dq=torch.randn(1,4,1,8,dtype=torch.float64);bias=torch.zeros(1,1,1,n,dtype=torch.float64);bias[...,I]=-torch.inf
    f=att(dq,k,v,bias);c=att(dq,k.index_select(-2,T),v.index_select(-2,T),torch.zeros(1,1,1,len(T),dtype=torch.float64))
    torch.testing.assert_close(c,f,atol=1e-12,rtol=1e-12)
    out.append(dict(n=n,physical_bypass_max_abs=bypass,hidden_exit_compact_attention_max_abs=compact,decode_compact_max_abs=float((c-f).abs().max())))
assert not torch.cuda.is_initialized()
atomic(O/'compact_mathematical_equivalence.json',dict(status='PASS',dtype='float64',cases=out,criterion='independent explicit softmax reference atol=rtol=1e-12',CUDA_initialized=False,does_not_replace_BF16_legacy_gate=True));print(json.dumps(out))
