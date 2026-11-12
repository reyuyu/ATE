"""Exactly bounded candidate classes: two Lt configs, row13, one Triton tile."""
import ctypes,pathlib,torch,triton,triton.language as tl
from .common import PRIVATE
class LtPlan:
 lib=None
 def __init__(self,m,n,k,workspace):
  if LtPlan.lib is None:
   l=ctypes.CDLL(str(PRIVATE/'build/liblt_bridge.so'));l.lt_plan.argtypes=[ctypes.c_int]*3+[ctypes.c_size_t,ctypes.POINTER(ctypes.c_int)];l.lt_plan.restype=ctypes.c_void_p;l.lt_count.argtypes=[ctypes.c_void_p];l.lt_count.restype=ctypes.c_int;l.lt_select.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t];l.lt_attr.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_int)];l.lt_run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p];l.lt_destroy.argtypes=[ctypes.c_void_p];l.lt_version.restype=ctypes.c_uint64;LtPlan.lib=l
  self.workspace=workspace;status=ctypes.c_int();self.p=self.lib.lt_plan(m,n,k,workspace.numel(),ctypes.byref(status));self.shape=(m,n,k)
  if not self.p:raise RuntimeError(('LtPlan failed',status.value,self.shape))
 def count(self):return self.lib.lt_count(self.p)
 def select(self,dense,index):return self.lib.lt_select(self.p,dense.p,index,self.workspace.numel())
 def config(self):
  result={}
  for name,i in [('algo_id',0),('tile_id',1),('split_k',2),('reduction_scheme',3),('cta_swizzle',4),('stages_id',6)]:
   v=ctypes.c_int();status=self.lib.lt_attr(self.p,i,ctypes.byref(v));result[name]=v.value if status==0 else 'unavailable:'+str(status)
  return result
 def run(self,x,w):
  m,n,k=self.shape;assert x.is_contiguous() and w.is_contiguous() and list(x.shape)==[m,k] and list(w.shape)==[n,k]
  out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16);status=self.lib.lt_run(self.p,w.data_ptr(),x.data_ptr(),out.data_ptr(),self.workspace.data_ptr(),self.workspace.numel(),torch.cuda.current_stream().cuda_stream)
  if status:raise RuntimeError(('LtMatmul failed',status,self.config()))
  return out
 def close(self):
  if self.p:self.lib.lt_destroy(self.p);self.p=None

@triton.jit
def _fixed_dot(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
 r=tl.program_id(0)*BM+tl.arange(0,BM);c=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.float32)
 for block in range(0,tl.cdiv(K,BK)):
  k=block*BK+kk;a=tl.load(X+r[:,None]*K+k[None,:],mask=(r[:,None]<M)&(k[None,:]<K),other=0);b=tl.load(W+c[None,:]*K+k[:,None],mask=(c[None,:]<N)&(k[:,None]<K),other=0)
  acc=tl.dot(a,b,acc)
 tl.store(Y+r[:,None]*N+c[None,:],acc.to(tl.bfloat16),mask=(r[:,None]<M)&(c[None,:]<N))

def fixed_triton(x,w):
 m,k=x.shape;n=w.shape[0];assert x.is_contiguous() and w.is_contiguous();y=torch.empty((m,n),device=x.device,dtype=x.dtype);kernel=_fixed_dot[(triton.cdiv(m,32),triton.cdiv(n,64))](x,w,y,m,n,k,num_warps=4,num_stages=3);return y,kernel
def row13(x,w):return torch.cat([torch.nn.functional.linear(x[i:i+13],w) for i in range(0,x.shape[0],13)],dim=0)
