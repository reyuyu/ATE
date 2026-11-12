"""CPU-only structural tests, without changing actual BF16 GPU kernels."""
from .common import *
from .adapter import pad_text,ProjectionRepair
import torch,types
def main():
 torch.manual_seed(20261110);n=321;keep=torch.cat((torch.arange(15),torch.arange(271,321)));assert len(keep)==65
 x=torch.randn(1,65,32,dtype=torch.float64);z=pad_text(x,n,keep)
 assert torch.equal(z.index_select(1,keep),x)
 image=torch.arange(15,271);assert torch.count_nonzero(z.index_select(1,image))==0
 modules=types.SimpleNamespace(k_proj=torch.nn.Linear(32,8).double(),v_proj=torch.nn.Linear(32,8).double(),q_proj=torch.nn.Linear(32,32).double())
 layer=types.SimpleNamespace(self_attn=modules);model=types.SimpleNamespace(language_model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[layer]*28)))
 ready={"roles":{"n":n,"image":image.tolist()},"ids":torch.ones(1,n,dtype=torch.long)}
 k=modules.k_proj.forward;q=modules.q_proj.forward
 direct=modules.k_proj(x)
 try:
  with ProjectionRepair(model,ready,{20:["k_proj","v_proj"]}) as repair:
   out=modules.k_proj(x);assert torch.allclose(out,direct,atol=1e-12,rtol=0)
   assert torch.equal(modules.k_proj(torch.ones(1,1,32,dtype=torch.float64)),k(torch.ones(1,1,32,dtype=torch.float64)))
   assert modules.q_proj.forward==q and len(repair.events)==1
   assert torch.allclose(out,modules.k_proj(z).index_select(1,keep),atol=1e-12,rtol=0)
   raise ValueError("cleanup mock")
 except ValueError:pass
 assert modules.k_proj.forward==k and modules.q_proj.forward==q
 p=OUT/"must_not_overwrite_atomic_test.bin";atomic_bytes(p,b"old")
 try:atomic_bytes(p,b"new");raise AssertionError("clobber permitted")
 except FileExistsError:pass
 assert p.read_bytes()==b"old";p.unlink()
 assert not torch.cuda.is_initialized()
 writej(OUT/"cpu_tests.json",{"status":"PASS","text_values_positions_preserved":True,"all_image_scratch_rows_exact_zero":True,"same_module_weights":True,"Q_unchanged_and_Decode_not_padded":True,"exception_restores_module":True,"atomic_write_cannot_overwrite":True,"mock_float64_atol":1e-12,"GPU_gate_unchanged":True,"CUDA_initialized":False})
 print("CPU_TESTS_PASS",flush=True)
if __name__=="__main__":main()
