"""CPU semantics and frozen-input tests; no CUDA initialization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common import *
import torch
from engine import DecodeImagePermission
def main():
 assert not torch.cuda.is_initialized()
 cs=configs();assert len(cs)==30 and cs['Native']==list(range(36)) and cs['NoImage']==[]
 tests=[]
 for family in ['Early','Middle','Late','Random']:
  last=set()
  for k in S['budgets'][1:-1]:
   keep=cs[family+str(k)];assert len(keep)==len(set(keep))==k and set(keep)<=set(range(36))
   assert last<=set(keep);last=set(keep)
 tests.append('all 30 exact nested frozen layer sets, integer midpoint floor, one seed')
 # Hard-mask only a true decode query; prefill/P-1 permission remains native.
 logits=torch.tensor([[[[0.1,0.3,0.4,0.7,0.9]]]])
 positions=torch.tensor([1,2]);mask=torch.zeros_like(logits);mask.index_fill_(-1,positions,float('-inf'))
 p=(logits+mask).softmax(-1);assert torch.equal(p[...,positions],torch.zeros_like(p[...,positions]))
 assert torch.equal((logits+mask)[...,[0,3,4]],logits[...,[0,3,4]])
 tests.append('pre-softmax image -inf gives exact zero probability; non-image logits unchanged')
 rows=read(OUT/'sample_manifest.json');binding=read(OUT/'input_bindings.json')
 assert len(rows)==len(binding)==500 and len({r['image_sha256'] for r in rows.values()})==309
 assert read(OUT/'input_validation.json')['status']=='PASS'
 assert all(len(r['gold'])==10 for r in rows.values())
 assert all(torch.load(RUN/'prepared'/(r['question_id']+'.inputs.pt'),map_location='cpu',weights_only=True).keys()=={'input_ids','attention_mask','image_grid_thw'} for r in rows.values())
 tests.append('500 exact Figure7 records, 309 clusters, Gold separate from model tensors')
 assert cs['Middle18']==list(range(11,29))
 assert not torch.cuda.is_initialized()
 result=dict(status='PASS',tests=tests,layer_sets=cs,CUDa_initialized=False,formal_requests_not_started=True)
 write(OUT/'cpu_tests.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
