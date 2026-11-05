import unittest,torch
from .common import *
from .engine import layer_mask
class Tests(unittest.TestCase):
 def setUp(self):
  self.n=11;self.images=torch.tensor([2,3,4]);self.base=torch.full((1,1,11,11),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(1)
 def expected(self,c,pre,layer):
  m=self.base.clone() if pre else torch.zeros(1,1,1,12,dtype=torch.bfloat16)
  spec=SPECS[c];allow=(spec['p1'] if pre else spec['decode']) and layer not in SKIP
  if not allow:m[0,0,-1,2:5]=-torch.inf
  if pre and layer>=22 and not spec['late_writer']:m[0,0,5:10,2:5]=-torch.inf
  return m
 def test_all_conditions_all_layers_prefill(self):
  for c in ['F']+CONDITIONS:
   for layer in range(36):self.assertTrue(torch.equal(layer_mask(self.base,self.images,self.n,True,layer,c),self.expected(c,True,layer)),(c,layer))
 def test_all_conditions_all_layers_decode(self):
  base=torch.zeros(1,1,1,12,dtype=torch.bfloat16)
  for c in ['F']+CONDITIONS:
   for layer in range(36):self.assertTrue(torch.equal(layer_mask(base,self.images,self.n,False,layer,c),self.expected(c,False,layer)),(c,layer))
 def test_A_B_C_only_non_skip_visual_restoration(self):
  for c in ['A','B','C']:
   for layer in SKIP:
    for pre in [True,False]:self.assertTrue(torch.isneginf(self.expected(c,pre,layer)[...,-1,2:5]).all())
 def test_D_mask_equal_F_but_early_attention_restored(self):
  self.assertTrue(SPECS['D']['early']);self.assertFalse(SPECS['F']['early'])
  for pre in [True,False]:
   for layer in range(36):self.assertTrue(torch.equal(self.expected('F',pre,layer),self.expected('D',pre,layer)))
 def test_E_only_late_R_changes(self):
  for layer in range(36):
   a=self.expected('E',True,layer);b=self.expected('F',True,layer);changed=a!=b;expected=torch.zeros_like(changed)
   if layer>=22:expected[...,5:10,2:5]=True
   self.assertTrue(torch.equal(changed,expected));self.assertTrue(torch.equal(a[...,-1,:],b[...,-1,:]))
 def test_mask_does_not_mutate_native_base(self):
  before=self.base.clone()
  for c in ['F']+CONDITIONS:
   for l in range(36):layer_mask(self.base,self.images,self.n,True,l,c)
  self.assertTrue(torch.equal(self.base,before))
 def test_exact_manifest_and_no_image_cross_shard(self):
  m=read(OUT/'sample_manifest.json');self.assertEqual(sha(OUT/'sample_manifest.json'),sha(HIST/'sample_manifest.json'))
  groups={}
  for r in m['samples']:groups.setdefault(r['image_sha256'],set()).add(r['rank'])
  self.assertTrue(all(len(v)==1 for v in groups.values()));self.assertEqual(len(m['samples']),1000)
 def test_accuracy_runner_has_no_benchmark_api(self):
  text=(ROOT/'vlm_research/qwen_recovery/engine.py').read_text()
  for forbidden in ['torch.cuda.Event','perf_counter','elapsed_time','reset_peak_memory','flops=']:self.assertNotIn(forbidden,text)
if __name__=='__main__':unittest.main()
