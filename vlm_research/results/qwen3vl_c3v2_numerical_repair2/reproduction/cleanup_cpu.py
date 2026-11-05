import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from vlm_research.qwen3vl_c3v2_numerical_repair2.repair import Repair2Execution
from vlm_research.qwen3vl_c3v2_numerical_repair2.probe import Probe
from vlm_research.qwen3vl_c3v2_numerical_repair2.common import *
config=Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]})
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=torch.nn.Module();self.model.language_model=qm.Qwen3VLTextModel(config);self.lm_head=torch.nn.Linear(32,64)
model=Toy();layout=read(PREVIOUS/'sample_manifest.json')['input_bindings'][ORIGINAL]
counts=lambda:[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
before=counts();eager=qm.eager_attention_forward;rope=qm.apply_rotary_pos_emb;forward=model.model.language_model.forward.__func__;linear=[m.forward.__func__ for m in model.model.language_model.layers[22].mlp.modules() if isinstance(m,torch.nn.Linear)]
for fail in [False,True]:
 try:
  with Probe(model,layout,'C').context(),Repair2Execution(model,layout,'C',audit=True,variant='FP32-late+decode').context():
   if fail:raise RuntimeError('injected_cleanup_test')
 except RuntimeError as e:assert str(e)=='injected_cleanup_test'
 assert counts()==before and qm.eager_attention_forward is eager and qm.apply_rotary_pos_emb is rope and model.model.language_model.forward.__func__ is forward
 assert [m.forward.__func__ for m in model.model.language_model.layers[22].mlp.modules() if isinstance(m,torch.nn.Linear)]==linear
write(OUT/'cpu_exception_cleanup.json',dict(normal_exit_restored=True,injected_exception_exit_restored=True,projection_forwards_restored=True,hook_counts_restored=True,gpu_used=False,model='Tiny random CPU36-layer Qwen; context-only tests; no frozen-data inference'))
print('PASS real CPU context normal and exceptional cleanup including projection monkeypatches')
