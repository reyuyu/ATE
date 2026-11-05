import torch,json,pathlib
from types import SimpleNamespace
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from vlm_research.qwen3vl_c3v2_numerical_debug.repair import RepairExecution
from vlm_research.qwen3vl_c3v2_numerical_debug.probe import Probe
from vlm_research.qwen3vl_c3v2_numerical_debug.common import OUT,ORIGINAL,read,write
config=Qwen3VLTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]})
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=torch.nn.Module();self.model.language_model=Qwen3VLTextModel(config);self.lm_head=torch.nn.Linear(32,64)
model=Toy();layout=read(OUT/'sample_manifest.json')['input_bindings'][ORIGINAL]
def counts():return [(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
before=counts();eager=qm.eager_attention_forward;rope=qm.apply_rotary_pos_emb;forward=model.model.language_model.forward.__func__
for failure in [False,True]:
 try:
  with Probe(model,layout,'C').context(),RepairExecution(model,layout,'C',audit=True).context():
   if failure:raise RuntimeError('intentional_cpu_exception')
 except RuntimeError as err:assert str(err)=='intentional_cpu_exception'
 assert counts()==before and qm.eager_attention_forward is eager and qm.apply_rotary_pos_emb is rope and model.model.language_model.forward.__func__ is forward
write(OUT/'cpu_exception_cleanup.json',dict(normal_exit_hooks_and_patches_restored=True,exception_exit_hooks_and_patches_restored=True,model='Tiny randomly initialized CPU-only 36-layer Qwen text model; no model/data inference; real hook/patch context paths',gpu_used=False))
with (OUT/'cpu_tests.txt').open('a') as f:f.write('\nAdditional real-context CPU cleanup: normal and injected-exception paths PASS; no GPU used.\n')
print('CPU normal/exception cleanup PASS')
