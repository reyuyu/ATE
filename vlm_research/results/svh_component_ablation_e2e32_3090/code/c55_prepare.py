"""CPU-only freeze of component-efficiency protocol and the existing30 inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from c55_common import *
import ast,inspect,shutil,collections
import torch
from transformers import CLIPImageProcessor
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
import c55_runtime as runtime
def main():
 assert not (N/'protocol.json').exists(),'Never overwrite an experiment protocol'
 verify_sources();samples=lines(OLD/'generation_length/input_manifest.jsonl.gz');mapping=read(OLD/'generation_length/input_path_mapping.json')
 assert len(samples)==30 and collections.Counter(s['benchmark'] for s in samples)=={'GQA':15,'TextVQA':15}
 processor=CLIPImageProcessor.from_pretrained(M,local_files_only=True);bindings=[];inputs=[];gold={}
 for s in samples:
  assert sha(mapping[s['key']])==s['image_sha256']
  raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(mapping[s['key']],processor).half())
  assert {k:tensor_hash(v) for k,v in raw.items()}==s['input_hashes']
  item=dict(s);gold[s['key']]=item.pop('answer');inputs.append(item)
  bindings.append(dict(key=s['key'],input_hashes=s['input_hashes'],image_sha256=s['image_sha256'],frozen_prompt_sha256=s['prompt_sha256']))
 assert runtime.NoSelectiveTransfer.forward is runtime.ORIGINAL_FACTORY.forward
 assert runtime.NoAttentionBypass.forward is runtime.ORIGINAL_FACTORY.forward
 tree=ast.parse((N/'code/c55_runtime.py').read_text());patched=[]
 for x in ast.walk(tree):
  if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='patch' and x.func.attr=='object':patched.append(x.args[1].value)
 assert set(patched)=={'MiddleWriter','kv_snapshot','Observer'}
 assert runtime.ref.generate is runtime.ORIGINAL_GENERATE
 assert not torch.cuda.is_initialized()
 writegz(N/'input_manifest.jsonl.gz',inputs);save(N/'answers.json',gold);save(N/'input_path_mapping.json',mapping)
 shutil.copyfile(OLD/'requirements.reference.txt',N/'requirements.reference.txt')
 save(N/'protocol.json',dict(status='FROZEN',created_ns=time.time_ns(),GPU_UUID=UUID,model='llava-hf/llava-1.5-7b-hf',revision=REV,precision='FP16',attention='eager',batch=1,greedy=True,conditions=CONDITIONS,length=32,ignore_eos=True,use_cache=True,logits_to_keep=1,real_decode_calls=31,sessions=2,formal_per_session=120,total_formal=240,inputs=dict(GQA=15,TextVQA_Paper_OCR=15,source=str(OLD/'generation_length/input_manifest.jsonl.gz'),source_sha256=sha(OLD/'generation_length/input_manifest.jsonl.gz')),quality_scores_source=str(OLD/'ablation/summary.csv'),quality_not_rerun=True,warmup=dict(Native32=40,each_condition=2,baseline_each_of_two_probes=10),calibration=dict(every_inputs=5,each_of_two_probes=3,windows_per_session=6,threshold=.02,action='WARN only; no resets/retries/deletions'),schedule='Per input cyclic four-condition rotation by input_index%4, same frozen schedule in both independent model-load sessions',timing='Original frozen e2e32_runtime.generate function object; controller init included; after context cleanup and CUDA sync; CPU/H2D/logs/post-timer physical KV audit excluded',KV='Actual live and unique storage bytes of K/V tensors at output32;31 Decode inputs cached; not total GPU memory',physical_components='Exactly reuse old s55_ablation_engine classes from accuracy experiment; no algorithm/Attention/model/timer edits',patch_scope=patched,no_other_benchmarks=True,no_training=True,no_history_overwrite=True,no_automatic_GitHub_push=True))
 filepaths=list(sorted((N/'code').glob('c55_*.py')))+[Path(runtime.ref.__file__),OLD/'code/s55_ablation_engine.py',OLD/'code/s55_common.py',OLD/'code/s55_monitor.py']
 save(N/'source_manifest.json',dict(at_ns=time.time_ns(),files={str(p):sha(p) for p in filepaths},frozen_computational_git_head=command(['git','-C',str(R),'rev-parse','HEAD']),old_source_manifest_sha256=sha(OLD/'source_manifest.json'),old_artifact_manifest_sha256=sha(OLD/'artifact_manifest.json'),model_receipt_sha256=sha(M/'.ate_revision.json'),original_generate_source_sha256=sha(runtime.ref.__file__),runtime_function_reused_without_edit=True))
 save(N/'cpu_preflight.json',dict(status='PASS',CUDA_initialized=False,inputs=30,all_images_and_three_tensor_hashes_exact=True,Gold_separate=True,physical_forward_inherited_unchanged=True,allowed_patches=patched,old_component_smoke=read(OLD/'ablation/smoke_validation.json')['status'],bindings=bindings))
 verify();print('COMPONENT_EFFICIENCY_CPU_PASS',flush=True)
if __name__=='__main__':main()
