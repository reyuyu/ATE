"""Post hoc, user-requested L20-L27 visual-writing restriction. Parent sources remain immutable."""
import argparse,json,csv,hashlib,os,time,gc,subprocess,shutil,sys
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import torch,transformers
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as e
from vlm_research.internvl3_visual_writing_figure3_3000.prepare import ROOT,MODEL,REV,sha
P=ROOT/"vlm_research/results/internvl3_visual_writing_figure3_3000"
O=ROOT/"vlm_research/results/internvl3_visual_writing_l20_l27_3000"
B=Path("/home/featurize/ate-runs/internvl3-figure3-3000")
R=B/"l20_l27_3000"
OLDCP=B/"checkpoints"
NAME="Keep-L0-L19"
PY="/home/featurize/.venvs/ate-frozen/bin/python"
ORIGINAL_WRITERS=e.writers
def writers(c):return list(range(20)) if c==NAME else ORIGINAL_WRITERS(c)
def save(d,n,v):
 d.mkdir(parents=True,exist_ok=True);p=d/n;t=p.with_name(p.name+".tmp")
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n");t.replace(p)
def command(*args):return subprocess.check_output(args,text=True).strip()
def read(d,n):return json.loads((d/n).read_text())
def cpu_tests(bindings):
 assert all(writers(c)==ORIGINAL_WRITERS(c) for c in e.CONDITIONS)
 cases=0
 for idx in [525,1928,1127]:
  role=bindings[idx]["roles"];n=role["n"];lo,hi=role["image"][0],role["image"][-1]+1
  base=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16)
  base.masked_fill_(torch.ones((n,n),dtype=torch.bool).tril(),0)
  for layer in range(28):
   m=e.edit_mask(base,role,True,layer>=20);x=torch.zeros_like(base,dtype=torch.bool)
   x[:,:,n-1,lo:hi]=True
   if layer>=20:x[:,:,hi:n-1,lo:hi]=True
   assert torch.equal(m!=base,x) and torch.isneginf(m[x]).all() and (m==0).any(-1).all()
   assert torch.equal(m[:,:,:hi],base[:,:,:hi]);cases+=1
  native=torch.zeros((1,1,1,n+4),dtype=torch.bfloat16)
  m=e.edit_mask(native,role,False,False)
  assert int((m!=native).sum())==256 and torch.isneginf(m[:,:,:,lo:hi]).all()
 return dict(passed=True,prefill_layer_cases=cases,decode_cases=3,allowed_R_layers=list(range(20)),
  forbidden_R_layers=list(range(20,28)),P1_and_Decode_image_off_all_28=True,non_target_exact=True)
def prepare():
 assert not O.exists(),"Never overwrite an existing extension"
 assert read(P,"validation.json")["passed"]
 assert read(P,"generation_complete.json")["requests"]==93000
 for name,h in read(P,"source_manifest.json").items():assert sha(name)==h
 assert transformers.__version__=="4.49.0" and torch.__version__=="2.6.0+cu124"
 O.mkdir(parents=True);R.mkdir(parents=True,exist_ok=False);(R/"checkpoints").mkdir()
 for src,dest in [("sample_manifest.json","sample_manifest.json"),("cpu_input_bindings.json","cpu_input_bindings.json")]:
  shutil.copy2(P/src,R/dest)
 shutil.copy2(P/"sample_manifest.json",O/"sample_manifest.json")
 samples=read(R,"sample_manifest.json")["samples"];bindings=read(R,"cpu_input_bindings.json")
 assert len(samples)==3000 and len({s["image_sha256"] for s in samples})==2409
 assert Counter(s["task"] for s in samples)==Counter({t:500 for t in e.TASKS})
 tests=cpu_tests(bindings);save(O,"cpu_tests.json",tests);save(R,"cpu_tests.json",tests)
 protocol=dict(experiment="User-requested additional L20-L27 OFF",condition=NAME,questions=3000,
  task_questions={t:500 for t in e.TASKS},image_clusters=2409,model="OpenGVLab/InternVL3-8B",revision=REV,
  GPU_UUID=read(P,"protocol.json")["gpu_uuid"],dtype="BF16",attention="eager",batch=1,
  torch="2.6.0+cu124",transformers="4.49.0",max_new_tokens=32,normal_EOS=True,greedy=True,
  logits_to_keep=1,max_num=1,visual_tokens=256,R_allowed_layers=list(range(20)),R_forbidden_layers=list(range(20,28)),
  P1_and_all_decode_to_image="OFF at all 28 layers",image_queries="native",
  physical_pruning=False,parent_result_directory=str(P),parent_protocol_sha256=sha(P/"protocol.json"),
  parent_sample_manifest_sha256=sha(P/"sample_manifest.json"),parent_per_question_sha256=sha(P/"per_question.csv"),
  original_frozen_protocol_unchanged=True,post_hoc=True,
  request="Boundary L20 was additionally requested after the L17-L27 results were known; exploratory, no search.",
  baseline_reuse="Existing same-GPU All-Writer and Keep-L0-L16; no 3000-question baseline rerun",
  statistics="20000 whole-image-SHA cluster paired bootstrap; ordinary unadjusted 95% exploratory CIs; question weighted",
  bootstrap_seed=20261110,figure="Retain old L17-L27 comparison and add L20-L27 as a third InternVL bar")
 save(O,"protocol.json",protocol);shutil.copy2(O/"protocol.json",R/"protocol.json")
 sources=dict(read(P,"source_manifest.json"));sources[str(Path(__file__).resolve())]=sha(__file__)
 save(O,"source_manifest.json",sources);shutil.copy2(O/"source_manifest.json",R/"source_manifest.json")
 save(O,"sample_manifest_reference.json",dict(path=str(P/"sample_manifest.json"),sha256=sha(P/"sample_manifest.json"),
  exact_same_order=True,no_image_rehash=True,no_resampling=True,questions=3000,image_clusters=2409))
 print("EXTENSION_CPU_PREPARE_PASS",flush=True)
def smoke(model,tok,loader,samples,bindings):
 checks=[]
 with patch.object(e,"writers",writers):
  for idx in [525,1928,1127]:
   s=samples[idx];prep=e.prepare(model,tok,loader,s);key=s["task"]+"__"+s["question_id"]
   old=read(OLDCP,key+".json")
   assert prep["input_binding"]==old["input_binding"]
   assert prep["binding"]["input_ids"]==bindings[idx]["input_ids"]
   native,ns,nt=e.generate(model,tok,prep,native=True)
   with e.OfficialChatCapture(model) as official:
    answer=model.chat(tok,prep["pixels"],prep["request"],dict(do_sample=False,num_beams=1,max_new_tokens=32,
     logits_to_keep=1,pad_token_id=tok.convert_tokens_to_ids(model.conv_template.sep.strip())))
   assert official.ids==native["token_ids"] and all(torch.equal(a,b) for a,b in zip(ns,official.scores))
   assert official.input_ids[0].tolist()==bindings[idx]["input_ids"] and answer.strip()==native["text"]
   assert len(ns)==len(official.scores) and torch.equal(official.embeddings,prep["inputs"]["inputs_embeds"].cpu())
   noop,noops,trace=e.generate(model,tok,prep,noop=True,audit=True,trace=True)
   assert noop["token_ids"]==native["token_ids"] and len(ns)==len(noops) and all(torch.equal(a,b) for a,b in zip(ns,noops))
   aw,aws,awt=e.generate(model,tok,prep,audit=True,trace=True)
   assert aw["token_ids"]==old["conditions"]["All-Writer"]["token_ids"]
   assert aw["all_step_logits_sha256"]==old["conditions"]["All-Writer"]["all_step_logits_sha256"]
   new,_,tr=e.generate(model,tok,prep,NAME,audit=True,trace=True)
   assert tr["prefix"]==trace["prefix"] and new["mask_audit"]["blocked_probability_max"]==0
   assert all(x["R_off"]==(x["layer"]>=20) for x in tr["mask_checks"] if x["step"]==0)
   assert new["mask_audit"]["real_decode"]>=1
   footprint=e.hooks_count(model)
   try:
    from transformers.cache_utils import DynamicCache
    with e.WriterMask(model,prep["roles"],NAME,DynamicCache()):raise RuntimeError("extension-cleanup")
   except RuntimeError as err:assert str(err)=="extension-cleanup"
   assert e.hooks_count(model)==footprint
   checks.append(dict(index=idx,question_id=s["question_id"],input_binding_matches=True,
    native_official_noop_all_logits_torch_equal=True,AllWriter_matches_saved_all_tokens_logits=True,
    original_image_queries_exact=True,mask=new["mask_audit"],allowed_R_layers=new["writer_layers"],
    non_target_and_position_exact=True,exception_cleanup=True))
   print("EXTENSION_SMOKE_PASS",idx,flush=True)
 save(R,"smoke_summary.json",dict(passed=True,samples=checks,protocol_sha256=sha(R/"protocol.json")))
@torch.no_grad()
def run():
 p=read(R,"protocol.json");ph=sha(R/"protocol.json")
 assert transformers.__version__=="4.49.0" and torch.__version__=="2.6.0+cu124"
 assert sha(R/"source_manifest.json")==p["source_manifest_sha256"]
 for path,h in read(R,"source_manifest.json").items():assert sha(path)==h
 assert command("nvidia-smi","--query-gpu=uuid","--format=csv,noheader")==p["GPU_UUID"]
 assert not command("nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader")
 torch.set_num_threads(4);torch.manual_seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=None
 try:
  from transformers import AutoModel,AutoTokenizer
  import platform
  save(R,"environment.json",dict(python=platform.python_version(),python_executable=sys.executable,
   torch=torch.__version__,torch_import=torch.__file__,transformers=transformers.__version__,
   transformers_import=transformers.__file__,cuda=torch.version.cuda,GPUs=command("nvidia-smi","--query-gpu=uuid,name,driver_version","--format=csv,noheader"),
   source_sha256=read(R,"source_manifest.json"),no_power_or_system_changes=True))
  model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,
   low_cpu_mem_usage=True,use_flash_attn=False,device_map={"":"cuda:0"},use_safetensors=True).eval()
  assert len(model.language_model.model.layers)==28 and model.num_image_token==256
  assert model.language_model.config._attn_implementation=="eager"
  assert {x.dtype for x in model.parameters()}=={torch.bfloat16}
  assert {str(x.device) for x in model.parameters()}=={"cuda:0"}
  assert not any(x.attn.use_flash_attn for x in model.vision_model.encoder.layers)
  tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
  loader=e.official_preprocessor(MODEL);samples=read(R,"sample_manifest.json")["samples"];bindings=read(R,"cpu_input_bindings.json")
  versions=[(x._version,x.data_ptr()) for x in model.parameters()];footprint=e.hooks_count(model)
  smoke(model,tok,loader,samples,bindings)
  parent_hashes=read(B,"mirror_complete.json")["all_files_SHA256"]
  count=0
  with patch.object(e,"writers",writers):
   for idx,s in enumerate(samples):
    key=s["task"]+"__"+s["question_id"]+".json";dest=R/"checkpoints"/key
    if dest.exists():
     prior=json.loads(dest.read_text())
     assert prior["protocol_sha256"]==ph and prior["sample_sha256"]==e.digest(s) and prior["GPU_UUID"]==p["GPU_UUID"]
     assert prior["complete"] and prior["index"]==idx and prior["generation"]["writer_layers"]==list(range(20))
     count+=1;continue
    raw=(OLDCP/key).read_bytes();assert hashlib.sha256(raw).hexdigest()==parent_hashes[key]
    old=json.loads(raw);assert old["complete"] and old["GPU_UUID"]==p["GPU_UUID"]
    prep=e.prepare(model,tok,loader,s)
    assert prep["input_binding"]==old["input_binding"]
    assert prep["prompt"]==bindings[idx]["prompt"] and prep["binding"]["roles"]==bindings[idx]["roles"]
    out,scores,_=e.generate(model,tok,prep,NAME)
    out["correct"]=int(out["prediction"]==s["answer"].lower());base=old["conditions"]["All-Writer"]
    out.update(correct_to_wrong=int(base["correct"] and not out["correct"]),
     wrong_to_correct=int(not base["correct"] and out["correct"]),
     first_token_flip=int(out["token_ids"][0]!=base["token_ids"][0]))
    record=dict(index=idx,condition=NAME,protocol_sha256=ph,parent_protocol_sha256=p["parent_protocol_sha256"],
     sample_sha256=e.digest(s),GPU_UUID=p["GPU_UUID"],question_id=s["question_id"],task=s["task"],gold=s["answer"],
     image_sha256=s["image_sha256"],input_binding=prep["input_binding"],input=prep["binding"],generation=out,
     parent_checkpoint_sha256=parent_hashes[key],complete=True)
    save(R/"checkpoints",key,record);count+=1;del prep,scores,old,record
    if count%10==0:
     assert command("nvidia-smi","--query-gpu=uuid","--format=csv,noheader")==p["GPU_UUID"]
     assert versions==[(x._version,x.data_ptr()) for x in model.parameters()] and e.hooks_count(model)==footprint
     save(R,"progress.json",dict(requests=count,target=3000,condition=NAME,GPU_UUID=p["GPU_UUID"]))
     print("EXTENSION_PROGRESS",count,"/3000",flush=True)
  assert count==3000
  save(R,"generation_complete.json",dict(passed=True,requests=3000,condition=NAME,protocol_sha256=ph,
   GPU_UUID=p["GPU_UUID"],weights_unchanged=versions==[(x._version,x.data_ptr()) for x in model.parameters()]))
 except BaseException:
  import traceback
  save(R,"failure.json",dict(traceback=traceback.format_exc(),protocol_sha256=ph));raise
 finally:
  if model is not None:del model
  gc.collect();torch.cuda.empty_cache();save(R,"gpu_release.json",dict(worker_finished=True,pid=os.getpid()))
def supervise():
 cfg=dict(worker=None,supervisor=os.getpid(),runtime=str(R),persistent=str(O))
 proc=subprocess.Popen([PY,"-u","-m","vlm_research.internvl3_visual_writing_l20_l27_3000","run"],cwd=ROOT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",TOKENIZERS_PARALLELISM="false"),stdout=(R/"worker.log").open("ab"),
  stderr=subprocess.STDOUT,start_new_session=True)
 cfg["worker"]=proc.pid;save(R,"active.json",cfg);seen={}
 while True:
  active=proc.poll() is None
  for src in (R/"checkpoints").glob("*.json"):
   st=src.stat();mark=(st.st_mtime_ns,st.st_size)
   if seen.get(src.name)==mark:continue
   data=src.read_bytes();dest=O/"checkpoints"/src.name;dest.parent.mkdir(exist_ok=True)
   temp=dest.with_suffix(".tmp");temp.write_bytes(data);temp.replace(dest)
   assert sha(dest)==hashlib.sha256(data).hexdigest();seen[src.name]=mark
  for src in R.glob("*.json"):
   if src.name in ("sample_manifest.json","cpu_input_bindings.json"):continue
   dest=O/src.name
   if not dest.exists() or sha(dest)!=sha(src):shutil.copy2(src,dest)
  save(R,"mirror_progress.json",dict(files=len(seen),updated=time.time(),worker_alive=active))
  if not active:break
  time.sleep(10)
 shutil.copy2(R/"worker.log",O/"worker.log")
 assert proc.returncode==0 and read(R,"generation_complete.json")["requests"]==3000
 hashes={p.name:sha(p) for p in (R/"checkpoints").glob("*.json")}
 assert len(hashes)==3000 and all(sha(O/"checkpoints"/n)==h for n,h in hashes.items())
 save(O,"persistence_complete.json",dict(passed=True,files=3000,SHA256=hashes))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
 subprocess.run([PY,"-u","-m","vlm_research.internvl3_visual_writing_l20_l27_3000","analyze"],cwd=ROOT,env=env,check=True)
 print("EXTENSION_ALL_COMPLETE_REVIEW_PENDING",flush=True)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("mode",choices=["prepare","run","supervise","analyze"])
 mode=a.parse_args().mode
 if mode=="analyze":
  from vlm_research.internvl3_visual_writing_l20_l27_analysis import main
  main()
 else:globals()[mode]()

