"""CPU-only reconstruction of the fixed historical B128 and strict prediction reuse."""
from .common import *
import torch,sqlite3,subprocess,importlib.util,sys,shutil
from vlm_research.internvl3_svh_seven_benchmarks import worker as w,scoring
def main():
 assert not torch.cuda.is_initialized()
 mf=read(PARENT/"sample_manifest.json");B=mf["GroupB"];sm=sorted(mf["Smoke"],key=lambda r:r["input_tokens"]);smoke=[sm[0],sm[len(sm)//2],sm[-1]]
 assert len(B)==128 and len({r["image_sha256"] for r in B})==128
 selected=sorted(B,key=lambda r:digest([SEED,"deep-exit-controls48",r["question_id"],r["image_sha256"]]))[:48]
 speed=sorted(B,key=lambda r:digest([SEED,"deep-exit-speed8",r["question_id"],r["image_sha256"]]))[:8]
 manifest=dict(parent_commit=PARENT_COMMIT,parent_manifest_sha256=sha(PARENT/"sample_manifest.json"),Main=B,Smoke=smoke,control_ids=[r["question_id"] for r in selected],speed_ids=[r["question_id"] for r in speed],seed=SEED,sample_role="frozen historical same-question comparison; previously used for R14; NOT new independent Holdout",main_order_preserved=True,control_selection="first48 fixed SHA256(seed,label,qid,imageSHA) rank; no outcome use",control_frozen_before_GPU=True)
 write(OUT/"sample_manifest.json",manifest)
 protocol=dict(status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER",historical_status="NUMERICAL_EQUIVALENCE_FAILED preserved",model="OpenGVLab/InternVL3-8B",revision="853e3a797a661694b1b8ece0cb72dc2b23e3dac9",GPU=GPU,previous_GPU=OLD_GPU,GPU_change_explicitly_authorized=True,baseline_origin="strictly bound historical predictions on previous GPU; no claim of current-GPU baseline output or timing equivalence",dtype="bfloat16",attention="eager",torch="2.6.0+cu124",transformers="4.49.0",batch=1,greedy=True,use_cache=True,logits_to_keep=1,image_shape=[1,3,448,448],visual_tokens=256,max_new_tokens=32,native_EOS=True,prompt_protocol="ATE_SVH_1TILE_SCORER_ALIGNED",instruction=INSTRUCTION,image_only=True,configs=CONFIGS,controls=CONTROLS,R_to_Image_off_from=20,SKIP_answer_attention_layers=[2,3,4,5],formal_quality_requests=352,Main_new=256,Controls_new=96,Main_reused=512,model_loads_target=1,quality_audit=False,bootstrap=dict(seed=SEED,images="whole image SHA paired clusters",repetitions=5000,CI=95,multiplicity="ordinary unadjusted exploratory"),speed_gate="A candidate is eligible iff its Main soft accuracy>=20% AND Main accuracy-minus-R0>=10pp. If none eligible, speed NOT_RUN.",speed=dict(images=8,warmups_each_method=2,formal_repeats_each_image=3,methods="Native,R0,eligible Deep candidates only",output_tokens=32,real_decode=31,ignore_EOS=True,boundary="GPU-ready input through true Vision/Projector/Prefill/Decode and necessary cleanup, synchronized wall; excludes loading/disk/tokenization/preprocessing/H2D/logging; no heavy Hooks"),hard_stop="structural Mask/RoPE/KV/position/physical-exit/skip errors, nonfinite, OOM, hash mismatch, invalid cap, cache pollution or hook leak",meta_policy="all generated outputs retained and officially scored; known BF16 or meta failures are not filtered",numerical_repair=False,source_parent_commit=PARENT_COMMIT,sample_manifest_sha256=sha(OUT/"sample_manifest.json"),model_path=str(MODEL),resources_local_SSD=True,GitHub_branch="codex/internvl3-svh-r-deep-exit-textvqa-3090")
 write(OUT/"protocol.json",protocol)
 tok=w.AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
 cfg=read(MODEL/"config.json");spec=importlib.util.spec_from_file_location("ate_deep_conversation",MODEL/"conversation.py");mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);template=mod.get_conv_template(cfg["template"])
 loader=w.Inputs(None,tok);old_bind=read(PARENT/"server_only/input_bindings.json");bindings={};paths={}
 for j,r in enumerate(B+smoke):
  q=r["question_id"];im=r["image_sha256"];path=Path("/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr/images/textvqa")/(r["image_id"]+".jpg")
  request="<image>\n"+r["question"]+"\n"+INSTRUCTION;assert request==r["request"]
  conv=template.copy();conv.append_message(conv.roles[0],request);conv.append_message(conv.roles[1],None)
  prompt=conv.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
  ids=tok(prompt,return_tensors="pt")["input_ids"]
  assert prompt==r["prompt"] and hashlib.sha256(prompt.encode()).hexdigest()==r["prompt_sha256"] and ids[0].tolist()==r["input_ids"]
  assert w.b.role_layout(ids[0].tolist(),tok.convert_tokens_to_ids("<IMG_CONTEXT>"))==r["roles"] and r["max_new_tokens"]==32 and len(r["gold"])==10
  pix=loader.pixels(str(path),im);assert tuple(pix.shape)==(1,3,448,448) and pix.dtype==torch.bfloat16
  details=dict(image_sha256=im,input_ids_sha256=w.tensor_sha(ids),pixels_bf16_sha256=w.tensor_sha(pix),prompt_sha256=r["prompt_sha256"]);binding=w.bind(details)
  assert old_bind[q]["details"]==details and old_bind[q]["binding"]==binding,("FROZEN_INPUT_MISMATCH",q)
  torch.save(dict(ids=ids,pixels=pix,roles=r["roles"],request=request,prompt=prompt),RUN/"prepared"/(im+".pt"))
  bindings[q]=dict(details=details,binding=binding);paths[q]=str(path)
  if j%32==0:print("PREPARED",j+1,len(B)+len(smoke),flush=True)
 write(RUN/"prepared/input_bindings.json",bindings);write(OUT/"input_bindings.json",bindings);write(OUT/"image_path_mapping.json",paths)
 db=sqlite3.connect("file:"+str(PARENT/"server_only/predictions.sqlite")+"?mode=ro&immutable=1",uri=True)
 ph=sha(PARENT/"protocol.json");sh=sha(PARENT/"source_manifest.json");reuse=[];audit=[]
 for r in B:
  for method in ["Native","R0","R6","R14"]:
   raw=db.execute("SELECT record FROM predictions WHERE phase='B' AND qid=? AND method=?",(r["question_id"],method)).fetchone()[0];x=json.loads(raw)
   assert x["protocol_sha256"]==ph and x["source_manifest_sha256"]==sh and x["input_binding"]==bindings[r["question_id"]]["binding"]
   assert x["sample_sha256"]==r["sample_sha256"] and x["image_sha256"]==r["image_sha256"] and x["gold"]==r["gold"]
   assert len(x["token_ids"])==x["generated_tokens"]<=32 and x["eos"]==(x["token_ids"][-1]==151645)
   a=scoring.score("TextVQA",x["text"],r)[1];b=scoring.independent_score("TextVQA",x["text"],r);assert abs(a-b)<1e-10 and abs(a-x["score"])<1e-10
   reuse.append(dict(record=x,original_record_sha256=hashlib.sha256(raw.encode()).hexdigest()));audit.append(dict(question_id=r["question_id"],method=method,record_sha256=hashlib.sha256(raw.encode()).hexdigest(),input_binding=x["input_binding"],strict_match=True))
 db.close();write(RUN/"prepared/reused_records.json",reuse);csvout("reuse_record_audit.csv",audit)
 source=read(PARENT/"source_manifest.json")["runtime_source_sha256"];checked={}
 for rel,h in source.items():
  p=RUN/"code"/rel;assert sha(p)==h;checked[rel]=h
 write(OUT/"parent_source_validation.json",dict(status="PASS",files=checked,parent_commit=PARENT_COMMIT))
 write(OUT/"input_reuse_validation.json",dict(status="PASS_CPU",main_questions=128,controls=48,smoke=len(smoke),strict_reused_records=512,all_prompt_image_BF16_pixels_token_ids_roles_model_and_scorer_bindings_match=True,Image_only=True,Gold_absent_from_prepared_model_inputs=True,parent_protocol_sha256=ph,parent_source_manifest_sha256=sh,parent_model_manifest_sha256=sha(PARENT/"model_manifest.json"),new_GPU=GPU,historical_GPU=OLD_GPU,GPU_change_authorized=True,no_claim_of_new_independent_Holdout=True))
 assert not torch.cuda.is_initialized();print("CPU_INPUT_REUSE_READY",len(reuse),flush=True)
if __name__=="__main__":main()
