"""CPU preparation and strict reuse verification for fixed R14 validation."""
from .common import *
import gzip,torch,importlib.util,sys,shutil
from collections import Counter
from vlm_research.internvl3_svh_seven_benchmarks import worker as w,scoring
def main():
 parent_manifest=read(PARENT/"results/sample_manifest.json");parent_protocol=read(PARENT/"results/protocol.json");parent_source=read(PARENT/"results/source_manifest.json");parent_ph=sha(PARENT/"results/protocol.json");parent_sh=sha(PARENT/"results/source_manifest.json")
 for path,h in parent_source["runtime_source_sha256"].items():assert sha(PARENT/"code"/path)==h,("PARENT_SOURCE_CHANGED",path)
 assert parent_protocol["revision"]=="853e3a797a661694b1b8ece0cb72dc2b23e3dac9" and parent_protocol["max_new_tokens"]==32 and parent_protocol["instruction"]==INSTRUCTION and parent_protocol["GPU"]==GPU
 A=parent_manifest["Holdout"];exclude={r["image_sha256"] for r in parent_manifest["Discovery"]+parent_manifest["Holdout"]+parent_manifest["Smoke"]}
 source=BASE/"manifests/TextVQA.jsonl.gz";views=[]
 with gzip.open(source,"rt") as f:
  for line in f:
   r=json.loads(line);r.pop("gold",None)
   if r["image_sha256"] not in exclude:views.append(r)
 views=sorted(views,key=lambda r:digest([SEED,"new-B-question",r["question_id"],r["image_sha256"]]))
 chosen=[];seen=set()
 for r in views:
  if r["image_sha256"] in seen:continue
  chosen.append(r);seen.add(r["image_sha256"])
  if len(chosen)==128:break
 assert len(chosen)==128 and not seen&exclude
 freeze=dict(seed=SEED,source_manifest_sha256=sha(source),remaining_question_count=len(views),remaining_image_count=len({r["image_sha256"] for r in views}),excluded_previous_images=len(exclude),sampling_rule="Uniform SHA256(seed,new-B-question,qid,imageSHA) rank over remaining questions; accept first question per new image until128; no question-type quotas, Gold, predictions, meta labels or scores",GroupB=[dict(question_id=r["question_id"],image_sha256=r["image_sha256"]) for r in chosen],frozen_before_any_R14_inference=True,frozen_before_historical_B_prediction_access=True)
 write(OUT/"group_B_sampling_freeze.json",freeze)
 selected={r["question_id"] for r in chosen};B=[]
 with gzip.open(source,"rt") as f:
  for line in f:
   r=json.loads(line)
   if r["question_id"] in selected:r["analysis_type"]=subtype(r["question"]);B.append(r)
 byid={r["question_id"]:r for r in B};B=[byid[r["question_id"]] for r in chosen]
 smoke=[parent_manifest["Smoke"][i] for i in [0,1,3,4,6,7]];allrows=A+B+smoke;assert len(allrows)==230 and len({r["image_sha256"] for r in allrows})==230
 speed=sorted(B,key=lambda r:rank(r,"R14-speed"))[:8]
 manifest=dict(seed=SEED,parent_commit="3af6596b1ce59d80e203ebf68c6760e242d4df47",parent_manifest_sha256=sha(PARENT/"results/sample_manifest.json"),group_B_sampling_freeze_sha256=sha(OUT/"group_B_sampling_freeze.json"),GroupA=A,GroupB=B,Smoke=smoke,speed_ids=[r["question_id"] for r in speed],Group_A_role="historical same-input supplementary contrast; NOT independent R14-versus-R6 confirmation",Group_B_role="primary independent image-disjoint validation",GroupB_overlap_with_parent_Discovery_Holdout_Smoke=0,one_question_per_image_B=True,question_types_analysis_only=True,subtype_counts={name:dict(Counter(r["analysis_type"] for r in rr)) for name,rr in [("A",A),("B",B)]},remaining_population_type_counts=dict(Counter(subtype(r["question"]) for r in views)))
 write(OUT/"sample_manifest.json",manifest);write(OUT/"search_space.json",SPACE)
 protocol=dict(status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER",historical_status="NUMERICAL_EQUIVALENCE_FAILED unchanged",model=parent_protocol["model"],revision=parent_protocol["revision"],GPU=GPU,dtype="bfloat16",attention="eager",torch="2.6.0+cu124",transformers="4.49.0",batch=1,greedy=True,use_cache=True,logits_to_keep=1,visual_tokens=256,image_shape=[1,3,448,448],max_new_tokens=32,native_EOS=True,prompt_protocol="ATE_SVH_1TILE_SCORER_ALIGNED",instruction=INSTRUCTION,image_only=True,parent_commit=manifest["parent_commit"],parent_protocol_sha256=parent_ph,parent_source_manifest_sha256=parent_sh,physical_engine_commit="9602d93869d4ccd11986573369d8fecc84b80f78",physical_engine="unchanged V2; no C1/C2 projection repair",R14_scope="L6-L19 only; NOT restoration of L0-L1 or L2-L5 answer attention, L20-L27 hidden/image access, or Native28-layer image access",physical_invariants="delete256 image hidden beforeL20; answer attention/QKV skippedL2-L5; onlyK persistent original imageKV; original logicalRoPE; allNormMLPResidual",search_space=SPACE,sample_manifest_sha256=sha(OUT/"sample_manifest.json"),formal_quality_budget=480,GroupA_new="R14 only96; Native/R0/R6 all reused after strict matching",GroupB_new="R0/R6/R14 each128; Native individual frozen predictions strictly bound",single_model_load=True,quality_audit=False,vision_feature_reuse=False,smoke="six parent Smoke images outsideA/B; Dense-R14 vs Physical-R14; heavy short/long audit, light logits onother4; NativeNoop strict onfirst; known BF16 logit errors recorded only",hard_stop="source/input mismatch, NaN/Inf, OOM, illegal cap, KV/Mask/RoPE/exit/skip/cache/hook structural failure",meta_policy="all anomalous and truncated outputs retained and officially scored, no stopping merely for known meta outputs",bootstrap=dict(unit="whole paired imageSHA clusters",repetitions=5000,seed=SEED,CI=95,multiplicity="ordinary unadjusted exploratory intervals"),soft_change_definitions=dict(improved_declined="difference in actual ten-answer soft score",zero_to_positive="0 to >0",positive_to_zero=">0 to0",not_full_to_full="<1 to1",full_to_not_full="1 to<1"),recovery_ratio="(R14-R0)/(Native-R0) within same group; undefined if denominator zero; no clamping",speed_gate="GroupB R14-R0 AND R14-R6 both have ordinary paired95%CI lower>0; otherwise NOT_RUN; no new conditions or sample expansion",speed=dict(methods=["Native","R0","R6","R14"],images=8,warmups_per_method=2,repeats=3,output_tokens=32,real_decode=31,ignore_EOS=True,boundary="GPU-ready input through real Vision/projector/prefill/31decode and necessary method cleanup; synchronized wall time; exclude model loading/disk/tokenizer/processing/H2D/write; no heavy audit",scratch="V2 transient zero K/V scratch remains; actual work included in timing, not persistent original visualKV"))
 write(OUT/"protocol.json",protocol)
 sources={str(p.relative_to(RUN/"code")):sha(p) for p in (RUN/"code").rglob("*.py")}
 assert sources["vlm_research/internvl3_svh_r_middle_reaccess_search/engine.py"]=="8751256f3a03d297d2cfc012101d53e2932a5d098ff8f49240e560a686617ff3"
 for path in ["vlm_research/internvl3_svh_r_middle_reaccess_search/engine.py","vlm_research/internvl3_svh_physical_d_pilot/engine.py","vlm_research/internvl3_visual_writing_figure3_3000/engine.py","vlm_research/internvl3_svh_seven_benchmarks/scoring.py"]:assert sources[path]==parent_source["runtime_source_sha256"][path]
 write(OUT/"source_manifest.json",dict(runtime_source_sha256=sources,core_forward_scorer_processor_unchanged=True,C1_C2_used=False))
 tok=w.AutoTokenizer.from_pretrained(w.MODEL,trust_remote_code=True,use_fast=False,local_files_only=True);cfg=read(w.MODEL/"config.json");spec=importlib.util.spec_from_file_location("ate_R14_conversation",w.MODEL/"conversation.py");conversation=importlib.util.module_from_spec(spec);sys.modules[spec.name]=conversation;spec.loader.exec_module(conversation);template=conversation.get_conv_template(cfg["template"])
 inputs=w.Inputs(None,tok);old_bind=read(PARENT/"prepared/input_bindings.json");current_bind=read(RUN/"prepared/input_bindings.json") if (RUN/"prepared/input_bindings.json").exists() else {};bindings={}
 for i,r in enumerate(allrows):
  request="<image>\n"+r["question"]+"\n"+INSTRUCTION;assert r["request"]==request
  c=template.copy();c.append_message(c.roles[0],request);c.append_message(c.roles[1],None);prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
  ids=tok(prompt,return_tensors="pt")["input_ids"];assert prompt==r["prompt"] and ids[0].tolist()==r["input_ids"] and w.sha(prompt)==r["prompt_sha256"]
  assert w.b.role_layout(ids[0].tolist(),tok.convert_tokens_to_ids("<IMG_CONTEXT>"))==r["roles"] and len(r["gold"])==10 and r["max_new_tokens"]==32
  im=r["image_sha256"];old_cache=PARENT/"prepared"/(im+".pt")
  if r["question_id"] in current_bind:
   pix=torch.load(RUN/"prepared"/(im+".pt"),map_location="cpu",weights_only=True);assert w.tensor_sha(pix)==current_bind[r["question_id"]]["details"]["pixels_bf16_sha256"]
  elif r["question_id"] in old_bind:
   pix=torch.load(old_cache,map_location="cpu",weights_only=True);assert w.tensor_sha(pix)==old_bind[r["question_id"]]["details"]["pixels_bf16_sha256"]
  else:pix=inputs.pixels(r["image_path"],im)
  assert pix.shape==(1,3,448,448) and pix.dtype==torch.bfloat16;torch.save(pix,RUN/"prepared"/(im+".pt"))
  details=dict(image_sha256=im,input_ids_sha256=w.tensor_sha(ids),pixels_bf16_sha256=w.tensor_sha(pix),prompt_sha256=r["prompt_sha256"]);bindings[r["question_id"]]=dict(details=details,binding=w.bind(details))
  if i%32==0:print("CPU_INPUTS",i+1,230,flush=True)
 write(RUN/"prepared/input_bindings.json",bindings);write(OUT/"input_bindings.json",bindings)
 parentdb=sqlite3.connect("file:"+str(PARENT/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True);legacy=sqlite3.connect("file:"+str(BASE/"checkpoints/TextVQA.sqlite")+"?mode=ro",uri=True);reuse={};receipt=[]
 for r in A:
  q=r["question_id"];reuse[q]={}
  for m in ["Native","R0","R6"]:
   raw=parentdb.execute("SELECT record FROM predictions WHERE phase='Holdout' AND qid=? AND method=?",(q,m)).fetchone()[0];x=json.loads(raw)
   assert x["protocol_sha256"]==parent_ph and x["source_manifest_sha256"]==parent_sh
   assert x["input_binding"]==bindings[q]["binding"]==old_bind[q]["binding"] and x["gold"]==r["gold"] and x["sample_sha256"]==r["sample_sha256"]
   assert abs(scoring.score("TextVQA",x["text"],r)[1]-x["score"])<1e-10 and abs(scoring.independent_score("TextVQA",x["text"],r)-x["score"])<1e-10
   reuse[q][m]=dict(record=x,original_record_sha256=w.sha(raw));receipt.append(dict(group="A",question_id=q,method=m,parent_record_sha256=w.sha(raw),input_binding=x["input_binding"],strict_match=True))
 native={}
 for r in B+smoke:
  raw=legacy.execute("SELECT record FROM predictions WHERE qid=? AND method='native'",(r["question_id"],)).fetchone()[0];x=json.loads(raw)
  assert x["input_binding_details"]==bindings[r["question_id"]]["details"] and x["input_binding"]==bindings[r["question_id"]]["binding"] and x["max_new_tokens"]==32 and x["gold"]==r["gold"] and x["sample_sha256"]==r["sample_sha256"]
  native[r["question_id"]]=dict(record=x,original_record_sha256=w.sha(raw))
  receipt.append(dict(group="B" if r["question_id"] in selected else "Smoke",question_id=r["question_id"],method="Native",parent_record_sha256=w.sha(raw),input_binding=x["input_binding"],strict_match=True))
 parentdb.close();legacy.close();write(RUN/"prepared/reused_A_records.json",reuse);write(RUN/"prepared/reused_native_B_smoke.json",native);csvout("reuse_record_audit.csv",receipt)
 write(OUT/"input_reuse_validation.json",dict(status="PASS_CPU_PENDING_SMOKE",GroupA_records=288,GroupB_Native_records=128,Smoke_Native_records=6,all_input_tensor_prompt_image_gold_cap_and_score_protocol_matches=True,parent_protocol_sha256=parent_ph,parent_source_manifest_sha256=parent_sh,parent_model_manifest_sha256=sha(PARENT/"results/model_manifest.json"),all_frozen_parent_runtime_source_hashes_verified=True,core_forward_engine_and_scorer_bytes_equal=True,GroupB_sampling_before_prediction_access=True,GroupB_image_overlap_with_previous200=0,Image_only=True,selected_inputs=230))
 write(OUT/"image_only_input_audit.json",dict(status="PASS",selected_questions=230,exact_native_template_original_question_frozen_instruction_and_ids=True,original_image_only_BF16_pixels=True,input_keys=["ids","pixels","roles","request","prompt"],auxiliary_OCR_caption_hints_gold_rationale_in_model_input=False,parent_population5000_audit_sha256=sha(PARENT/"results/image_only_input_audit.json"),source_binding_sha256=sha(RUN/"prepared/input_bindings.json")))
 write(OUT/"cpu_tests.json",dict(status="PASS",input_bindings=230,GroupA_strict_reuse=288,new_B_Native_strict_reuse=128,image_disjoint=True,source_frozen_V2=True,GPU_initialized=torch.cuda.is_initialized()))
 assert not torch.cuda.is_initialized();print("CPU_READY_R14",len(A),len(B),manifest["subtype_counts"],flush=True)
if __name__=="__main__":main()
