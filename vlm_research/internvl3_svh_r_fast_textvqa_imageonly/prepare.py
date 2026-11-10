"""CPU-only Image-only audit, outcome-blind sampling, and selected pixel cache."""
from .common import *
import gzip,importlib.util,sys,subprocess,torch
from collections import Counter
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
def main():
 source=BASE/"manifests/TextVQA.jsonl.gz"
 # Predictions are not opened here. Gold is discarded from the sampling view.
 pool=[]
 with gzip.open(source,"rt") as f:
  for line in f:
   r=json.loads(line);r.pop("gold",None);r["analysis_type"]=subtype(r["question"]);pool.append(r)
 assert len(pool)==5000 and len({r["question_id"] for r in pool})==5000
 cfg=read(w.MODEL/"config.json")
 spec=importlib.util.spec_from_file_location("ate_frozen_conversation",w.MODEL/"conversation.py");conv=importlib.util.module_from_spec(spec);sys.modules[spec.name]=conv;spec.loader.exec_module(conv)
 template=conv.get_conv_template(cfg["template"])
 tok=w.AutoTokenizer.from_pretrained(w.MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
 audit=[]
 for r in pool:
  request="<image>\n"+r["question"]+"\n"+INSTRUCTION
  assert r["request"]==request,("AUXILIARY_CONTENT_OR_PROMPT_MISMATCH",r["question_id"])
  c=template.copy();c.append_message(c.roles[0],request);c.append_message(c.roles[1],None)
  prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
  assert prompt==r["prompt"] and w.sha(prompt)==r["prompt_sha256"],("CHAT_TEMPLATE_MISMATCH",r["question_id"])
  ids=tok(prompt,return_tensors="pt")["input_ids"]
  assert ids[0].tolist()==r["input_ids"] and w.b.role_layout(ids[0].tolist(),tok.convert_tokens_to_ids("<IMG_CONTEXT>"))==r["roles"]
  assert r["benchmark"]=="TextVQA" and r["max_new_tokens"]==32 and r["protocol_label"]=="ATE_SVH_1TILE_SCORER_ALIGNED"
  audit.append({"question_id":r["question_id"],"prompt_sha256":r["prompt_sha256"],"token_ids_sha256":w.tensor_sha(ids),"image_sha256":r["image_sha256"],"exact_question_and_fixed_instruction":True})
 print("PROMPT_AUDIT_PASS",len(audit),flush=True)
 images=sorted({r["image_sha256"] for r in pool},key=lambda im:digest([SEED,"image-partition",im]));split=len(images)//2;dimages=set(images[:split]);himages=set(images[split:])
 def choose(allowed,label):
  chosen=[];seen=set()
  for typ in TYPES:
   candidates=sorted([r for r in pool if r["image_sha256"] in allowed and r["analysis_type"]==typ],key=lambda r:rank(r,label))
   for r in candidates:
    if r["image_sha256"] not in seen:
     chosen.append(r);seen.add(r["image_sha256"])
     if sum(x["analysis_type"]==typ for x in chosen)>=QUOTAS[typ]:break
  # Any insufficient question-text stratum is filled by global hash rank, never by gold or outputs.
  for r in sorted([r for r in pool if r["image_sha256"] in allowed],key=lambda r:rank(r,label+":fill")):
   if len(chosen)>=96:break
   if r["image_sha256"] not in seen:chosen.append(r);seen.add(r["image_sha256"])
  assert len(chosen)==96
  return chosen
 disc=choose(dimages,"Discovery");hold=choose(himages,"Holdout")
 first=[]
 for typ in TYPES:
  rr=sorted([r for r in disc if r["analysis_type"]==typ],key=lambda r:rank(r,"StageA"))
  first+=rr[:len(rr)//2]
 for r in sorted(disc,key=lambda r:rank(r,"StageA:fill")):
  if len(first)>=48:break
  if r["question_id"] not in {x["question_id"] for x in first}:first.append(r)
 assert len(first)==48
 first=sorted(first,key=lambda r:rank(r,"StageA:order"));idsA={r["question_id"] for r in first}
 disc=first+sorted([r for r in disc if r["question_id"] not in idsA],key=lambda r:rank(r,"StageB:order"));hold=sorted(hold,key=lambda r:rank(r,"Holdout:order"))
 freeze={"seed":SEED,"source_manifest_sha256":sha(source),"sampling_inputs_only":["question_id","question_text","image_sha256","question_text_derived_type"],"source_size":5000,"image_partition":"sorted SHA256(seed,image-partition,imageSHA); first half Discovery, second half Holdout","stratum_quotas":QUOTAS,"fallback":"insufficient question-text strata filled by global hash, one question per image","Discovery":[{k:r[k] for k in ["question_id","image_sha256","analysis_type"]} for r in disc],"Holdout":[{k:r[k] for k in ["question_id","image_sha256","analysis_type"]} for r in hold],"stageA":"first 48 of Discovery; deterministic half of actual question-text strata","Gold_or_predictions_used":False,"frozen_before_Native_checkpoint_access":True}
 write(OUT/"sampling_freeze.json",freeze)
 selected_images={r["image_sha256"] for r in disc+hold}
 unused=[];seen=set(selected_images)
 for r in sorted(pool,key=lambda r:rank(r,"independent-smoke")):
  if r["image_sha256"] not in seen:unused.append(r);seen.add(r["image_sha256"])
 unused=sorted(unused,key=lambda r:(r["input_tokens"],rank(r,"smoke-length")))
 smoke_views=unused[:3]+unused[len(unused)//2:len(unused)//2+2]+unused[-3:]
 assert len(smoke_views)==8
 freeze["Smoke"]=[{k:r[k] for k in ["question_id","image_sha256","analysis_type"]} for r in smoke_views]
 freeze["smoke_formal_image_overlap"]=0;write(OUT/"sampling_freeze.json",freeze)
 selected={r["question_id"] for r in disc+hold+smoke_views};full={}
 with gzip.open(source,"rt") as f:
  for line in f:
   r=json.loads(line)
   if r["question_id"] in selected:r["analysis_type"]=subtype(r["question"]);assert len(r["gold"])==10;full[r["question_id"]]=r
 disc=[full[r["question_id"]] for r in disc];hold=[full[r["question_id"]] for r in hold]
 smoke=[full[r["question_id"]] for r in smoke_views]
 speed=sorted(disc,key=lambda r:rank(r,"speed"))[:12]
 assert len({r["image_sha256"] for r in disc+hold})==192
 manifest={"sampling_freeze_sha256":sha(OUT/"sampling_freeze.json"),"seed":SEED,"Discovery":disc,"Holdout":hold,"Smoke":smoke,"stage_A_ids":[r["question_id"] for r in disc[:48]],"stage_B_ids":[r["question_id"] for r in disc[48:]],"smoke_ids":[r["question_id"] for r in smoke],"speed_ids":[r["question_id"] for r in speed],"image_overlap":0,"unique_selected_images":192,"subtypes":{name:dict(Counter(r["analysis_type"] for r in rr)) for name,rr in [("Discovery",disc),("Holdout",hold)]},"types_based_on_question_only":True}
 write(OUT/"sample_manifest.json",manifest);write(OUT/"search_space.json",SPACE)
 protocol={"status":"EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER","historical_numerical_status":"NUMERICAL_EQUIVALENCE_FAILED remains unchanged","model":"OpenGVLab/InternVL3-8B","revision":"853e3a797a661694b1b8ece0cb72dc2b23e3dac9","GPU":GPU,"dtype":"bfloat16","attention":"eager","torch":"2.6.0+cu124","transformers":"4.49.0","batch":1,"greedy":True,"use_cache":True,"logits_to_keep":1,"image_shape":[1,3,448,448],"visual_tokens":256,"max_new_tokens":32,"native_EOS":True,"prompt_protocol":"ATE_SVH_1TILE_SCORER_ALIGNED","instruction":INSTRUCTION,"image_only":True,"input_allowlist":"native chat template + original image and question + exact frozen short-answer instruction; no auxiliary OCR/hint/caption/Gold","single_model_load":True,"parent_commit":"03ae6763a02bca30ae7bb553e1ff1c006d8c2552","engine_commit":"9602d93869d4ccd11986573369d8fecc84b80f78","engine":"unmodified V2; no C1/C2 projection repair","zero_only_scratch":"retained; real cost included if speed gate passes; not persistent original visual KV","sample_manifest_sha256":sha(OUT/"sample_manifest.json"),"search_space":SPACE,"selection_A":"rank highest official soft accuracy first; ties lower meta rate, lower actual mean quality-EOS KV capacity, fewer layers, frozen order; select top3 reaccess plus R0","selection_B":{"best_quality":"highest soft accuracy; same tie-breaks","best_efficiency":"within 2.0 percentage points of highest soft accuracy; choose fewest retained layers, then lower actual KV capacity, lower meta rate, higher accuracy, frozen order; R0 eligible","holdout_no_reselection":True},"bootstrap":{"unit":"whole paired image SHA256 clusters","repetitions":5000,"seed":SEED,"CI":95,"multiplicity":"ordinary unadjusted exploratory CI; not confirmatory noninferiority"},"hard_stop":"input/source hash mismatch, Native/Noop failure, NaN/Inf, OOM, illegal token length, KV/Mask/RoPE/exit/skip/cache/hook structural error","output_policy":"all meta/repeat/empty/truncated/wrong answers retained and scored; known frequent Physical meta outputs do not stop this task","smoke":"8 frozen Discovery; Native/Noop strict; Dense/Physical R0,R4-M,R8; full structural audit shortest and longest, light logits for remaining six; BF16 errors recorded not gated","Native_reuse":"sample freeze first, exact input/model/processor/tokenizer/cap/source binding; Holdout prediction text/tokens/scores not loaded before final selection","formal_quality_max":816,"counts":{"A":336,"B_max":192,"Holdout_max":288},"quality_feature_reuse":False,"speed_gate":"at least one preselected non-R0 candidate has Holdout soft gain>0 AND paired image-cluster ordinary95%CI lower>0; speed precedence fixed as best_quality then best_efficiency among confirmed candidates; no Holdout reranking, new layers or quality generation","speed":{"images":12,"warmups_per_method":2,"repeats_per_image":3,"methods":"Native,R0,confirmed final candidate","output_tokens":32,"real_decode":31,"ignore_EOS":True,"boundary":"GPU-ready input through Vision/projector/prefill/decode/method cleanup, synchronized wall time; exclude disk/tokenizer/preprocessor/H2D/write","no_positive_confirmed_gain":"NOT_RUN; never reuse GQA timing"}}
 write(OUT/"protocol.json",protocol)
 sources={str(p.relative_to(RUN/"code")):sha(p) for p in (RUN/"code").rglob("*.py")}
 assert sources["vlm_research/internvl3_svh_r_middle_reaccess_search/engine.py"]=="8751256f3a03d297d2cfc012101d53e2932a5d098ff8f49240e560a686617ff3"
 assert sources["vlm_research/internvl3_svh_physical_d_pilot/engine.py"]=="de5871a49416e507f6cd662646d52929ad8929cde9565ceff4f07b0707655b96"
 write(OUT/"source_manifest.json",{"runtime_source_sha256":sources,"C1_C2_used":False,"V2_engine_unchanged":True})
 inp=w.Inputs(None,tok);bindings={};pixel_hashes={}
 previous=read(RUN/"prepared/input_bindings.json") if (RUN/"prepared/input_bindings.json").exists() else {}
 known_pixels={x["details"]["image_sha256"]:x["details"]["pixels_bf16_sha256"] for x in previous.values()}
 for i,r in enumerate(disc+hold+smoke):
  im=r["image_sha256"];cached=RUN/"prepared"/(im+".pt")
  if cached.exists() and im in known_pixels:
   pix=torch.load(cached,map_location="cpu",weights_only=True);assert w.tensor_sha(pix)==known_pixels[im]
  else:pix=inp.pixels(r["image_path"],im)
  assert pix.dtype==torch.bfloat16 and pix.shape==(1,3,448,448)
  torch.save(pix,RUN/"prepared"/(im+".pt"));pixel_hashes[im]=w.tensor_sha(pix)
  ids=torch.tensor([r["input_ids"]],dtype=torch.long);details={"image_sha256":im,"input_ids_sha256":w.tensor_sha(ids),"pixels_bf16_sha256":pixel_hashes[im],"prompt_sha256":r["prompt_sha256"]};bindings[r["question_id"]]={"details":details,"binding":w.bind(details)}
  if i%32==0:print("CPU_INPUTS",i+1,192,flush=True)
 write(RUN/"prepared/input_bindings.json",bindings)
 # Access only historical Native metadata after sampling freeze; never load Holdout answers/tokens/scores here.
 db=sqlite3.connect(BASE/"checkpoints/TextVQA.sqlite")
 keys=["input_binding_details","input_binding","sample_sha256","gold","max_new_tokens","prompt_sha256","source_manifest_sha256","protocol_sha256"]
 query="SELECT "+",".join("json_extract(record,'$."+k+"')" for k in keys)+" FROM predictions WHERE qid=? AND method='native'"
 signatures=set()
 for r in disc+hold+smoke:
  values=db.execute(query,(r["question_id"],)).fetchone();assert values
  rec=dict(zip(keys,values));rec["input_binding_details"]=json.loads(rec["input_binding_details"]);rec["gold"]=json.loads(rec["gold"])
  assert rec["input_binding_details"]==bindings[r["question_id"]]["details"] and rec["input_binding"]==bindings[r["question_id"]]["binding"]
  assert rec["sample_sha256"]==r["sample_sha256"] and rec["gold"]==r["gold"] and rec["max_new_tokens"]==32 and rec["prompt_sha256"]==r["prompt_sha256"]
  signatures.add((rec["source_manifest_sha256"],rec["protocol_sha256"]))
 db.close()
 parentprotocol=read(BASE/"results/protocol.json");assert parentprotocol["revision"]==protocol["revision"] and parentprotocol["gpu_uuid"]==GPU and parentprotocol["caps"]["TextVQA"]==32
 audit_summary={"status":"PASS","population_prompts_audited":5000,"all_requests_exact_question_plus_frozen_instruction":True,"all_population_native_template_and_input_ids_equal":True,"selected_image_files_SHA_verified":200,"selected_BF16_pixel_tensors_prepared_and_hash_bound":200,"selected_model_tensor_shape":[1,3,448,448],"GPU_input_keys":["ids","pixels","roles","request","prompt"],"auxiliary_OCR_tokens_text_sidecars_captions_hints_gold_rationale_passed_to_model":False,"dataset_directory_contains_ocr":True,"directory_name_not_used_as_evidence":True,"native_conversation_source_sha256":sha(w.MODEL/"conversation.py"),"population_input_audit_sha256":digest(audit),"selected_binding_sha256":sha(RUN/"prepared/input_bindings.json"),"all_methods_share_same_prepared_inputs":True}
 write(OUT/"image_only_input_audit.json",audit_summary);csvout("population_prompt_audit.csv",audit)
 write(OUT/"native_reuse_validation.json",{"status":"CPU_BINDING_PASS_PENDING_GPU_SMOKE","questions":200,"selected_ids_frozen_before_any_checkpoint_access":True,"historical_Native_prediction_payloads_read_during_CPU_prepare":0,"Holdout_prediction_payloads_read_before_final_selection":0,"all_exact_input_tensor_prompt_gold_cap_bindings_equal":True,"parent_protocol_sha256":sha(BASE/"results/protocol.json"),"parent_source_manifest_sha256":sha(BASE/"results/source_manifest.json"),"historical_record_signatures":[list(x) for x in sorted(signatures)],"core_native_engine_sha256":sources["vlm_research/internvl3_svh_physical_d_pilot/engine.py"],"historical_Physical_reused":False})
 write(OUT/"cpu_tests.json",{"status":"PASS","GPU_initialized":torch.cuda.is_initialized(),"image_disjoint":True,"selected_pixel_and_token_bindings":True,"source_V2_unchanged":True,"selected_questions":192,"separate_smoke_questions":8,"population_prompt_audit":5000})
 assert not torch.cuda.is_initialized();print("CPU_PREPARED_TEXTVQA",len(bindings),manifest["subtypes"],flush=True)
if __name__=="__main__":main()
