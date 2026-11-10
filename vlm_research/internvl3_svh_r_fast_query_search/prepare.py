"""CPU-only protocol freeze and selected-input preparation. No CUDA initialization."""
from .common import *
from collections import Counter
import torch,transformers,sys,subprocess
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
def main():
 old=read(OLD/"results/sample_manifest.json")
 def take(split,q,n,label):
  pool=old[split];out=[]
  for c,target in [("query",q)]+[(c,n//4) for c in ["choose","compare","logical","verify"]]:
   a=sorted([r for r in pool if r["category"]==c],key=lambda r:rank(r,label))
   out+=a[:target]
  return sorted(out,key=lambda r:rank(r,label+":order"))
 # Deterministic nested stages from the same frozen 120; no model answers/gold consulted in ranking.
 discovery=take("Discovery",80,40,"fast-discovery")
 A=[]
 for c,target in [("query",48)]+[(c,4) for c in ["choose","compare","logical","verify"]]:
  A+=sorted([r for r in discovery if r["category"]==c],key=lambda r:rank(r,"fast-stage-A"))[:target]
 A=sorted(A,key=lambda r:rank(r,"fast-stage-A:order"));aid={r["question_id"] for r in A};B=[r for r in discovery if r["question_id"] not in aid]
 holdout=take("Holdout",104,56,"fast-holdout")
 # Three shortest and three longest Query inputs, then shortest/longest non-query; ensures length diversity without predictions.
 query=sorted([r for r in discovery if r["category"]=="query"],key=lambda r:(r["input_tokens"],rank(r,"smoke")))
 nonq=sorted([r for r in discovery if r["category"]!="query"],key=lambda r:(r["input_tokens"],rank(r,"smoke")))
 smoke=query[:3]+query[-3:]+[nonq[0],nonq[-1]]
 speed=[];seen=set()
 for c,target in [("query",8)]+[(c,1) for c in ["choose","compare","logical","verify"]]:
  available=sorted([r for r in discovery if r["category"]==c],key=lambda r:rank(r,"speed"))
  for r in available:
   if r["image_sha256"] not in seen:
    speed.append(r);seen.add(r["image_sha256"])
    if sum(x["category"]==c for x in speed)==target:break
 assert len(A)==64 and len(B)==56 and len(holdout)==160 and len(speed)==12
 assert not {r["image_sha256"] for r in discovery}&{r["image_sha256"] for r in holdout}
 manifest={"parent_manifest_sha256":sha(OLD/"results/sample_manifest.json"),"seed":SEED,"selection_uses_gold_or_predictions":False,"rule":"SHA256(seed,label,qid) rank within fixed structural classes; nested 64+56 from 120; exact code prepare.py frozen before GPU","Discovery":discovery,"StageA_ids":[r["question_id"] for r in A],"StageB_ids":[r["question_id"] for r in B],"Holdout":holdout,"smoke_ids":[r["question_id"] for r in smoke],"speed_ids":[r["question_id"] for r in speed],"counts":{n:dict(Counter(r["category"] for r in rr)) for n,rr in [("A",A),("B",B),("Holdout",holdout)]},"image_overlap":0,"holdout_status":"never generated in previous failed search; image-disjoint from Discovery"}
 write(OUT/"sample_manifest.json",manifest);write(OUT/"search_space.json",SPACE)
 protocol={"status":"EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER","historical_numerical_status":"NUMERICAL_EQUIVALENCE_FAILED preserved; not reclassified","model":"OpenGVLab/InternVL3-8B","revision":"853e3a797a661694b1b8ece0cb72dc2b23e3dac9","gpu_uuid":GPU,"dtype":"bfloat16","attention":"eager","torch":"2.6.0+cu124","transformers":"4.49.0","batch":1,"greedy":True,"max_new_tokens":16,"use_cache":True,"logits_to_keep":1,"visual_tokens":256,"image_shape":[1,3,448,448],"single_model_load":True,"engine_parent_commit":"9602d93869d4ccd11986573369d8fecc84b80f78","diagnosis_commit":"2d671f809c0866083a56fe5ecfcb7832fd92f884","base_commit":"3297559f21d3de55a6630a5491c9ecf66301c4b7","engine":"unchanged SVH-R V2; no C1/C2 repair","scratch":"transient zero-only K/V scratch retained and real overhead included in timing; not persistent visual KV","quality_audit":False,"CPU_input_cache":"all selected images prepared before model loading; CPU BF16 tensors on ext4 SSD","sample_manifest_sha256":sha(OUT/"sample_manifest.json"),"search_space":SPACE,"stage_A_selection":{"score":"Query_gain_pp - max(0,NonQuery_loss_pp) - 100*max(0,anomaly_rate-R0_rate) - 0.5*max(0,mean_actual_KV_capacity_MiB-R0_capacity)","selected":"top three reaccess candidates plus R0","ties":"higher Query, lower measured KV capacity, fewer layers, frozen candidate order","anomaly":"any meta/repetition/empty/truncation/explanation flag; all included in scoring"},"stage_B_selection":{"best_quality":"highest Query accuracy; ties lower actual KV capacity, fewer layers, frozen order","best_efficiency":"within 2.5pp of highest Query and NonQuery no worse than R0-2.5pp; minimize number of layers, then actual KV capacity; if no eligible candidate use R0","includes_R0":True,"measured_KV_cost":"arithmetic mean of actual quality-EOS cache capacity; output lengths differ and are recorded; fixed32 KV measured separately","holdout_selection_frozen_before_predictions":True},"smoke_hard_gates":{"structural":"any permission/KV/position/skip/exit/finite/input/Native-Noop/hook failure stops","severe_output":"per condition >=3 of 8 meta/empty/repetitive outputs stops","widespread_disagreement":"per condition >=4 of 8 parsed-answer disagreements Dense vs Physical stops","known_BF16_logit_error":"record only; no old MaxAbs gate","multitoken_coverage":"at least one observed valid multi-token answer; otherwise pause, do not add outcome-selected samples"},"quality_stop_policy":{"structural":"immediate stop","systematic_output":"two consecutive nonoverlapping 32-request windows within a condition each have meta>=25% or empty/repeat>=12.5% or truncation>=50%; OR ten consecutive empty/repetitive outputs. Individual anomalies always retained and scored"},"bootstrap":{"unit":"image_sha256 whole paired clusters","repetitions":5000,"seed":SEED,"CI":95,"multiplicity":"ordinary unadjusted exploratory intervals; selection and many subgroups prevent confirmatory claims"},"speed":{"images":12,"conditions":"Native,R0,best_efficiency deduplicated","warmups_per_condition":2,"repeats":3,"output_tokens":32,"decode_steps":31,"ignore_EOS":True,"boundary":"GPU-ready input through Vision/projector/prefill/decode and method cleanup; synchronized wall time; no disk/tokenizer/H2D/results writing; unmodified frozen engine","interleaved":True},"budget":{"A":448,"B_max":224,"Holdout_max":480,"quality_max_excluding_unreusable_Native":1152}}
 write(OUT/"protocol.json",protocol)
 # Verify frozen sources before opening cached model/tokenizer.
 code=RUN/"code";sources={str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")}
 assert sources["vlm_research/internvl3_svh_r_middle_reaccess_search/engine.py"]=="8751256f3a03d297d2cfc012101d53e2932a5d098ff8f49240e560a686617ff3"
 assert sources["vlm_research/internvl3_svh_physical_d_pilot/engine.py"]=="de5871a49416e507f6cd662646d52929ad8929cde9565ceff4f07b0707655b96"
 write(OUT/"source_manifest.json",{"runtime_source_sha256":sources,"engine_parent_commit":protocol["engine_parent_commit"],"C1_C2_used":False})
 tok=w.AutoTokenizer.from_pretrained(w.MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
 inp=w.Inputs(None,tok);allrows={r["question_id"]:r for r in discovery+holdout+smoke};imgs={}
 for r in allrows.values():imgs.setdefault(r["image_sha256"],r)
 pixhash={}
 for i,(im,r) in enumerate(imgs.items()):
  pix=inp.pixels(r["image_path"],im);assert pix.shape==(1,3,448,448) and pix.dtype==torch.bfloat16
  pixhash[im]=w.tensor_sha(pix);torch.save(pix,RUN/"prepared"/(im+".pt"))
  if i%25==0:print("CPU_IMAGES",i+1,len(imgs),flush=True)
 db=sqlite3.connect(BASE/"checkpoints/GQA.sqlite");bindings={};native={};errors=[]
 for q,r in allrows.items():
  ids=tok(r["prompt"],return_tensors="pt")["input_ids"];assert ids[0].tolist()==r["input_ids"]
  assert w.sha(r["prompt"])==r["prompt_sha256"] and w.request_text("GQA",r)==r["request"]
  assert w.b.role_layout(ids[0].tolist(),tok.convert_tokens_to_ids("<IMG_CONTEXT>"))==r["roles"]
  details={"image_sha256":r["image_sha256"],"input_ids_sha256":w.tensor_sha(ids),"pixels_bf16_sha256":pixhash[r["image_sha256"]],"prompt_sha256":r["prompt_sha256"]}
  bindings[q]={"details":details,"binding":w.bind(details)}
  raw=db.execute("SELECT record FROM predictions WHERE qid=? AND method='native'",(q,)).fetchone();assert raw
  rec=json.loads(raw[0]);ok=rec["input_binding_details"]==details and rec["input_binding"]==w.bind(details) and rec["sample_sha256"]==r["sample_sha256"] and rec["gold"]==r["gold"] and rec["max_new_tokens"]==16 and rec["prompt_sha256"]==r["prompt_sha256"]
  if not ok:errors.append(q)
  native[q]=rec
 assert not errors,("NATIVE_BINDING_MISMATCH",errors)
 write(RUN/"prepared/input_bindings.json",bindings);write(RUN/"prepared/historical_native.json",native)
 # All historical model/source/protocol receipts; no old timing or old Physical scores reused.
 parentprotocol=read(BASE/"results/protocol.json");assert parentprotocol["revision"]==protocol["revision"] and parentprotocol["gpu_uuid"]==GPU
 write(OUT/"native_reuse_validation.json",{"status":"PASS_CPU_BINDINGS_PENDING_GPU_SMOKE","questions":len(native),"all_image_prompt_ids_pixel_gold_cap_bindings_equal":True,"parent_protocol_sha256":sha(BASE/"results/protocol.json"),"parent_source_manifest_sha256":sha(BASE/"results/source_manifest.json"),"Native_engine_sha256":sources["vlm_research/internvl3_svh_physical_d_pilot/engine.py"],"model_integrity_receipt":str(BASE/"results/model_and_library_integrity_v2.json"),"model_integrity_receipt_sha256":sha(BASE/"results/model_and_library_integrity_v2.json"),"new_R0_required":True,"historical_Physical_reused":False,"holdout_scores_not_used_for_selection":True})
 write(OUT/"cpu_tests.json",{"status":"PASS","selected_images_verified":len(imgs),"prepared_rows":len(allrows),"GPU_initialized":torch.cuda.is_initialized(),"classes_and_counts_frozen":True,"Discovery_Holdout_image_disjoint":True,"K_subset_L10_L19":all(set(k)<=set(range(10,20)) for k in SPACE.values()),"source_engine_unchanged":True,"request_bindings":True})
 assert not torch.cuda.is_initialized()
 print("CPU_PREPARED",len(allrows),len(imgs),flush=True)
if __name__=="__main__":main()
