"""Freeze image-disjoint Discovery/Holdout and rules without reading any predictions."""
from .common import *
import gzip,collections,re
def rank(q,label):return sha(str(SEED)+":"+label+":"+str(q))
def subtype(r):
 q=r["question"].lower();t=r["types"]
 if re.search(r"^how many\b|\bnumber of\b",q):return "quantity"
 if re.search(r"\bcolou?r\b",q):return "color"
 if t["detailed"]=="positionQuery" or re.search(r"^where\b|^on which side\b",q):return "spatial"
 if t["semantic"]=="rel":return "relation"
 if t["semantic"] in ("cat","obj"):return "object"
 if t["semantic"]=="attr":return "attribute"
 return "other"
def take(a,n,label,used):
 a=sorted(a,key=lambda r:rank(r["question_id"],label));out=[]
 for unique in (True,False):
  for x in a:
   if x in out or (unique and x["image_sha256"] in used):continue
   out.append(x);used.add(x["image_sha256"])
   if len(out)==n:return out
 return out
def freeze():
 OUT.mkdir(exist_ok=True)
 assert not (RUN/"checkpoints/search.sqlite").exists(),"FREEZE_AFTER_GENERATION_FORBIDDEN"
 with gzip.open(RUN/"references/GQA_full_manifest.jsonl.gz","rt") as f:rows=[json.loads(x) for x in f]
 assert len(rows)==12578 and len({x["question_id"] for x in rows})==12578
 assert all(sha(x["prompt"])==x["prompt_sha256"] and "\n" in x["request"] and x["max_new_tokens"]==16 for x in rows)
 hashes=sorted({x["image_sha256"] for x in rows},key=lambda x:rank(x,"image-partition"))
 assert len(hashes)==398
 parts={"Discovery":set(hashes[:199]),"Holdout":set(hashes[199:])}
 sets={};summary={}
 for split,images in parts.items():
  pool=[x for x in rows if x["image_sha256"] in images];used=set();chosen=[]
  for typ in ["choose","compare","logical","verify"]:
   chosen+=take([x for x in pool if x["category"]==typ],20,split+typ,used)
  query=[x for x in pool if x["category"]=="query"];sel=[]
  for st in ["object","color","spatial","relation","quantity","attribute","other"]:
   sel+=take([x for x in query if subtype(x)==st],min(20,sum(subtype(x)==st for x in query)),split+"query-"+st,used)
  ids={x["question_id"] for x in sel}
  sel+=take([x for x in query if x["question_id"] not in ids],max(0,160-len(sel)),split+"query-fill",used)
  chosen+=sel
  chosen=sorted(chosen,key=lambda x:rank(x["question_id"],split+"final"))
  sets[split]=chosen
  summary[split]={"N":len(chosen),"image_clusters":len({x["image_sha256"] for x in chosen}),"categories":dict(collections.Counter(x["category"] for x in chosen)),"query_subtypes":dict(collections.Counter(subtype(x) for x in chosen if x["category"]=="query")),"available_categories":dict(collections.Counter(x["category"] for x in pool))}
 assert not ({x["image_sha256"] for x in sets["Discovery"]}&{x["image_sha256"] for x in sets["Holdout"]})
 assert all(len(s)==240 and sum(x["category"]=="query" for x in s)==160 for s in sets.values())
 # Input-length quantiles, different images, no answers or predictions used.
 efficiency=[];seen=set()
 for typ,n in [("query",12),("choose",2),("compare",2),("logical",2),("verify",2)]:
  pool=sorted([r for r in sets["Discovery"] if r["category"]==typ],key=lambda r:(r["input_tokens"],rank(r["question_id"],"efficiency")))
  for j in range(n):
   target=j*(len(pool)-1)/max(1,n-1)
   candidates=sorted(enumerate(pool),key=lambda z:(abs(z[0]-target),z[0]))
   x=next(x for i,x in candidates if x["image_sha256"] not in seen);efficiency.append(x["question_id"]);seen.add(x["image_sha256"])
 assert len(efficiency)==20 and len(seen)==20
 discovery=sets["Discovery"];smoke=[]
 for typ in ["query","choose","compare","logical","verify"]:
  a=[r for r in discovery if r["category"]==typ];smoke.append(min(a,key=lambda r:rank(r["question_id"],"smoke"))["question_id"])
 for r in [min(discovery,key=lambda x:x["input_tokens"]),max(discovery,key=lambda x:x["input_tokens"])]:
  if r["question_id"] not in smoke:smoke.append(r["question_id"])
 for r in sorted(discovery,key=lambda r:rank(r["question_id"],"smoke-fill")):
  if len(smoke)==8:break
  if r["question_id"] not in smoke:smoke.append(r["question_id"])
 samples={"seed":SEED,"source_manifest_sha256":sha(RUN/"references/GQA_full_manifest.jsonl.gz"),"source_question_count":12578,"source_image_clusters":398,"sampling_uses_predictions_or_Gold":False,"image_partition":"SHA256(seed:image-partition:imageSHA) order; first199 Discovery/remaining199 Holdout","sampling_rule":"20 each non-query class; up to20 each query subtype then deterministic fill to160; prefer previously unselected image within split; hash order; no score/Gold used","fallback":"Per stratum min(target,available), no image repartition or outcome-dependent substitution; insufficient target reported before GPU","query_subtype_rule":"question-only regex plus real GQA semantic/detailed types, implemented in prepare.py","summary":summary,"Discovery":sets["Discovery"],"Holdout":sets["Holdout"],"efficiency_ids":efficiency,"smoke_ids":smoke,"smoke_multitoken_coverage_rule":"If initial8 lack a multi-token historical Native answer, replace last smoke with Discovery-only maximal historical Native token length (qid hash tie); only coverage, never formal sample selection or Gold correctness. Save finalized smoke IDs before GPU.","holdout_label":"INTERNAL_FROZEN_VALIDATION; original full GQA summaries previously viewed; not external confirmation","diagnostic_panel":[]}
 writej(OUT/"sample_manifest.json",samples)
 writej(OUT/"search_space.json",{"frozen_before_GPU":True,"conditions":SPACE,"candidate_order":list(SPACE),"Native_reference":True,"visual_hidden_exit_before_layer":20,"skip_answer_attention_layers":[2,3,4,5],"permitted_reaccess_layers":list(range(6,20))})
 writej(OUT/"selection_rule.json",{"frozen_before_GPU":True,"eligibility":["structure/numerical gates PASS","Non-Query point difference vs R0 >= -1pp"],"order":"Query Accuracy descending","near_optimal_Query_gap_pp":1,"best_cost":"minimum measured actual KV Capacity among near-optimal; then fewer reaccess layers; then frozen candidate order","best_query":"max Query Accuracy; then frozen candidate order","max_holdout_candidates":2,"no_holdout_inference_or_candidate_predictions_before_selection_SHA":True,"threshold_interpretation":"exploratory engineering screening, not noninferiority proof","confirmation_rule":"SEARCH_CONFIRMED only if selected cost candidate Holdout Query improvement vs R0 >0pp and Non-Query difference >=-1pp; report CI uncertainty separately, never retune"})
 gate={"native_noop":"torch.equal all effective logits and complete tokens","physical_dense":{"per_step_logits_max_abs_le":0.125,"per_step_logits_mean_abs_le":0.01,"per_layer_shared_hidden_relative_RMS_le":0.015625,"all_step_top1_equal":True,"complete_tokens_equal":True,"all_finite":True},"same_prefix_comparison":True,"rationale":"Reuse original pre-waiver BF16 engineering bounds unchanged (max .125, mean .01, hidden1/64). Previous waiver is not inherited.","failure_policy":"No formal Physical-R search on structural/No-op/KV failure; a numeric failure is diagnosed and remains FAILED, never waived or relaxed based on accuracy."}
 protocol={"experiment":"InternVL3-SVH-R middle reaccess small-sample search","parent_result_commit":PARENT_COMMIT,"model":"OpenGVLab/InternVL3-8B","revision":REV,"GPU_UUID":GPU,"dtype":"BF16","attention":"eager","torch":"2.6.0+cu124","transformers":"4.49.0","batch_size":1,"max_num":1,"visual_tokens":256,"quality_max_new_tokens":16,"normal_EOS":True,"greedy":True,"logits_to_keep":1,"use_cache":True,"prompt_scorer":"Exact parent ATE_SVH_1TILE_SCORER_ALIGNED GQA, image-only","sample_manifest_sha256":sha(OUT/"sample_manifest.json"),"search_space_sha256":sha(OUT/"search_space.json"),"selection_rule_sha256":sha(OUT/"selection_rule.json"),"numerical_gate":gate,"legacy_Physical_D_status":"NUMERICAL_EQUIVALENCE_FAILED; prior user waiver retained as history, does not authorize new SVH-R failure","implementation_v1":"Independent Control/CompactCache. At skipped prefill L2-5 only: append one transient zero K/V row with a fully masked P-1 column, preserving original AV reduction K dimension; P-1 QKV/QK/AV/o_proj never executed, no skipped-layer persistent cache. All selected K<=19 keep real256 imageKV. Deep Hidden exit20 unchanged.","baseline_reuse":"Native only if exact parent source/prompt/inputTensor/generation tokens verified; R0 implementation source differs due prefill reduction-shape compatibility so regenerate same-GPU R0, never silently reuse legacy Physical-D.","bootstrap_reps":20000,"bootstrap_seed":SEED,"cluster_bootstrap":"paired Image SHA, question-weighted ordinary95%, no multiplicity correction","efficiency":{"inputs":20,"query_min":12,"tokens":32,"prefill":1,"decode":31,"ignore_EOS":True,"warmups_per_method":2,"repeats":5,"order":"input-major, repeat-major; rotate condition order by input+repeat","timer":"GPU-ready pixels/IDs through vision+projector+injection+prefill+31decode+method install/restoration and transient cleanup; synchronize wallclock","excludes":["model load","disk I/O","tokenization","image preprocessing","H2D","output writes","audit hooks"],"KV":"real tensor bytes for Live; unique underlying storage.nbytes for Capacity, cache kept live when measured"},"output_monitor":{"nonoverlap_window":40,"consecutive_windows":2,"empty_fraction":0.1,"repetition_fraction":0.1,"meta_or_format_fraction":0.25,"truncation_fraction":0.5,"consecutive_empty_or_repetition":10,"meta_regex":"^answering\\b|does not require|as an ai|cannot (?:see|determine)|unable to","wrong_answers_not_stop_signal":True,"individual_anomalies_scored_unchanged":True},"no_full_benchmark":True,"no_training":True,"no_layer_expansion":True,"GitHub_branch":BRANCH,"prepared_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 writej(OUT/"protocol.json",protocol)
 state("CPU_SAMPLE_FREEZE_COMPLETE",summary=summary,search_space_sha256=sha(OUT/"search_space.json"))
 print("SAMPLES_FROZEN",summary,"efficiency",len(efficiency),"smoke",smoke,flush=True)
if __name__=="__main__":freeze()
