"""Reuse the exact B128 and verified GPU-ready inputs; freeze one user-requested window."""
from .common import *
import torch,shutil,subprocess
from vlm_research.internvl3_svh_seven_benchmarks import worker as w,scoring
OLD_RUN=Path("/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090")
def main():
 assert not torch.cuda.is_initialized()
 for name in ["sample_manifest.json","input_bindings.json","model_manifest.json"]:shutil.copy2(PARENT/name,OUT/name)
 mf=read(OUT/"sample_manifest.json");bindings=read(OUT/"input_bindings.json")
 for r in mf["Main"]+mf["Smoke"]:
  p=OLD_RUN/"prepared"/(r["image_sha256"]+".pt");x=torch.load(p,weights_only=True,map_location="cpu")
  assert set(x)=={"ids","pixels","roles","request","prompt"} and x["prompt"]==r["prompt"] and x["request"]==r["request"] and x["roles"]==r["roles"]
  details=dict(image_sha256=r["image_sha256"],input_ids_sha256=w.tensor_sha(x["ids"]),pixels_bf16_sha256=w.tensor_sha(x["pixels"]),prompt_sha256=hashlib.sha256(x["prompt"].encode()).hexdigest())
  assert details==bindings[r["question_id"]]["details"] and w.bind(details)==bindings[r["question_id"]]["binding"]
 protocol=read(PARENT/"protocol.json")
 protocol.update(configs=CONFIGS,controls={},formal_quality_requests=128,Main_new=128,Controls_new=0,Main_reused=1664,model_loads_target=1,source_parent_commit=PARENT_COMMIT,speed_gate="NOT_APPLICABLE: user-requested single-window quality supplement only",speed=dict(status="NOT_RUN",reason="Exactly one all24-exceptL2-L5 condition; quality only, no speed request"),GitHub_branch="codex/internvl3-svh-r-all24-textvqa-3090",purpose="User requested all answer reaccess layers except L2-L5: exactly24 KEEP layers; same historical128, NOT independent validation",baseline_origin="same previous 128 questions: Native/R0/R6/R14 originally old GPU, All previously tested physical windows same current GPU",sample_manifest_sha256=sha(OUT/"sample_manifest.json"),input_tensors_from=str(OLD_RUN/"prepared"))
 protocol.update(terminal_cutoff28_semantics="All28 language blocks retain256 image Hidden; remove only afterL27 before tokenwise final norm; NO Prefill language-block visual-row saving",new_engine_scope="byte-identical reuse of the parent terminal-cutoff28 engine; only KEEP expanded to all24 non-SKIP layers; no engine modification")
 write(OUT/"protocol.json",protocol)
 db=sqlite3.connect("file:"+str(PARENT/"server_only/predictions.sqlite")+"?mode=ro&immutable=1",uri=True);assert db.execute("pragma integrity_check").fetchone()==("ok",)
 audit=[];count=0
 for phase,qid,m,origin,h,raw in db.execute("select phase,qid,method,origin,checksum,record from predictions where phase='Main'"):
  assert hashlib.sha256(raw.encode()).hexdigest()==h;x=json.loads(raw);r=next(r for r in mf["Main"] if r["question_id"]==qid)
  assert x["input_binding"]==bindings[qid]["binding"] and x["sample_sha256"]==r["sample_sha256"] and x["gold"]==r["gold"]
  a=scoring.score("TextVQA",x["text"],r)[1];b=scoring.independent_score("TextVQA",x["text"],r);assert abs(a-b)<1e-10 and abs(a-x["score"])<1e-10
  audit.append(dict(phase=phase,question_id=qid,method=m,original_record_sha256=h,origin=origin,input_binding=x["input_binding"]));count+=1
 db.close();assert count==1664
 csvout("reuse_record_audit.csv",audit)
 old=read(PARENT/"source_manifest.json")["runtime_source_sha256"]
 for p,h in old.items():assert sha(RUN/"code"/p)==h
 write(OUT/"input_reuse_validation.json",dict(status="PASS_CPU",questions=128,prepared_inputs_verified=131,reused_records=1664,all_input_binding_Gold_prompt_pixel_ids_roles_model_and_scoring_verified=True,prepared_inputs_contain_no_Gold=True,new_formal_quality=128,no_GQA_or_unrequested_windows=True,explicit_followup_not_independent=True,parent_commit=PARENT_COMMIT,parent_protocol_sha256=sha(PARENT/"protocol.json"),parent_source_manifest_sha256=sha(PARENT/"source_manifest.json")))
 print("ALL24_CPU_REUSE_PASS",flush=True)
if __name__=="__main__":main()
