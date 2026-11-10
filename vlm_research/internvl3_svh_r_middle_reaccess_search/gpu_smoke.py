"""GPU correctness only; hard numerical gate, no inherited waiver, no quality scoring."""
from .common import *
from . import engine as e
import os,sqlite3,gc,traceback,inspect,shutil
import torch
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
def sources():
 d={}
 for folder in ["internvl3_svh_r_middle_reaccess_search","internvl3_svh_physical_d_pilot","internvl3_visual_writing_figure3_3000","internvl3_svh_seven_benchmarks"]:
  for p in sorted((RUN/"code/vlm_research"/folder).glob("*.py")):d[str(p.relative_to(RUN/"code"))]=sha(p)
 out={"status":"FROZEN_BEFORE_GPU_SMOKE","parent_commit":PARENT_COMMIT,"inference_source_SHA256":d,"protocol_sha256":sha(OUT/"protocol.json"),"search_space_sha256":sha(OUT/"search_space.json"),"sample_manifest_sha256":sha(OUT/"sample_manifest.json"),"selection_rule_sha256":sha(OUT/"selection_rule.json")}
 writej(OUT/"source_manifest.json",out);return out
def compare(q,name,dense,physical,ds,ps,dc,pc):
 gate=readj(OUT/"protocol.json")["numerical_gate"]["physical_dense"];rows=[];passed=True
 for step,(a,b) in enumerate(zip(ds,ps)):
  diff=(a.float()-b.float()).abs();mx=float(diff.max());mean=float(diff.mean());top=bool(a.argmax()==b.argmax())
  ok=mx<=gate["per_step_logits_max_abs_le"] and mean<=gate["per_step_logits_mean_abs_le"] and top
  rows.append(dict(question_id=q,condition=name,kind="logits",step=step,layer=27,max_abs=mx,mean_abs=mean,relative_RMS=None,top1_equal=top,passed=ok));passed &=ok
 for key,(pos,pv) in pc.hidden.items():
  step,layer=key;dp,dv=dc.hidden[key]
  idx={v:i for i,v in enumerate(dp.tolist())};dv=dv[:,[idx[i] for i in pos.tolist()]].float();pv=pv.float()
  diff=(pv-dv);rel=float(diff.square().mean().sqrt()/dv.square().mean().sqrt().clamp_min(1e-6))
  ok=rel<=gate["per_layer_shared_hidden_relative_RMS_le"];passed &=ok
  rows.append(dict(question_id=q,condition=name,kind="hidden",step=step,layer=layer,max_abs=float(diff.abs().max()),mean_abs=float(diff.abs().mean()),relative_RMS=rel,top1_equal=None,passed=ok))
 return passed,rows
def main():
 model=None;inputs=None;numerical=[];checks=[];kv=[];final_ids=[]
 try:
  cpu=readj(OUT/"cpu_tests.json");assert cpu["status"]=="PASS" and cpu["engine_sha256"]==sha(RUN/"code"/PKG/"engine.py")
  frozen=sources();w.gpu_idle();samples=readj(OUT/"sample_manifest.json");discovery=samples["Discovery"];index={x["question_id"]:x for x in discovery};final_ids=list(samples["smoke_ids"])
  # Coverage adjustment uses only token length of already frozen Discovery Native.
  db=sqlite3.connect("file:"+str(PARENT_RUN/"checkpoints/GQA.sqlite")+"?mode=ro",uri=True)
  historical={}
  for r in discovery:
   raw=db.execute("SELECT record FROM predictions WHERE qid=? AND method='native'",(r["question_id"],)).fetchone()
   assert raw is not None;historical[r["question_id"]]=json.loads(raw[0])
  if not any(len(historical[q]["token_ids"])>2 for q in final_ids):
   best=min(discovery,key=lambda r:(-len(historical[r["question_id"]]["token_ids"]),sha(str(SEED)+r["question_id"])))["question_id"]
   final_ids[-1]=best
  db.close()
  assert len(final_ids)==len(set(final_ids))==8
  writej(OUT/"smoke_sample_manifest.json",{"frozen_before_model_load":True,"ids":final_ids,"source":"Discovery only; native length coverage only, no accuracy or Gold examined","historical_native_lengths":{q:len(historical[q]["token_ids"]) for q in final_ids}})
  state("SMOKE_LOADING_MODEL");model,tok=w.load_model();writej(OUT/"environment.json",w.environment(model,tok));inputs=w.Inputs(model,tok)
  all_pass=True;multi=False
  for sample_no,q in enumerate(final_ids):
   r=index[q];ready,ib,binding=inputs.ready(r)
   native,ns,nc=e.generate(model,tok,ready,"Native",[],audit=True)
   noop,zs,zc=e.generate(model,tok,ready,"No-op",[],audit=True)
   assert native["token_ids"]==noop["token_ids"] and all(torch.equal(a,b) for a,b in zip(ns,zs)),"NATIVE_NOOP_FAILED"
   assert ib==historical[q]["input_binding"] and binding==historical[q]["input_binding_details"]
   assert native["token_ids"]==historical[q]["token_ids"],"HISTORICAL_NATIVE_TOKEN_MISMATCH"
   with e.frozen.OfficialChatCapture(model) as capture:
    text=model.chat(tok,ready["pixels"],ready["request"],dict(do_sample=False,num_beams=1,max_new_tokens=16,min_new_tokens=0,logits_to_keep=1,pad_token_id=tok.convert_tokens_to_ids(model.conv_template.sep.strip())))
   assert capture.ids==native["token_ids"] and text.strip()==native["text"]
   assert capture.input_ids[0].tolist()==r["input_ids"] and len(ns)==len(capture.scores) and all(torch.equal(a.float(),b) for a,b in zip(ns,capture.scores)),"OFFICIAL_CHAT_LOGITS_FAILED"
   multi |= len(native["token_ids"])>2
   for name,K in SPACE.items():
    dense,ds,dc=e.generate(model,tok,ready,name,K,audit=True,physical=False)
    physical,ps,pc=e.generate(model,tok,ready,name,K,audit=True,physical=True)
    # Same-prefix replay prevents autoregressive divergence being mislabeled roundoff.
    teacher,ts,tc=e.generate(model,tok,ready,name,K,max_tokens=len(dense["token_ids"]),fixed=True,audit=True,forced=dense["token_ids"],physical=True)
    ok,rows=compare(q,name,dense,teacher,ds,ts,dc,tc)
    token_equal=physical["token_ids"]==dense["token_ids"];ok &= token_equal
    numerical.extend(rows);all_pass &=ok
    record={"question_id":q,"condition":name,"reaccess_layers":K,"native_noop_official_logits_strict_equal":True,"historical_native_input_tokens_equal":True,"input_binding":ib,"input_binding_details":binding,"dense_tokens":dense["token_ids"],"physical_tokens":physical["token_ids"],"dense_text":dense["text"],"physical_text":physical["text"],"full_tokens_equal":token_equal,"numerical_gate_pass":ok,"max_logits_abs":max(x["max_abs"] for x in rows if x["kind"]=="logits"),"max_hidden_relative_RMS":max(x["relative_RMS"] for x in rows if x["kind"]=="hidden"),"physical_execution":physical["audit"]}
    # One actual fixed8 cache/attention extension per condition, after numeric PASS.
    if sample_no==0 and ok:
     a,ads,adc=e.generate(model,tok,ready,name,K,max_tokens=8,fixed=True,audit=True,physical=False)
     b,bds,bdc=e.generate(model,tok,ready,name,K,max_tokens=8,fixed=True,audit=True,forced=a["token_ids"],physical=True)
     extra,extra_rows=compare(q,name,a,b,ads,bds,adc,bdc);numerical.extend(extra_rows);ok &=extra;all_pass &=extra
     record["fixed8_teacher_prefix_gate_pass"]=extra;record["fixed8_physical_audit"]=b["audit"]
     del a,ads,adc,b,bds,bdc
    checks.append(record)
    kv.extend(dict(question_id=q,condition=name,**row) for row in physical["kv_layout"])
    if not ok:
     raw=RUN/"diagnostics";raw.mkdir(exist_ok=True)
     torch.save({"question_id":q,"condition":name,"dense_scores":ds,"same_prefix_physical_scores":ts,"free_physical_scores":ps,"dense_tokens":dense["token_ids"],"physical_tokens":physical["token_ids"],"input_binding":binding},raw/(name+"_"+q+"_FAILED_scores.pt"))
    csvwrite(OUT/"numerical_comparison.csv",numerical);csvwrite(OUT/"kv_layout.csv",kv)
    writej(OUT/"smoke_validation.json",{"status":"RUNNING","checks":checks,"all_numeric_pass_so_far":all_pass,"old_numerical_failure_and_waiver_not_inherited":True,"formal_discovery":"NOT_RUN","formal_holdout":"NOT_RUN","formal_efficiency":"NOT_RUN"})
    print("SMOKE",q,name,"PASS" if ok else "FAILED",record["max_logits_abs"],record["max_hidden_relative_RMS"],flush=True)
    del dense,ds,dc,physical,ps,pc,teacher,ts,tc;gc.collect()
   del ready,native,ns,nc,noop,zs,zc,capture;gc.collect()
   if not all_pass:break
  status="PASS" if all_pass and len(checks)==8*len(SPACE) and multi else "FAILED"
  writej(OUT/"smoke_validation.json",{"status":status,"checks":checks,"all_numeric_pass":all_pass,"multi_token_answer_covered":multi,"completed_samples":len({x["question_id"] for x in checks}),"planned_samples":8,"old_numerical_failure_and_waiver_not_inherited":True,"formal_discovery":"NOT_RUN","formal_holdout":"NOT_RUN","formal_efficiency":"NOT_RUN","numerical_gate":readj(OUT/"protocol.json")["numerical_gate"],"source_manifest_sha256":sha(OUT/"source_manifest.json")})
  state("SMOKE_PASS" if status=="PASS" else "SMOKE_FAILED_NO_FORMAL_SEARCH")
 except Exception as x:
  writej(OUT/"smoke_validation.json",{"status":"FAILED","error":str(x),"traceback":traceback.format_exc(),"checks":checks,"formal_discovery":"NOT_RUN","formal_holdout":"NOT_RUN","formal_efficiency":"NOT_RUN"})
  state("SMOKE_FAILED_EXCEPTION",error=str(x));print(traceback.format_exc(),flush=True)
 finally:
  if inputs is not None:inputs.pixels.cache_clear()
  model=None;inputs=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("SMOKE_WORKER_GPU_RELEASED",flush=True)
if __name__=="__main__":main()
