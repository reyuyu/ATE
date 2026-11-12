"""Paths, immutable evidence writes and original numerical gate."""
from pathlib import Path
import os,json,hashlib,csv,uuid,time
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-l20-numerical-diagnosis-3090");RUN=Path("/home/featurize/ate-runs/internvl3-svh-r-l20-numerical-diagnosis-3090");PKG=Path("vlm_research/internvl3_svh_r_l20_numerical_diagnosis");REL=Path("vlm_research/results/internvl3_svh_r_l20_numerical_diagnosis_3090");OUT=RUN/"results"
BRANCH="codex/internvl3-svh-r-l20-numerical-diagnosis-3090";PARENT="9602d93869d4ccd11986573369d8fecc84b80f78";GPU="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e";REV="853e3a797a661694b1b8ece0cb72dc2b23e3dac9"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def atomic_bytes(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
 try:
  with tmp.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.link(tmp,p)  # atomic no-clobber: raises FileExistsError on any old artifact
 finally:
  tmp.unlink(missing_ok=True)
def writej(p,data):atomic_bytes(p,(json.dumps(data,indent=2,ensure_ascii=False)+"\n").encode())
def csvonce(path,rows,fields=None):
 import io
 b=io.StringIO(newline="");fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 w=csv.DictWriter(b,fieldnames=fields);w.writeheader()
 for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
 atomic_bytes(path,b.getvalue().encode())
def freeze():
 prior=readj(RUN/"references/protocol.json")
 writej(OUT/"protocol.json",{"parent_commit":PARENT,"GPU_UUID":GPU,"model":"OpenGVLab/InternVL3-8B","revision":REV,"dtype":"BF16","attention":"eager","torch":"2.6.0+cu124","transformers":"4.49.0","greedy":True,"batch_size":1,"max_num":1,"image_size":448,"visual_tokens":256,"quality_max_new_tokens":16,"native_EOS":True,"use_cache":True,"logits_to_keep":1,"numerical_gate":prior["numerical_gate"],"sample_binding_sha256":sha(OUT/"sample_binding.json"),"scope":"A/B frozen R0 L20 operator diagnosis; <=2 local repairs; complete R0 first sample, further prior 7 Smoke only if first passes; no Discovery/Holdout/quality scoring/E2E32 benchmark","A_variants":{"A1":"original321 normalized Hidden native Q/K/V then gather65 text rows","A2":"gather65 actual text normalized Hidden then same native Q/K/V","A3":"zero321 scratch populated only at same65 original text positions; native Q/K/V then gather65"},"B":"Independent native-operator replay, same frozen L20 Hidden/RoPE/mask, compare legal original logical keys only; no QK norm exists in actual Qwen2","C1":"L20 K/V projections only, Prefill only: zero pad original sequence rows, gather text results, no persistent padding or visual data","C2":"NOT_YET_DEFINED; may target only next operator proven after C1, separately frozen before test; no precision/kernel/layer search","cost":{"warmup":25,"repeats":100,"order":"rotate A1/A2/A3 by repeat","CUDA_event":True,"include_in_repair":"allocation+zero+index_copy+native K/V+gather","no_CPU_audit_in_timing":True,"inputs":"primary frozen sample; additional frozen Smoke lengths only if needed","not_E2E32":True},"raw_evidence":"unique phase paths with atomic no-clobber writes; full BF16 tensors server-only, SHA/shape/dtype metadata GitHub","old_waiver_inherited":False,"prepared_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
 writej(OUT/"candidate_repairs.json",{"max_candidates":2,"C1":{"layer":20,"modules":["k_proj","v_proj"],"stage":"Prefill only","input":"zero scratch at original positions, no original image Hidden","cache":"compact unchanged","condition":"run only if A/B establish shape-related K/V differences"},"C2":{"status":"UNDEFINED_NOT_RUN","freeze_rule":"next independently proven local divergence only; immutable separate candidate document before GPU execution"}})
 writej(OUT/"source_manifest.json",{"parent_commit":PARENT,"protocol_sha256":sha(OUT/"protocol.json"),"frozen_before_GPU":True,"source_sha256":{str(p.relative_to(RUN/"code")):sha(p) for d in ["internvl3_svh_r_middle_reaccess_search","internvl3_svh_physical_d_pilot","internvl3_visual_writing_figure3_3000","internvl3_svh_seven_benchmarks",PKG.name] for p in sorted((RUN/"code/vlm_research"/d).glob("*.py"))},"original_evidence_sha256":readj(RUN/"references/prior_evidence_manifest.json")})
 print("PROTOCOL_AND_SOURCES_FROZEN",sha(OUT/"protocol.json"),flush=True)
if __name__=="__main__":freeze()
