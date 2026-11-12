"""Stop policy v2: observed output behavior is scored; execution integrity remains mandatory."""
from .common import *
from collections import defaultdict
import numpy as np
TARGETS=["GQA","TextVQA","OCRBench"]
PARENT="6c294aae67ddb2a5cb3706ded76dcf3edb26c859"
OLD_RUN=Path("/home/featurize/ate-runs/internvl3-svh-seven-benchmarks-3090")
OLD_ROOT=Path("/home/featurize/work/ATE-internvl3-svh-seven-benchmarks-3090")
PKG=Path("vlm_research/internvl3_svh_seven_benchmarks")
BRANCH="codex/internvl3-svh-stop-policy-v2-three-benchmarks-3090"
class HardStop(RuntimeError):pass
def policy():
    return {"version":"stop_policy_v2","parent_commit":PARENT,"original_protocol_sha256":sha(OUT/"protocol.json"),"original_source_manifest_sha256":sha(OUT/"source_manifest.json"),"generation_protocol_unchanged":True,"changes_only":"Continuation policy and audit; no prompt, cap, model, preprocessing, physical mechanism, greedy/EOS, data or scorer changes.","targets":TARGETS,"reused_full_benchmarks":["ScienceQA-IMG","POPE","MME","MMBench"],"output_anomaly_continue":["wrong or unparseable answer","unrelated explanation","repeated words/phrases","truncation at the original task cap"],"hard_stop":["NaN/Inf","OOM","more output tokens than cap","KV/RoPE/position/attention structural failure","input/source SHA mismatch","Native/No-op/official Chat mismatch","damaged checkpoint","paired input mismatch"],"systemic_monitor":{"window_questions_per_method":100,"consecutive_nonoverlapping_windows":2,"empty_fraction":0.10,"repetition_fraction":0.10,"other_behavior_anomaly_fraction":0.25,"truncation_fraction":0.50,"consecutive_empty_or_repetitive_questions":10,"accuracy_or_gold_never_used_for_stopping":True,"never_automatically_adjust_thresholds_or_generation":True},"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER"}
def check_sources():
    for rel,h in readj(OUT/"source_manifest.json")["runtime_source_sha256"].items():
        if sha(RUN/"code"/rel)!=h:raise HardStop("FROZEN_RUNTIME_SOURCE_SHA_MISMATCH "+rel)
    ex=readj(OUT/"execution_source_manifest_v2.json")
    for rel,h in ex["continuation_source_sha256"].items():
        if sha(RUN/"code"/rel)!=h:raise HardStop("V2_CONTINUATION_SOURCE_SHA_MISMATCH "+rel)
    if sha(OUT/"stop_policy_v2.json")!=ex["stop_policy_sha256"]:raise HardStop("POLICY_SHA_MISMATCH")
def validate_record(x,r):
    if x["sample_sha256"]!=r["sample_sha256"] or x["prompt_sha256"]!=r["prompt_sha256"] or x["image_sha256"]!=r["image_sha256"]:raise HardStop("INPUT_BINDING_MISMATCH")
    if x["protocol_sha256"]!=sha(OUT/"protocol.json") or x["source_manifest_sha256"]!=sha(OUT/"source_manifest.json"):raise HardStop("GENERATION_PROTOCOL_SOURCE_MISMATCH")
    ids=x["token_ids"];n=x["generated_tokens"]
    if len(ids)!=n or not 1<=n<=CAPS[r["benchmark"]] or x["max_new_tokens"]!=CAPS[r["benchmark"]]:raise HardStop("TOKEN_COUNT_EXCEEDS_CAP_OR_INVALID")
    if bool(x["eos"])!=(ids[-1]==151645) or bool(x["truncated"])!=(n==CAPS[r["benchmark"]] and ids[-1]!=151645):raise HardStop("EOS_OR_LENGTH_METADATA_MISMATCH")
    details=x["input_binding_details"]
    if bind(details)!=x["input_binding"] or details["prompt_sha256"]!=r["prompt_sha256"] or details["image_sha256"]!=r["image_sha256"]:raise HardStop("INPUT_DETAIL_SHA_MISMATCH")
    expected=sha(np.asarray(r["input_ids"],dtype="<i8").tobytes())
    if details["input_ids_sha256"]!=expected:raise HardStop("INPUT_ID_TENSOR_SHA_MISMATCH")
    logical=r["input_tokens"]+n-1;live=0
    if len(x["kv_layout"])!=28 or sorted(k["layer"] for k in x["kv_layout"])!=list(range(28)):raise HardStop("KV_LAYER_COUNT_MISMATCH")
    for k in x["kv_layout"]:
        layer=k["layer"];native=x["method"]=="native";length=logical if native else (0 if layer in (2,3,4,5) else logical-256)
        if k["logical_length"]!=logical or k["physical_length"]!=length or k["visual_kv"]!=(256 if native else 0):raise HardStop("KV_PHYSICAL_OR_LOGICAL_LAYOUT_MISMATCH")
        live+=k["live_bytes"]
    if abs(live/1048576-x["KV_live_MiB"])>1e-10:raise HardStop("KV_LIVE_BYTES_MISMATCH")
def behavior(x):
    # Accuracy / gold is deliberately absent. Wrong answers remain ordinary scored predictions.
    flags=set(x.get("flags",[]));cap=bool(x["truncated"])
    other=bool(flags-{"ABNORMAL_STOP_OR_TRUNCATION","TRUNCATED","TRUNCATION","EMPTY_OUTPUT","REPETITION"})
    return {"empty":bool(x["empty"]),"repetition":bool(x["repetition"]),"other":other,"truncation":cap}
class Monitor:
    def __init__(self,bench):
        self.bench=bench;self.cfg=readj(OUT/"stop_policy_v2.json")["systemic_monitor"];self.lengths={m:[] for m in METHODS};self.counts={m:defaultdict(int) for m in METHODS};self.block={m:[] for m in METHODS};self.previous={m:set() for m in METHODS};self.streak={m:0 for m in METHODS};self.windows=[]
    def observe(self,pair,qid):
        violations=[]
        for x in pair:
            m=x["method"];z=behavior(x);self.lengths[m].append(x["generated_tokens"])
            for k,v in z.items():self.counts[m][k]+=int(v)
            self.counts[m]["output_anomaly"]+=int(bool(x["flags"]) or x["parse_failure"])
            self.counts[m]["eos"]+=int(x["eos"])
            self.streak[m]=self.streak[m]+1 if z["empty"] or z["repetition"] else 0
            if self.streak[m]>=self.cfg["consecutive_empty_or_repetitive_questions"]:violations.append({"method":m,"rule":"10_consecutive_empty_or_repetitive","question_id":qid})
            self.block[m].append(z)
            if len(self.block[m])==self.cfg["window_questions_per_method"]:
                rates={k:sum(v[k] for v in self.block[m])/len(self.block[m]) for k in z}
                thresholds={"empty":self.cfg["empty_fraction"],"repetition":self.cfg["repetition_fraction"],"other":self.cfg["other_behavior_anomaly_fraction"],"truncation":self.cfg["truncation_fraction"]}
                active={k for k in rates if rates[k]>=thresholds[k]}
                hit=active&self.previous[m]
                self.windows.append({"method":m,"ending_question_id":qid,"rates":rates,"thresholds":thresholds,"active":sorted(active),"two_consecutive_windows":sorted(hit)})
                if hit:violations.append({"method":m,"question_id":qid,"rule":"two_consecutive_windows","metrics":sorted(hit)})
                self.previous[m]=active;self.block[m]=[]
        return violations
    def save(self,processed,violations=None):
        report={}
        for m,a in self.lengths.items():
            if a:report[m]={"N":len(a),"mean_tokens":float(np.mean(a)),"P50":float(np.quantile(a,.5)),"P95":float(np.quantile(a,.95)),"max_tokens":int(max(a)),"counts":dict(self.counts[m]),"rates":{k:v/len(a) for k,v in self.counts[m].items()}}
        writej(OUT/self.bench/"output_monitor_v2.json",{"status":"SYSTEMIC_OUTPUT_STOP" if violations else "MONITORING","frozen_order_questions_processed":processed,"methods":report,"windows":self.windows,"violations":violations or [],"stop_policy_sha256":sha(OUT/"stop_policy_v2.json"),"quality_scores_not_used_for_stop":True})
