"""Descriptive CPU-only audit of a short meta-answer pattern missed by the frozen heuristic.
No inference, score, stopping threshold, source freeze or raw prediction is changed.
"""
from .common import *
import sqlite3,gzip,collections
def run():
    targets=["GQA","TextVQA","OCRBench"];audit={};cases=[]
    for bench in targets:
        db=sqlite3.connect("file:"+str(RUN/"checkpoints"/(bench+".sqlite"))+"?mode=ro",uri=True)
        rows={(q,m):json.loads(raw) for q,m,raw in db.execute("SELECT qid,method,record FROM predictions")};db.close()
        with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:order=[json.loads(x)["question_id"] for x in f]
        methods={}
        for method in METHODS:
            selected=[rows[(q,method)] for q in order if (q,method) in rows]
            flags=[]
            for x in selected:
                text=x["text"].strip().lower()
                hit=text.startswith("answering") or "does not require" in text
                flags.append(hit)
                if hit:cases.append({"benchmark":bench,"question_id":x["question_id"],"method":method,"text":x["text"],"token_ids":x["token_ids"],"original_flags":x["flags"],"audit_flag":"META_RESPONSE_PATTERN_POSTHOC","included_in_official_scoring":True,"prediction_unchanged":True})
            windows=[{"start_index":i,"N":len(flags[i:i+100]),"meta_response_count":sum(flags[i:i+100])} for i in range(0,len(flags),100)]
            full=[x for x in windows if x["N"]==100]
            consecutive=any(full[i-1]["meta_response_count"]>=25 and full[i]["meta_response_count"]>=25 for i in range(1,len(full)))
            methods[method]={"N":len(selected),"meta_response_count":sum(flags),"meta_response_fraction":sum(flags)/max(1,len(selected)),"windows":windows,"two_full_consecutive_windows_at_least_25_percent":consecutive}
        complete=all(v["N"]==EXPECTED[bench] for v in methods.values())
        systematic=any(v["two_full_consecutive_windows_at_least_25_percent"] for v in methods.values())
        audit[bench]={"methods":methods,"full_generation_already_complete":complete,"systematic_meta_response_pattern":systematic,"action":"NO_FURTHER_GENERATION_TASK_ALREADY_COMPLETE_MONITORING_DEVIATION" if systematic and complete else "STOP_AFFECTED_TASK_REQUIRED" if systematic else "NO_META_PATTERN_STOP_SIGNAL"}
    from .analyze import csv_gz
    if cases:csv_gz(OUT/"meta_response_cases_v2.csv.gz",cases)
    missed=audit["TextVQA"]["systematic_meta_response_pattern"]
    result={"status":"KNOWN_MONITOR_FALSE_NEGATIVES_TEXTVQA" if missed else "NO_SYSTEMATIC_META_PATTERN_DETECTED","benchmarks":audit,"descriptive_detector":"lowercase output startswith answering OR contains does not require; posthoc discovery, not a newly preregistered stopping policy","frozen_original_monitor_unchanged":True,"original_policy_and_predictions_unchanged":True,"TextVQA_deviation":"A systematic irrelevant short meta-answer pattern was not captured by the original format/length heuristic. Detected after all 5000 pairs had already been generated. The benchmark was not stopped at the intended earlier point; this is an execution/monitoring deviation. All original answers and scores are retained. No rerun or parameter change." if missed else None,"checked_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    writej(OUT/"systemic_output_audit_v2.json",result)
    print("META_RESPONSE_AUDIT",result["status"],{b:{m:d["meta_response_count"] for m,d in x["methods"].items()} for b,x in audit.items()},flush=True)
    return result
if __name__=="__main__":run()
