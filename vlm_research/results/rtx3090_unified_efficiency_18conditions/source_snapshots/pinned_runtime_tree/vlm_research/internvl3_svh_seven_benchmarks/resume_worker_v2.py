"""Resume only missing predictions using the byte-identical v1 inference functions."""
from .common import *
from . import worker as w
from .stop_policy_v2 import *
import argparse,traceback,gc,sqlite3
import torch
def provenance(db,x,origin):
    raw=canonical(x);old=db.execute("SELECT record_sha256 FROM v2_provenance WHERE qid=? AND method=?",(x["question_id"],x["method"])).fetchone()
    if old:
        if old[0]!=sha(raw):raise HardStop("PREDICTION_MUTATED_AFTER_SAVE")
    else:
        db.execute("INSERT INTO v2_provenance VALUES (?,?,?,?,?,?)",(x["question_id"],x["method"],origin,sha(raw),sha(OUT/"stop_policy_v2.json"),sha(OUT/"execution_source_manifest_v2.json")));db.commit()
def paired(model,tok,inputs,db,r,index,ph,src,counters):
    order=METHODS[index%2:]+METHODS[:index%2];results=[];ready=None
    for method in order:
        old=w.get(db,r["question_id"],method)
        if old is not None:
            validate_record(old,r);provenance(db,old,"V2_GENERATED" if old.get("phase")=="formal_stop_policy_v2" else "V1_REUSED")
            results.append(old);continue
        if ready is None:ready,ib,binding=inputs.ready(r)
        # Exact v1 forward, generation loop, EOS, tokenizer, processor and cap. Only phase labels differ.
        x=w.request(model,tok,inputs,r,method,"formal_stop_policy_v2",ph,src,ready,ib,binding)
        try:validate_record(x,r)
        except Exception:
            writej(OUT/r["benchmark"]/"hard_stop_record_v2.json",x);raise
        w.put(db,x);provenance(db,x,"V2_GENERATED");counters["new_predictions"]+=1;results.append(x)
    if results[0]["input_binding"]!=results[1]["input_binding"]:raise HardStop("PAIRED_INPUT_MISMATCH")
    return results
def main():
    a=argparse.ArgumentParser();a.add_argument("--benchmark",required=True,choices=TARGETS);bench=a.parse_args().benchmark
    check_sources()
    if readj(OUT/"resume_validation_v2.json")["status"]!="PASS":raise HardStop("CPU_RESUME_VALIDATION_MISSING")
    if source_fingerprints()!=readj(OUT/"protocol.json")["source_sha256"]:raise HardStop("FROZEN_PHYSICAL_ROOT_SHA_MISMATCH")
    gpu_idle();rows=w.manifest(bench);meta=readj(OUT/"dataset_manifest.json")[bench]
    if sha(RUN/"manifests"/(bench+".jsonl.gz"))!=meta["manifest_sha256"]:raise HardStop("INPUT_MANIFEST_SHA_MISMATCH")
    ph=sha(OUT/"protocol.json");src=sha(OUT/"source_manifest.json");out=OUT/bench;db=model=inputs=None
    monitor=Monitor(bench);counters={"new_predictions":0};current=None
    try:
        db=w.database(bench,ph,src)
        for name,expected in [("stop_policy_v2_sha256",sha(OUT/"stop_policy_v2.json")),("execution_source_manifest_v2_sha256",sha(OUT/"execution_source_manifest_v2.json"))]:
            value=db.execute("SELECT value FROM metadata WHERE key=?",(name,)).fetchone()
            if value is None or value[0]!=expected:raise HardStop("V2_CHECKPOINT_METADATA_MISMATCH "+name)
        initial={q for q,n in db.execute("SELECT qid,count(*) FROM predictions GROUP BY qid") if n==2}
        complete=len(initial)
        state(bench,"LOADING_MODEL",complete_questions=complete,target=len(rows),stop_policy="v2")
        model,tok=w.load_model();writej(out/"environment_v2.json",w.environment(model,tok));inputs=w.Inputs(model,tok)
        smoke=w.smoke(model,tok,inputs,rows,meta["preflight_question_ids"])
        old=readj(OUT/"history/stop_policy_v1"/bench/"smoke_validation.json")
        previous={x["question_id"]:x for x in old["checks"]}
        for x in smoke["checks"]:
            prior=previous[x["question_id"]]
            if x["input_binding"]!=prior["input_binding"] or x["native_tokens"]!=prior["native_tokens"] or x["physical_tokens"]!=prior["physical_tokens"]:raise HardStop("RESUME_V1_SMOKE_REPRODUCTION_MISMATCH")
        smoke["v1_input_and_full_smoke_tokens_reproduced"]=True
        smoke["stop_policy_sha256"]=sha(OUT/"stop_policy_v2.json");writej(out/"smoke_validation_v2.json",smoke)
        state(bench,"FULL_GENERATION",complete_questions=complete,target=len(rows),new_predictions=0,stop_policy="v2")
        print("RESUME_STARTED",bench,"reused_pairs",complete,flush=True)
        for i,r in enumerate(rows):
            current=r;was_complete=r["question_id"] in initial
            pair=paired(model,tok,inputs,db,r,i,ph,src,counters)
            if not was_complete:complete+=1
            violations=monitor.observe(pair,r["question_id"])
            if violations:
                monitor.save(i+1,violations)
                state(bench,"PAUSED_SYSTEMIC_OUTPUT_V2",question_id=r["question_id"],complete_questions=complete,target=len(rows),new_predictions=counters["new_predictions"],violations=violations);return
            if (i+1)%25==0 or i+1==len(rows):
                monitor.save(i+1);check_sources()
                state(bench,"FULL_GENERATION",complete_questions=complete,frozen_order_processed=i+1,target=len(rows),new_predictions=counters["new_predictions"],generation_rows=db.execute("SELECT count(*) FROM predictions").fetchone()[0],stop_policy="v2")
                print("PROGRESS",bench,complete,len(rows),"NEW",counters["new_predictions"],flush=True)
        if db.execute("SELECT count(*) FROM predictions").fetchone()[0]!=2*len(rows):raise HardStop("INCOMPLETE_FINAL_CHECKPOINT")
        check_sources();monitor.save(len(rows))
        state(bench,"GENERATION_COMPLETE",complete_questions=len(rows),generation_rows=2*len(rows),new_predictions=counters["new_predictions"],stop_policy="v2")
        writej(out/"image_consumption_integrity_v2.json",{"status":"PASS","verified_images":inputs.verified,"checked_at_first_consumption":True,"CPU_pixels_cache_size":8,"inherited_verified_images":"Parent v1 image-consumption and CPU checkpoint proof retained in history; reused predictions are byte-identical."})
        print("GENERATION_COMPLETE_V2",bench,counters,flush=True)
    except Exception as e:
        err={"status":"FAILED_RUNTIME_V2","question_id":None if current is None else current["question_id"],"error":str(e),"traceback":traceback.format_exc(),"new_predictions":counters["new_predictions"],"GPU_allocated_MiB":torch.cuda.memory_allocated()/1048576 if torch.cuda.is_initialized() else None,"GPU_reserved_MiB":torch.cuda.memory_reserved()/1048576 if torch.cuda.is_initialized() else None}
        writej(out/"failure_v2.json",err);state(bench,"FAILED_RUNTIME_V2",**{k:v for k,v in err.items() if k!="status"});print(err["traceback"],flush=True)
    finally:
        if db is not None:
            try:db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:db.close()
        if inputs is not None:inputs.pixels.cache_clear()
        inputs=model=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        print("WORKER_EXIT_V2",bench,flush=True)
if __name__=="__main__":main()
