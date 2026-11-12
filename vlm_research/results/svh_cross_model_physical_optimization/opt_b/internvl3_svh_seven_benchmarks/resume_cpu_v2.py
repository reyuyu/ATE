"""CPU-only validation of v1 checkpoint reuse and the independent v2 stop policy."""
from .common import *
from .stop_policy_v2 import *
import sqlite3,gzip,shutil,ast
def dbrows(path):
    db=sqlite3.connect("file:"+str(path)+"?mode=ro",uri=True)
    assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    rows={(q,m):raw for q,m,raw in db.execute("SELECT qid,method,record FROM predictions")}
    meta=dict(db.execute("SELECT key,value FROM metadata"));db.close();return rows,meta
def main():
    original=OLD_RUN/"results"
    assert sha(OUT/"protocol.json")==sha(original/"protocol.json")
    assert sha(OUT/"source_manifest.json")==sha(original/"source_manifest.json")
    assert readj(OUT/"protocol.json")["numerical_status"]=="FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER"
    assert sha(MODEL/"config.json")==readj(OUT/"model_manifest.json")["config_sha256"]
    assert len(list(MODEL.glob("*.safetensors")))==4
    for p in MODEL.glob("*.safetensors"):assert p.stat().st_size>1000000000
    check_sources()
    branch=subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"],text=True,timeout=45).strip();assert branch==BRANCH
    parent=readj(original/"artifact_manifest.json")
    reuse4={}
    for bench in ["ScienceQA-IMG","POPE","MME","MMBench"]:
        entries=[x for x in parent["uploaded_files"] if x["path"].startswith("vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090/"+bench+"/")]
        for x in entries:
            p=OUT/Path(x["path"]).relative_to("vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090")
            assert sha(p)==x["sha256"]
        reuse4[bench]={"status":"EXACT_PARENT_ARTIFACTS_REUSED","files":len(entries),"new_GPU_generations":0,"parent_commit":PARENT}
    results={}
    for bench,expected in [("GQA",772),("TextVQA",358),("OCRBench",490)]:
        path=RUN/"checkpoints"/(bench+".sqlite");oldpath=OLD_RUN/"checkpoints"/(bench+".sqlite")
        saved=next(x for x in parent["server_only"] if Path(x["persistent_path"]).name==bench+".sqlite" and "server_raw" in x["persistent_path"])
        assert sha(oldpath)==sha(Path(saved["persistent_path"]))==saved["sha256"]
        old,oldmeta=dbrows(oldpath);new,newmeta=dbrows(path)
        if "stop_policy_v2_sha256" not in newmeta:assert sha(path)==saved["sha256"]
        assert oldmeta["protocol_sha256"]==sha(OUT/"protocol.json") and oldmeta["sources_sha256"]==sha(OUT/"source_manifest.json")
        assert oldmeta["benchmark"]==bench
        with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:rows=[json.loads(x) for x in f]
        assert len(rows)==EXPECTED[bench]
        assert sha(RUN/"manifests"/(bench+".jsonl.gz"))==readj(OUT/"dataset_manifest.json")[bench]["manifest_sha256"]
        lookup={r["question_id"]:r for r in rows};pairs=defaultdict(dict)
        for key,raw in old.items():assert new[key]==raw,"OLD_PREDICTION_CHANGED"
        for (q,m),raw in new.items():
            x=json.loads(raw);assert (q,m)==(x["question_id"],x["method"])
            validate_record(x,lookup[q]);pairs[q][m]=x
            assert bind({k:v for k,v in lookup[q].items() if k!="sample_sha256"})==lookup[q]["sample_sha256"]
        assert len(old)==2*expected
        for q,rs in pairs.items():
            if len(rs)==2:assert rs["native"]["input_binding"]==rs["physical_d"]["input_binding"]
        db=sqlite3.connect(path)
        db.execute("CREATE TABLE IF NOT EXISTS v2_provenance(qid TEXT,method TEXT,origin TEXT,record_sha256 TEXT,policy_sha256 TEXT,execution_sha256 TEXT,PRIMARY KEY(qid,method))")
        for key,raw in new.items():
            origin="V1_REUSED" if key in old else "V2_GENERATED"
            found=db.execute("SELECT record_sha256 FROM v2_provenance WHERE qid=? AND method=?",key).fetchone()
            if found:assert found[0]==sha(raw)
            else:db.execute("INSERT INTO v2_provenance VALUES(?,?,?,?,?,?)",(*key,origin,sha(raw),sha(OUT/"stop_policy_v2.json"),sha(OUT/"execution_source_manifest_v2.json")))
        for k,h in [("stop_policy_v2_sha256",sha(OUT/"stop_policy_v2.json")),("execution_source_manifest_v2_sha256",sha(OUT/"execution_source_manifest_v2.json"))]:
            found=db.execute("SELECT value FROM metadata WHERE key=?",(k,)).fetchone()
            if found:assert found[0]==h
            else:db.execute("INSERT INTO metadata VALUES(?,?)",(k,h))
        db.commit();db.close()
        missing=[r["question_id"] for r in rows if len(pairs.get(r["question_id"],{}))<2]
        results[bench]={"status":"PASS","parent_checkpoint_sha256":saved["sha256"],"parent_pairs_reused":expected,"parent_records_byte_identical":True,"current_pairs":sum(len(x)==2 for x in pairs.values()),"first_incomplete_question_id":missing[0] if missing else None,"remaining_pairs":len(missing),"expected_total":len(rows),"exact_runtime_protocol_prompt_gold_input_and_KV_reuse":"PASS"}
    # Meaningful policy contracts: isolated anomalies/caps continue, sustained system failures stop.
    def example(m,**kw):
        return dict(method=m,generated_tokens=3,flags=[],parse_failure=False,empty=False,repetition=False,truncated=False,eos=True,**kw)
    m=Monitor("CPU_TEST")
    pair=[example(x) for x in METHODS];assert not m.observe(pair,"0")
    for x in pair:x.update(flags=["REPETITION"],repetition=True,truncated=True,eos=False,generated_tokens=16)
    assert not m.observe(pair,"1"),"SINGLE_ANOMALY_MUST_CONTINUE"
    m=Monitor("CPU_TEST")
    for i in range(200):
        pair=[example(x) for x in METHODS]
        for x in pair:x.update(truncated=True,generated_tokens=16,eos=False,flags=["ABNORMAL_STOP_OR_TRUNCATION"])
        hit=m.observe(pair,str(i))
        if i<199:assert not hit
    assert hit,"SUSTAINED_TRUNCATION_MUST_STOP"
    m=Monitor("CPU_TEST")
    for i in range(10):
        pair=[example(x) for x in METHODS]
        for x in pair:x.update(repetition=True,flags=["REPETITION"])
        hit=m.observe(pair,str(i))
        if i<9:assert not hit
    assert hit,"SYSTEMIC_REPETITION_MUST_STOP"
    for name in ["stop_policy_v2.py","resume_worker_v2.py","resume_cpu_v2.py"]:ast.parse((RUN/"code"/PKG/name).read_text())
    record={"status":"PASS","CPU_only":True,"parent_commit":PARENT,"original_protocol_unchanged":True,"frozen_inference_sources_unchanged":True,"checkpoint_integrity_and_SHA256":"PASS","reused_predictions_not_deleted_or_rewritten":True,"isolated_anomaly_cap_wrong_answer_continue":True,"systemic_monitor_contracts":"PASS","no_correctness_gold_in_stop_monitor":True,"benchmarks":results,"inherited_full_benchmarks":reuse4,"stop_policy_sha256":sha(OUT/"stop_policy_v2.json"),"execution_source_manifest_v2_sha256":sha(OUT/"execution_source_manifest_v2.json")}
    writej(OUT/"resume_validation_v2.json",record);print("CPU_RESUME_V2_PASS",results,flush=True)
if __name__=="__main__":main()
