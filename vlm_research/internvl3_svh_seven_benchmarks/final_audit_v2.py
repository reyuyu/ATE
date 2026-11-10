"""Final CPU-only v2 audit; no new inference or scorer changes."""
from .common import *
from .stop_policy_v2 import *
from . import validate_artifacts
import sqlite3,gzip,collections
def main():
    from .output_response_audit_v2 import run as response_audit
    response=response_audit()
    check_sources();validate_artifacts.main()
    generation=readj(OUT/"protocol.json")
    assert sha(OUT/"protocol.json")==sha(OLD_RUN/"results/protocol.json")
    assert generation["numerical_status"]=="FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER"
    parent=readj(OLD_RUN/"results/artifact_manifest.json");reuse={}
    prefix="vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090/"
    for bench in ["ScienceQA-IMG","POPE","MME","MMBench"]:
        entries=[x for x in parent["uploaded_files"] if x["path"].startswith(prefix+bench+"/")]
        for x in entries:assert sha(OUT/Path(x["path"]).relative_to(prefix))==x["sha256"]
        reuse[bench]={"status":"EXACT_PARENT_BYTES","files_checked":len(entries),"new_GPU_generations":0}
    audit={}
    for bench in TARGETS:
        db=sqlite3.connect("file:"+str(RUN/"checkpoints"/(bench+".sqlite"))+"?mode=ro",uri=True)
        assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        records={(q,m):raw for q,m,raw in db.execute("SELECT qid,method,record FROM predictions")}
        old=sqlite3.connect("file:"+str(OLD_RUN/"checkpoints"/(bench+".sqlite"))+"?mode=ro",uri=True)
        prior={(q,m):raw for q,m,raw in old.execute("SELECT qid,method,record FROM predictions")};old.close()
        for k,raw in prior.items():assert records[k]==raw
        origins=collections.Counter()
        for q,m,origin,h,ph,ex in db.execute("SELECT qid,method,origin,record_sha256,policy_sha256,execution_sha256 FROM v2_provenance"):
            assert sha(records[(q,m)])==h and ((q,m) in prior)==(origin=="V1_REUSED")
            assert ph==sha(OUT/"stop_policy_v2.json") and ex==sha(OUT/"execution_source_manifest_v2.json")
            origins[origin]+=1
        assert sum(origins.values())==len(records);db.close()
        sm=readj(OUT/bench/"smoke_validation_v2.json")
        assert sm["v1_input_and_full_smoke_tokens_reproduced"] and all(x["native_noop_official_logits_strict_equal"] for x in sm["checks"])
        env=readj(OUT/bench/"environment_v2.json")
        assert env["GPU_UUID"]==GPU and env["torch"]=="2.6.0+cu124" and env["transformers"]=="4.49.0"
        assert env["model_revision"]==REV and env["weights_all_gpu"] and env["all_float_weights_bf16"] and env["language_attention"]=="eager"
        val=readj(OUT/bench/"validation.json")
        if val["status"]=="COMPLETE":assert len(records)==2*EXPECTED[bench]
        audit[bench]={"status":val["status"],"predictions":len(records),"V1_REUSED":origins["V1_REUSED"],"V2_GENERATED":origins["V2_GENERATED"],"all_original_JSON_bytes_preserved":True,"SQLite_integrity":"PASS","prediction_provenance_SHA256":"PASS","Native_noop_and_full_v1_smoke_tokens":"PASS","generation_protocol_unchanged":True}
    gpu=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,memory.used,temperature.gpu","--format=csv,noheader"],capture_output=True,text=True,check=True,timeout=30).stdout.strip()
    assert GPU in gpu
    apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],capture_output=True,text=True,check=True,timeout=30).stdout.strip()
    assert not apps,("GPU_NOT_RELEASED",apps)
    result={"status":"PASS","benchmarks":audit,"unchanged_full_benchmarks":reuse,"original_protocol_sha256":sha(OUT/"protocol.json"),"stop_policy_sha256":sha(OUT/"stop_policy_v2.json"),"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER","GPU_UUID":GPU,"GPU_final_state":gpu,"GPU_compute_processes":apps,"GPU_released":True,"no_CUDA_context_or_model_load_in_audit":True,"server_not_shutdown":True,"checked_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    result.update(output_monitor_coverage_status=response["status"],TextVQA_monitoring_deviation=response["TextVQA_deviation"],artifact_integrity_status="PASS",unqualified_monitoring_PASS=False)
    writej(OUT/"final_validation_v2.json",result)
    val=readj(OUT/"validation.json");val.update(independent_final_v2_validation="PASS",all_inherited_four_artifacts_byte_identical=True,original_protocol_unchanged=True,GPU_released=True,output_monitor_coverage_status=response["status"],TextVQA_monitoring_deviation=response["TextVQA_deviation"]);writej(OUT/"validation.json",val)
    report=OUT/"report_zh.md";text=report.read_text();marker="\n## V2最终验收\n";text=text.split(marker)[0]
    text=text.replace("# InternVL3-SVH 七项 Benchmark 单图块质量评测", "# InternVL3-SVH 七项 Benchmark 单图块质量评测\n\n**执行偏差：TextVQA Physical-D出现系统性无关元回答，原监测漏检，发现时已完成5000对，未能及时停止。所有记录和得分保留；不宣称本轮输出监测完全通过。详见末尾验收及systemic_output_audit_v2.json。**")
    text+=marker+"\nCPU独立输入/Gold/Prompt/Token/EOS/逐层KV核验通过；原记录JSON及新记录出处SHA全部通过；原四项完整产物逐文件SHA与父Commit一致，未重新推理或计分。原protocol.json与九个冻结运行文件未改变，v2只改变停止/继续执行策略。\n\n"
    text+="**TextVQA监测执行偏差：** 全量后内容审计发现Physical-D有1782/5000（35.64%）条以answering开头的无关元回答；Native对应26/5000，宽匹配包含does not require时为27/5000。原格式/长度检测未充分识别短元回答，未及时触发应有的系统性停止。发现时该任务已全部生成，没有后续TextVQA请求；未重跑、删题或改参数。全部原文、Token及得分保留，详见systemic_output_audit_v2.json和meta_response_cases_v2.csv.gz。此处的PASS指文件/输入/KV/评分完整性，不代表输出监测无偏差或Physical-D质量通过。\n\n"
    for b,d in audit.items():text+=f'- {b}：{d["status"]}；复用 {d["V1_REUSED"]} 条原预测，新增 {d["V2_GENERATED"]} 条预测。\n'
    text+="\n指定RTX3090无计算进程，GPU已释放，服务器保持运行。详细证据见final_validation_v2.json；最终Commit以publication.json/远端分支头为准。\n"
    report.write_text(text);print("FINAL_V2_VALIDATION_PASS_GPU_RELEASED",audit,flush=True)
if __name__=="__main__":main()
