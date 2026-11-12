"""V2 completion analysis: reuse official scorers/statistics; preserve output anomalies."""
from .common import *
from .stop_policy_v2 import *
from . import analyze as a
import sqlite3,gzip,collections,argparse
def analyze_one(bench):
    a.analyze_one(bench)
    db=sqlite3.connect(RUN/"checkpoints"/(bench+".sqlite"))
    original=sqlite3.connect("file:"+str(OLD_RUN/"checkpoints"/(bench+".sqlite"))+"?mode=ro",uri=True)
    old={(q,m):raw for q,m,raw in original.execute("SELECT qid,method,record FROM predictions")};original.close()
    records={(q,m):raw for q,m,raw in db.execute("SELECT qid,method,record FROM predictions")}
    for key,raw in old.items():assert records[key]==raw,"PARENT_PREDICTION_MUTATED"
    origins=[];count=collections.Counter()
    for q,m,origin,h,policy_h,execution_h in db.execute("SELECT qid,method,origin,record_sha256,policy_sha256,execution_sha256 FROM v2_provenance"):
        assert sha(records[(q,m)])==h
        assert policy_h==sha(OUT/"stop_policy_v2.json") and execution_h==sha(OUT/"execution_source_manifest_v2.json")
        assert (origin=="V1_REUSED")==((q,m) in old)
        count[origin]+=1;origins.append({"benchmark":bench,"question_id":q,"method":m,"origin":origin,"raw_prediction_sha256":h,"generation_protocol_sha256":sha(OUT/"protocol.json"),"stop_policy_sha256":policy_h,"execution_source_manifest_v2_sha256":execution_h})
    assert len(origins)==len(records);db.close()
    a.csv_gz(OUT/bench/"prediction_provenance_v2.csv.gz",origins)
    val=readj(OUT/bench/"validation.json")
    val.update(stop_policy="v2",stop_policy_sha256=sha(OUT/"stop_policy_v2.json"),original_inference_protocol_unchanged=True,parent_predictions_byte_identical=True,parent_predictions_reused=count["V1_REUSED"],new_predictions_generated=count["V2_GENERATED"],parent_commit=PARENT,no_output_anomaly_excluded=True)
    writej(OUT/bench/"validation.json",val)
    abnormal=[]
    with gzip.open(OUT/bench/"per_question.csv.gz","rt") as f:
        for x in csv.DictReader(f):
            flags=json.loads(x["flags"]);score=float(x["score"])
            if score==0:flags=flags+["INCORRECT_OFFICIAL_ANSWER"]
            elif score<1:flags=flags+["PARTIAL_TEN_ANSWER_SOFT_SCORE"]
            if flags:abnormal.append({**x,"classification":"OUTPUT_ANOMALY","v2_audit_flags":flags,"included_in_official_scoring":True,"correctness_not_used_for_stopping":True})
    a.csv_gz(OUT/bench/"output_anomalies_v2.csv.gz",abnormal)
def aggregate():
    a.aggregate()
    v=readj(OUT/"validation.json")
    v.update(stop_policy="v2",generation_protocol_unchanged=True,original_protocol_sha256=sha(OUT/"protocol.json"),stop_policy_sha256=sha(OUT/"stop_policy_v2.json"),parent_commit=PARENT,inherited_benchmarks=["ScienceQA-IMG","POPE","MME","MMBench"],only_resumed_benchmarks=TARGETS)
    writej(OUT/"validation.json",v)
    all_anomalies=[]
    for bench in TARGETS:
        f=OUT/bench/"output_anomalies_v2.csv.gz"
        if f.exists():
            with gzip.open(f,"rt") as source:all_anomalies.extend(csv.DictReader(source))
    if all_anomalies:a.csv_gz(OUT/"output_anomalies_v2.csv.gz",all_anomalies)
    report=OUT/"report_zh.md";text=report.read_text()
    text=text.replace("https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-physical-d-seven-benchmarks-3090","https://github.com/reyuyu/ATE/tree/"+BRANCH)
    text=text.replace("预检及正式异常记录完整保留；未通过长度门槛的任务暂停，不将子集成绩写成全集成绩。","v1预检、暂停和异常原文均保留在history/stop_policy_v1及父Commit。本轮stop_policy_v2仅改变继续执行策略：单题解释、重复、错误/解析失败或原上限截断均保留并计分，不据此停止全集；结构性错误、NaN/Inf/OOM、哈希不符和冻结系统性监测规则仍硬停止。")
    text=text.replace("异常输出项按门槛暂停并保留原文。","单题异常保留、计分并继续；仅结构错误或预先冻结的系统性异常触发停止。")
    text=text.replace("| MME Perception | 2374 |","| MME Perception | 2114 |").replace("| MME Cognition | 2374 |","| MME Cognition | 260 |")
    prefix="# stop_policy_v2 补完：原推理协议不变\n\n"
    prefix+="父结果Commit："+PARENT+"。本分支保留原protocol.json字节及全部原始暂停证据；v2不改变模型、Prompt、图片、评分器、原生EOS、生成上限或Physical-D。原四项全量结果按父Commit直接复用，不重新推理或重计分。\n\n"
    text=prefix+text
    notes=["","## V2恢复与异常保留",""]
    for bench in TARGETS:
        f=OUT/bench/"validation.json"
        if f.exists():
            d=readj(f);notes.append(f'- {bench}：{d["status"]}；复用原预测 {d.get("parent_predictions_reused","待分析")} 条，新生成 {d.get("new_predictions_generated","待分析")} 条。完整原始JSON字节对照见prediction_provenance_v2.csv.gz。')
    notes+=["","原父版本暂停样本（GQA无关解释、TextVQA长书名截断、OCRBench重复标题）均原样保留且纳入本轮评分，没有重生成、跳过或修正。TextVQA始终为32 Token；任何未来64/128 Token版本须独立完整运行，不能混入本轮。",
    "OUTPUT_ANOMALY完整清单见各任务及根目录output_anomalies_v2.csv.gz：包括解释、重复、截断、零分/解析失败及部分TextVQA软评分。错误与部分分数只用于质量审计，不用于停止判断，也不改变官方评分。",
    "系统性监测按100题/方法的非重叠窗口：连续两个窗口空输出≥10%、重复≥10%、其他行为格式异常≥25%或截断≥50%时暂停；连续10题空输出/重复也暂停。阈值在新增GPU生成前冻结，不使用Gold或Accuracy来决定停止、不自动调整。逐窗口及长度P50/P95/最大值见各任务output_monitor_v2.json。",
    "MME完整Benchmark为2374条；表中Perception为2114条、Cognition为260条，分别按官方原始积分计算。MMBench为4377条已发布旋转记录、1176个Circular组。原四项汇总与逐题产物的SHA256保持父版本一致。",
    "数值状态仍是FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，Native/No-op及新结构性错误仍为硬门槛。所有比较仍是单图块ATE协议，不能冒充作者完整多图块排行榜结果。"]
    report.write_text(text+"\n"+"\n".join(notes)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--benchmark",choices=TARGETS);args=p.parse_args()
    if args.benchmark:analyze_one(args.benchmark)
    aggregate()
