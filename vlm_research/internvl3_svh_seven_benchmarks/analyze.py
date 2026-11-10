"""Independent CPU rescoring, native-unit paired statistics and light evidence export."""
from .common import *
from .scoring import *
from . import statistics as stats
import sqlite3,gzip,collections,contextlib,io,traceback,math
import numpy as np
PERCEPTION=["existence","count","position","color","posters","celebrity","scene","landmark","artwork","OCR"]
COGNITION=["commonsense_reasoning","numerical_calculation","text_translation","code_reasoning"]
def manifests(b):
    with gzip.open(RUN/"manifests"/(b+".jsonl.gz"),"rt") as f:return {r["question_id"]:r for r in map(json.loads,f)}
def csv_gz(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(path,"wt",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def measure(bench,rows,metric,subset="all"):
    if bench=="MME":
        tasks=PERCEPTION if subset=="Perception" else COGNITION if subset=="Cognition" else PERCEPTION+COGNITION
        n,p,ci,clusters=stats.mme(rows,tasks);unit="raw_points"
    elif bench=="MMBench":
        (n,p,ci,clusters),groups=stats.circular(rows);unit="Circular_pp"
    elif metric=="F1":n,p,ci,clusters=stats.pope_f1(rows);unit="F1_pp"
    else:n,p,ci,clusters=stats.linear(rows,1000 if bench=="OCRBench" else 100);unit="raw_points_0_1000" if bench=="OCRBench" else "pp"
    return {"benchmark":bench,"metric":metric,"subset":subset,"N":len(rows),"native_score":n,"physical_score":p,"delta":p-n,"ci_low":ci[0],"ci_high":ci[1],"units":unit,"image_clusters":clusters,"status":"COMPLETE","bootstrap_reps":20000,"bootstrap_seed":20261110}
def official_checks(bench,rows):
    result={"scalar_rescoring":"PASS"}
    if bench=="OCRBench":
        for method in METHODS:
            data=[dict(r,text=r[method+"_text"]) for r in rows];orig=official_ocr_batch(data)
            assert orig==[int(r[method+"_score"]) for r in rows]
        result["original_MultimodalOCR_loop"]="PASS"
    if bench=="POPE":
        f=ast_scope(RUN/"references/InternVL/internvl_chat/eval/pope/eval_pope.py",{"eval_pope"},{"json":json})["eval_pope"]
        for subset in ["random","popular","adversarial"]:
            rs=[r for r in rows if r["category"]==subset];path=RUN/("pope_labels_"+subset+".jsonl");path.write_text("".join(json.dumps({"label":r["gold"]})+"\n" for r in rs))
            for m in METHODS:
                s=io.StringIO()
                with contextlib.redirect_stdout(s):value=f([{"text":r[m+"_text"]} for r in rs],str(path))
                tp=sum(r["gold"]=="yes" and r[m+"_parsed"]=="yes" for r in rs);fp=sum(r["gold"]=="no" and r[m+"_parsed"]=="yes" for r in rs);fn=sum(r["gold"]=="yes" and r[m+"_parsed"]=="no" for r in rs)
                expected=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
                assert abs(value-expected)<1e-12
        result["official_POPE_confusion_and_F1"]="PASS"
    if bench=="MMBench":
        # Execute official prefetch_circular_group verbatim, with its frozen matcher.
        path=RUN/"references/VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py"
        src=path.read_text();tree=ast.parse(src);wanted=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ("build_choices","prefetch_answer","prefetch_circular_group")]
        import pandas as pd
        scope={"can_infer":official_matcher(),"np":np,"pd":pd,"string":string,"logger":logging.getLogger(__name__)}
        exec(compile(ast.Module(body=wanted,type_ignores=[]),str(path),"exec"),scope)
        import pandas as pd
        groups=collections.defaultdict(list)
        for r in rows:groups[r["group"]].append(r)
        for rs in groups.values():
            for m in METHODS:
                frame=pd.DataFrame([dict(index=int(r["question_id"]),prediction=r[m+"_text"],GT=r["gold"],**r["choices"]) for r in rs])
                out=scope["prefetch_circular_group"](frame,verbose=False)
                # None requires judge; frozen exact_matching policy scores it zero.
                hit=out["hit"] if out is not None else 0
                assert hit==int(all(r[m+"_score"]==1 for r in rs))
        result["official_circular_prefetch_exact_matching"]="PASS"
    return result
def analyze_one(bench):
    path=RUN/"checkpoints"/(bench+".sqlite")
    if not path.exists():return
    out=OUT/bench;out.mkdir(exist_ok=True)
    index=manifests(bench);db=sqlite3.connect(path);records=[json.loads(x[0]) for x in db.execute("SELECT record FROM predictions ORDER BY qid,method")];db.close()
    grouped=collections.defaultdict(dict);exports=[];anomaly_rows=[];layouts=[]
    for x in records:
        r=index[x["question_id"]];assert x["sample_sha256"]==r["sample_sha256"]
        parsed,value=score(bench,x["text"],r);v2=independent_score(bench,x["text"],r);assert abs(value-v2)<1e-12,(bench,r["question_id"],value,v2)
        e=dict(x,score=value,parsed_answer=parsed);layout=e.pop("kv_layout")
        for k in layout:layouts.append({"benchmark":bench,"question_id":r["question_id"],**k})
        e.update(category=r["category"],group=r["group"],dataset_name=r.get("dataset_name",""),question=r["question"],choices=r.get("choices",{}))
        exports.append(e);grouped[r["question_id"]][x["method"]]=e
        if x["flags"] or x["parse_failure"]:anomaly_rows.append({k:e[k] for k in ["benchmark","question_id","method","text","token_ids","eos","truncated","flags","parse_failure","score"]})
    paired=[]
    for q,pair in grouped.items():
        if set(pair)!=set(METHODS):continue
        a=pair["native"];b=pair["physical_d"];r=index[q]
        assert a["input_binding"]==b["input_binding"] and a["prompt_sha256"]==b["prompt_sha256"] and a["image_sha256"]==b["image_sha256"]
        paired.append(dict(r,native_score=a["score"],physical_score=b["score"],native_text=a["text"],physical_text=b["text"],native_parsed=a["parsed_answer"],physical_parsed=b["parsed_answer"],first_token_flip=a["token_ids"][0]!=b["token_ids"][0],complete_token_change=a["token_ids"]!=b["token_ids"],text_change=a["text"]!=b["text"]))
    csv_gz(out/"per_question.csv.gz",exports);csv_gz(out/"kv_layout_per_layer.csv.gz",layouts);csvwrite(out/"output_anomalies.csv",anomaly_rows,["benchmark","question_id","method","text","token_ids","eos","truncated","flags","parse_failure","score"])
    length=[]
    for method in METHODS:
        rs=[x for x in exports if x["method"]==method];values=np.array([x["generated_tokens"] for x in rs])
        if not len(rs):continue
        length.append({"benchmark":bench,"method":method,"N":len(rs),"mean":float(values.mean()),"P50":float(np.median(values)),"P95":float(np.percentile(values,95)),"max":int(values.max()),"EOS_rate":sum(x["eos"] for x in rs)/len(rs),"truncation_rate":sum(x["truncated"] for x in rs)/len(rs),"parse_failure_rate":sum(x["parse_failure"] for x in rs)/len(rs),"explanation_count":sum(x["explanation"] for x in rs),"repetition_count":sum(x["repetition"] for x in rs)})
    csvwrite(out/"generation_length_audit.csv",length)
    current=readj(RUN/"status"/(bench+".json"));full=current["status"]=="GENERATION_COMPLETE" and len(paired)==EXPECTED[bench] and len(exports)==2*EXPECTED[bench]
    if full:
        official=official_checks(bench,paired)
        if bench=="MME":summary=[measure(bench,paired,"MME","Perception"),measure(bench,paired,"MME","Cognition"),measure(bench,paired,"MME","Total")]
        else:
            summary=[measure(bench,paired,"Circular Accuracy" if bench=="MMBench" else "OCRBench" if bench=="OCRBench" else "Soft Accuracy" if bench=="TextVQA" else "Accuracy")]
            if bench=="POPE":
                summary.append(measure(bench,paired,"F1"))
                for sub in ["random","popular","adversarial"]:
                    rs=[r for r in paired if r["category"]==sub];summary.extend([measure(bench,rs,"Accuracy",sub),measure(bench,rs,"F1",sub)])
        csvwrite(out/"accuracy_summary.csv",summary)
        changes={"native_correct_to_physical_wrong":sum(r["native_score"]==1 and r["physical_score"]==0 for r in paired),"native_wrong_to_physical_correct":sum(r["native_score"]==0 and r["physical_score"]==1 for r in paired),"first_token_flips":sum(r["first_token_flip"] for r in paired),"complete_token_changes":sum(r["complete_token_change"] for r in paired),"full_text_changes":sum(r["text_change"] for r in paired)}
        writej(out/"paired_change_counts.json",changes)
        subgroups=[]
        for cat in sorted({r["category"] for r in paired}):
            rs=[r for r in paired if r["category"]==cat]
            subgroups.append({"benchmark":bench,"category":cat,"N":len(rs),"native_correct_or_soft_sum":sum(r["native_score"] for r in rs),"physical_correct_or_soft_sum":sum(r["physical_score"] for r in rs),"delta_pp":100*sum(r["physical_score"]-r["native_score"] for r in rs)/len(rs),"note":"MME category raw official scores in main metric; this row describes question-level answers only." if bench=="MME" else ""})
        csvwrite(out/"category_diagnostics.csv",subgroups)
        csv_gz(out/"harm_cases.csv.gz",[{k:r[k] for k in ["question_id","image_sha256","question","gold","category","native_text","physical_text","native_score","physical_score"]} for r in paired if r["physical_score"]<r["native_score"]])
        status="COMPLETE"
    else:
        official={"scalar_rescoring":"PASS"};status=current["status"]
    validation={"status":status,"expected_questions":EXPECTED[bench],"saved_generation_rows":len(exports),"paired_questions":len(paired),"full_split_score_reported":bool(full),"all_raw_predictions_rescored_twice":True,"scorers":official,"all_input_bindings_equal":True,"no_large_output_discarded":True,"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER"}
    writej(out/"validation.json",validation);print("ANALYZED",bench,status,len(exports),flush=True)
def aggregate():
    summaries=[];lengths=[];pre=[];anomalies_all=[];valid={};states={}
    for bench in BENCHES:
        status=RUN/"status"/(bench+".json");states[bench]=readj(status) if status.exists() else {"status":"NOT_STARTED"}
        out=OUT/bench
        for name,dst in [("accuracy_summary.csv",summaries),("generation_length_audit.csv",lengths),("prompt_preflight.csv",pre),("output_anomalies.csv",anomalies_all)]:
            p=out/name
            if p.exists():
                with p.open() as f:dst.extend(list(csv.DictReader(f)))
        if (out/"validation.json").exists():valid[bench]=readj(out/"validation.json")
        elif states[bench]["status"]=="BLOCKED_SCORER_OR_DATA":valid[bench]=states[bench]
    csvwrite(OUT/"accuracy_summary.csv",summaries,["benchmark","metric","subset","N","native_score","physical_score","delta","ci_low","ci_high","units","image_clusters","status","bootstrap_reps","bootstrap_seed"])
    csvwrite(OUT/"paired_comparisons.csv",summaries,["benchmark","metric","subset","N","native_score","physical_score","delta","ci_low","ci_high","units","image_clusters","status","bootstrap_reps","bootstrap_seed"])
    csvwrite(OUT/"generation_length_audit.csv",lengths,["benchmark","method","N","mean","P50","P95","max","EOS_rate","truncation_rate","parse_failure_rate","explanation_count","repetition_count"])
    csvwrite(OUT/"prompt_preflight.csv",pre,["benchmark","question_id","method","prompt_sha256","image_sha256","visual_tokens","input_tokens","generated_tokens","max_new_tokens","eos","truncated","text","parsed_answer","parse_failure","explanation","repetition","flags","physical_minus_native_length"])
    csvwrite(OUT/"output_anomalies.csv",anomalies_all,["benchmark","question_id","method","text","token_ids","eos","truncated","flags","parse_failure","score"])
    with gzip.open(OUT/"per_question.csv.gz","wt",newline="",encoding="utf8") as dest:
        writer=None
        for b in BENCHES:
            f=OUT/b/"per_question.csv.gz"
            if not f.exists():continue
            with gzip.open(f,"rt") as src:
                reader=csv.DictReader(src)
                if writer is None:writer=csv.DictWriter(dest,fieldnames=reader.fieldnames);writer.writeheader()
                for row in reader:writer.writerow(row)
    complete=[b for b,v in valid.items() if v["status"]=="COMPLETE"]
    writej(OUT/"validation.json",{"status":"COMPLETE" if len(complete)==7 else "PARTIAL","completed_benchmarks":complete,"benchmarks":valid,"runtime_states":states,"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER","protocol_sha256":sha(OUT/"protocol.json")})
    lines=["# InternVL3-SVH 七项 Benchmark 单图块质量评测","",f"当前已完整完成 {len(complete)}/7 项。只有 COMPLETE 行为全集正式分数；其余不推断缺失分数。","","| Benchmark | N | Native | Physical-D | Δ | 95% CI | 单位 / 状态 |","|---|---:|---:|---:|---:|---|---|"]
    for bench in BENCHES:
        ss=[r for r in summaries if r["benchmark"]==bench and (r["subset"]=="all" or (bench=="MME" and r["subset"] in ("Perception","Cognition"))) and r["metric"]!="F1"]
        if ss:
            for r in ss:lines.append(f'| {bench} {r["subset"] if bench=="MME" else ""} | {r["N"]} | {float(r["native_score"]):.3f} | {float(r["physical_score"]):.3f} | {float(r["delta"]):+.3f} | [{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}] | {r["units"]} / COMPLETE |')
        else:lines.append(f'| {bench} | — | — | — | — | — | {states[bench]["status"]} |')
    lines+=["","主标签：ATE_SVH_1TILE_SCORER_ALIGNED。不是 InternVL 作者完整排行榜配置；固定 1 tile、256 视觉 Token，与多图块、CoT、作者 max_new_tokens 等设置存在差异。MMBench 为 EN Dev20230712 Circular，采用官方 exact_matching，不调用 LLM judge。TextVQA 为 image-only，保留全部十答案共识软评分。MME 保留已知历史图片 SHA 差异，报告 Perception/Cognition 原始积分。","",
    "数值状态保持 FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER，不宣称 NUMERICAL_PASS。Physical-D 核心来自 e288eee271b7aeb64e5df59783d3201bb2a6b313，SHA256 不变。Native/No-op/官方 Chat 严格一致及物理结构检查独立记录在各任务 smoke_validation.json。","",
    "任务安全上限：POPE 8；MME/GQA/ScienceQA/MMBench 16；TextVQA 32；OCRBench 128。Greedy、B1、正常 EOS、每次全新 Cache。预检及正式异常记录完整保留；未通过长度门槛的任务暂停，不将子集成绩写成全集成绩。","",
    "95%CI 为固定 seed=20261110、20,000 次 Image SHA256 整簇配对 Bootstrap 的普通区间，未作多重比较校正，不证明等价。OCRBench 使用固定1000题尺度的簇重采样均值；MME 每次重采样重新计算各类 acc+acc_plus；Circular 按完整发布题组聚合。不同原生单位不平均。","",
    "每个任务的逐题完整答案、Token IDs、Gold、输入绑定和分数见对应 per_question.csv.gz；原始 SQLite 检查点保留在服务器 SSD，并经哈希校验同步持久化目录。图像读取时校验实际 SHA，未重新下载模型或图片。","",
    "本轮不执行固定32 Token效率测试。自然生成耗时或长度不能解释为物理加速。","",
    "| 任务 | 当前状态 / 错误 |","|---|---|"]
    for b in BENCHES:lines.append(f'| {b} | {states[b]["status"]}: {str(states[b].get("error",""))[:500]} |')
    lines+=["","GPU UUID："+GPU,"","服务器持久化目录："+str(PERSIST),"","GitHub 分支：https://github.com/reyuyu/ATE/tree/codex/internvl3-svh-physical-d-seven-benchmarks-3090",""]
    (OUT/"report_zh.md").write_text("\n".join(lines))
    plot(summaries)
def plot(summary):
    if not summary:
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));ax.axis("off")
        statuses=[b+": "+(readj(RUN/"status"/(b+".json"))["status"] if (RUN/"status"/(b+".json")).exists() else "NOT_STARTED") for b in BENCHES]
        ax.text(.02,.98,"No complete benchmark scores yet\n\n"+"\n".join(statuses),va="top",family="monospace")
        fig.tight_layout();fig.savefig(OUT/"figure_quality.png",dpi=160);fig.savefig(OUT/"figure_quality.pdf");plt.close(fig)
        return
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[r for r in summary if r["metric"]!="F1" and (r["subset"]=="all" or (r["benchmark"]=="MME" and r["subset"] in ("Perception","Cognition")))]
    fig,axes=plt.subplots(len(rows),1,figsize=(7,2*len(rows)),squeeze=False)
    for ax,r in zip(axes[:,0],rows):
        vals=[float(r["native_score"]),float(r["physical_score"])];ax.barh(["Native","Physical-D"],vals,color=["#607d8b","#e87838"]);ax.invert_yaxis();ax.set_title(r["benchmark"]+" "+(r["subset"] if r["benchmark"]=="MME" else "")+" ("+r["units"]+")")
        for j,v in enumerate(vals):ax.text(v,j,f" {v:.2f}",va="center")
        ax.set_xlim(0,max(vals)*1.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("ATE single-tile quality; numerical waiver retained",fontsize=11);fig.tight_layout();fig.savefig(OUT/"figure_quality.png",dpi=160);fig.savefig(OUT/"figure_quality.pdf");plt.close(fig)
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--benchmark");args=a.parse_args()
    if args.benchmark:analyze_one(args.benchmark)
    aggregate()
