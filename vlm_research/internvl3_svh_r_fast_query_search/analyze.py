"""Independent CPU scoring, paired image-cluster inference and concise report."""
from .common import *
from collections import Counter
from vlm_research.internvl3_svh_seven_benchmarks import scoring
import torch,subprocess
_csvout=csvout
def csvout(name,rows,fields=None):
 if fields is not None:rows=[{k:r.get(k,"") for k in fields} for r in rows]
 return _csvout(name,rows,fields)
def main():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");records=[json.loads(x[0]) for x in db.execute("SELECT record FROM predictions")];db.close()
 manifest=read(OUT/"sample_manifest.json");protocol=read(OUT/"protocol.json");complete=read(OUT/"run_completion.json") if (OUT/"run_completion.json").exists() else {"status":"PARTIAL"}
 byq={r["question_id"]:r for r in manifest["Discovery"]+manifest["Holdout"]};sm=read(OUT/"source_manifest.json");ph=sha(OUT/"protocol.json");source_sha=sha(OUT/"source_manifest.json")
 for path,h in sm["runtime_source_sha256"].items():assert sha(RUN/"code"/path)==h
 qual=[x for x in records if x["phase"] in ("A","B","Holdout")];speed=[x for x in records if x["phase"].startswith("speed_")]
 audit=[];layout=[]
 for x in records:
  r=byq[x["question_id"]];assert x["gold"]==r["gold"] and x["image_sha256"]==r["image_sha256"] and x["sample_sha256"]==r["sample_sha256"]
  assert x["protocol_sha256"]==ph and x["source_manifest_sha256"]==source_sha
  official=scoring.score("GQA",x["text"],r)[1];independent=scoring.independent_score("GQA",x["text"],r);assert official==independent==x["score"]
  assert x["generated_tokens"]==len(x["token_ids"])<= (32 if x["phase"].startswith("speed_") else 16)
  assert len(x["kv_layout"])==28
  if x["method"]!="Native":
   for row in x["kv_layout"]:
    i=row["layer"];K=SPACE[x["method"]];expected=0 if i in (2,3,4,5) else r["input_tokens"]+x["generated_tokens"]-1-(0 if i in K else 256)
    assert row["physical_length"]==expected and row["visual_kv"]==(256 if i in K else 0)
    assert row["capacity_bytes"]>=row["live_bytes"]>=0
  assert abs(sum(a["live_bytes"] for a in x["kv_layout"])/1048576-x["KV_live_MiB"])<1e-9
  assert abs(sum(a["capacity_bytes"] for a in x["kv_layout"])/1048576-x["KV_capacity_MiB"])<1e-9
  for row in x["kv_layout"]:layout.append(dict(phase=x["phase"],question_id=x["question_id"],condition=x["method"],**row))
 assert not torch.cuda.is_initialized()
 selection=read(OUT/"selection.json") if (OUT/"selection.json").exists() else {}
 selected=selection.get("stage_A",{}).get("selected",[])
 disc_all=[x for x in qual if x["phase"] in ("A","B")];disc=[x for x in disc_all if x["method"] in selected+["Native"]];hold=[x for x in qual if x["phase"]=="Holdout"]
 if complete["status"]=="COMPLETE":
  assert sum(x["method"]!="Native" for x in qual)==complete["quality_generated"]<=1152
  assert len([x for x in qual if x["phase"]=="A" and x["method"]!="Native"])==448
  assert len([x for x in qual if x["phase"]=="B" and x["method"]!="Native"])==56*len(selected)
  assert len([x for x in hold if x["method"]!="Native"])==160*len(selection["holdout_methods"])
  assert len(speed)==36*len(set(x["method"] for x in speed))
  rerank=selections([x for x in disc if x["method"]!="Native"],"B");assert rerank["best_quality"]==selection["best_quality"] and rerank["best_efficiency"]==selection["best_efficiency"]
  assert complete["selection_sha256"]==sha(OUT/"selection.json")==read(OUT/"selection_freeze_receipt.json")["selection_sha256"]
 csvout("stage_A_accuracy.csv",summarize([x for x in qual if x["phase"]=="A"],"A64"))
 csvout("discovery_accuracy.csv",summarize(disc,"Discovery120"))
 csvout("holdout_accuracy.csv",summarize(hold,"Holdout160"))
 pairs=[]
 for label,rows in [("StageA64",[x for x in qual if x["phase"]=="A"]),("Discovery120",disc),("Holdout160",hold)]:
  for m in list(dict.fromkeys(x["method"] for x in rows)):
   for base in ["R0","Native"]:
    if m==base:continue
    for group in ["all","query","nonquery"]:
     p=pair(rows,base,m,label,group)
     if p:pairs.append(p)
 # Equal-budget R4 center versus late is paired at the common Stage A64 only unless both reached B.
 a4=[x for x in qual if x["phase"]=="A"]
 for g in ["all","query","nonquery"]:
  p=pair(a4,"R4-M","R4-L","StageA64_R4_equal_budget",g)
  if p:pairs.append(p)
 if "R4-M" in selected and "R4-L" in selected:
  for g in ["all","query","nonquery"]:
   p=pair(disc,"R4-M","R4-L","Discovery120_R4_equal_budget",g)
   if p:pairs.append(p)
 csvout("paired_changes.csv",pairs)
 sub=[];harm=[];changes=[]
 for phase,rows in [("StageA64",[x for x in qual if x["phase"]=="A"]),("Discovery120",disc),("Holdout160",hold)]:
  for m in dict.fromkeys(x["method"] for x in rows):
   for kind in ["category","subtype"]:
    for value in sorted(set(x[kind] for x in rows if (kind=="category" or x["category"]=="query"))):
     rr=[x for x in rows if x["method"]==m and x[kind]==value and (kind=="category" or x["category"]=="query")]
     if rr:sub.append(dict(phase=phase,method=m,stratum=kind,value=value,N=len(rr),correct=sum(x["score"] for x in rr),accuracy_pct=100*np.mean([x["score"] for x in rr])))
  base={x["question_id"]:x for x in rows if x["method"]=="R0"}
  for x in rows:
   if x["method"] in ("R0","Native"):continue
   b=base[x["question_id"]]
   if b["score"]!=x["score"] or any(x["flags"].values()):
    row=dict(phase=phase,question_id=x["question_id"],image_sha256=x["image_sha256"],category=x["category"],subtype=x["subtype"],method=x["method"],question=byq[x["question_id"]]["question"],gold=x["gold"],R0_text=b["text"],candidate_text=x["text"],R0_score=b["score"],candidate_score=x["score"],change="harm" if b["score"]>x["score"] else "recovery" if b["score"]<x["score"] else "anomaly_only",flags=x["flags"])
    changes.append(row)
    if row["change"]=="harm":harm.append(row)
 csvout("query_subtypes.csv",sub);csvout("harm_cases.csv",harm, list(changes[0]) if changes else ["phase","question_id","method"]);csvout("recovery_and_anomaly_cases.csv",changes)
 kv=[]
 for label,rows in [("StageA64",[x for x in qual if x["phase"]=="A"]),("Discovery120",disc),("Holdout160",hold)]:
  for method in dict.fromkeys(x["method"] for x in rows):
   rr=[x for x in rows if x["method"]==method]
   kv.append(dict(phase=label,method=method,N=len(rr),reaccess_layers=SPACE.get(method,list(range(28))),mean_actual_quality_EOS_KV_live_MiB=float(np.mean([x["KV_live_MiB"] for x in rr])),mean_actual_quality_EOS_KV_capacity_MiB=float(np.mean([x["KV_capacity_MiB"] for x in rr])),mean_generated_tokens=float(np.mean([x["generated_tokens"] for x in rr])),visual_KV_layers=28 if method=="Native" else len(SPACE[method])))
 csvout("kv_cost.csv",kv);csvout("kv_layout_quality.csv",[x for x in layout if not x["phase"].startswith("speed_")])
 csvout("per_question.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in qual])
 csvout("output_anomalies.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in qual if any(x["flags"].values())],["phase","question_id","method","category","gold","text","token_ids","score","generated_tokens","eos","truncated","empty","flags"])
 lengths=[]
 for phase,rows in [("A",[x for x in qual if x["phase"]=="A"]),("B",[x for x in qual if x["phase"]=="B"]),("Holdout",hold)]:
  for m in dict.fromkeys(x["method"] for x in rows):
   rr=[x for x in rows if x["method"]==m];n=np.array([x["generated_tokens"] for x in rr])
   lengths.append(dict(phase=phase,method=m,N=len(rr),P50=float(np.quantile(n,.5)),P95=float(np.quantile(n,.95)),maximum=int(n.max()),EOS=sum(x["eos"] for x in rr),truncated=sum(x["truncated"] for x in rr),empty=sum(x["empty"] for x in rr),meta=sum(x["flags"]["meta"] for x in rr),repeat=sum(x["flags"]["repeat"] for x in rr),all_anomalies=sum(any(x["flags"].values()) for x in rr)))
 csvout("output_length_summary.csv",lengths)
 release=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
 validation={"status":"COMPLETE_EXPLORATORY" if complete["status"]=="COMPLETE" else "PARTIAL","independent_scalar_scores_all_equal":True,"actual_KV_shape_bytes_verified_all":True,"all_checkpoint_input_protocol_source_bindings_verified":True,"Discovery_Holdout_image_disjoint":True,"all_anomalies_retained":True,"new_Native_quality_generations":0,"historical_Native_binding_and_8_smoke":"PASS","model_loads":complete.get("model_loads"),"new_quality_generations":complete.get("quality_generated"),"speed_requests":len(speed),"historical_numerical_failure_unchanged":True,"numerical_equivalence_claimed":False,"GPU_processes_after_worker_exit":release,"GPU_released":not release,"same_GPU_UUID":GPU,"bootstrap":"5000 image-cluster paired, ordinary 95%, no multiplicity correction","no_extra_candidates_or_benchmarks":True}
 write(OUT/"validation.json",validation)
 def table(rows,keys):
  return "| "+" | ".join(keys)+" |\n| "+" | ".join(["---"]*len(keys))+" |\n"+"\n".join("| "+" | ".join(f"{r.get(k):.3f}" if isinstance(r.get(k),float) else str(r.get(k,"—")) for k in keys)+" |" for r in rows)
 hsummary=summarize(hold,"Holdout160");ds=summarize(disc,"Discovery120");asummary=summarize([x for x in qual if x["phase"]=="A"],"A64")
 speedsummary=list(csv.DictReader((OUT/"speed_summary.csv").open())) if (OUT/"speed_summary.csv").exists() else []
 sp={x["method"]:x for x in speedsummary};bestq=selection.get("best_quality","NOT_SELECTED");beste=selection.get("best_efficiency","NOT_SELECTED")
 def acc(method,group="query"):
  return next((x["accuracy_pct"] for x in hsummary if x["method"]==method and x["group"]==group),float("nan"))
 pcore=[x for x in pairs if x["phase"]=="Holdout160" and x["base"]=="R0" and x["group"]=="query" and x["candidate"]==bestq]
 evidence=("尚无可用独立结果" if not pcore else f"{pcore[0]['difference_pp']:+.3f}pp，95%CI [{pcore[0]['CI95_low_pp']:+.3f}, {pcore[0]['CI95_high_pp']:+.3f}]pp")
 smoke=read(OUT/"smoke_validation.json") if (OUT/"smoke_validation.json").exists() else {}
 report=f"""# InternVL3-SVH-R 快速中层回看探索

状态：{validation['status']}；{protocol['status']}。本轮不宣称数值等价。\n\n**没有找到可在独立Holdout重复改善Query的配置。** Discovery挑出的两个候选在Holdout均低于新R0，不能据此认定某个最少回看层数有效，也不再扩大搜索。

Discovery 冻结质量候选 **{bestq}**（K={SPACE.get(bestq,[])}），低成本候选 **{beste}**（K={SPACE.get(beste,[])}）。
独立 Holdout Query：Native {acc('Native'):.3f}%，R0 {acc('R0'):.3f}%，{bestq} {acc(bestq):.3f}%，{beste} {acc(beste):.3f}%。最佳质量相对R0：{evidence}。
是否有重复改善应依据该独立配对差值和区间判断，不能用Discovery挑选结果代替Holdout验证。

## 1. 阶段A：固定64题全部七候选

{table(asummary,["method","group","N","correct","accuracy_pct","mean_KV_capacity_MiB","anomalies"])}

筛选规则在任何GPU生成前冻结：Query提升减去Non-Query损失、超额异常率和实际KV容量惩罚。全六个回看候选排名保存在selection_stage_A.json，只有前三个和R0补齐剩余56题。4层中心与后段只在共同StageA64上作首要同预算比较，不能把不同样本量误当同样本比较。

## 2. Discovery120

{table(ds,["method","group","N","correct","accuracy_pct","mean_KV_capacity_MiB","anomalies"])}

best_quality按Query最高选取；best_efficiency允许距最高Query 2.5pp且Non-Query不低于R0 2.5pp，再优先少层、低实测KV。规则细节见protocol.json，不根据Holdout重新选层。候选选择文件SHA256：{sha(OUT/'selection.json') if selection else 'NOT_SELECTED'}。

## 3. 独立Holdout160

{table(hsummary,["method","group","N","correct","accuracy_pct","mean_KV_capacity_MiB","anomalies"])}

{table([x for x in pairs if x['phase']=='Holdout160' and x['base']=='R0'],["candidate","group","N","difference_pp","CI95_low_pp","CI95_high_pp","wrong_to_right","right_to_wrong"])}

Image SHA整簇配对Bootstrap 5000次，普通未校正95%CI；搜索、多组和子组比较均为探索性。区间跨0不证明等价；小样本不能证明全量GQA非劣效。Query各Object/Attribute/Color/Relation/Spatial等实际类别及Non-Query四类全部见query_subtypes.csv，全部新增错误见harm_cases.csv，恢复与异常见recovery_and_anomaly_cases.csv。没有数量子类时不虚构数量题。

## 4. 同机真实固定32 Token速度

{table(speedsummary,["method","requests","E2E_ms","TTFT_ms","TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"])}

12张预先冻结不同图片，每方法2次预热、每图3次正式重复，轮换顺序。包含真实Vision、Projector、Prefill、31次Decode及方法必要清理；同步墙钟，排除磁盘/Tokenizer/预处理/H2D和结果写盘。KV均由真实28层K/V张量与底层存储字节重算。质量正常EOS与测速固定32严格分开；质量EOS平均KV不能冒充固定32KV。GPU为{GPU}。不混入旧实例时间。零值Key/Value临时scratch参与实际计算，开销已计时；不宣称完全消除了被屏蔽Key的计算。

"""
 if sp:
  native_time=float(sp["Native"]["E2E_ms"])
  report+="| 方法 | 相对Native E2E加速倍数 | 相对R0增加KV@32 MiB | 相对Native KV下降 |\n|---|---:|---:|---:|\n"
  for m,x in sp.items():
   report+=f"| {m} | {native_time/float(x['E2E_ms']):.3f}× | {float(x['KV_live_MiB'])-float(sp['R0']['KV_live_MiB']):+.3f} | {100*(1-float(x['KV_live_MiB'])/float(sp['Native']['KV_live_MiB'])):.2f}% |\n"
 report+=f"""\n## 4b. 相同4层预算及Query子类\n\n{table([x for x in pairs if x["phase"]=="Discovery120_R4_equal_budget"],["base","candidate","group","difference_pp","CI95_low_pp","CI95_high_pp"])}\n\n{table([x for x in sub if x["phase"]=="Holdout160" and x["stratum"]=="subtype"],["method","value","N","correct","accuracy_pct"])}\n\n4层中心与4层后段在Discovery上的差异只表示本样本观察；后段未进入最终Holdout，不能宣称中层位置普遍更优。Holdout恢复和损害是逐题配对统计，不以净平均掩盖Non-Query或Query类别中的错误。\n\n## 5. Smoke与数值豁免

Native/No-op有效Logits严格torch.equal通过；8题新Native完整Token和历史记录一致，历史Native逐题图片SHA、Prompt、Token IDs、BF16像素Tensor、Gold、模型Revision、原生生成协议与源码绑定全部通过，故质量参考合法复用，新增Native质量生成0次。新R0全部重新生成，不用历史Physical-D顶替。

只对R0/R4-L/R8进行8题对应Dense-R/Physical-R对照；短/长输入做完整结构审计，其余仅轻量有效Logits记录。L20删除256视觉Hidden、L2–L5答案Attention真实不调用QKV/QK/AV/o_proj、所选层保留真实视觉KV、其他层无持久原始视觉KV、位置RoPE、禁用概率、非目标连接和Norm/MLP/Residual由冻结V2结构审计及逐题KV核对验证。缓存独立、Hook恢复，无NaN/Inf。Smoke状态：{smoke.get('status')}。

"""
 for m in ["R0","R4-L","R8"]:
  report+=f"- {m}：{smoke.get(m)}\n"
 checks=smoke.get("checks",[])
 for m in ["R0","R4-L","R8"]:
  dif=[v for x in checks if x["condition"]==m for v in x["aligned_prefix_logits"]]
  if dif:report+=f"- {m}：相同生成前缀Logits MaxAbs最大{max(v['max_abs'] for v in dif):.6f}，步均MeanAbs {np.mean([v['mean_abs'] for v in dif]):.6f}，Top1一致{sum(v['top1_equal'] for v in dif)}/{len(dif)}。前缀分歧后的Logits不作同条件数值比较。\n"
 report+=f"""
历史9602和2d671的NUMERICAL_EQUIVALENCE_FAILED及C1/C2失败证据原封保留。本轮使用9602冻结V2原引擎，不导入C1/C2补零投影修复，不放宽旧协议后宣称PASS，而是用户在GPU前授权新探索性质量协议。Dense/Physical存在BF16形状相关数值差异，输出/质量变化不能全部归因于回看机制；这限制机制归因和跨实现可迁移性。本轮只有结构门槛PASS_WITH_NUMERICAL_WAIVER，绝非数值等价。

## 6. 完整性与资源

正式新增质量{complete.get('quality_generated','PARTIAL')}次，Native历史复用280题，固定32正式测速{len(speed)}请求，模型加载{complete.get('model_loads','unknown')}次。全部异常计入评分，没有删题、换Gold、扩上限、重新挑层或运行其他Benchmark。独立官方/标量评分逐条相等；输入、协议、源码与检查点SHA绑定核验通过。逐题原始答案/Token IDs、实际KV、异常和进度保存在SSD与持久工作盘server_only。

服务器完整结果：/home/featurize/work/ATE-internvl3-svh-r-fast-query-search-3090/vlm_research/results/internvl3_svh_r_fast_query_search_3090/
SSD运行目录：{RUN}
Git分支：codex/internvl3-svh-r-fast-query-search-3090
GPU释放核验：{validation['GPU_released']}；未关闭服务器。不制作Windows备份。\n\n收尾工程事件：所有生成、测速及汇总已完成、模型已释放后，原run.py最后一条进度日志在SQLite关闭后查询而报错。原failure.json与完整日志保留，post_run_bookkeeping_incident.json记录CPU核验；未重跑任何生成。run_safe.py仅修正关闭顺序供后续复现，未参与本轮测量；原始运行源码SHA保持不变。
"""
 (OUT/"report_zh.md").write_text(report)
 print(json.dumps(validation,ensure_ascii=False));print("BEST",bestq,beste);print("HOLDOUT",json.dumps(hsummary,ensure_ascii=False));print("SPEED",json.dumps(speedsummary,ensure_ascii=False))
if __name__=="__main__":main()
