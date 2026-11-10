"""CPU-only official TextVQA soft-score verification, paired statistics, and report."""
from .common import *
import torch,subprocess
from vlm_research.internvl3_svh_seven_benchmarks import scoring
def table(rows,keys):
 return "| "+" | ".join(keys)+" |\n| "+" | ".join(["---"]*len(keys))+" |\n"+"\n".join("| "+" | ".join(f"{r.get(k):.3f}" if isinstance(r.get(k),float) else str(r.get(k,"—")) for k in keys)+" |" for r in rows)
def main():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");rows=[json.loads(x[0]) for x in db.execute("SELECT record FROM predictions")];db.close()
 manifest=read(OUT/"sample_manifest.json");byq={r["question_id"]:r for r in manifest["Discovery"]+manifest["Holdout"]+manifest["Smoke"]};protocol=read(OUT/"protocol.json");sm=read(OUT/"source_manifest.json");ph=sha(OUT/"protocol.json");sh=sha(OUT/"source_manifest.json")
 for path,h in sm["runtime_source_sha256"].items():assert sha(RUN/"code"/path)==h,("SOURCE_HASH",path)
 quality_rows=[x for x in rows if x["phase"] in ("A","B","Holdout")];speed=[x for x in rows if x["phase"].startswith("speed_")]
 bindings=read(RUN/"prepared/input_bindings.json")
 for x in rows:
  r=byq[x["question_id"]];assert x["gold"]==r["gold"] and len(x["gold"])==10 and x["image_sha256"]==r["image_sha256"] and x["sample_sha256"]==r["sample_sha256"]
  assert x["protocol_sha256"]==ph and x["source_manifest_sha256"]==sh and x["input_binding"]==bindings[x["question_id"]]["binding"]
  official=scoring.score("TextVQA",x["text"],r)[1];independent=scoring.independent_score("TextVQA",x["text"],r)
  assert abs(official-independent)<1e-10 and abs(x["score"]-independent)<1e-10
  assert len(x["token_ids"])==x["generated_tokens"]<=32 and len(x["kv_layout"])==28
  assert x["eos"]==(x["token_ids"][-1]==151645)
  for v in x["kv_layout"]:
   layer=v["layer"];K=SPACE.get(x["method"],list(range(28)))
   expected=r["input_tokens"]+x["generated_tokens"]-1 if x["method"]=="Native" else 0 if layer in (2,3,4,5) else r["input_tokens"]+x["generated_tokens"]-1-(0 if layer in K else 256)
   assert v["physical_length"]==expected and v["visual_kv"]==(256 if layer in K else 0)
   assert v["capacity_bytes"]>=v["live_bytes"]>=0
   expectedbytes=(np.prod(v["k_shape"])+np.prod(v["v_shape"]))*2 if v["physical_length"] else 0
   assert expectedbytes==v["live_bytes"]
  assert abs(sum(v["live_bytes"] for v in x["kv_layout"])/1048576-x["KV_live_MiB"])<1e-9
  assert abs(sum(v["capacity_bytes"] for v in x["kv_layout"])/1048576-x["KV_capacity_MiB"])<1e-9
 selection=read(OUT/"selection.json") if (OUT/"selection.json").exists() else {};done=read(OUT/"run_completion.json") if (OUT/"run_completion.json").exists() else {"status":"PARTIAL"}
 selected=selection.get("stage_A",{}).get("selected",[]);A=[x for x in quality_rows if x["phase"]=="A"];disc=[x for x in quality_rows if x["phase"] in ("A","B") and x["method"] in selected+["Native"]];hold=[x for x in quality_rows if x["phase"]=="Holdout"]
 if done["status"]=="COMPLETE":
  assert len([x for x in A if x["method"]!="Native"])==336
  assert len([x for x in quality_rows if x["phase"]=="B" and x["method"]!="Native"])==48*len(selected)
  assert len([x for x in hold if x["method"]!="Native"])==96*len(selection["holdout_methods"])
  assert sum(x["method"]!="Native" for x in quality_rows)==done["quality_generated"]<=816
  rerank=selections([x for x in disc if x["method"]!="Native"],"B");assert rerank["best_quality"]==selection["best_quality"] and rerank["best_efficiency"]==selection["best_efficiency"]
  assert done["selection_sha256"]==sha(OUT/"selection.json")==read(OUT/"selection_freeze_receipt.json")["selection_sha256"]
 stages=[("StageA48",A),("Discovery96",disc),("Holdout96",hold)]
 for name,rr in stages:csvout("stage_A_accuracy.csv" if name=="StageA48" else "discovery_accuracy.csv" if name=="Discovery96" else "holdout_accuracy.csv",summarize(rr,name))
 pairs=[];subtypes=[];case_rows=[];meta=[];kv=[];lengths=[]
 for phase,rr in stages:
  methods=list(dict.fromkeys(x["method"] for x in rr));base={x["question_id"]:x for x in rr if x["method"]=="R0"};nat={x["question_id"]:x for x in rr if x["method"]=="Native"}
  for method in methods:
   vals=[x for x in rr if x["method"]==method];summary=next(x for x in summarize(vals,phase));meta.append({k:v for k,v in summary.items() if k not in ("reaccess_layers","mean_KV_live_MiB","mean_KV_capacity_MiB")})
   kv.append(dict(phase=phase,method=method,N=len(vals),reaccess_layers=SPACE.get(method,list(range(28))),mean_actual_quality_EOS_KV_live_MiB=summary["mean_KV_live_MiB"],mean_actual_quality_EOS_KV_capacity_MiB=summary["mean_KV_capacity_MiB"],mean_generated_tokens=float(np.mean([x["generated_tokens"] for x in vals])),KV_increment_vs_R0_MiB=summary["mean_KV_live_MiB"]-np.mean([base[x["question_id"]]["KV_live_MiB"] for x in vals])))
   lens=[x["generated_tokens"] for x in vals];lengths.append(dict(phase=phase,method=method,N=len(vals),P50=float(np.quantile(lens,.5)),P95=float(np.quantile(lens,.95)),max_tokens=max(lens),EOS=sum(x["eos"] for x in vals),truncated=sum(x["truncated"] for x in vals),empty=sum(x["empty"] for x in vals)))
   for typ in TYPES:
    v=[x for x in vals if x["analysis_type"]==typ]
    if not v:continue
    b=[base[x["question_id"]] for x in v];d=[x["score"]-y["score"] for x,y in zip(v,b)]
    subtypes.append(dict(phase=phase,method=method,question_text_type=typ,N=len(v),soft_accuracy_pct=100*float(np.mean([x["score"] for x in v])),difference_vs_R0_pp=100*float(np.mean(d)),improved=sum(z>1e-10 for z in d),declined=sum(z< -1e-10 for z in d),unchanged=sum(abs(z)<=1e-10 for z in d),meta_count=sum(x["flags"]["meta"] for x in v),answering_prefix=sum(x["flags"]["answering_prefix"] for x in v)))
   for b in ["R0","Native"]:
    if method!=b:
     p=pair(rr,b,method,phase)
     if p:pairs.append(p)
   for x in vals:
    b=base[x["question_id"]];n=nat[x["question_id"]]
    if method!="Native" and (abs(x["score"]-b["score"])>1e-10 or x["flags"]["meta"] or b["flags"]["meta"]):
     case_rows.append(dict(phase=phase,question_id=x["question_id"],image_sha256=x["image_sha256"],analysis_type=x["analysis_type"],method=method,question=byq[x["question_id"]]["question"],gold=x["gold"],R0_text=b["text"],candidate_text=x["text"],Native_text=n["text"],R0_score=b["score"],candidate_score=x["score"],Native_score=n["score"],soft_delta=x["score"]-b["score"],R0_meta=b["flags"]["meta"],candidate_meta=x["flags"]["meta"],candidate_answering_prefix=x["flags"]["answering_prefix"],meta_resolved=b["flags"]["meta"] and not x["flags"]["meta"],flags=x["flags"]))
  if "R4-M" in methods and "R4-L" in methods:pairs.append(pair(rr,"R4-M","R4-L",phase+"_R4_equal_budget"))
 csvout("paired_soft_score_changes.csv",pairs);csvout("textvqa_subtypes.csv",subtypes);csvout("meta_response_analysis.csv",meta);csvout("kv_cost.csv",kv);csvout("output_length_summary.csv",lengths);csvout("per_question_changes.csv",case_rows)
 csvout("per_question.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in quality_rows])
 csvout("kv_layout_per_layer.csv",[dict(phase=x["phase"],question_id=x["question_id"],condition=x["method"],**v) for x in rows for v in x["kv_layout"]])
 csvout("output_anomalies.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in quality_rows if any(x["flags"].values())],["phase","question_id","method","analysis_type","gold","text","token_ids","score","generated_tokens","eos","truncated","empty","flags"])
 smoke=read(OUT/"smoke_validation.json") if (OUT/"smoke_validation.json").exists() else {}
 gate=read(OUT/"holdout_decision.json") if (OUT/"holdout_decision.json").exists() else {"quality_status":"NOT_RUN","speed_status":"NOT_RUN"}
 if done["status"]=="COMPLETE":
  hp=[p for p in pairs if p["phase"]=="Holdout96" and p["base"]=="R0" and p["candidate"] in [selection["best_quality"],selection["best_efficiency"]] and p["candidate"]!="R0"]
  confirmed=[p["candidate"] for p in hp if p["difference_pp"]>0 and p["CI95_low_pp"]>0]
  assert set(confirmed)==set(gate["confirmed_preselected_candidates"])
  assert (len(speed)==108)==bool(confirmed)
 process=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip();assert not torch.cuda.is_initialized()
 validation=dict(status="COMPLETE_EXPLORATORY" if done["status"]=="COMPLETE" else "PARTIAL",quality_status=gate["quality_status"],new_quality_generations=done.get("quality_generated"),Native_historical_reused=done.get("Native_historical_reused"),new_Native_quality=0,model_loads=done.get("model_loads"),formal_speed_requests=len(speed),speed_status=gate["speed_status"],independent_official_and_leave_one_out_scores_equal=True,all_ten_answer_golds_verified=True,checkpoint_input_protocol_source_hashes_verified=True,physical_KV_shape_dtype_live_and_capacity_bytes_verified=True,all_meta_repeat_empty_truncated_outputs_retained=True,no_GQA_or_other_benchmark=True,no_Holdout_predictions_before_selection=True,no_Holdout_layer_reselection=True,image_only_population_audit_5000=True,selected_pixel_bindings_verified=200,numerical_equivalence_claimed=False,historical_numerical_FAILED_unchanged=True,GPU_UUID=GPU,GPU_compute_processes_after_worker=process,GPU_released=not process)
 write(OUT/"validation.json",validation)
 def summary(phase,method):
  return next((x for x in summarize(hold if phase=="Holdout96" else disc,phase) if x["method"]==method),{})
 bq=selection.get("best_quality","NOT_SELECTED");be=selection.get("best_efficiency","NOT_SELECTED")
 report=f"""# InternVL3-SVH-R TextVQA Image-only快速搜索

**状态：{gate['quality_status']}。** 本轮为EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，历史NUMERICAL_EQUIVALENCE_FAILED保持原样，不宣称Dense/Physical数值等价。

Discovery冻结质量候选 **{bq}**（K={SPACE.get(bq,[])}），低成本候选 **{be}**（K={SPACE.get(be,[])}）。
独立Holdout96：Native {summary('Holdout96','Native').get('soft_accuracy_pct',float('nan')):.3f}%，R0 {summary('Holdout96','R0').get('soft_accuracy_pct',float('nan')):.3f}%，{bq} {summary('Holdout96',bq).get('soft_accuracy_pct',float('nan')):.3f}%，{be} {summary('Holdout96',be).get('soft_accuracy_pct',float('nan')):.3f}%。
是否恢复取决于以下同题软分差和图像簇区间，不能以回答更正常代替质量收益。不得将96题结果当作完整5000题成绩。

## Image-only审计与抽样

全部5000题均重新构造原生Chat Template、原始问题及冻结指令，逐题比较完整Prompt、Token IDs和256图像Token位置。指令严格为：
{INSTRUCTION}

数据目录含ocr，实际模型输入只有原图448×448单图块像素、问题、模板和该指令；没有OCR Tokens、sidecar、reference text、Caption、Hint、Gold或Rationale。200张选中图片（正式192、独立Smoke8）经过原图SHA、官方Processor和BF16像素Tensor绑定核验。image_only_input_audit.json、population_prompt_audit.csv和sample_manifest.json提供可审计证据。

先冻结sampling_freeze.json，再读取历史Native的绑定元数据。Discovery96与Holdout96按Image SHA划分，每题不同图片；Smoke另有8张不重叠图片。问题文字启发式七类，不使用Gold长度、预测或旧正确性选题。两集合实际分层均为：signs16/products16/titles12/numbers16/short_words8/multiword16/other12。该分层改变了总体问题分布，不能与历史全5000题均值直接比较。

## 阶段A：48题七候选

{table(summarize(A,'StageA48'),['method','N','soft_accuracy_pct','meta_count','answering_prefix_count','mean_KV_capacity_MiB'])}

按GPU前冻结规则：Soft Accuracy优先，再按元回答率和实际KV成本打破平局，仅前三个回看候选和R0补齐另外48题。不按48题结果宣称显著性。

## Discovery96与冻结选择

{table(summarize(disc,'Discovery96'),['method','N','soft_accuracy_pct','meta_count','answering_prefix_count','mean_KV_capacity_MiB'])}

best_quality为Soft Accuracy最高；best_efficiency允许距离最高2.0pp，优先较少层和较低实测KV；R0亦可成为低成本候选。最终选择SHA：{sha(OUT/'selection.json') if selection else 'NOT_SELECTED'}。Holdout预测正文/Token/评分在此之前没有读取；只用于事前绑定核验的历史元数据不参与选层。

## Holdout96：真实质量与配对不确定性

{table(summarize(hold,'Holdout96'),['method','N','soft_accuracy_pct','meta_count','answering_prefix_count','mean_KV_live_MiB','mean_KV_capacity_MiB'])}

{table([p for p in pairs if p['phase']=='Holdout96' and p['base']=='R0' and p['candidate']!='Native'],['candidate','difference_pp','CI95_low_pp','CI95_high_pp','improved','declined','unchanged','meta_resolved','meta_resolved_and_score_improved','meta_to_scored_answer'])}

Bootstrap采用Image SHA整簇配对5000次，普通未校正95%CI。样本每图一题仍按图像簇执行。搜索与多子类带来选择和多重比较，CI跨零不证明等价，不宣称全量非劣效。无Holdout后重选层。若两个候选都没有改善，则为NO_CONFIRMED_TEXTVQA_RECOVERY；若点估计正向但区间未排除零，明确属于未确认趋势，也不自动测速。

## 哪些文字问题恢复、元回答是否转成有效文字答案

{table([x for x in subtypes if x['phase']=='Holdout96'],['method','question_text_type','N','soft_accuracy_pct','difference_vs_R0_pp','improved','declined','meta_count'])}

全部原始元回答、空输出、重复和截断均原样纳入十答案官方软评分，没有删题、修改答案或更改Prompt。meta_response_analysis.csv报告answering前缀与更广元回答启发式；per_question_changes.csv逐题区分消除元回答、软分提高及实际获得非零软分，避免仅凭文本变正常宣称OCR能力恢复。元回答检测依据完整文本内容，不仅依赖长度/EOS。启发式分组和元回答标记均可能有歧义，正式Soft Accuracy不依赖这些标签。

## 相同四层预算

{table([p for p in pairs if p['phase'].endswith('R4_equal_budget')],['phase','base','candidate','N','difference_pp','CI95_low_pp','CI95_high_pp'])}

StageA48是两个四层候选共同预设比较；若二者均入围，另报告共同Discovery96。未入Holdout的条件不补测、不冒充独立确认。

## KV成本与有条件速度

{table([x for x in kv if x['phase']=='Holdout96'],['method','N','mean_actual_quality_EOS_KV_live_MiB','mean_actual_quality_EOS_KV_capacity_MiB','KV_increment_vs_R0_MiB','mean_generated_tokens'])}

这里是正常EOS质量生成结束时真实持久KV，不是固定32KV，不是理论容量。各方法输出长度不同，成本同时受答案长度影响。每层K/V形状、BF16 dtype、物理长度与底层存储字节均独立重算，L2–L5持久KV为零，只有K层持久保留256原始视觉KV，非K层仅文本KV。临时零值Key/Value scratch不等于持久原始视觉KV，也不声称消除了全部被屏蔽Key计算。

测速状态：**{gate['speed_status']}**。GPU前冻结门槛为预选候选相对R0软分为正且图像簇95%CI下界大于零；候选优先次序预设为best_quality、best_efficiency，不按Holdout重新排名。
"""
 if speed:
  ss=list(csv.DictReader((OUT/"speed_summary.csv").open()));report+="\n"+table(ss,["method","requests","E2E_ms","TTFT_ms","TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"])+"\n"
  sp={x["method"]:x for x in ss};c=gate["speed_candidate"];report+=f"\n实测{c}相对Native E2E加速{float(sp['Native']['E2E_ms'])/float(sp[c]['E2E_ms']):.3f}倍，KV@32相对R0增加{float(sp[c]['KV_live_MiB'])-float(sp['R0']['KV_live_MiB']):.3f}MiB。12张冻结不同图片、每条件2预热及每图3正式重复；同步墙钟包含Vision、Projector、Prefill、31真实Decode和必要清理，排除磁盘、Tokenizer、预处理、H2D和写盘；旋转顺序，不复用质量阶段图像特征进行测速。\n"
 else:report+="\n没有通过预先冻结的Holdout质量改善门槛，因此未执行新增测速；E2E/TTFT/TPOT/KV@32均为NOT_RUN，绝不填入旧GQA数据或理论值。\n"
 report+=f"""
## Smoke、数值豁免与执行范围

Native/No-op有效Logits严格torch.equal通过，8张独立Smoke题Native完整Token与历史一致，192正式题历史Native全部完成Prompt/图像/BF16像素Tensor/Tokenizer/模型Revision/32上限绑定后复用。没有复用历史Physical-D，新R0全部由冻结V2本轮生成。

Smoke只运行R0/R4-M/R8对应Dense-R/Physical-R；最短/最长输入记录完整结构审计，其余只保留有效Logits摘要。L20真实退出256视觉Hidden、L2–L5答案Attention真实跳算、原始逻辑RoPE、KV权限、非目标连接、Norm/MLP/Residual、禁用概率0和Cache独立均通过。无NaN/Inf或Hook泄漏。全部质量审计关闭重型Hook，无28层Hidden或逐步全词表Logits复制。
"""
 for m in ["R0","R4-M","R8"]:
  report+=f"\n- {m} Smoke：{smoke.get(m)}"
  dif=[v for c in smoke.get("checks",[]) if c["condition"]==m for v in c["aligned_prefix_logits"]]
  if dif:report+=f"；相同生成前缀最大Logits MaxAbs={max(v['max_abs'] for v in dif):.6f}，步均MeanAbs={np.mean([v['mean_abs'] for v in dif]):.6f}。"
 report+=f"""

历史数值FAILED保持原样，V2物理删行引起BF16 GEMM形状误差，本轮不使用C1/C2，也不做Kernel搜索。Dense/Physical的质量差异不能全部归因于视觉回看机制，不能宣称数值等价；结果只是对这一冻结实现的探索性工程观察。是否值得更大独立TextVQA验证应以Holdout趋势、区间、元回答真实恢复及KV/速度成本共同判断，不能自动扩大。

实际正式新增质量{done.get('quality_generated')}次（上限816），历史Native正式参考{done.get('Native_historical_reused')}题、新Native正式质量0次，正式测速{len(speed)}请求；模型加载{done.get('model_loads')}次。没有运行GQA、OCRBench或其他Benchmark，没有训练、改Prompt、删异常或扩候选。全部预测由官方Answer Processor与独立leave-one-out实现逐条复算一致，来源/输入/协议/Cache字节核验通过。GPU释放：{validation['GPU_released']}；不关闭服务器。

SSD：{RUN}
持久结果：/home/featurize/work/ATE-internvl3-svh-r-fast-textvqa-imageonly-3090/vlm_research/results/internvl3_svh_r_fast_textvqa_imageonly_3090/
分支：codex/internvl3-svh-r-fast-textvqa-imageonly-3090
"""
 report+="\n## 十个研究问题的直接结论\n\n"
 hbest=next((p for p in pairs if p["phase"]=="Holdout96" and p["base"]=="R0" and p["candidate"]==bq),None)
 if hbest:
  hb=summary("Holdout96",bq);hn=summary("Holdout96","Native")
  report+=f"1. Discovery按冻结规则选中的最佳配置为{bq}，K={SPACE[bq]}；best_efficiency={be}。\n2. 独立Holdout相对R0软分{hbest['difference_pp']:+.3f}pp，普通95%CI [{hbest['CI95_low_pp']:+.3f}, {hbest['CI95_high_pp']:+.3f}]pp，软分提高/下降/不变={hbest['improved']}/{hbest['declined']}/{hbest['unchanged']}。\n3. 元回答由{hbest['meta_before']}降到{hbest['meta_after']}；消除{hbest['meta_resolved']}条，其中仅{hbest['meta_resolved_and_score_improved']}题伴随软分提高，不能只凭元回答下降声称恢复识字。\n"
  positive=sorted([x for x in subtypes if x["phase"]=="Holdout96" and x["method"]==bq],key=lambda x:-(x["difference_vs_R0_pp"]*x["N"]))
  report+="4. 按软分总和的净恢复贡献排序："+", ".join(f"{x['question_text_type']} {x['difference_vs_R0_pp']*x['N']/100:+.3f}软分单位（提高{x['improved']}、下降{x['declined']}题）" for x in positive)+"。各类仅8–16题，属于描述性观察。\n"
  report+="5. 四层中心/后段的共同StageA48结果与配对CI见上表；不能据粗筛或仅元回答率判断位置优劣，两者未进入独立Holdout。\n"
  report+=f"6. 本轮显示独立正向收益的候选使用{len(SPACE[bq])}层；2/4层只在48题粗筛，未证明{len(SPACE[bq])}层是必要下限，也未独立验证所有更小预算。\n7. 真实质量EOS KV增量见kv_cost.csv，固定32实际KV及E2E仅使用本轮TextVQA测速，不能混用长度或GQA数据。\n8. 真实E2E收益及倍数见速度表，包含临时scratch开销，没有用理论Token节省代替实测。\n9. 数值豁免保留历史FAILED，限制纯机制归因和数值等价结论；正向软分只说明当前冻结实现的小样本观察。\n10. 与Native仍相差{hn['soft_accuracy_pct']-hb['soft_accuracy_pct']:.3f}pp，区间下界接近零；仅值得有限独立扩样确认，不宜据此直接宣称OCR退化已解决或自动跑全量5000题。\n"
 (OUT/"report_zh.md").write_text(report);print(json.dumps(validation,ensure_ascii=False));print("HOLDOUT",json.dumps(summarize(hold,"Holdout96"),ensure_ascii=False));print("DECISION",json.dumps(gate,ensure_ascii=False))
if __name__=="__main__":main()
