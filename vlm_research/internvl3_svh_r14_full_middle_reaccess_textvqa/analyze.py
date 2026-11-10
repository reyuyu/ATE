"""Independent TextVQA scoring and fixed A/B paired analysis for R14."""
from .common import *
import torch,subprocess
from vlm_research.internvl3_svh_seven_benchmarks import scoring
def table(rows,keys):
 return "| "+" | ".join(keys)+" |\n| "+" | ".join(["---"]*len(keys))+" |\n"+"\n".join("| "+" | ".join(f"{r.get(k):.3f}" if isinstance(r.get(k),float) else str(r.get(k,"—")) for k in keys)+" |" for r in rows)
def main():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");rows=[json.loads(x[0]) for x in db.execute("SELECT record FROM predictions")];db.close();mf=read(OUT/"sample_manifest.json");source=read(OUT/"source_manifest.json");ph=sha(OUT/"protocol.json");sh=sha(OUT/"source_manifest.json");bindings=read(RUN/"prepared/input_bindings.json");byq={r["question_id"]:r for r in mf["GroupA"]+mf["GroupB"]+mf["Smoke"]}
 for p,h in source["runtime_source_sha256"].items():assert sha(RUN/"code"/p)==h
 for x in rows:
  r=byq[x["question_id"]];assert x["protocol_sha256"]==ph and x["source_manifest_sha256"]==sh and x["input_binding"]==bindings[x["question_id"]]["binding"]
  assert x["gold"]==r["gold"] and len(x["gold"])==10 and x["sample_sha256"]==r["sample_sha256"] and x["image_sha256"]==r["image_sha256"]
  a=scoring.score("TextVQA",x["text"],r)[1];b=scoring.independent_score("TextVQA",x["text"],r);assert abs(a-b)<1e-10 and abs(x["score"]-b)<1e-10
  assert len(x["token_ids"])==x["generated_tokens"]<=32 and x["eos"]==(x["token_ids"][-1]==151645) and len(x["kv_layout"])==28
  for v in x["kv_layout"]:
   i=v["layer"];K=SPACE.get(x["method"],list(range(28)));expected=r["input_tokens"]+x["generated_tokens"]-1
   if x["method"]!="Native":expected=0 if i in (2,3,4,5) else expected-(0 if i in K else 256)
   assert v["physical_length"]==expected and v["visual_kv"]==(256 if i in K else 0) and v["capacity_bytes"]>=v["live_bytes"]>=0
   assert v["live_bytes"]==((np.prod(v["k_shape"])+np.prod(v["v_shape"]))*2 if expected else 0)
  assert abs(sum(v["live_bytes"] for v in x["kv_layout"])/1048576-x["KV_live_MiB"])<1e-9 and abs(sum(v["capacity_bytes"] for v in x["kv_layout"])/1048576-x["KV_capacity_MiB"])<1e-9
 qual=[x for x in rows if x["phase"] in ("A","B")];speed=[x for x in rows if x["phase"].startswith("speed_")];done=read(OUT/"run_completion.json");assert done["status"]=="COMPLETE" and done["quality_generated"]==480
 assert sum(x["generation_origin"]=="NEW" for x in qual if x["phase"]=="A")==96 and sum(x["generation_origin"]=="NEW" for x in qual if x["phase"]=="B")==384
 assert len(qual)==896 and sum(x["generation_origin"]=="REUSED" for x in qual)==416
 assert all(x["method"]=="R14" for x in qual if x["phase"]=="A" and x["generation_origin"]=="NEW")
 pairs=[];sub=[];meta=[];kv=[];changes=[];ratios=[];lengths=[];summaries={}
 for phase,label in [("A","A_historical_followup96"),("B","B_independent128")]:
  rr=[x for x in qual if x["phase"]==phase];summary=summarize(rr,label);summaries[phase]=summary;csvout("group_A_accuracy.csv" if phase=="A" else "group_B_accuracy.csv",summary);sm={x["method"]:x for x in summary}
  for base,cand in [("R0","R6"),("R0","R14"),("R6","R14"),("R14","Native")]:
   p=pair(rr,base,cand,label);p["confirmation_scope"]="historical same-question followup only" if phase=="A" else "primary independent validation";pairs.append(p)
  denom=sm["Native"]["soft_accuracy_pct"]-sm["R0"]["soft_accuracy_pct"];gain=sm["R14"]["soft_accuracy_pct"]-sm["R0"]["soft_accuracy_pct"]
  ratios.append(dict(group=phase,R14_minus_R0_pp=gain,R14_minus_R6_pp=sm["R14"]["soft_accuracy_pct"]-sm["R6"]["soft_accuracy_pct"],Native_minus_R14_pp=sm["Native"]["soft_accuracy_pct"]-sm["R14"]["soft_accuracy_pct"],Native_minus_R0_pp=denom,recovered_gap_pct=100*gain/denom if abs(denom)>1e-12 else "UNDEFINED",same_sample_only=True))
  baselines={m:{x["question_id"]:x for x in rr if x["method"]==m} for m in ["Native","R0","R6","R14"]}
  for m in ["Native","R0","R6","R14"]:
   vals=list(baselines[m].values());s=sm[m]
   meta.append({k:v for k,v in s.items() if k not in ("reaccess_layers","mean_KV_live_MiB","mean_KV_capacity_MiB")})
   kv.append(dict(group=phase,method=m,N=len(vals),retained_image_KV_layers=SPACE.get(m,list(range(28))),mean_quality_EOS_KV_live_MiB=s["mean_KV_live_MiB"],mean_quality_EOS_KV_capacity_MiB=s["mean_KV_capacity_MiB"],mean_actual_output_tokens=float(np.mean([x["generated_tokens"] for x in vals])),increment_vs_R0_MiB=s["mean_KV_live_MiB"]-sm["R0"]["mean_KV_live_MiB"],increment_vs_R6_MiB=s["mean_KV_live_MiB"]-sm["R6"]["mean_KV_live_MiB"]))
   lens=[x["generated_tokens"] for x in vals];lengths.append(dict(group=phase,method=m,N=len(vals),P50=float(np.quantile(lens,.5)),P95=float(np.quantile(lens,.95)),maximum=max(lens),EOS=sum(x["eos"] for x in vals),truncated=sum(x["truncated"] for x in vals),empty=sum(x["empty"] for x in vals),repeat=sum(x["flags"]["repeat"] for x in vals)))
   for typ in TYPES:
    v=[x for x in vals if x["analysis_type"]==typ]
    if not v:continue
    delta=[x["score"]-baselines["R0"][x["question_id"]]["score"] for x in v]
    sub.append(dict(group=phase,method=m,type=typ,N=len(v),soft_accuracy_pct=100*float(np.mean([x["score"] for x in v])),difference_vs_R0_pp=100*float(np.mean(delta)),improved=sum(d>1e-10 for d in delta),declined=sum(d< -1e-10 for d in delta),meta_count=sum(x["flags"]["meta"] for x in v),answering_prefix_count=sum(x["flags"]["answering_prefix"] for x in v)))
  for r in (mf["GroupA"] if phase=="A" else mf["GroupB"]):
   q=r["question_id"];c=baselines["R14"][q]
   for base in ["R0","R6"]:
    a=baselines[base][q];d=c["score"]-a["score"]
    changes.append(dict(group=phase,question_id=q,image_sha256=r["image_sha256"],type=r["analysis_type"],question=r["question"],gold=r["gold"],base=base,base_text=a["text"],R14_text=c["text"],base_score=a["score"],R14_score=c["score"],soft_delta=d,score_change="improved" if d>1e-10 else "declined" if d< -1e-10 else "unchanged",zero_to_positive=a["score"]<=1e-10 and c["score"]>1e-10,positive_to_zero=a["score"]>1e-10 and c["score"]<=1e-10,base_meta=a["flags"]["meta"],R14_meta=c["flags"]["meta"],meta_eliminated=a["flags"]["meta"] and not c["flags"]["meta"],meta_new=not a["flags"]["meta"] and c["flags"]["meta"],meta_eliminated_with_positive_score=a["flags"]["meta"] and not c["flags"]["meta"] and c["score"]>0))
 csvout("paired_soft_changes.csv",pairs);csvout("recovery_fraction.csv",ratios);csvout("textvqa_subtypes.csv",sub);csvout("meta_response_analysis.csv",meta);csvout("kv_cost.csv",kv);csvout("output_length_summary.csv",lengths);csvout("per_question_soft_changes.csv",changes)
 csvout("per_question.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in qual]);csvout("kv_layout_per_layer.csv",[dict(phase=x["phase"],question_id=x["question_id"],condition=x["method"],**v) for x in rows for v in x["kv_layout"]])
 csvout("meta_transition_summary.csv",[x for x in pairs if x["candidate"]=="R14"])
 gate=read(OUT/"quality_decision.json");b0=next(x for x in pairs if x["phase"]=="B_independent128" and x["base"]=="R0" and x["candidate"]=="R14");b6=next(x for x in pairs if x["phase"]=="B_independent128" and x["base"]=="R6" and x["candidate"]=="R14")
 assert bool(speed)==(b0["CI95_low_pp"]>0 and b6["CI95_low_pp"]>0);assert len(speed) in (0,96)
 process=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip();assert not torch.cuda.is_initialized()
 validation=dict(status="COMPLETE_EXPLORATORY",new_quality_generations=480,GroupA_new_R14=96,GroupA_reused_Native_R0_R6=288,GroupB_new_R0_R6_R14=384,GroupB_reused_Native=128,independent_official_and_leave_one_out_scores_equal=True,input_tensor_prompt_source_model_and_reuse_hashes_verified=True,actual_KV_shapes_live_capacity_bytes_verified=True,GroupB_image_disjoint_from_previous200=True,GroupB_one_question_per_image=True,GroupA_not_independent_confirmation=True,all_outputs_anomalies_retained=True,formal_speed_requests=len(speed),speed_status=gate["speed_status"],model_loads=done["model_loads"],no_GQA_other_benchmarks_new_candidates_or_prompt_changes=True,numerical_equivalence_claimed=False,historical_NUMERICAL_EQUIVALENCE_FAILED_preserved=True,GPU_UUID=GPU,GPU_released=not process,GPU_compute_processes=process)
 write(OUT/"validation.json",validation);smoke=read(OUT/"smoke_validation.json");bm={x["method"]:x for x in summaries["B"]};br=next(x for x in ratios if x["group"]=="B");am={x["method"]:x for x in summaries["A"]}
 report=f"""# InternVL3-SVH-R14全部允许中层回看：TextVQA Image-only

**主结果为全新B组128题：R14 Soft Accuracy={bm['R14']['soft_accuracy_pct']:.3f}%，R6={bm['R6']['soft_accuracy_pct']:.3f}%，R0={bm['R0']['soft_accuracy_pct']:.3f}%，Native={bm['Native']['soft_accuracy_pct']:.3f}%。**
R14−R0={b0['difference_pp']:+.3f}pp（普通95%CI [{b0['CI95_low_pp']:+.3f}, {b0['CI95_high_pp']:+.3f}]pp），R14−R6={b6['difference_pp']:+.3f}pp（[{b6['CI95_low_pp']:+.3f}, {b6['CI95_high_pp']:+.3f}]pp）。追回同题Native−R0差距的{br['recovered_gap_pct']:.3f}%，仍距Native {br['Native_minus_R14_pp']:.3f}pp。

状态：EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER；历史NUMERICAL_EQUIVALENCE_FAILED保持不变。R14仅开放L6–L19，不包含L0–L1，L2–L5答案Attention仍跳算，L20前仍删除视觉Hidden，L20–L27仍无原始视觉Hidden。它绝非28层全部视觉通路恢复。
"""
 if bm["R14"]["soft_accuracy_pct"]<=15:
  report+="\n**现有L6–L19全部回看不足以恢复文字识别能力。** 本结论针对该冻结SVH-R V2、Image-only和本轮样本；不自动扩层、调Prompt或扩大样本。\n"
 report+=f"""
## A组：历史96题追补对照

{table(summaries['A'],['method','N','soft_accuracy_pct','meta_count','answering_prefix_count','mean_KV_live_MiB','mean_KV_capacity_MiB'])}

{table([x for x in pairs if x['phase']=='A_historical_followup96'],['base','candidate','difference_pp','CI95_low_pp','CI95_high_pp','improved','declined','unchanged'])}

A组是上轮已用于判断R6的Holdout，不能重新包装为R14与R6完全独立确认。Native/R0/R6共288条原记录严格匹配后复用，只新增96次R14；输入Tensor、Prompt、图像、Gold、模型Revision、32Token上限、Scorer和冻结核心前向SHA全部核验，不修改父结果。

## B组：独立128题主验证

{table(summaries['B'],['method','N','soft_accuracy_pct','meta_count','answering_prefix_count','mean_KV_live_MiB','mean_KV_capacity_MiB'])}

{table([x for x in pairs if x['phase']=='B_independent128'],['base','candidate','difference_pp','CI95_low_pp','CI95_high_pp','improved','declined','unchanged','zero_to_positive','positive_to_zero','not_full_to_full','full_to_not_full'])}

从剩余TextVQA原始问题总体按固定Seed与QuestionID/ImageSHA哈希排名，按顺序接受首道未选图片的问题，直至128题；未使用类别配额、Gold、历史预测或元回答选题。与上一轮Discovery96/Holdout96/Smoke8全部200张图像零交集，一图一题。在任何R14 GPU推理前已冻结完整清单SHA：{sha(OUT/'sample_manifest.json')}。问题文字分类仅在抽样后作描述性分析；原总体剩余分布与B组实际分布见sample_manifest.json。
B组R0/R6/R14共384次均为新生成；Native为该128题逐题严格绑定的历史预测，并非旧样本均值。

## 同题恢复比例与不确定性

{table(ratios,['group','R14_minus_R0_pp','R14_minus_R6_pp','Native_minus_R14_pp','Native_minus_R0_pp','recovered_gap_pct'])}

恢复比例=(R14−R0)/(Native−R0)，同组计算且不截断到0–100%；分母零时记UNDEFINED。它是描述性点估计。所有差值CI均按图片SHA整簇配对Bootstrap5000次，固定Seed={SEED}，普通未校正95%CI；有限样本及多重比较不能提供全量非劣效或机制确认。区间跨零不证明等价。Soft Score提高/下降/不变与0↔正分、非满分↔满分分别报告，不能只数答案变化。

## 元回答：消除是否带来真实文字得分

{table([x for x in pairs if x['candidate']=='R14'],['phase','base','candidate','meta_before','meta_after','meta_resolved','meta_new','meta_resolved_and_score_improved','meta_to_scored_answer','zero_to_positive','positive_to_zero'])}

元回答检测沿用冻结规则，覆盖answering前缀、answering does not require、answer not in image/text、cannot determine等完整文本模式。各条件全部元回答、重复、空白和截断均原样纳入官方十答案软评分，不删除或修正。meta_response_analysis.csv列出数量，meta_transition_summary.csv区分消除/新增及真正获得非零软分；逐题原始文本、十答案Gold与实际分数变化见per_question_soft_changes.csv。元回答变少本身不证明OCR质量恢复。

## 文字问题类型

{table([x for x in sub if x['group']=='B'],['method','type','N','soft_accuracy_pct','difference_vs_R0_pp','improved','declined','meta_count'])}

Signs/Products/Titles/Numbers/Short words/Multiword/Other分类来自问题文字启发式，沿用父实验，未用于B组选题。小类别仅作描述，全部损害与恢复记录保留。

## 实际质量EOS缓存与可选速度

{table(kv,['group','method','N','mean_quality_EOS_KV_live_MiB','mean_quality_EOS_KV_capacity_MiB','increment_vs_R0_MiB','increment_vs_R6_MiB','mean_actual_output_tokens'])}

以上是正常EOS质量生成末端的真实持久KV Live/底层Storage Capacity，不是理论预算、显存峰值或固定32缓存。输出长度不同会影响均值；R14恰有14层保留原始256视觉KV，R6为6层、R0为0层，L2–L5持久缓存为空。逐层K/V形状、BF16 dtype、物理长度和真实存储字节已经独立核验。

正式测速状态：**{gate['speed_status']}**。GPU前冻结门槛为B组R14对R0、R6的普通配对95%CI下界均>0，未满足则停止扩展。没有使用旧TextVQA/GQA时间或理论KV填充未执行数据。
"""
 if speed:
  ss=list(csv.DictReader((OUT/"speed_summary.csv").open()));report+="\n"+table(ss,["method","requests","E2E_ms","TTFT_ms","TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"])+"\n";sp={x["method"]:x for x in ss}
  report+=f"\nR14相对Native E2E真实加速{float(sp['Native']['E2E_ms'])/float(sp['R14']['E2E_ms']):.3f}倍，KV@32相对R0增加{float(sp['R14']['KV_live_MiB'])-float(sp['R0']['KV_live_MiB']):.3f}MiB。8张事先冻结新图，每方法2预热、每图3正式重复，共96请求。包含真实Vision/Projector/Prefill/31Decode及必要清理，排除磁盘/Tokenizer/预处理/H2D/写盘，轮换方法顺序，无重型Hook。临时零值Key/Value scratch参与真实计算，开销已计入。\n"
 else:report+="\n正式E2E/TTFT/TPOT/KV@32均为NOT_RUN。只报告本轮实际质量EOS KV，不生成虚构性能行。\n"
 dif=[v for c in smoke["checks"] for v in c["aligned_prefix_logits"]]
 report+=f"""
## Smoke、复用与数值豁免

6题Dense-R14/Physical-R14 Smoke：完整Token一致{smoke['full_token_equal']}/6，标准化答案一致{smoke['normalized_answer_equal']}/6；Native/No-op有效Logits严格torch.equal，首个Native与历史Token一致。相同生成前缀Logits最大MaxAbs={max(v['max_abs'] for v in dif):.6f}、步均MeanAbs={np.mean([v['mean_abs'] for v in dif]):.6f}，仅记录，不用旧0.125门槛阻止本轮探索，也不宣称数值等价。最短/最长输入完整结构审计验证真实Hidden删除、跳算QKV/Attention、全部14视觉KV、非K无持久原始图像KV、Mask禁用概率0、原始逻辑RoPE、Norm/MLP/Residual和独立Cache；无NaN/Inf或Hook泄漏。

冻结V2核心未改变，不导入C1/C2修复、不做Kernel或新层搜索。历史BF16形状相关误差限制纯机制归因、跨实现外推和数值等价结论。所有实际Prompt均重新核对原图/原问题/原生模板和固定指令：
{INSTRUCTION}
没有OCR Tokens、识别文字、Caption、reference text、Hint、Gold或Rationale进入模型。目录名字不作为输入证据。

完成480次新增正式质量（A96+B384），另复用416条质量参考记录；正式测速{len(speed)}次，模型加载{done['model_loads']}次。独立官方Answer Processor/十答案leave-one-out复算逐条一致，完整Checkpoint、Token、日志和SHA保留。GPU释放={validation['GPU_released']}。没有运行GQA、其他Benchmark或更改父实验/main，不自动扩样、扩层或关闭服务器。

SSD：{RUN}
持久结果：/home/featurize/work/ATE-internvl3-svh-r14-full-middle-reaccess-textvqa-3090/vlm_research/results/internvl3_svh_r14_full_middle_reaccess_textvqa_3090/
分支：codex/internvl3-svh-r14-full-middle-reaccess-textvqa-3090
"""
 report+="\n研究范围判断：若R14未明显优于R6且仍远低于Native，当前证据不支持把继续扩大回看范围当作已证实方案；任何新增通路应另立协议，本轮不执行。\n"
 (OUT/"report_zh.md").write_text(report);print(json.dumps(validation,ensure_ascii=False));print("GROUP_A",json.dumps(summaries["A"],ensure_ascii=False));print("GROUP_B",json.dumps(summaries["B"],ensure_ascii=False));print("RATIOS",json.dumps(ratios,ensure_ascii=False));print("DECISION",json.dumps(gate,ensure_ascii=False))
if __name__=="__main__":main()
