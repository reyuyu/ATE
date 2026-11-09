"""Independent CPU scoring and exploratory paired statistics for the appended boundary."""
import json,csv,shutil,subprocess,os,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from vlm_research.internvl3_visual_writing_l20_l27_3000 import P,O,R,B,OLDCP,NAME,MODEL,PY,ROOT,read,save,sha,e
from vlm_research.internvl3_visual_writing_figure3_3000.analyze import bootstrap
def csvout(name,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for row in rows for k in row))
 with (O/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def csvread(p):
 with p.open() as f:return list(csv.DictReader(f))
def main():
 assert read(O,"generation_complete.json")["requests"]==3000 and read(O,"persistence_complete.json")["passed"]
 assert read(O,"smoke_summary.json")["passed"]
 protocol=read(O,"protocol.json");ph=sha(O/"protocol.json")
 assert sha(P/"protocol.json")==protocol["parent_protocol_sha256"]
 assert sha(P/"per_question.csv")==protocol["parent_per_question_sha256"]
 for p,h in read(O,"source_manifest.json").items():assert sha(p)==h
 samples=read(O,"sample_manifest.json")["samples"];tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
 fn=read(P/"references/qwen/vlm_research/results/qwen3vl_visual_writing_necessity_3000","official_gqa_prediction.json")["function_source"]
 scope={};exec(fn,scope);scorer=scope["gqa_prediction"]
 values=np.empty((3000,3),dtype=np.int8);allrecords=[];newrows=[];hashes={}
 for i,s in enumerate(samples):
  key=s["task"]+"__"+s["question_id"]+".json";r=read(O/"checkpoints",key);g=r["generation"]
  assert r["complete"] and r["index"]==i and r["sample_sha256"]==e.digest(s) and r["protocol_sha256"]==ph
  assert r["GPU_UUID"]==protocol["GPU_UUID"] and r["input_binding"]==e.digest(r["input"])
  assert g["writer_layers"]==list(range(20)) and g["P1_Decode_image_off"] and g["logits_to_keep"]==1
  ids=g["token_ids"];text=tok.decode(ids,skip_special_tokens=True).split("<|im_end|>")[0].strip();pred=scorer(text)
  assert text==g["text"] and pred==g["prediction"] and g["correct"]==int(pred==s["answer"].lower())
  assert g["mask_audit"]["blocked_probability_max"]==0 and g["mask_audit"]["steps"]==len(ids)
  n=len(r["input"]["input_ids"]);assert g["final_cache_lengths"]==[n+len(ids)-1]*28 and g["independent_fresh_cache"]
  old=read(P/"checkpoints",key);assert sha(P/"checkpoints"/key)==r["parent_checkpoint_sha256"]
  assert old["input_binding"]==r["input_binding"] and old["GPU_UUID"]==r["GPU_UUID"]
  base=old["conditions"]["All-Writer"];keep=old["conditions"]["Keep-L0-L16"]
  values[i]=[base["correct"],keep["correct"],g["correct"]]
  assert g["correct_to_wrong"]==int(base["correct"] and not g["correct"])
  assert g["wrong_to_correct"]==int(not base["correct"] and g["correct"])
  hashes[key]=sha(O/"checkpoints"/key)
  row=dict(index=i,task=s["task"],question_id=s["question_id"],image_id=s["image_id"],image_sha256=s["image_sha256"],
   condition=NAME,gold=s["answer"],text=text,prediction=pred,correct=g["correct"],token_ids=json.dumps(ids),
   generated_tokens=len(ids),correct_to_wrong=g["correct_to_wrong"],wrong_to_correct=g["wrong_to_correct"],
   first_token_flip=g["first_token_flip"],eos=g["eos"],truncated=g["truncated"],empty=g["empty"],
   repeated_trigram=g["repeated_trigram"],input_binding=r["input_binding"],GPU_UUID=r["GPU_UUID"],
   protocol_sha256=ph,parent_checkpoint_sha256=r["parent_checkpoint_sha256"])
  newrows.append(row);allrecords.append(g)
 csvout("per_question.csv",newrows)
 image=np.array([s["image_sha256"] for s in samples]);task=np.array([s["task"] for s in samples])
 summary=[];pairs=[];names=["All-Writer","Keep-L0-L16",NAME]
 for subset_no,subset in enumerate(["Overall"]+e.TASKS):
  take=np.arange(3000) if subset=="Overall" else np.flatnonzero(task==subset)
  v=values[take];draws=bootstrap(v,image[take],20261110+subset_no);point=v.mean(0)*100
  for j,c in enumerate(names):
   loss=(draws[:,0]-draws[:,j])*100;ci=np.quantile(loss,[.025,.975])
   summary.append(dict(subset=subset,condition=c,questions=len(take),correct=int(v[:,j].sum()),accuracy_pct=point[j],
    loss_vs_AllWriter_pp=point[0]-point[j],loss_CI95_low=ci[0],loss_CI95_high=ci[1],
    correct_to_wrong=int(((v[:,0]==1)&(v[:,j]==0)).sum()),wrong_to_correct=int(((v[:,0]==0)&(v[:,j]==1)).sum()),
    source="new 3000 generation" if j==2 else "reused frozen same-GPU parent",post_hoc=(j==2)))
  for ref,cand in [(0,2),(1,2)]:
   diff=(draws[:,cand]-draws[:,ref])*100;ci=np.quantile(diff,[.025,.975])
   pairs.append(dict(subset=subset,reference=names[ref],candidate=names[cand],
    candidate_minus_reference_pp=point[cand]-point[ref],CI95_low=ci[0],CI95_high=ci[1],
    bootstrap=20000,image_clusters=len(set(image[take])),post_hoc=True,multiplicity="ordinary unadjusted exploratory 95%"))
  print("EXTENSION_BOOTSTRAP",subset,flush=True)
 csvout("accuracy_summary.csv",summary);csvout("paired_comparisons.csv",pairs)
 shutil.copy2(P/"plot_layerwise.csv",O/"plot_layerwise.csv")
 assert sha(P/"plot_layerwise.csv")==sha(O/"plot_layerwise.csv")
 deep=csvread(P/"plot_deep.csv");row=next(x for x in summary if x["subset"]=="Overall" and x["condition"]==NAME)
 for d in deep:
  if d["model"]=="InternVL3-8B":
   d.update(additional_deep_start=20,additional_restricted_accuracy_pct=row["accuracy_pct"],
    additional_loss_pp=row["loss_vs_AllWriter_pp"],additional_ci95_low=row["loss_CI95_low"],
    additional_ci95_high=row["loss_CI95_high"],additional_post_hoc=True)
 csvout("plot_deep.csv",deep)
 subprocess.run([PY,str(O/"plot_figure3.py")],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),check=True)
 gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"],text=True).strip()
 assert not gpu
 assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()==protocol["GPU_UUID"]
 for src in [R/"pipeline.log",R/"worker.log"]:
  if src.exists():shutil.copy2(src,O/("saved_"+src.name))
 report=["# InternVL3-8B 追加关闭 L20–L27：3000题配对验证","",
  "追加请求在原 L17–L27 结果已知后提出，属于探索性分析；未搜索其他边界。原93,000条记录、协议和原图保持不变。",
  "只新增3000次完整生成，复用同一3090上的All-Writer及Keep-L0-L16。原P−1和全部Decode仍在全部28层禁止访问Image Key；本轮只关闭R在L20–L27的写入权限，没有物理删除。",
  "","| 条件 | 正确/3000 | Accuracy % | 相对All-Writer损失 pp | 普通配对95%CI pp |","|---|---:|---:|---:|---|"]
 for s in summary:
  if s["subset"]=="Overall":
   report.append(f"| {s['condition']} | {s['correct']} | {s['accuracy_pct']:.4f} | {s['loss_vs_AllWriter_pp']:+.4f} | [{s['loss_CI95_low']:+.4f}, {s['loss_CI95_high']:+.4f}] |")
 comp=next(r for r in pairs if r["subset"]=="Overall" and r["reference"]=="Keep-L0-L16")
 report+=["",f"相对旧L17–L27 OFF，新窗口Accuracy差为 {comp['candidate_minus_reference_pp']:+.4f} pp，95%CI [{comp['CI95_low']:+.4f}, {comp['CI95_high']:+.4f}] pp。",
  "普通95%CI基于20,000次Image SHA整簇、同题配对、题数加权Bootstrap；新增比较未作多重校正，不能按确认性检验解读。CI跨0不能证明等价或无损。",
  "","| Task | All-Writer % | L17–L27 OFF % | L20–L27 OFF % |","|---|---:|---:|---:|"]
 for t in e.TASKS:
  ts=[r for r in summary if r["subset"]==t];report.append("| "+t+" | "+" | ".join(f"{r['accuracy_pct']:.2f}" for r in ts)+" |")
 report+=["","最短/典型/最长输入官方Chat与Native/No-op全部有效Logits严格torch.equal；重载后的All-Writer与旧记录全部Token及Logits哈希严格一致。新增全部3000题的输入绑定完全一致，被禁概率为0，Cache/位置/独立评分通过。",
  "新图(a)(b)复用原固定CSV，未修改曲线数据；InternVL(c)保留旧窗口并新增L20–L27，后者标注为post hoc。热图同时标示L17与L20。原LLaVA/Qwen数据不重跑。",
  "Dense权限限制不能证明物理移除视觉Hidden、KV或Attention路径后的行为，不能声称实现加速。","",
  f"结果：{O}",f"父实验：{P}","GitHub未推送；GPU无计算进程，服务器未关闭。",""]
 (O/"report_zh.md").write_text("\n".join(report))
 save(O,"validation.json",dict(passed=True,new_requests=3000,reused_reference_rows=6000,questions=3000,
  image_clusters=2409,same_GPU_UUID=protocol["GPU_UUID"],all_input_bindings_match=True,
  independent_scores_match=True,forbidden_probability_zero=True,all_28_cache_lengths_correct=True,
  checkpoint_SHA256_verified=True,old_protocol_and_results_unchanged=True,post_hoc=True,GPU_released=True))
 save(O,"checkpoint_hashes.json",hashes)
 entries={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob("*"))
  if p.is_file() and p.name not in ("SHA256SUMS","evidence_manifest.json")}
 save(O,"evidence_manifest.json",dict(status="PASS",files=entries,parent_protocol_sha256=protocol["parent_protocol_sha256"],
  model_path=str(MODEL),model_revision=protocol["revision"],source_files=read(O,"source_manifest.json")))
 entries["evidence_manifest.json"]=dict(sha256=sha(O/"evidence_manifest.json"))
 (O/"SHA256SUMS").write_text("".join(v["sha256"]+"  "+n+"\n" for n,v in sorted(entries.items())))
 print("EXTENSION_FINAL_VALIDATION_PASS",json.dumps(row),flush=True)

