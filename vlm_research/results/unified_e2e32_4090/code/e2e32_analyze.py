"""CPU-only finalization of immutable unified E2E@32 measurements."""
from pathlib import Path
import csv,json,gzip,sqlite3,statistics,itertools,hashlib,subprocess,shutil,tarfile
from datetime import datetime,timezone
from e2e32_prepare import *

def output_csv(name,rows):
 if not rows:return
 path=OUT/name
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def checkpoint(table):
 db=sqlite3.connect('file:'+str(OUT/'checkpoint.sqlite')+'?mode=ro',uri=True);rows=[]
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM '+table):
  r=json.loads(payload);assert digest(r)==h;rows.append(r)
 db.close();return rows
def get(r,metric):return r['generation']['cache']['live_bytes']/2**20 if metric=='KV_MiB' else r['generation'][metric]
def main():
 protocol=read(OUT/'protocol.json');progress=read(OUT/'progress.json');statuses=read(OUT/'method_status.json');rows=checkpoint('requests');probes=checkpoint('probes')
 keyorder={k:i for i,k in enumerate(protocol['keys'])};rows.sort(key=lambda r:(keyorder[r['key']],METHODS.index(r['method'])))
 metrics=['KV_MiB','E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']
 records={(r['key'],r['method']):r for r in rows};assert len(records)==len(rows)
 observed=sorted({r['GPU_UUID'] for r in rows+probes});assert observed in ([UUID],[])
 assert not any(r['generation']['formal_layer_observer_hooks'] for r in rows)
 assert all(r['generation']['output_tokens']==32 and r['generation']['real_decode_calls']==31 and r['generation']['prefill_calls']==r['generation']['vision_forward_calls']==1 for r in rows)
 assert all(abs(get(r,'TPOT_ms')-(get(r,'E2E_ms')-get(r,'TTFT_ms'))/31)<1e-9 for r in rows)
 groups={m:[r for r in rows if r['method']==m] for m in METHODS};approved=[]
 completed=progress.get('phase')=='COMPLETE' and progress.get('status') in ('PASS','PARTIAL')
 for m,rs in groups.items():
  if len(rs)==30 and [r['key'] for r in rs]==protocol['keys'] and statuses.get(m,{}).get('status')=='PASS' and completed:approved.append(m)
 per=[];layouts=[]
 for r in rows:
  g=r['generation'];base=dict(method=r['method'],method_label=LABELS[r['method']],key=r['key'],benchmark=r['benchmark'],formal_index=r['formal_index'],order_in_input=r['order_in_input'],session=r['session'],GPU_UUID=r['GPU_UUID'],input_tokens=r['input_tokens'],output_tokens=32,real_decode_calls=31,seed=r['seed'],sample_sha256=r['sample_sha256'],input_ids_sha256=r['input_hashes']['input_ids'],pixel_values_sha256=r['input_hashes']['pixel_values'])
  base.update({m:get(r,m) for m in metrics});base.update(KV_storage_MiB=g['cache']['storage_bytes']/2**20,wall_minus_event_ms=get(r,'TPOT_ms')-get(r,'CUDA_Event_Decode_TPOT_ms'),begin_unix_time=r['begin_unix_time'],end_unix_time=r['end_unix_time'],audited_for_main_table=r['method'] in approved);per.append(base)
  for layer in g['cache']['layers']:
   row=dict(method=r['method'],method_label=LABELS[r['method']],key=r['key'],benchmark=r['benchmark'],GPU_UUID=r['GPU_UUID']);row.update(layer)
   # SparseVLM uses compact positions after merging. The raw runtime's generic
   # string label missed this case; actual traced positions/rotary checks are
   # compact and correct. Correct ONLY the CSV annotation, retain raw evidence.
   if r['method']=='SparseVLM':row['position_convention']='compact (merged visual representations); raw string annotation corrected in CPU export'
   for field in ('K_shape','V_shape'):
    if field in row:row[field]=json.dumps(row[field])
   layouts.append(row)
 # Absent physical SVH layers do not carry position/hash metadata. Uniform CSV.
 fields=list(dict.fromkeys(k for row in layouts for k in row))
 layouts=[{k:r.get(k) for k in fields} for r in layouts]
 output_csv('efficiency_per_input.csv',per);output_csv('kv_layout_per_layer.csv',layouts)
 summaries=[];allmeans={}
 for method in METHODS:
  rs=groups[method]
  for split in ('All30','GQA','TextVQA'):
   subset=rs if split=='All30' else [r for r in rs if r['benchmark']==split]
   summary=dict(method=method,method_label=LABELS[method],split=split,n=len(subset),audited=method in approved,status='PASS' if method in approved else '—')
   for metric in metrics:
    values=[get(r,metric) for r in subset]
    summary[metric+'_mean']=statistics.mean(values) if values else None;summary[metric+'_median']=statistics.median(values) if values else None
   if split=='All30':allmeans[method]=summary
   summaries.append(summary)
 native=allmeans['Native']
 for summary in summaries:
  comparison=next(s for s in summaries if s['method']=='Native' and s['split']==summary['split'])
  for metric in ('KV_MiB','E2E_ms','TPOT_ms'):
   n=comparison[metric+'_mean'];v=summary[metric+'_mean'];summary[metric+'_reduction_vs_Native_pct']=100*(1-v/n) if n and v is not None else None
 output_csv('efficiency_summary.csv',summaries)
 pairs=[]
 for left,right in itertools.combinations(METHODS,2):
  keys=[k for k in protocol['keys'] if (k,left) in records and (k,right) in records]
  if not keys:continue
  for metric in metrics:
   delta=[get(records[k,right],metric)-get(records[k,left],metric) for k in keys]
   pairs.append(dict(reference=left,method=right,metric=metric,n=len(keys),audited=left in approved and right in approved,mean_difference_method_minus_reference=statistics.mean(delta),median_difference_method_minus_reference=statistics.median(delta),minimum_difference=min(delta),maximum_difference=max(delta),mean_paired_relative_change_pct=statistics.mean(100*(get(records[k,right],metric)/get(records[k,left],metric)-1) for k in keys)))
 output_csv('paired_comparison.csv',pairs)
 with gzip.open(OUT/'formal_requests.jsonl.gz','wt') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 with gzip.open(OUT/'drift_probe_requests.jsonl.gz','wt') as f:
  for r in probes:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 checks=[read(f) for f in sorted((OUT/'sessions').glob('session_*/drift_after_*.json'))]
 drift=[dict(session=int(f.parent.name.split('_')[-1]),**read(f)) for f in sorted((OUT/'sessions').glob('session_*/drift_after_*.json'))]
 save('drift_summary.json',dict(status='PASS' if len(checks)==6 and all(c['status']=='PASS' for c in checks) else 'INCOMPLETE_OR_DRIFT',checks=drift,probe_requests=len(probes),excluded_from_formal=True))
 telemetry=[];others=[]
 for path in (OUT/'sessions').glob('session_*/gpu_telemetry.jsonl'):
  for line in path.read_text().splitlines():
   item=json.loads(line);names=item.get('gpu_fields',[]);values=item.get('gpu','').split(', ')
   if len(names)==len(values):
    vals=dict(zip(names,values));assert vals['uuid']==UUID;telemetry.append(vals)
   for process in item.get('processes','').splitlines():
    if process:
     pid=int(process.split(',')[0]);session=read(path.parent/'progress.json')
     if pid!=session['pid']:others.append(process)
 ranges={}
 for field in ('clocks.current.sm','clocks.current.memory','power.draw','power.limit','temperature.gpu','utilization.gpu'):
  values=[]
  for t in telemetry:
   try:values.append(float(t[field]))
   except (ValueError,KeyError):pass
  ranges[field]=dict(min=min(values),max=max(values),mean=statistics.mean(values)) if values else None
 save('gpu_telemetry_summary.json',dict(GPU_UUID=UUID,samples=len(telemetry),ranges=ranges,other_compute_processes=others))
 assert not others
 inputmap={s['key']:s for s in lines(OUT/'input_manifest.jsonl.gz')}
 assert all(r['input_hashes']==inputmap[r['key']]['input_hashes'] and r['sample_sha256']==inputmap[r['key']]['sample_sha256'] for r in rows)
 validation=dict(status='PASS' if len(approved)==8 and len(rows)==240 and all(c['status']=='PASS' for c in checks) and len(checks)==6 else 'PARTIAL_OR_BLOCKED',formal_requests=len(rows),audited_methods=approved,rows_per_method={m:len(rs) for m,rs in groups.items()},GPU_UUID=observed,all_input_SHA_equal=True,physical_KV_layer_rows=len(layouts),one_prefill_31_decode=True,quality_requests=0,formal_model_loads=len(list((OUT/'sessions').glob('session_*'))),source_verification='PASS',raw_annotation_correction='SparseVLM position_convention CSV string corrected to compact; recorded positions and numerical RoPE tests unchanged')
 verify_sources()
 # Check every running source against the already-frozen protocol fingerprint.
 for path,h in protocol['frozen_source_SHA256'].items():assert sha(path)==h,path
 for name,h in protocol['runner_source_SHA256'].items():assert sha(OUT/'code'/name)==h,name
 save('validation.json',validation)
 report=['# ATE 八方法统一 KV Cache@32 / E2E@32 效率复现','',f"状态：{validation['status']}；正式请求 {len(rows)}/240；每种通过审计的方法 30 题（GQA15+TextVQA15）。本次未运行任何质量 Benchmark、训练或 OCRBench。",'', '| Method | KV Cache@32 MiB ↓ | E2E@32 ms ↓ | TTFT ms | TPOT ms/token |','|---|---:|---:|---:|---:|']
 for m in METHODS:
  s=allmeans[m];values=[s['KV_MiB_mean'],s['E2E_ms_mean'],s['TTFT_ms_mean'],s['TPOT_ms_mean']]
  report.append('| '+LABELS[m]+' | '+' | '.join(f'{v:.3f}' if m in approved else '—' for v in values)+' |')
 report+=['','以上为同一批 30 题算术均值；中位数及 GQA/TextVQA 分组均值见 efficiency_summary.csv。KV 是真实持久缓存的 live bytes，不包含权重、峰值显存或临时张量。32 个输出只写入 31 个生成 Token 到缓存。','', '| Method | KV 减少 | E2E 减少 | TPOT 减少 |','|---|---:|---:|---:|']
 for m in approved:
  s=allmeans[m];report.append('| '+LABELS[m]+' | '+' | '.join(f"{s[k+'_reduction_vs_Native_pct']:.2f}%" for k in ('KV_MiB','E2E_ms','TPOT_ms'))+' |')
 report+=['',f"本轮 Native TPOT 为 {native['TPOT_ms_mean']:.3f} ms/token，仍接近59。FastV E2E 比 Native 慢 {-allmeans['FastV']['E2E_ms_reduction_vs_Native_pct']:.2f}%，SVH/SVH-R 分别比 Native 快 {allmeans['SVH']['E2E_ms_reduction_vs_Native_pct']:.2f}% / {allmeans['SVH-R']['E2E_ms_reduction_vs_Native_pct']:.2f}%。外部方法之间约百分之几以内的小差异不宜过度解读；本次是单轮同题配对计时，不声明统计显著性。"]
 report+=['','## 可比性与漂移','',f'- GPU UUID 始终为 `{UUID}`；Driver 610.57.04。正式阶段一个进程加载一次模型，统一外层 Runtime、相同输入 SHA、32-token 生命周期与同步边界；原始逐请求记录带 UUID。',
 '- FP16 / Eager / Batch1 / Greedy / use_cache=True / logits_to_keep=1；Torch 2.6.0+cu124、Transformers 4.49.0，实际导入路径和源码 SHA 见 environment.json。SparseVLM 沿用官方手写 Eager FP16 softmax，并未换成融合 SDPA/Flash。',
 '- 所有方法均计入 Vision、Projector、Prefill、31 次 Decode、必要的选择/聚类/剪枝与缓存管理。图片读取、Tokenizer、预处理、输入 H2D、模型加载、最终 KV 审计与日志 I/O 在计时之外。TextVQA 复用原30题清单中的冻结OCR上下文视图；本次不调用外部OCR，也不是重新选择500题Image-only清单。',
 '- 每方法两个相同 Smoke/预热输入；按冻结题序交错方法并轮换顺序。Native/No-op Token 与首 Logits 完全一致；独立紧凑位置兼容循环与 stock Native 32-token/首 Logits 完全一致。实际 Hidden、Mask、RoPE、QKV、每层缓存及至少 8 个真实 Decode 步均通过 Smoke。',
 f"- Native 漂移检查 {len(checks)}/6，状态：{read(OUT/'drift_summary.json')['status']}；每 5 题使用同两题探针，对比本次三轮暖机基线，5% 为暂停门槛。探针共 {len(probes)} 请求，均不计正式结果。"]
 if checks:
  for metric in ('E2E_ms','TPOT_ms'):
   ds=[c['relative_change'][metric]*100 for c in checks];report.append(f'- Native {metric} 相对暖机基线变化范围 {min(ds):+.2f}% 至 {max(ds):+.2f}%。')
 report.append('- GPU 监控范围：'+json.dumps(ranges,ensure_ascii=False)+'；其他计算进程：'+str(others)+'。')
 report+=['','## 官方工作点与限制','',
 '- Native、FastV、V²Drop、SVH、SVH-R 直接复用此前冻结实现；各方法独立控制器的必要开销保留。三种新方法使用同一组冻结 HF 权重与模块，兼容层单独保存，官方源文件未改。',
 '- FastV：固定 d1659729b5bf1be225e99ee15783deeea80f63b1，K3/R2⁄3 静态 KV，L0–L2 保留 576，进入 L3 后 192。属于当前已验证静态 KV 工作点，不能直接绑定公开论文其他 K/缓存分支的质量分数。',
 '- V²Drop：固定官方 2a9431150f8ace3a1d9abcdc96a7d46acca5597df，保持已有 L3/L17/L22 的 576→288→144→1 配置。与论文 192-token 质量工作点不同。',
 '- SparseVLM：固定 ICML v1.5 734d9003879689713a132b83297dfad060a5c9af；main 已转为 SparseVLM+，本次未使用。名义192预算 [300,200,110] 加入官方回收聚类后，实际视觉 KV 分布为 576×3层、309×4层、204×9层、113×16层，平均206.5；没有人为更改为恒定192。官方旧类 SDPA/flash_attn 接线和废弃接口由独立 HF 桥接绕过，官方手写 Attention 函数、评分与剪枝/合并 AST 原样执行。不能把本次效率无条件配到论文其他版本或精确192-token质量数值。',
 '- ApET：固定 70cc01b215f4d74c5661c699919fe3aeda48bed0；官方288个视觉表示进入LLM、L16起96个，平均192；FPS basis=10，官方重建误差排序、基底替换、合并和位置重编号原样执行。匹配公开192预算的超参数类型；本次未验证论文质量分数。Vision桥接读取相同hidden_states[-2]，不保留或stack原版encode_images完全丢弃的辅助CLIP Attention输出；此项是移除未使用的导出，不改变算法输入或排序/合并。',
 '- DUET-VLM：固定 cd69e489f39a2a51e3b278a470dc8dc32c00e0fb；dominant300（含CLS）+contextual7，cluster_width4，307→153→0，进入L16/L24剪枝，层平均191.75。当前官方函数使用全部图后文本 Token 排名，对应 C+all；论文主表 C 使用最后文本 Token，二者不可冒充同一质量工作点。本次没有使用训练后的 DUET 权重。',
 '- llava-SVH：视觉 Hidden 在进入L20前移除，答案阶段不持久保留原始视觉KV；SVH-R 在进入L25前移除Hidden，只于L11–L24持久保留视觉KV。两者L2–L5物理跳过答案位置Attention，普通Prefill其他位置仍执行，不宣称整层Prefill全部跳算。冻结源码未改。',
 '- Native/FastV/V²Drop 使用原始位置或其子集；ApET/DUET/SparseVLM 按官方实现压缩后重编号。真实逐层 KV 长度与视觉表示数量见 kv_layout_per_layer.csv。',
 '- CSV 导出时修正了原始运行器对 SparseVLM 位置规则的通用字符串注释：应为 compact。原始记录、实际位置、数值 RoPE 检查及计时未作改动。','',
 '结论：本次通过完整审计的行可以作为同机、同输入、固定32-token条件下的效率比较。论文主表须注明上述具体版本/预算/缓存分支及DUET C+all变体；不能直接与不匹配的公开质量分数拼成八方法联合质量–效率表。','',
 '官方参考：[SparseVLM ICML](https://github.com/Gumpest/SparseVLMs/tree/v1.5)、[ApET](https://github.com/Maqkccx/ApET)、[DUET](https://github.com/AMD-AGI/DUET-VLM)、[DUET论文配置与变体](https://arxiv.org/html/2602.18846v1)。','',
 '## 路径','',f'- 结果：`{OUT}`',f'- 完整官方源码与兼容层：`{EXT}`',f'- 模型：`{M}`；Revision `{protocol["revision"]}`。',
 '- protocol.json、environment.json、smoke_summary.json、input_verification.json、efficiency_per_input.csv、efficiency_summary.csv、paired_comparison.csv、kv_layout_per_layer.csv、formal_requests.jsonl.gz、checkpoint.sqlite、drift_summary.json、gpu_telemetry_summary.json、validation.json、evidence_manifest.json。','',
 '本次没有修改冻结算法、GPU设置或远端GitHub。完成后释放GPU；释放记录另见gpu_release.json。']
 (OUT/'report_zh.md').write_text('\n'.join(report)+'\n')
 save('main_table.json',dict(rows=[dict(method=m,label=LABELS[m],audited=m in approved,**{k:allmeans[m][k] for k in allmeans[m] if k.endswith('_mean') or k.endswith('_pct')}) for m in METHODS],validation=validation))
 # Keep every imported frozen source and all independent bridges with the result.
 for path,h in protocol['frozen_source_SHA256'].items():
  src=Path(path);dest=OUT/'code/frozen_sources'/src.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==h
  else:shutil.copyfile(src,dest)
 evidence={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('evidence_manifest.json','run.lock') and not p.name.endswith(('-shm','-wal')) and '__pycache__' not in p.parts}
 save('evidence_manifest.json',dict(at=datetime.now(timezone.utc).isoformat(),files=evidence,formal_requests=len(rows),GPU_UUID=UUID))
 print(json.dumps(read(OUT/'main_table.json'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
