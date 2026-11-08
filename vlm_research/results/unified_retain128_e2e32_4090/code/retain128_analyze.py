"""Independent CPU-only validation/statistics. No model or CUDA initialization."""
from pathlib import Path
import csv,gzip,hashlib,itertools,json,math,sqlite3,statistics,subprocess,sys,shutil
from datetime import datetime,timezone
from retain128_prepare import OUT as O,E,R,EXT,METHODS,REQUESTED,LABELS,UUID,read,sha,digest,save,lines,expected_layout,verify_sources
def emit_csv(name,rows,fields=None):
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def cmd(*a):return subprocess.check_output(a,text=True,timeout=30).strip()
def main():
 p=read(O/'protocol.json');state=read(O/'progress.json');smoke=read(O/'smoke_summary.json');status=read(O/'method_status.json');verify_sources()
 assert smoke['no_op_Native_exact'] and smoke['physical_cache_32_layers_checked'] and smoke['GPU_UUID']==UUID
 assert state['phase']=='COMPLETE' and state['status'] in ('PASS','PARTIAL')
 db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro',uri=True);records=[]
 for key,payload,h in db.execute('SELECT key,payload,sha256 FROM requests'):
  r=json.loads(payload);assert digest(r)==h and key==r['key']+'|'+r['method'];records.append(r)
 native_refs=read(O/'historical_token_references.json');inputs={s['key']:s for s in lines(O/'input_manifest.jsonl.gz')};bindings=read(O/'input_verification.json')
 assert bindings['status']=='PASS' and bindings['inputs']==30 and len(inputs)==30
 available=smoke['available'];assert available[0]=='Native' and set(available).issubset(METHODS)
 keys={(s,m) for s in p['keys'] for m in available};assert {(r['key'],r['method']) for r in records}==keys and len(records)==len(keys)
 groups={m:{} for m in available};per=[];layers=[];metrics=['KV_MiB','E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','Actual_Avg_Visual_Tokens']
 for r in records:
  method=r['method'];g=r['generation'];s=inputs[r['key']]
  assert r['GPU_UUID']==UUID and r['input_hashes']==s['input_hashes'] and r['sample_sha256']==s['sample_sha256']
  assert g['output_tokens']==32 and len(g['token_ids'])==32 and g['real_decode_calls']==31 and g['prefill_calls']==g['vision_forward_calls']==1
  assert g['all_logits_finite'] and g['use_cache'] and g['logits_to_keep']==1 and not g['formal_layer_observer_hooks']
  assert math.isclose((g['E2E_ms']-g['TTFT_ms'])/31,g['TPOT_ms'],rel_tol=1e-10)
  assert all(math.isfinite(g[m]) and g[m]>0 for m in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms'])
  kv=g['cache'];assert kv['cached_decode_inputs']==31 and kv['storage_bytes']==kv['live_bytes']
  assert len(kv['layers'])==32;visual=[];live=storage=0
  for i,v in enumerate(kv['layers']):
   k=v['K_shape'];w=v['V_shape'];assert k==w and len(k)==4 and k[0]==1 and k[1]==32 and k[3]==128 and v['dtype']=='torch.float16'
   assert k[2]==v['used_tokens']==v['capacity_tokens']==r['input_tokens']-576+v['visual_tokens']+31
   # Independent arithmetic on the recorded REAL tensor shapes/dtype.
   physical_bytes=math.prod(k)*2*2;assert physical_bytes==v['live_bytes']==v['storage_bytes']
   live+=physical_bytes;storage+=v['storage_bytes'];visual.append(v['visual_tokens'])
   layers.append(dict(key=r['key'],method=method,layer=i,visual_tokens=v['visual_tokens'],used_tokens=v['used_tokens'],capacity_tokens=v['capacity_tokens'],K_shape=json.dumps(k),V_shape=json.dumps(w),dtype=v['dtype'],live_bytes=physical_bytes,storage_bytes=v['storage_bytes'],position_convention=v['position_convention']))
  assert live==kv['live_bytes'] and storage==kv['storage_bytes'] and visual==expected_layout(method)
  if method=='Native':assert g['token_ids']==native_refs[r['key']+'|Native']
  pos=p['keys'].index(r['key']);order=available[pos%len(available):]+available[:pos%len(available)];assert order[r['order_in_input']]==method
  v=dict(key=r['key'],benchmark=r['benchmark'],method=method,GPU_UUID=UUID,session=r['session'],input_tokens=r['input_tokens'],sample_sha256=r['sample_sha256'],input_hashes=json.dumps(r['input_hashes'],sort_keys=True),
   Actual_Avg_Visual_Tokens=sum(visual)/32,KV_MiB=live/2**20,KV_storage_MiB=storage/2**20,E2E_ms=g['E2E_ms'],TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],output_tokens=32,real_decode_calls=31,token_ids=json.dumps(g['token_ids']),begin_unix_time=r['begin_unix_time'],end_unix_time=r['end_unix_time'])
  per.append(v);groups[method][r['key']]=v
 per.sort(key=lambda v:(p['keys'].index(v['key']),available.index(v['method'])))
 emit_csv('efficiency_per_input.csv',per);emit_csv('kv_layout_per_layer.csv',layers)
 summaries=[];means={}
 for m in REQUESTED:
  rs=list(groups.get(m,{}).values());row=dict(method=m,label=LABELS[m],status='PASS_30' if len(rs)==30 else status.get(m,{}).get('status','MISSING'),requests=len(rs),paper_point=p['configuration_audit'][m].get('paper_point'))
  if rs:
   means[m]={k:statistics.mean(r[k] for r in rs) for k in metrics}
   for k in metrics:row[k+'_mean']=means[m][k];row[k+'_median']=statistics.median(r[k] for r in rs)
   for k in ['KV_MiB','E2E_ms','TTFT_ms','TPOT_ms']:row[k+'_improvement_vs_Native_pct']=(1-means[m][k]/means['Native'][k])*100
  summaries.append(row)
 emit_csv('efficiency_summary.csv',summaries)
 paired=[]
 for a,b in itertools.combinations(available,2):
  for metric in metrics[:5]:
   ds=[groups[b][k][metric]-groups[a][k][metric] for k in p['keys']]
   paired.append(dict(reference=a,challenger=b,metric=metric,pairs=30,mean_delta_challenger_minus_reference=statistics.mean(ds),median_delta=statistics.median(ds),improvement_ratio_of_means_pct=(1-means[b][metric]/means[a][metric])*100))
 emit_csv('paired_comparison.csv',paired)
 drift=[];telemetry=[];other=[]
 for seg in sorted((O/'sessions').glob('session_*')):
  assert not (seg/'gpu_conflict.json').exists()
  for q in sorted(seg.glob('drift_after_*.json')):
   d=read(q);assert d['status']=='PASS' and d['GPU_UUID']==UUID;drift.append(d)
  for q in seg.glob('gpu_telemetry.jsonl'):
   for line in q.read_text().splitlines():
    t=json.loads(line);assert t['gpu'].split(',')[0].strip()==UUID;telemetry.append(t)
 assert len(drift)==6
 ranges={}
 for field in ['clocks.current.sm','clocks.current.memory','power.draw','power.limit','temperature.gpu','utilization.gpu']:
  values=[float(dict(zip(t['gpu_fields'],[x.strip() for x in t['gpu'].split(',')]))[field]) for t in telemetry]
  ranges[field]=dict(min=min(values),max=max(values),mean=statistics.mean(values))
 save('gpu_telemetry_summary.json',dict(records=len(telemetry),GPU_UUID=UUID,other_compute_conflicts=0,ranges=ranges))
 save('drift_summary.json',dict(status='PASS',checks=len(drift),max_absolute_relative_change=max(abs(x) for d in drift for x in d['relative_change'].values()),records=drift))
 # Same-instance previous results are context, not concurrent paired timing.
 old={m:[] for m in available}
 for r in lines(E/'formal_requests.jsonl.gz'):
  if r['method'] in old:
   assert r['GPU_UUID']==UUID and r['input_hashes']==inputs[r['key']]['input_hashes']
   g=r['generation'];old[r['method']].append(dict(key=r['key'],KV_MiB=g['cache']['live_bytes']/2**20,E2E_ms=g['E2E_ms'],TPOT_ms=g['TPOT_ms'],Actual_Avg_Visual_Tokens=sum(l['visual_tokens'] for l in g['cache']['layers'])/32))
 oldmeans={m:{k:statistics.mean(r[k] for r in rs) for k in rs[0] if k!='key'} for m,rs in old.items()}
 comparison=[]
 for m in available:
  assert len(old[m])==30
  comparison.append(dict(method=m,old_actual_avg_visual=oldmeans[m]['Actual_Avg_Visual_Tokens'],new_actual_avg_visual=means[m]['Actual_Avg_Visual_Tokens'],old_KV_MiB=oldmeans[m]['KV_MiB'],new_KV_MiB=means[m]['KV_MiB'],KV_reduction_pct=(1-means[m]['KV_MiB']/oldmeans[m]['KV_MiB'])*100,
   old_E2E_ms=oldmeans[m]['E2E_ms'],new_E2E_ms=means[m]['E2E_ms'],raw_E2E_reduction_pct=(1-means[m]['E2E_ms']/oldmeans[m]['E2E_ms'])*100,
   Native_normalized_E2E_reduction_pct=(1-(means[m]['E2E_ms']/means['Native']['E2E_ms'])/(oldmeans[m]['E2E_ms']/oldmeans['Native']['E2E_ms']))*100,
   limitation='Separate sessions; same UUID/input, no paired randomization between budgets; old V2Drop was demo576/288/144/1, not paper192' if m=='V2Drop' else 'Separate sessions; same UUID/input; Native normalization is contextual, not a controlled within-session budget A/B'))
 emit_csv('comparison_with_previous192.csv',comparison)
 with gzip.open(O/'formal_requests.jsonl.gz','wt') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 with gzip.open(O/'smoke_records.jsonl.gz','wt') as f:
  for key,payload,h in db.execute('SELECT key,payload,sha256 FROM smoke'):
   r=json.loads(payload);assert digest(r)==h;f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 with gzip.open(O/'drift_probe_requests.jsonl.gz','wt') as f:
  for key,payload,h in db.execute('SELECT key,payload,sha256 FROM probes'):
   r=json.loads(payload);assert digest(r)==h;f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 db.close()
 # Preserve all executed repository modules in a compact standalone source snapshot.
 env=read(O/'environment.json');sources=p['source_SHA256']
 for path,h in sources.items():
  src=Path(path);assert sha(src)==h
  if src.is_relative_to(R):dest=O/'core_sources'/src.relative_to(R)
  else:dest=O/'core_sources/installed_dependencies'/str(src).lstrip('/')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==h
  else:shutil.copyfile(src,dest)
 release=dict(at=datetime.now(timezone.utc).isoformat(),GPU=cmd('nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'),compute_processes=cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'))
 assert not release['compute_processes'] and release['GPU'].startswith(UUID)
 save('gpu_release.json',release)
 validation=dict(status='PASS_AUDITED_CONDITIONS_PARTIAL_REQUESTED_METHODS' if len(available)<6 else 'PASS',formal_requests=len(records),requested_max=180,formal_inputs_each=30,methods=available,blocked={m:v for m,v in status.items() if v['status']!='PASS'},
  source_integrity='PASS',input_tensor_SHA256='PASS_30',same_GPU_UUID=UUID,physical_tensor_KV_independent_arithmetic='PASS',Native_historical_first32_tokens='EXACT_30',fixed_32_outputs_and_31_true_decode=True)
 validation.update(fixed_32_outputs_and_31_true_decode=True,rotating_method_order=True,drift_checks=6,drift_status='PASS',GPU_released=True,quality_requests=0,OCRBench_requests=0,Windows_backups=0,GitHub_upload='PENDING')
 save('validation.json',validation);save('main_table.json',summaries)
 table=['| Method | Actual Avg Visual Tokens | KV@32 MiB | E2E@32 ms | TTFT ms | TPOT ms |','|---|---:|---:|---:|---:|---:|']
 for row in summaries:
  if row['requests']==30:table.append('| '+row['label']+' | '+' | '.join(f"{row[k+'_mean']:.3f}" for k in ['Actual_Avg_Visual_Tokens','KV_MiB','E2E_ms','TTFT_ms','TPOT_ms'])+' |')
  else:table.append('| '+row['label']+' | — | — | — | — | — |')
 report='\n'.join(['# ATE Retain-128 统一 E2E@32 效率复现','',f'完成 {len(records)}/180 个正式请求：Native 与 {len(available)-1} 个外部方法各30题。FastV 缺少可核验的官方平均128工作点，未运行，不填数值。只运行效率实验，OCRBench/质量/训练请求均为0。','',*table,'',
 '配置与论文对应关系：','',
 '- SparseVLM：官方 retained_tokens=128，top-k [303,110,36]，保留原始回收聚类，实测层布局 576×3、312×4、117×9、39×16，均值145.40625；不能称实际平均128。',
 '- V²Drop：论文 Appendix A 的保留比 [0.28,0.25,0]，进入L3/L17/L22执行；原始 floor/min1 不变，576→161→40→1，均值131。旧“192”复现是演示参数576→288→144→1，并非论文192工作点。',
 '- ApET：官方已发布192→64、L16、FPS/basis10组合，在同一32层骨干的均值128；此组合出现在LLaVA-NeXT脚本，LLaVA1.5脚本公开288→96及96→32。按论文128预算迁移已发布参数，未变更算法；缺少证明它就是LLaVA1.5论文质量表128行的独立配置记录，不能宣称完全对应。',
 '- DUET-VLM：论文附录Table12 的170 dominant（含CLS）+35 contextual、width4，205→102→0，均值128。固定官方推理源码默认所有图后文本参与排序，故明确为C+all；C末Token路径仅为注释，没有改写成C，不使用训练权重。',
 '- FastV：官方固定Commit提供K/R机制和静态KV分支，但没有可唯一确认的平均128 preset。不能把深层保留128当作层平均128，也没有反解K/R凑预算；本项BLOCKED_EVIDENCE。','',
 '同机改善（相对本轮Native，按30题算术均值）：','',
 *[f"- {LABELS[m]}：KV减少{(1-means[m]['KV_MiB']/means['Native']['KV_MiB'])*100:.2f}%，E2E减少{(1-means[m]['E2E_ms']/means['Native']['E2E_ms'])*100:.2f}%。" for m in available if m!='Native'],'',
 '测量及核验：统一FP16/Eager/Batch1/Greedy、use_cache=True、logits_to_keep=1，同一冻结GQA15+TextVQA15输入Tensor SHA256；含Vision/Projector/Prefill/31次Decode与必要清理，排除预处理/H2D/写盘。KV以真实终端K/V形状、dtype、有效长度独立重算，未使用总显存或理论预算替代。每题32输出Token、忽略EOS。',
 f'全程GPU UUID：`{UUID}`。输入轮换方法顺序，统一预热；每5题Native探针检查，共6次全部PASS，最大相对漂移{max(abs(x) for d in drift for x in d["relative_change"].values())*100:.3f}%，未自动重设基线。Native与原已验收32Token结果逐题完全一致；Native/Bridge、V2Drop禁用No-op的Token/首Logit完全一致，真实Hidden/Mask/RoPE/KV与31步Decode Smoke通过。','',
 '与此前约192档比较：见 comparison_with_previous192.csv，保留真实旧均值。虽然GPU UUID和输入相同，两个预算属于不同运行时段，不能把跨时段差值当作严格的配对预算因果效果；同时给出以各轮Native归一化的E2E变化。旧V²Drop参数本就不对应论文192，不能做论文192→128质量效率联合结论。','',
 '用途限制：可作为同一冻结HF模型、Eager、静态真实KV、32输出Token条件下的效率证据；必须显示真实视觉均值和DUET C+all标签。它不是六方法均严格平均128的齐全主表，也不能绑定到未验证的公开质量分数。','',
 '官方来源：',
 '- FastV：https://github.com/pkunlp-icler/FastV/tree/d1659729b5bf1be225e99ee15783deeea80f63b1',
 '- SparseVLM：https://github.com/Gumpest/SparseVLMs/tree/734d9003879689713a132b83297dfad060a5c9af',
 '- V²Drop：https://arxiv.org/html/2509.01552v2 （Appendix A）；源码Commit 2a9431150f8ace3a1d9abcdc96a7d46acca597df。',
 '- ApET：https://arxiv.org/html/2602.19870v1 ；https://github.com/Maqkccx/ApET/tree/70cc01b215f4d74c5661c699919fe3aeda48bed0',
 '- DUET：https://arxiv.org/html/2602.18846v1 （Appendix C/Table12）；https://github.com/AMD-AGI/DUET-VLM/tree/cd69e489f39a2a51e3b278a470dc8dc32c00e0fb','',
 f'持久化结果根目录：`{O}`。独立适配器：`{EXT}/retain128_compatibility/`。所有主要文件绝对路径见absolute_file_paths.txt。检查点与完整Smoke轨迹留工作盘，GitHub发布的实际文件范围与未上传项由github_publication_manifest.json记录。本任务不向Windows传输备份。GPU已释放，无计算进程，不关闭服务器实例。',''])
 (O/'report_zh.md').write_text(report)
 print(json.dumps(dict(status=validation['status'],formal_requests=len(records),table=[{k:r.get(k) for k in ['method','requests','KV_MiB_mean','E2E_ms_mean','TTFT_ms_mean','TPOT_ms_mean']} for r in summaries],drift_max=max(abs(x) for d in drift for x in d['relative_change'].values()),GPU_release=release),ensure_ascii=False))
if __name__=='__main__':main()
