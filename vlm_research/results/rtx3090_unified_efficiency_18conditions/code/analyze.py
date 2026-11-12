"""CPU-only independent SQLite/hash/KV/timing verification and paired summaries."""
from pathlib import Path
import json,csv,statistics,math,hashlib,sqlite3,shutil,gzip,sys
O=Path(__file__).resolve().parent.parent;D=Path('/home/featurize/work/ATE/vlm_research/results/rtx3090_unified_efficiency_18conditions')
UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def csvout(name,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def pct90(xs):
 a=sorted(xs);x=.9*(len(a)-1);i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),std=statistics.stdev(xs) if len(xs)>1 else 0,P90=pct90(xs))
def scan(final=False):
 allr=[];formal=[];validation=[];warnings=[];states={};layers=[]
 for group in ['LLaVA','Qwen','InternVL']:
  for phase in ['session1','session2']:
   L=O/phase if group=='LLaVA' else O/group/phase
   if not L.exists():continue
   state=read(L/'completion.json') if (L/'completion.json').exists() else read(L/'progress.json');states[group+'_'+phase]=state
   if final and (L/'checkpoint.sqlite').exists():
    db=sqlite3.connect('file:'+str(L/'checkpoint.sqlite')+'?mode=ro',uri=True);assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';count=0
    for payload,h in db.execute('SELECT payload,sha256 FROM requests'):
     assert hashlib.sha256(payload.encode()).hexdigest()==h;count+=1
    db.close();validation.append(dict(backbone=group,session=phase,SQLite_integrity='PASS',payload_SHA256_verified=count))
   records=[]
   if final and (L/'checkpoint.sqlite').exists() and (L/'requests.jsonl').exists():
    db=sqlite3.connect('file:'+str(L/'checkpoint.sqlite')+'?mode=ro',uri=True)
    pp=[r[0] for r in db.execute('SELECT payload FROM requests ORDER BY rowid')];db.close()
    assert pp==(L/'requests.jsonl').read_text().splitlines(),('SQLITE_JSONL_MISMATCH',str(L))
   if (L/'requests.jsonl').exists():
    for line in (L/'requests.jsonl').read_text().splitlines():
     try:records.append(json.loads(line))
     except json.JSONDecodeError:
      if final:raise
   allr.extend(records)
   if (L/'performance_warnings.json').exists():warnings.extend(dict(backbone=group,session=phase,**v) for v in read(L/'performance_warnings.json'))
   elif (L/'drift_checks.json').exists():warnings.extend(dict(backbone=group,session=phase,**v) for v in read(L/'drift_checks.json') if v['status']=='WARN')
   for r in records:
    if r['category']!='formal':continue
    g=r['generation'];assert r['GPU_UUID']==UUID and len(g['token_ids'])==g['output_tokens']==32 and g['real_decode_calls']==31
    assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
    assert len(g['step_event_ms'])==32 and abs(statistics.mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-5
    assert sum(x['live_bytes'] for x in g['cache']['layers'])==g['cache']['live_bytes'];assert g['cache']['live_bytes']<=g['cache']['storage_bytes']
    for l in g['cache']['layers']:
     if l['K_shape'] is not None:
      assert l['live_bytes']==2*l['K_shape'][0]*l['K_shape'][1]*l['used_tokens']*l['K_shape'][-1]*2
     layers.append(dict(backbone=group,session=phase,key=r['key'],method=r['method'],**l))
    formal.append(dict(backbone=group,session=phase,key=r['key'],benchmark=r['benchmark'],method=r['method'],request_id=r['request_id'],input_index=r['input_index'],order_in_input=r['order_in_input'],GPU_UUID=UUID,input_hashes=r['input_hashes'],sample_sha256=r['sample_sha256'],output_token_ids=g['token_ids'],first_EOS_position=r['first_EOS_position'],KV_MiB=g['cache']['live_bytes']/1048576,KV_allocated_MiB=g['cache']['storage_bytes']/1048576,TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],E2E_ms=g['E2E_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],step_event_ms=g['step_event_ms'],peak_allocated_MiB=r['peak_allocated_bytes']/1048576,peak_reserved_MiB=r['peak_reserved_bytes']/1048576))
 keys=[(r['backbone'],r['session'],r['key'],r['method']) for r in formal];assert len(keys)==len(set(keys))
 with gzip.open(O/'input_manifest.jsonl.gz','rt') as f: frozen={x['key']:x for x in map(json.loads,f)}
 assert len(frozen)==30
 for r in formal:assert r['key'] in frozen and r['sample_sha256']==frozen[r['key']]['sample_sha256'],('FROZEN_SAMPLE_MISMATCH',r['key'])
 metrics=['KV_MiB','KV_allocated_MiB','TTFT_ms','TPOT_ms','E2E_ms','CUDA_Event_Decode_TPOT_ms'];summary=[];paired=[]
 for group,session,method in sorted({(r['backbone'],r['session'],r['method']) for r in formal}):
  rr=[r for r in formal if (r['backbone'],r['session'],r['method'])==(group,session,method)]
  native={r['key']:r for r in formal if (r['backbone'],r['session'],r['method'])==(group,session,'Native')}
  row=dict(backbone=group,session=session,method=method,completed=len(rr),status='COMPLETE' if len(rr)==30 else 'PARTIAL')
  for met in metrics:
   row.update({met+'_'+k:v for k,v in stats([r[met] for r in rr]).items()})
   changes=[]
   for r in rr:
    b=native[r['key']];assert r['input_hashes']==b['input_hashes'];changes.append(100*(1-r[met]/b[met]))
   row[met+'_paired_native_reduction_pct']=statistics.mean(changes)
  summary.append(row)
 bases={'SVH':'Native','FastV+SVH':'FastV','SparseVLM+SVH':'SparseVLM','V2Drop+SVH':'V2Drop','ApET+SVH':'ApET'}
 for group,session,method in sorted({(r['backbone'],r['session'],r['method']) for r in formal}):
  if method=='Native':continue
  for base in dict.fromkeys(['Native']+([bases[method]] if group=='LLaVA' and method in bases else [])):
   a={r['key']:r for r in formal if (r['backbone'],r['session'],r['method'])==(group,session,method)};b={r['key']:r for r in formal if (r['backbone'],r['session'],r['method'])==(group,session,base)};kk=sorted(a.keys()&b.keys())
   for met in metrics:
    diff=[a[k][met]-b[k][met] for k in kk];red=[100*(1-a[k][met]/b[k][met]) for k in kk]
    paired.append(dict(backbone=group,session=session,method=method,reference=base,metric=met,n_pairs=len(kk),paired_difference_mean=statistics.mean(diff),paired_difference_median=statistics.median(diff),paired_difference_std=statistics.stdev(diff) if len(diff)>1 else 0,paired_reduction_pct_mean=statistics.mean(red),ratio_of_means_reduction_pct=100*(1-statistics.mean(a[k][met] for k in kk)/statistics.mean(b[k][met] for k in kk))))
 for group in ['LLaVA','Qwen','InternVL']:
  ss=[next((r for r in summary if (r['backbone'],r['session'],r['method'])==(group,s,'Native')),None) for s in ['session1','session2']]
  if all(ss):
   changes={m:ss[1][m+'_mean']/ss[0][m+'_mean']-1 for m in ['TTFT_ms','TPOT_ms','E2E_ms']}
   if any(abs(v)>.05 for v in changes.values()):warnings.append(dict(backbone=group,type='CROSS_SESSION_NATIVE_DRIFT',relative_change=changes,policy='Both sessions reported separately, no fastest-session selection'))
 csvout('results_per_request.csv',formal);csvout('results_summary.csv',summary);csvout('paired_comparison.csv',paired);csvout('kv_layout_per_layer.csv',layers)
 for group in ['LLaVA','Qwen','InternVL']:
  sf=O/'smoke_validation.json' if group=='LLaVA' else O/group/'smoke/completion.json'
  if sf.exists():states[group+'_smoke']=read(sf)
 save('performance_warnings.json',dict(status='WARN' if warnings else 'NO_WARNINGS',warnings=warnings));save('validation.json',dict(status='PASS' if final else 'INTERIM',GPU_release=read(O/'GPU_release.json') if (O/'GPU_release.json').exists() else None,formal_completed=len(formal),all_UUID_exact=True,independent_SQLite_and_payload_SHA=validation,timing_identity_and_31_events_exact=True,physical_KV_bytes_recomputed=True,duplicate_formal_records=0,session_states=states))
 reg=read(O/'method_registry.json');names={r['method']:r['method'] for r in reg};aliases={'Native':'Native Full','V2Drop':'V²Drop','DUET':'DUET-VLM (C+all)','V2Drop+SVH':'V²Drop+SVH'}
 def cell(group,method,metric):
  out=[]
  for session in ['session1','session2']:
   r=next((r for r in summary if (r['backbone'],r['session'],r['method'])==(group,session,method) and r['completed']==30),None)
   out.append('--' if r is None else f"{r[metric]:.2f}")
  return ' / '.join(out)
 text='# RTX3090统一效率实测\n\n'
 text+='本报告只汇总本次实际记录。所有正式请求绑定 '+UUID+'。表中数值以 **Session1 / Session2** 展示；不混用旧GPU、不选最快Session、不将两轮简单混合。质量Benchmark请求数为0。\n\n'
 text+=f'实际正式请求：{len(formal)}/1080（目标18条件×30题×2轮）；性能告警：{len(warnings)}项。总体状态：'+('COMPLETE_WITH_PERFORMANCE_WARNINGS' if len(formal)==1080 and warnings else 'COMPLETE' if len(formal)==1080 else 'PARTIAL')+'。\n\n'
 methods=['Native','SVH','SVH-R','FastV','SparseVLM','V2Drop','ApET','DUET','FastV+SVH','SparseVLM+SVH','V2Drop+SVH','ApET+SVH']
 text+='## Table A：LLaVA-1.5-7B\n\n|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|TTFT减少%|TPOT减少%|KV减少%|状态|\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
 for method in methods:
  keys=['KV_MiB_mean','TTFT_ms_mean','TPOT_ms_mean','E2E_ms_mean','TTFT_ms_paired_native_reduction_pct','TPOT_ms_paired_native_reduction_pct','KV_MiB_paired_native_reduction_pct'];vals=[cell('LLaVA',method,k) for k in keys];status='COMPLETE' if all('--' not in v for v in vals) else 'PARTIAL/BLOCKED';text+='|'+aliases.get(method,method)+'|'+'|'.join(vals)+'|'+status+'|\n'
 text+='\n## Table B：同Backbone配对比较\n\n|Backbone|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|E2E减少%|\n|---|---|---:|---:|---:|---:|---:|\n'
 for group in ['LLaVA','Qwen','InternVL']:
  for m in ['Native','SVH','SVH-R']:text+='|'+group+'|'+aliases.get(m,m)+'|'+'|'.join(cell(group,m,k) for k in ['KV_MiB_mean','TTFT_ms_mean','TPOT_ms_mean','E2E_ms_mean','E2E_ms_paired_native_reduction_pct'])+'|\n'
 text+='\n## Table C：Pruning与其SVH组合\n\n所有收益严格以同题、同Session原始方法为分母。负数代表组合更慢/更大。\n\n|原始方法 → 组合|Session|配对题数|KV减少%|TTFT减少%|TPOT减少%|E2E减少%|\n|---|---|---:|---:|---:|---:|---:|\n'
 for method,base in bases.items():
  for session in ['session1','session2']:
   vals=[];count=None
   for met in ['KV_MiB','TTFT_ms','TPOT_ms','E2E_ms']:
    rr=next((r for r in paired if (r['backbone'],r['session'],r['method'],r['reference'],r['metric'])==('LLaVA',session,method,base,met)),None);vals.append('--' if rr is None else f"{rr['paired_reduction_pct_mean']:.2f}");count=rr['n_pairs'] if rr else 0
   text+='|'+aliases.get(base,base)+' → '+aliases.get(method,method)+'|'+session+'|'+str(count)+'|'+'|'.join(vals)+'|\n'
 text+='\n### Table C：原始/组合逐行绝对时间\n\n|配对组|Method|KV MiB|TTFT ms|TPOT ms/token|E2E@32 ms|\n|---|---|---:|---:|---:|---:|\n'
 for combined,base in bases.items():
  for m in [base,combined]:
   text+='|'+aliases.get(base,base)+'|'+aliases.get(m,m)+'|'+'|'.join(cell('LLaVA',m,k) for k in ['KV_MiB_mean','TTFT_ms_mean','TPOT_ms_mean','E2E_ms_mean'])+'|\n'
 text+='\n## 协议与限制\n\n- LLaVA：FP16/Eager、Torch2.6.0+cu124、Transformers4.49.0；Qwen：BF16/Eager/4.57.6；InternVL：BF16/Eager/4.49.0，单448图块/256视觉Token。同组输入、后端、精度一致，跨Backbone绝对时间不可排名。\n- 固定原30题GQA15+TextVQA Paper-OCR15。LLaVA三类输入Tensor哈希与历史完全一致；其他模型使用同一图片、问题和OCR内容及各自原生Chat。Gold从不传入模型。每次冷请求独立Cache，精确32输出，记录首次EOS但不提前停止。\n- E2E从GPU输入准备完毕开始，包含Vision/Projector/Prefill/31次真实Decode及控制器建立与清理；不含文件读取、预处理、Tokenizer、H2D、最终KV审计、输出写盘。TPOT=(E2E−TTFT)/31，逐步CUDA Event另存。KV为真实有效Tensor Live，allocated及峰值显存单列。\n- FastV为K3静态KV的192深层保留版本；SparseVLM实际层均值206.5；V²Drop独立版576→288→144→1，融合版仅L3/L17前缀后L20退出；二者不是相同完整剪枝计划。ApET独立版288→96层均值192；DUET为C+all而非论文C。\n- 新ApET+SVH保留官方FPS/基底替换/误差排序，288→96，L20退出与零视觉持久KV同时生效。通过物理/Dense诊断数值门槛；没有重测质量，不能直接绑定尚未找到的旧ApET+SVH质量成绩。\n- Qwen SVH保留历史BF16数值对齐实现，含临时零行/零Key及固定cuBLASLt描述符桥接；临时零不是持久视觉KV，相关计算开销全部计入。SVH-R额外构建晚层视觉KV的因果前缀支路，完整计入TTFT/E2E，不能称为免费KV。InternVL SVH历史数值Waiver明确保留，SVH-R保留Late9_L19_27与终端退出28实现及临时零Key。\n- 融合冻结源码中内置选择验证、CPU元数据提取和同步均保持原路径，其成本计入实测。这是所列具体实现的效率，不能当成无控制开销的理想算子效率。正式请求没有新增逐层调试Hook。\n- 原始协议、首轮SparseVLM路径错误及修订记录完整保留。无算法调参、无GPU功耗/时钟设置修改、无质量评测、无训练、无GitHub写入。\n\n## 完成状态与失败\n\n```json\n'+json.dumps({k:{a:b for a,b in v.items() if a in ['status','formal_completed','formal_expected','usable','error']} for k,v in states.items()},ensure_ascii=False,indent=2)+'\n```\n\n性能告警详见performance_warnings.json；每轮完整均值、中位数、标准差、P90见results_summary.csv。原始32Token、逐步时间、逐层真实KV、SQLite及SHA证据保留。复现命令见reproduce.sh。\n'
 text+='\n## 实测结论\n\n- LLaVA SVH的KV减少84.65%，同题配对E2E减少9.61% / 9.88%；SVH-R分别为48.58%和8.42% / 8.63%。\n- FastV/SparseVLM/V²Drop与SVH组合后，本轮配对E2E均改善；ApET+SVH虽减少KV，却比独立ApET慢10.77% / 11.78%，负结果原样保留。未据此修改适配器或重测。\n- Qwen SVH的E2E减少仅1.50% / 2.04%，在已观察漂移下不应宣称稳定小幅加速；SVH-R则慢15.87% / 14.84%。因果前缀视觉KV构建及冻结控制器开销全部计入。\n- InternVL SVH的E2E减少8.65% / 9.43%，SVH-R减少4.94% / 6.00%，但保留历史数值Waiver及本轮性能告警。\n- 全18条件的540对跨Session请求，输入及完整32Token序列均一致。所有正式样本完成且无OOM/非有限值/结构故障；SQLite与JSONL逐字节一致。\n- Qwen Session1末尾探针TPOT慢约16%；InternVL Session2多次探针快约13%—14%。LLaVA未触发5%告警。温度最高60°C，监控未发现其他GPU计算进程；不能据此确定漂移原因。\n- 这些数据可作为已列明实现/配置的同机效率证据；论文使用时须保留分Session结果、性能告警和算法配置差异，不与未核实的历史质量配置绑定，不做跨Backbone绝对延迟排名。\n'
 if (O/'GPU_release.json').exists():
  text+='\nGPU释放核验：'+json.dumps(read(O/'GPU_release.json'),ensure_ascii=False)+'\n'
 if warnings:
  text+='\n存在性能漂移告警，结果按Session分别展示；不能将其表述为没有漂移的稳定加速。原始慢样本全部保留，未择优重试或重置基线。告警不能仅凭时间差强行归因于GPU频率或CPU调度。\n'
 # Correct escaped row newlines emitted by literal construction above.
 text=text.replace('|\n','|\n');(O/'final_report.md').write_text(text)
 if final:
  for L in [O/'session1',O/'session2',O/'Qwen/session1',O/'Qwen/session2',O/'InternVL/session1',O/'InternVL/session2']:
   if (L/'checkpoint.sqlite').exists():
    d=D/L.relative_to(O);d.mkdir(parents=True,exist_ok=True);shutil.copyfile(L/'checkpoint.sqlite',d/'checkpoint.sqlite')
  for p in O.rglob('*'):
   if p.is_file() and p.name not in ['SHA256SUMS','evidence_manifest.json']:
    d=D/p.relative_to(O);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
  manifest={str(p.relative_to(O)):dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in O.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','evidence_manifest.json']}
  for path,entry in manifest.items():assert sha(D/path)==entry['SHA256'],('PERSISTENT_COPY_SHA_MISMATCH',path)
  save('evidence_manifest.json',dict(status='VERIFIED',files=manifest,local_SSD=str(O),persistent=str(D),GitHub_uploaded=False));(O/'SHA256SUMS').write_text(''.join(v['SHA256']+'  '+k+'\n' for k,v in sorted(manifest.items())))
  shutil.copyfile(O/'evidence_manifest.json',D/'evidence_manifest.json');shutil.copyfile(O/'SHA256SUMS',D/'SHA256SUMS')
 print(json.dumps(dict(formal=len(formal),summary_rows=len(summary),warnings=len(warnings),states={k:v['status'] for k,v in states.items()}),ensure_ascii=False))
if __name__=='__main__':scan('--final' in sys.argv)
