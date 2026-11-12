"""CPU-only audit and independent session summaries; never start GPU inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sqlite3,hashlib,statistics,math,subprocess,ast
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
O=Path(__file__).resolve().parents[1];R=O.parents[2];U=R/'vlm_research/results/unified_e2e32_4090'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(a):return statistics.mean(a) if a else None
def fmt(v,n=2):return '—' if v is None else f'{v:.{n}f}'
def csvfile(n,a):
 if not a:return
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def main():
 p=read(O/'protocol.json');refs=read(O/'historical_native_tokens.json');assert digest(refs)==p['historical_native_token_SHA256']
 for path,h in p['sources'].items():assert sha(path)==h,path
 for name,h in p['entry_sources_SHA256'].items():assert sha(O/'code'/name)==h,name
 assert sha(p['model_path']+'/.ate_revision.json')==p['model_receipt_sha256']
 db=sqlite3.connect('file:'+str(O/'checkpoint.sqlite')+'?mode=ro',uri=True);data={}
 for t in ['formal','smoke','warmup','probes','failures']:
  rows=[]
  for key,payload,h in db.execute('select key,payload,sha256 from '+t+' order by rowid'):
   r=json.loads(payload);assert digest(r)==h;r['ledger_key']=key;rows.append(r)
  data[t]=rows
  with (O/(t+'_requests.jsonl')).open('w') as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 db.close();formal=data['formal'];keys={(r['session'],r['key']) for r in formal};assert len(keys)==len(formal)
 bindings={s['key']:s for s in read(O/'input_verification.json')['records']}
 for t in ['formal','smoke','warmup','probes']:
  for r in data[t]:
   g=r['generation'];assert r['protocol_SHA256']==digest(p) and r['GPU_UUID']==p['GPU_UUID']
   assert r['input_hashes']==bindings[r['key']]['input_hashes'] and r['sample_SHA256']==bindings[r['key']]['sample_sha256']
   assert r['runtime_SHA256']==p['sources'][str(U/'code/e2e32_runtime.py')] and r['entry_SHA256']==p['entry_sources_SHA256']
   assert g['method']=='Native' and g['token_ids']==refs[r['key']] and g['output_tokens']==32 and g['real_decode_calls']==31
   assert g['prefill_calls']==g['vision_forward_calls']==1 and g['all_logits_finite']
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6 and len(g['step_event_ms'])==32
   assert abs(sum(g['step_event_ms'][1:])/31-g['CUDA_Event_Decode_TPOT_ms'])<1e-8
   assert g['cache']['cached_decode_inputs']==31 and len(g['cache']['layers'])==32 and g['cache']['live_bytes']==g['cache']['storage_bytes']
   if t=='formal':assert r['role']=='formal' and g['smoke']==g['formal_layer_observer_hooks']==False
 flat=[]
 for r in formal:
  g=r['generation'];flat.append(dict(Session=r['session'],key=r['key'],benchmark=r['benchmark'],formal_index=r['formal_index'],GPU_UUID=r['GPU_UUID'],E2E_ms=g['E2E_ms'],TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],wall_minus_event_TPOT_ms=g['TPOT_ms']-g['CUDA_Event_Decode_TPOT_ms'],output_tokens=32,real_decode_calls=31,historical_32_tokens_exact=r['historical_32_tokens_exact'],sample_SHA256=r['sample_SHA256'],input_ids_SHA256=r['input_hashes']['input_ids'],attention_mask_SHA256=r['input_hashes']['attention_mask'],pixel_values_SHA256=r['input_hashes']['pixel_values'],runtime_SHA256=r['runtime_SHA256'],timestamp=r['at'],token_ids=json.dumps(g['token_ids'])))
 csvfile('native_per_input.csv',flat);summaries=[];environments=[];smokes=[];drifts=[];telemetry=[];states=[]
 for index in range(1,4):
  seg=O/'sessions'/('session_%02d'%index);state=read(seg/'progress.json') if (seg/'progress.json').exists() else dict(status='NOT_STARTED',session=index);states.append(state)
  rows=[r for r in formal if r['session']==index];gs=[r['generation'] for r in rows];e=[g['E2E_ms'] for g in gs]
  assert [r['key'] for r in rows]==p['keys'][:len(rows)]
  d=read(seg/'drift_summary.json') if (seg/'drift_summary.json').exists() else dict(status='NOT_COMPLETED',checks=[]);drifts.append(dict(session=index,**d))
  summary=dict(Session=index,GPU_UUID=p['GPU_UUID'],status=state['status'],formal_requests=len(rows),E2E_mean_ms=mean(e),E2E_median_ms=statistics.median(e) if e else None,E2E_sample_std_ms=statistics.stdev(e) if len(e)>1 else None,TTFT_mean_ms=mean([g['TTFT_ms'] for g in gs]),TPOT_mean_ms=mean([g['TPOT_ms'] for g in gs]),CUDA_Event_Decode_TPOT_mean_ms=mean([g['CUDA_Event_Decode_TPOT_ms'] for g in gs]),max_abs_probe_E2E_drift_pct=max([abs(c['relative_change']['E2E_ms'])*100 for c in d['checks']],default=None),max_abs_probe_TPOT_drift_pct=max([abs(c['relative_change']['TPOT_ms'])*100 for c in d['checks']],default=None),Drift=d['status'])
  summaries.append(summary)
  if (seg/'environment.json').exists():environments.append(read(seg/'environment.json'))
  if (seg/'smoke_validation.json').exists():smokes.append(dict(session=index,**read(seg/'smoke_validation.json')))
  if (seg/'telemetry.jsonl').exists():
   for line in (seg/'telemetry.jsonl').read_text().splitlines():telemetry.append(dict(session=index,**json.loads(line)))
 csvfile('native_session_summary.csv',summaries);save('environment.json',dict(sessions=environments));save('smoke_validation.json',dict(status='PASS' if len(smokes)==3 and all(s['status']=='PASS' for s in smokes) else 'INCOMPLETE',sessions=smokes))
 gpu_summaries=[]
 for index in range(1,4):
  fr=[r for r in formal if r['session']==index];lo=min([r['begin_unix_time'] for r in fr],default=0);hi=max([r['end_unix_time'] for r in fr],default=0);parsed=[]
  for t in telemetry:
   if t['session']!=index or 'error' in t:continue
   ts=datetime.fromisoformat(t['at']).timestamp()
   if lo<=ts<=hi:parsed.append(dict(zip(t['query_fields'].split(','),[v.strip() for v in t['GPU'].split(',')])))
  g=dict(session=index,formal_phase_samples=len(parsed),pstates=dict(Counter(t['pstate'] for t in parsed)),throttle_masks=dict(Counter(t['clocks_throttle_reasons.active'] for t in parsed)))
  for key in ['temperature.gpu','power.draw','power.limit','clocks.current.sm','clocks.current.memory']:
   values=[float(t[key]) for t in parsed];g[key]=dict(min=min(values) if values else None,mean=mean(values),max=max(values) if values else None)
  gpu_summaries.append(g)
 save('drift_summary.json',dict(sessions=drifts,definition=p['drift'],all_probes_excluded_from_formal=True));save('gpu_telemetry.json',dict(formal_phase_summaries=gpu_summaries,records=telemetry))
 complete=len(formal)==90 and all(s['status']=='PASS' and s['formal_requests']==30 for s in summaries)
 means=[s['E2E_mean_ms'] for s in summaries] if complete else [];median=statistics.median(means) if means else None
 spread=(max(means)-min(means)) if means else None;maxdev=max(abs(v/median-1) for v in means) if means else None
 conflicts=[t for t in telemetry if t.get('other_processes')];errors=[t for t in telemetry if 'error' in t]
 consistent=all(r['historical_32_tokens_exact'] for r in formal) and bool(smokes) and all(s['status']=='PASS' for s in smokes)
 stable=complete and all(d['status']=='PASS' for d in drifts) and not conflicts
 recommended=consistent and stable and maxdev is not None and maxdev<=.02
 validation=dict(status='PASS' if recommended else 'INCOMPLETE_OR_UNSTABLE',formal_completed=len(formal),formal_expected=90,complete_three_sessions=complete,input_verification=read(O/'input_verification.json')['status'],historical_32_token_consistency=consistent,within_session_drift_gates_passed=stable,session_E2E_means_ms=means,median_of_three_E2E_means_ms=median,session_means_min_ms=min(means) if means else None,session_means_max_ms=max(means) if means else None,session_means_range_ms=spread,session_means_range_pct_of_median=spread/median*100 if means else None,max_session_mean_deviation_from_median_pct=maxdev*100 if maxdev is not None else None,recommended_as_3090_same_machine_reference=recommended,recommendation_rule='All90 exact historical inputs/tokens, all3 sessions and drift gates pass, no competing GPU process, and all3 session means within2% of their median. Never select a subset or retry for favorable timing.',other_GPU_process_events=len(conflicts),telemetry_query_errors=len(errors),runtime_identical_to_frozen_5ee2=True,native_only=True,natural_EOS_or_quality_requests=0,analysis_SHA256=sha(__file__),GPU_used_for_analysis=False)
 save('validation.json',validation)
 differences=[r['generation']['TPOT_ms']-r['generation']['CUDA_Event_Decode_TPOT_ms'] for r in formal]
 event_audit=dict(formal_requests=len(formal),E2E_equals_TTFT_plus_31_TPOT=True,CUDA_Event_decode_sum_equals_31_event_TPOT=True,mean_wall_minus_event_TPOT_ms=mean(differences),max_absolute_wall_minus_event_TPOT_ms=max([abs(v) for v in differences],default=None))
 save('timing_consistency.json',event_audit)
 # Read both historical measurements without executing either experiment.
 def csvrows(path):
  with Path(path).open(newline='') as f:return list(csv.DictReader(f))
 hist=[r for r in csvrows(U/'efficiency_per_input.csv') if r['method']=='Native'];assert len(hist)==30
 old192=dict(label='4090 八方法统一192档',E2E=mean([float(r['E2E_ms']) for r in hist]),TPOT=mean([float(r['TPOT_ms']) for r in hist]))
 V=R/'vlm_research/results/unified_retain128_e2e32_4090';h128=next(r for r in csvrows(V/'efficiency_summary.csv') if r['method']=='Native')
 old128=dict(label='4090 Retain-128实验 Native',E2E=float(h128['E2E_ms_mean']),TPOT=float(h128['TPOT_ms_mean']))
 def function_ast(path):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='generate'),include_attributes=False)
 ast_equal=function_ast(U/'code/e2e32_runtime.py')==function_ast(V/'code/retain128_runtime.py');assert ast_equal
 historical=dict(rows=[old192,old128],new_Native_runtime_SHA256=sha(U/'code/e2e32_runtime.py'),retain128_runtime_SHA256=sha(V/'code/retain128_runtime.py'),generate_function_AST_exact_between_4090_runs=ast_equal,note='Current experiment calls the exact5ee2 Native loop. Retain128 overall runtime has different imports and offline/method-specific audit branches, but generate function AST is exact. Only existing Native results are shown, no retain128 method is executed.')
 save('historical_comparison.json',historical)
 report=['# RTX 3090 Native Full E2E@32 基线复测','',f'状态：**{validation["status"]}**；正式请求 **{len(formal)}/90**，三轮各30题；预热、Smoke、基线与校准均独立保留且不计入正式统计。','', '## 一致性与执行路径','',f'GPU UUID `{p["GPU_UUID"]}`。模型 `{p["model"]}` / Revision `{p["revision"]}`。FP16、Eager、batch1、greedy、use_cache=True、logits_to_keep=1；PyTorch2.6.0+cu124、Transformers4.49.0。每次调用原冻结 `e2e32_runtime.generate(a,layout,Native)`，没有重写模型、Attention或计时循环；固定32个输出，1次Prefill+31次真实Decode，忽略EOS。', '', '30个原始ID、顺序、图片SHA、完整Prompt/OCR、input_ids/attention_mask/pixel_values三个Tensor的SHA与5ee2原件完全一致。GQA此处沿用原效率实验的Paper预处理与Prompt，不混用自然EOS任务的另一套输入。只生成Native Full，No-op仅使用冻结identity_attention包装Native并验证首步Logits及全部32Token精确一致；没有运行任何剪枝/退出方法或质量评测。', '', '历史32Token在每次实际请求中逐项核对；完整原件在 frozen_reference/，新图片路径仅保存在 input_path_mapping.json，未改旧Manifest。模型完整SHA复用本实例已通过的下载流LFS验收，重新核对固定Revision、凭据文件SHA与17个文件尺寸；未重复扫描14GB权重。', '', '## 三个独立Session','',table(['Session','GPU UUID','n','E2E Mean ms','TTFT Mean ms','TPOT Mean ms','Event Decode TPOT ms','Drift'],[[s['Session'],s['GPU_UUID'],s['formal_requests'],fmt(s['E2E_mean_ms']),fmt(s['TTFT_mean_ms']),fmt(s['TPOT_mean_ms']),fmt(s['CUDA_Event_Decode_TPOT_mean_ms']),s['Drift']] for s in summaries]),'',table(['Session','E2E Median ms','E2E样本标准差ms','最大探针E2E偏移','最大探针TPOT偏移','状态'],[[s['Session'],fmt(s['E2E_median_ms']),fmt(s['E2E_sample_std_ms']),fmt(s['max_abs_probe_E2E_drift_pct'])+'%',fmt(s['max_abs_probe_TPOT_drift_pct'])+'%',s['status']] for s in summaries]),'',f'三轮E2E均值的中位数：**{fmt(median)}ms**；轮均值范围：{fmt(validation["session_means_min_ms"])}–{fmt(validation["session_means_max_ms"])}ms，极差{fmt(spread)}ms（中位数的{fmt(validation["session_means_range_pct_of_median"])}%）。没有把90次请求直接混成一个基线数。','', '## 计时与稳定性','',p['timing']+'。'+p['included']+'。排除：'+p['excluded']+'。', '', '每个Session独立进程、模型仅加载一次，所有输入CPU预处理/H2D在计时外。Native至少预热180秒，完整双探针循环保留；之后6轮双探针建立本轮基线。每5题执行同样的两题探针，第30题后的探针兼作轮后检查。每项E2E/TPOT独立统计连续超2%窗口数；任一指标连续2窗超限即暂停，不重设本轮基线、不挑选新轮替代。', '', '逐请求严格核对 E2E=TTFT+31×TPOT；CUDA Events独立计量31次Decode。标准差使用样本标准差（ddof=1）。GPU每约2秒记录UUID、P-state、SM/Memory Clock、温度、功耗/限制、各Throttle理由、其他计算进程，同时保存CPU负载与压力；完整CPU线程/亲和性、CUDA与依赖/安装源码SHA见各Session环境记录。未修改GPU时钟、功耗或全局驱动；没有启用FlashAttention/SDPA/compile。','',f'其他GPU进程事件：{len(conflicts)}；监控查询异常：{len(errors)}。每轮详细探针均值、相对偏移和暂停判断见 drift_summary.json；原始探针见 probes_requests.jsonl。','', '## 与4090历史数据并列','',table(['硬件/实验','Native E2E@32 ms','Native TPOT ms'],[[h['label'],fmt(h['E2E']),fmt(h['TPOT'])] for h in [old192,old128]]+[[f'3090 Session{s["Session"]}',fmt(s['E2E_mean_ms']),fmt(s['TPOT_mean_ms'])] for s in summaries]),'', '本轮Native计时函数与5ee2正式源码逐字节一致。Retain128的整体运行器另有文件，这里只读取其已有Native统计，未执行其任何方法。3090与4090硬件不同，上表用于核对量级和波动，不能将绝对延迟比称为算法Speedup，不能替换4090原主表。','', '## 验收结论','',f'输入一致性：{validation["input_verification"]}；三轮Native/No-op与历史32Token一致性：{consistent}；完整三轮及轮内漂移门禁：{stable}；三轮均值相对其中位数的最大偏离：{fmt(validation["max_session_mean_deviation_from_median_pct"])}%。', '', '**适合作为后续3090同机效率实验的Native参考。**' if recommended else '**尚不能认定为通过验收的3090 Native参考。** 不补造缺失Session，不剔除异常请求，不用4090历史数值替代。','', '推荐标准同时考虑完整90题、历史Token/输入一致、轮内门禁和三轮均值相对中位数不超过2%的重复性。即使通过，后续八方法实验仍需同机同时期校准，不应无条件使用跨轮绝对延迟计算Speedup。','', '## 持久化与异常','',f'目录：`{O}`。完整CSV、JSONL、SQLite不可变台账、每轮日志/环境/Smoke/探针/GPU遥测、参考源码与SHA均保留。没有修改旧4090结果或main；未Push/PR；无Windows结果备份。']
 for s in states:
  if s.get('failure'):report+=['',f'Session{s["session"]}保留异常：`{s["failure"]}`。']
 if not complete:report+=['','一致性通过仅指已执行的请求和Smoke；NOT_STARTED的Session没有产生结果，不能解释为已经完成其一致性验收。Session1如被暂停，不用新的有利轮次替换，也不把模型重载后的结果拼成原来的一次加载Session。']
 if (O/'diagnosis.json').exists():
  dg=read(O/'diagnosis.json');report+=['','## 暂停后的只读排查','', '未启动额外GPU请求。六轮基线双探针E2E均值为：'+str([round(x,2) for x in dg['baseline_round_E2E_means_ms']])+'ms；峰峰变化相对基线均值为'+fmt(dg['baseline_round_max_minus_min_fraction']*100)+'%。这表明已执行180秒以上预热并不等于本机已达到2%以内的稳定状态。','', '同题同Token的校准窗口从2567.11ms基线降至第10题后2288.14ms、第15题后2273.50ms；对应最终E2E偏移−11.44%、TPOT偏移−12.50%。CUDA Event Decode TPOT同步变化，不能用输出长度差异解释。未观察到其他GPU计算进程；GPU功耗限制未改动，Memory Clock记录保持9501MHz。SM Clock变化、主机调度与请求间I/O占空比仍可能影响结果，现有证据不足以唯一归因。', '', '遥测实际间隔中位数'+fmt(dg['telemetry_gap_median_s'])+'秒，最大'+fmt(dg['telemetry_gap_max_s'])+'秒；超过5秒共'+str(dg['telemetry_gaps_over5s'])+'次，因此不能把缺失时段描述成连续覆盖。原始同时间探针/遥测、CPU线程配置和压力记录见diagnosis.json。正式请求仅完成前15道GQA；TextVQA正式15题未执行（TextVQA固定探针和Smoke已执行），该部分均值不能代表原30题均值。']
 report+=['','逐请求计时恒等式与CUDA Event汇总校验：`'+json.dumps(event_audit,ensure_ascii=False)+'`。']
 report+=['','历史4090两行都是完整30题；当前3090 Session1仅GQA15题，尚无同一30题完整均值，不能把两者的部分/完整均值差解释为同输入集硬件性能差。主机CPU也不同：旧4090为AMD EPYC7453、14个可用CPU，新3090为Intel Xeon Gold6142、6个可用CPU；Torch intra-op线程均1，默认inter-op分别14/6，未调优或修改。详见diagnosis.json及各环境原件。']
 report+=['','## GPU正式阶段实测范围','',table(['Session','采样数','温度℃','SM Clock MHz','Memory Clock MHz','功耗W','P-state/Throttle masks'],[[g['session'],g['formal_phase_samples'],str(g['temperature.gpu']['min'])+'–'+str(g['temperature.gpu']['max']),str(g['clocks.current.sm']['min'])+'–'+str(g['clocks.current.sm']['max']),str(g['clocks.current.memory']['min'])+'–'+str(g['clocks.current.memory']['max']),str(g['power.draw']['min'])+'–'+str(g['power.draw']['max']),str(g['pstates'])+'/'+str(g['throttle_masks'])] for g in gpu_summaries]),'','上述采样包括正式题目之间的探针与间隔；详细理由（SW Power Cap、HW/SW Thermal、HW Power Brake等）均有同次查询的独立字段，见gpu_telemetry.json。不能将P2或某一次时钟采样解释为固定时钟运行。', '',f'两个4090历史实验的generate函数AST精确一致：{ast_equal}；整体文件的差异为方法导入、方法特定布局/离线审计等。本轮调用的5ee2原始运行器SHA：`{historical["new_Native_runtime_SHA256"]}`。']
 report+=['',f'所有表台账行数：{ {k:len(v) for k,v in data.items()} }。断点复用仅允许跳过完整已通过Session；不把模型重载后的部分数据静默拼成同一个独立Session。']
 (O/'report_zh.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(dict(validation=validation,sessions=summaries),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
