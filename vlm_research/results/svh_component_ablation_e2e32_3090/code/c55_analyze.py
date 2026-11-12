"""CPU completeness, independent KV arithmetic and paired efficiency summary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from c55_common import *
import sqlite3,csv,collections,statistics,math,bisect,hashlib,datetime
def csvread(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def stats(xs):return dict(mean=statistics.mean(xs),median=statistics.median(xs),std=statistics.stdev(xs) if len(xs)>1 else 0,min=min(xs),max=max(xs),n=len(xs))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def main():
 assert read(N/'execution_progress.json')['status']=='BOTH_SESSIONS_COMPLETE_PENDING_CPU_ANALYSIS';verify()
 samples={s['key']:s for s in lines(N/'input_manifest.jsonl.gz')};expected={(k,c) for k in samples for c in CONDITIONS};formal=[];layers=[];summary=[];paired=[];component_pairs=[];drifts=[];states={};tokens={};telemetry=[];warning_breakdown=[];metrics=('KV_MiB','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms')
 for session in (1,2):
  p=N/f'session{session}';state=read(p/'progress.json');assert state['status'] in ('COMPLETE','COMPLETE_WITH_PERFORMANCE_WARNINGS');states[str(session)]=state
  smoke=read(p/'smoke_validation.json');assert smoke['status']=='PASS' and len(smoke['cases'])==8 and all(x['quality_path_all32_logits_bitwise_exact'] for x in smoke['cases'])
  wc=collections.Counter((x['type'],x.get('category','calibration' if x['type']=='calibration_over2pct' else 'warmup')) for x in read(p/'performance_warnings.json'))
  warning_breakdown.extend([dict(session=session,type=t,category=c,count=n) for (t,c),n in sorted(wc.items())])
  gpu=[]
  with (p/'gpu_telemetry.jsonl').open() as f:
   for line in f:
    x=json.loads(line)
    if 'SM_MHz' in x:gpu.append(x)
  times=[r['at_ns'] for r in gpu];assert times==sorted(times) and gpu
  telemetry.append(dict(session=session,samples=len(gpu),temperature_C=stats([x['temperature_C'] for x in gpu]),SM_MHz=stats([x['SM_MHz'] for x in gpu]),power_mW=stats([x['power_mW'] for x in gpu]),throttle_masks=dict(collections.Counter(str(x['throttle_mask']) for x in gpu))))
  db=sqlite3.connect('file:'+str(p/'checkpoint.sqlite')+'?mode=ro',uri=True);assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok';current=[];seen=set();counts=collections.Counter()
  for rid,category,key,condition,payload,h in db.execute('SELECT * FROM requests ORDER BY id'):
   r=json.loads(payload);assert digest(r)==h and r['id']==rid and r['GPU_UUID']==UUID;g=r['generation'];counts[category]+=1
   assert r['input_hashes']==samples[key]['input_hashes'] and g['output_tokens']==32 and len(g['token_ids'])==32 and g['real_decode_calls']==31 and g['all_logits_finite']
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
   assert abs(statistics.mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
   assert g['cache']['storage_bytes']==g['cache']['live_bytes']
   if category!='formal':continue
   assert (key,condition) not in seen;seen.add((key,condition));s=samples[key];n=len(s['input_ids']);tokens[(session,key,condition)]=g['token_ids']
   actual=0
   for row in g['cache']['layers']:
    i=row['layer'];absent=condition in ('wo_selective','SVH') and 2<=i<=5;visual=576 if condition=='Native' else 0;used=0 if absent else n-576+visual+31;want=used*2*32*128*2
    assert row['used_tokens']==used and row['visual_tokens']==(0 if absent else visual) and row['live_bytes']==row['storage_bytes']==want
    assert row['K_shape']==row['V_shape']==(None if absent else [1,32,used,128]);actual+=row['live_bytes']
    layers.append(dict(session=session,input_id=key,condition=condition,**row))
   assert actual==g['cache']['live_bytes']
   ix=max(0,bisect.bisect_right(times,r['begin_ns'])-1);start=gpu[ix];during=[x for x in gpu[ix:] if x['at_ns']<=r['end_ns']] or [start]
   row=dict(session=session,GPU_UUID=UUID,request_id=rid,input_id=key,benchmark=s['benchmark'],condition=condition,output_tokens=32,KV_MiB=actual/2**20,TTFT_ms=g['TTFT_ms'],TPOT_ms=g['TPOT_ms'],CUDA_Event_Decode_TPOT_ms=g['CUDA_Event_Decode_TPOT_ms'],E2E_ms=g['E2E_ms'],begin_ns=r['begin_ns'],end_ns=r['end_ns'],SM_MHz_min=min(x['SM_MHz'] for x in during),SM_MHz_max=max(x['SM_MHz'] for x in during),temperature_C_max=max(x['temperature_C'] for x in during),power_W_mean=statistics.mean(x['power_mW'] for x in during)/1000,Pstates=sorted(set(x['pstate'] for x in during)),throttle_masks=sorted(set(x['throttle_mask'] for x in during)),thread_CPU_ms=r['thread_CPU_ms'],thread_runqueue_ms=r['thread_runqueue_ms'],early_vs_late_decode_change=r['early_vs_late_decode_change']);current.append(row);formal.append(row)
  assert seen==expected and len(current)==120 and counts=={'smoke':14,'warmup':48,'baseline':20,'formal':120,'calibration':36},counts
  csvwrite(p/'per_request.csv',current);native={r['input_id']:r for r in current if r['condition']=='Native'}
  for condition in CONDITIONS:
   rs=[r for r in current if r['condition']==condition];assert len(rs)==30;item=dict(session=session,condition=condition,requests=30)
   for metric in metrics:
    st=stats([r[metric] for r in rs])
    for k,v in st.items():item[metric+'_'+k]=v
    nm=statistics.mean(x[metric] for x in native.values());item[metric+'_reduction_percent']=(1-st['mean']/nm)*100
    ds=[native[r['input_id']][metric]-r[metric] for r in rs];sds=stats(ds);ci=2.045229642*sds['std']/math.sqrt(30)
    paired.append(dict(session=session,condition=condition,metric=metric,n=30,native_minus_condition_mean=sds['mean'],median=sds['median'],std=sds['std'],CI95_low=sds['mean']-ci,CI95_high=sds['mean']+ci,condition_better_inputs=sum(x>0 for x in ds),condition_worse_inputs=sum(x<0 for x in ds),note='Exploratory paired input t interval; timing correlation/system drift; no multiple-testing correction'))
   item['E2E_speedup']=statistics.mean(x['E2E_ms'] for x in native.values())/item['E2E_ms_mean'];summary.append(item)
  full={r['input_id']:r for r in current if r['condition']=='SVH'}
  for condition in ('wo_selective','wo_bypass'):
   rs=[r for r in current if r['condition']==condition]
   for metric in metrics:
    ds=[r[metric]-full[r['input_id']][metric] for r in rs];st=stats(ds);ci=2.045229642*st['std']/math.sqrt(30)
    component_pairs.append(dict(session=session,condition=condition,metric=metric,n=30,condition_minus_full_mean=st['mean'],median=st['median'],std=st['std'],CI95_low=st['mean']-ci,CI95_high=st['mean']+ci,full_lower_inputs=sum(x>0 for x in ds),full_higher_inputs=sum(x<0 for x in ds),condition_vs_full_increase_percent=(statistics.mean(r[metric] for r in rs)/statistics.mean(r[metric] for r in full.values())-1)*100,note='Positive difference means Full SVH uses less KV or time; exploratory paired t interval'))
  cal=read(p/'calibration.json')['checks'];assert len(cal)==12 and collections.Counter(x['window'] for x in cal)=={i:2 for i in range(1,7)};drifts.extend([dict(session=session,**x) for x in cal])
  for mode in ('gpu','cpu','processes'):assert not read(p/(mode+'_collector_summary.json'))['errors']
  db.close()
 comparisons=[]
 for key in samples:
  for condition in CONDITIONS:
   x=tokens[(1,key,condition)];y=tokens[(2,key,condition)];ex=x.index(2) if 2 in x else None;ey=y.index(2) if 2 in y else None
   comparisons.append(dict(key=key,condition=condition,all32_exact=x==y,first_difference_0based=next((i for i,(a,b) in enumerate(zip(x,y)) if a!=b),None),EOS1_0based=ex,EOS2_0based=ey,through_EOS_exact=ex==ey and (x[:ex+1]==y[:ey+1] if ex is not None else x==y),session1_tokens=x,session2_tokens=y))
 assert all(x['through_EOS_exact'] for x in comparisons),'Cross-session pre-EOS output mismatch requires audit'
 save(N/'cross_session_tokens.json',dict(status='PASS',all32_exact=sum(x['all32_exact'] for x in comparisons),comparisons=comparisons))
 old_accuracy=csvread(OLD/'ablation/summary.csv');combined=[]
 for condition in CONDITIONS:
  rs=[r for r in summary if r['condition']==condition];assert len(rs)==2;item=dict(condition=condition,label=LABELS[condition],sessions=2,requests=60)
  for bench,field in [('ScienceQA-IMG','SQA_accuracy_percent'),('POPE','POPE_accuracy_percent')]:item[field]=float(next(x['accuracy_percent'] for x in old_accuracy if x['condition']==condition and x['benchmark']==bench))
  for metric in metrics:
   values=[r[metric+'_mean'] for r in rs];item[metric]=statistics.mean(values);item[metric+'_session_mean_std']=statistics.stdev(values);item[metric+'_session1']=values[0];item[metric+'_session2']=values[1];item[metric+'_session2_vs_session1_percent']=(values[1]/values[0]-1)*100;item[metric+'_reduction_percent_mean_within_session']=statistics.mean(r[metric+'_reduction_percent'] for r in rs)
  item['E2E_speedup_mean_within_session']=statistics.mean(r['E2E_speedup'] for r in rs);combined.append(item)
 csvwrite(N/'per_request.csv',formal);csvwrite(N/'kv_per_layer.csv',layers);csvwrite(N/'session_summary.csv',summary);csvwrite(N/'paired_statistics.csv',paired);csvwrite(N/'component_effects.csv',component_pairs);csvwrite(N/'summary.csv',combined);save(N/'GPU_telemetry_summary.json',telemetry);save(N/'warning_breakdown.json',warning_breakdown)
 drift=dict(checks=drifts,warn_checks=sum(x['warn'] for x in drifts),max_abs_E2E_drift=max(abs(x['relative_drift']['E2E_ms']) for x in drifts),max_abs_TPOT_drift=max(abs(x['relative_drift']['TPOT_ms']) for x in drifts),no_baseline_resets=True);save(N/'calibration.json',drift)
 status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if any(x['status']=='COMPLETE_WITH_PERFORMANCE_WARNINGS' for x in states.values()) or not all(x['all32_exact'] for x in comparisons) else 'COMPLETE'
 validation=dict(status=status,execution_correctness='PASS',sessions=2,formal_requests=240,input_hashes='ALL30_EXACT',same_quality_components='ALL32_LOGITS_BITWISE_EXACT_SMOKE',physical_KV_independent_byte_arithmetic='PASS',timing_identity='PASS',all_original_generate_source_unchanged=True,quality_scores_reused_from_original_summary_sha256=sha(OLD/'ablation/summary.csv'),session_status=states,summary=combined,calibration_warns=drift['warn_checks']);save(N/'validation.json',validation);verify()
 out=['# SVH Component Ablation：RTX3090 E2E@32 效率补测','',f'状态：**{status}**；执行与正确性：PASS。两个独立Session各120条，合计240条正式请求，全部保留。GPU UUID `{UUID}`。','',
  '四个配置与原SQA/POPE消融使用完全相同的物理实现和模型Revision。Accuracy直接引用已验证旧结果，不重新评测。**效率输入为原冻结GQA15+TextVQA Paper-OCR15，并非SQA/POPE全集的自然EOS延迟。** 固定输出32Token，忽略EOS，1次Prefill与31次真实Decode。FP16/Eager/Batch1/Greedy、torch2.6.0+cu124、transformers4.49.0。',
  '', '## 四配置综合表','', '下表时间为两个Session均值的平均；±为两个Session均值的样本标准差。完整逐题离散度、逐Session均值/中位数/标准差及同轮配对区间另存CSV。', '',table(['Configuration','SQA %','POPE %','KV@32 MiB','TTFT ms','TPOT ms/token','E2E@32 ms','E2E speedup'],[[x['label'],f'{x["SQA_accuracy_percent"]:.2f}',f'{x["POPE_accuracy_percent"]:.2f}',f'{x["KV_MiB"]:.4f} ± {x["KV_MiB_session_mean_std"]:.4f}',f'{x["TTFT_ms"]:.2f} ± {x["TTFT_ms_session_mean_std"]:.2f}',f'{x["TPOT_ms"]:.3f} ± {x["TPOT_ms_session_mean_std"]:.3f}',f'{x["E2E_ms"]:.2f} ± {x["E2E_ms_session_mean_std"]:.2f}',f'{x["E2E_speedup_mean_within_session"]:.4f}×'] for x in combined]),
  '', '## 逐Session实测','',table(['Session','Condition','KV MiB','TTFT ms','TPOT ms/token','CUDA Event TPOT','E2E ms','E2E reduction %'],[[x['session'],LABELS[x['condition']],f'{x["KV_MiB_mean"]:.4f}',f'{x["TTFT_ms_mean"]:.3f}',f'{x["TPOT_ms_mean"]:.3f}',f'{x["CUDA_Event_Decode_TPOT_ms_mean"]:.3f}',f'{x["E2E_ms_mean"]:.3f}',f'{x["E2E_ms_reduction_percent"]:+.3f}'] for x in summary]),
  '', '## 相对同轮Native的变化','', '降低比例先在各Session计算，再取两轮平均；负值表示比Native更高或更慢。', '',table(['Condition','KV reduction %','TTFT reduction %','TPOT reduction %','E2E reduction %'],[[x['label']]+[f'{x[m+"_reduction_percent_mean_within_session"]:+.3f}' for m in ('KV_MiB','TTFT_ms','TPOT_ms','E2E_ms')] for x in combined]),
  '', '## 执行路径与物理KV','',
  '- 直接调用冻结`unified_e2e32_4090/code/e2e32_runtime.py`原`generate`函数对象，未复制或修改其计时循环、Greedy、模型或Attention。仅在函数外选择已验证消融类；新增KV审计在原计时结束后运行，投影/位置/RoPE观察器仅用于Smoke，正式运行无观察钩子。控制器和掩码初始化仍位于原计时范围内。',
  '- w/o Selective Transfer：实际复用质量评测NoSelectiveTransfer类，视觉隐藏状态通过全部32层Prefill，Relays全层读取图像；答案不直接访问图像，L2–5物理Attention Bypass保持，答案持久KV仍为文本。这一消融不应被误写成保留全部图像KV。',
  '- w/o Attention Bypass：实际复用NoAttentionBypass类，L20视觉隐藏提前退出与图像KV移除不变；L2–5恢复正常Attention及真实文本KV，32层均有文本KV。',
  '- 两探针×四配置的32步logits与对应质量路径逐位相同。数值影子验证仅使用实际Greedy输出作为固定前缀，逐步argmax也必须一致，不使用Gold。Native/No-op及Native/Full-SVH原运行器的全部32步logits严格一致；真实Q/K/V调用、隐藏行数和RoPE检查通过。',
  '- KV@32由实际K/V形状、有效长度、FP16元素大小及独立存储指针统计；生成第32个输出时缓存中有31个Decode输入。逐题逐层核验独立字节公式，不能用总GPU显存代替KV。',
  '', '## 稳定性与配对统计','',
  '每轮独立进程加载模型；固定40次Native完整32Token预热，加四条件各2次预热，双探针各10次建立独立中位数基线。每题交错四条件，顺序按题号循环轮换，两轮调度完全相同。每5题双探针各3次校准，性能漂移只WARN，无重置基线、重跑或删除请求。5Hz GPU、1Hz CPU和其他GPU进程遥测保留。',
  '',f'24项校准中{drift["warn_checks"]}项超过2%；最大绝对E2E漂移{drift["max_abs_E2E_drift"]*100:.3f}%，最大绝对TPOT漂移{drift["max_abs_TPOT_drift"]*100:.3f}%。两轮同题同配置32Token输出完全一致的数量为{sum(x["all32_exact"] for x in comparisons)}/120；所有比较在首EOS之前及EOS位置一致。',
  '',table(['Session','Window','Probe','E2E drift %','TPOT drift %','Flag'],[[x['session'],x['window'],x['key'],f'{x["relative_drift"]["E2E_ms"]*100:+.3f}',f'{x["relative_drift"]["TPOT_ms"]*100:+.3f}','WARN' if x['warn'] else 'within2%'] for x in drifts]),
  '',table(['Session','Condition','E2E faster /30','Paired Native−condition E2E ms','Exploratory95% CI ms'],[[x['session'],LABELS[x['condition']],x['condition_better_inputs'],f'{x["native_minus_condition_mean"]:.3f}',f'[{x["CI95_low"]:.3f}, {x["CI95_high"]:.3f}]'] for x in paired if x['metric']=='E2E_ms' and x['condition']!='Native']),
  '', '区间为30题配对差的探索性t区间，含系统状态相关性，未作多重检验校正。两轮绝对时延变化、逐请求异常和所有慢请求均保留；即使全部执行完成，性能WARN也不能称稳定PASS。',
  '', '## 结果与复现','', '`summary.csv`为四条件综合表；`session_summary.csv`、`paired_statistics.csv`、`per_request.csv`、`kv_per_layer.csv`、`calibration.json`及两轮SQLite保存完整证据。`protocol.json`和`source_manifest.json`为测量前冻结记录，`artifact_manifest.json`为完成后清单。原SQA/POPE结果与此前长度实验均未覆盖。',
  '', '仅在新目录、核验相同冻结源码与输入并适配实例UUID后运行；已存在的协议与Session不允许覆盖。', '', '```bash',f'source {REST}/activate.sh',f'cd {N}',f'{P} -B code/c55_prepare.py',f'{P} -u -B code/c55_run.py --session 1',f'{P} -u -B code/c55_run.py --session 2',f'{P} -B code/c55_analyze.py','```']
 out+=['','## 相对Full SVH的组件影响','', '正值表示关闭该组件后更慢或KV更大；这里按同题、同Session与Full SVH配对。', '',table(['Session','Condition','Metric','Condition−Full mean','Exploratory95% CI','Increase vs Full %'],[[x['session'],LABELS[x['condition']],x['metric'],f'{x["condition_minus_full_mean"]:+.4f}',f'[{x["CI95_low"]:.4f}, {x["CI95_high"]:.4f}]',f'{x["condition_vs_full_increase_percent"]:+.3f}'] for x in component_pairs if x['metric']!='CUDA_Event_Decode_TPOT_ms'])]
 out+=['','## 性能警告分类','',table(['Session','Type','Category','Count'],[[x['session'],x['type'],x['category'],x['count']] for x in warning_breakdown]) if warning_breakdown else '未记录到上述性能警告。','', '请求内早/晚Decode差异超过5%是诊断标记，不能单凭它判定GPU降频；Smoke含数值捕获与观察钩子，其计时不用于正式统计。Native校准、正式请求内标记和预热/Smoke标记分别列出，不能混为同一种稳定性结论。']
 for condition in ('wo_selective','wo_bypass'):
  values={m:statistics.mean(x['condition_vs_full_increase_percent'] for x in component_pairs if x['condition']==condition and x['metric']==m) for m in ('KV_MiB','TTFT_ms','TPOT_ms','E2E_ms')}
  out.append(f'{LABELS[condition]}相对Full SVH，两轮同轮比值的平均变化：KV {values["KV_MiB"]:+.3f}%，TTFT {values["TTFT_ms"]:+.3f}%，TPOT {values["TPOT_ms"]:+.3f}%，E2E {values["E2E_ms"]:+.3f}%。是否能推广到其他输入和GPU状态，仍受固定30题、两个Session与上述性能警告限制。')
 (N/'report_zh.md').write_text('\n'.join(out)+'\n');print('COMPONENT_EFFICIENCY_CPU_FINAL_PASS',status,flush=True)
if __name__=='__main__':main()
