"""CPU-only readback audit and seven-metric report; no inference or source edits."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from seven_common import *
from collections import defaultdict,Counter
from datetime import datetime,timezone
import ast,csv,sqlite3,subprocess,statistics,shutil
from vlm_research.information_horizon_alignment import analyze
from vlm_research.mme_seed_quality.common import TASKS
def writecsv(name,rows):
 if rows:
  with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
state=read(O/'progress.json');audit=read(O/'input_audit.json');protocol=read(O/'protocol.json');binding=digest(protocol)
amendment=read(O/'efficiency_protocol_amendment.json')
assert sha(O/'run_seven_v1.py')==protocol['sources']['run_seven.py']==amendment['runner_v1_sha256']
assert sha(O/'run_seven.py')==amendment['runner_v2_sha256']
assert amendment['quality_protocol_unchanged'] and amendment['timing_requests_before_amendment']==0
for name,h in protocol['sources'].items():
 if name!='run_seven.py':assert sha(O/name)==h,name
assert sha(V/'official_sources/V2Drop.py')=='e39ed8d99c3e1f23bd156067bfd28179827771df3556631b6daed4427a7b8b61'
assert sha(V/'hf_adapter.py')=='b172ded154847b3f9bdfa4d46b48708d12f9b0e469d109af35e347e666aad4ae'
samples=gz(O/'input_manifest.jsonl.gz');by={s['key']:s for s in samples};records={};timings=[]
with sqlite3.connect('file:'+str(O/'predictions.sqlite')+'?mode=ro',uri=True) as db:
 assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
 for k,p,h in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(p);s=by[k];assert digest(r)==h and r['binding']==binding and r['sample_sha256']==digest(s) and r['gold']==s['answer'] and r['image_sha256']==s['image_sha256']
  assert score(s,r['generation']['text'])==r['scoring']
  g=r['generation'];assert g['empty']==(not bool(g['text'])) and g['truncated']==(len(g['token_ids'])==s['max_new_tokens'] and not g['eos']) and g['all_effective_logits_finite']
  assert [e['layer'] for e in r['pruning_events']]==[3,17] and [e['visual_after'] for e in r['pruning_events']]==[288,144]
  records[k]=r
 for k,p,h in db.execute('SELECT key,payload,sha256 FROM timings'):
  r=json.loads(p);assert digest(r)==h and r['binding']==binding
  out=r['result'];assert out['fixed'] and out['output_tokens']==r['length'] and out['decode_steps']==r['length']-1 and out['all_logits_finite']
  assert out['visual_KV_residency_percent']==0 and out['cache']['retained_image_layers']==[]
  assert abs(out['KV_live_MiB']-sum(x['live_bytes'] for x in out['cache']['layers'])/2**20)<1e-12
  assert abs(out['decode_GPU_TPOT_ms']-sum(out['step_gpu_ms'][1:])/(r['length']-1))<1e-10
  assert abs(out['decode_TPOT_ms']-(out['E2E_ms']-out['TTFT_ms'])/(r['length']-1))<1e-10
  timings.append(r)
if len(timings)==60:
 assert {(r['key'],r['length']) for r in timings}=={(k,length) for k in audit['efficiency_keys'] for length in [32,128]}
summary_rows=[];details={};stats={}
for benchmark,expected in audit['expected'].items():
 ss=[s for s in samples if s['benchmark']==benchmark];rs=[records[s['key']] for s in ss if s['key'] in records];complete=len(rs)==expected
 stats[benchmark]=dict(completed=len(rs),expected=expected,new_generated=sum(r['provenance']=='NEW_GPU_GENERATION' for r in rs),reused=sum(r['provenance']=='REUSED_VERIFIED_PREDICTION' for r in rs),empty=sum(r['generation']['empty'] for r in rs),truncated=sum(r['generation']['truncated'] for r in rs),parse_failed=sum(r['scoring'].get('parse_failed',False) for r in rs))
 value=None;unit='percent';label='Accuracy'
 if complete:
  values=[records[s['key']]['scoring']['score'] for s in ss]
  if benchmark=='MME':
   totals,tasks=analyze.mme_score(ss,values);ind={}
   for task in sum(TASKS.values(),[]):
    rows=[(s,v) for s,v in zip(ss,values) if s['task']==task];pairs=defaultdict(list)
    for s,v in rows:pairs[s['pair_id']].append(v)
    assert all(len(p)==2 for p in pairs.values())
    ind[task]=100*sum(v for s,v in rows)/len(rows)+100*sum(all(v) for v in pairs.values())/len(pairs)
   check={k:sum(ind[t] for t in subtasks) for k,subtasks in dict(**TASKS,Total=sum(TASKS.values(),[])).items()}
   assert all(abs(totals[k]-check[k])<1e-9 for k in totals)
   assert totals==read(FM/'score_summary.json')['scores']
   details['MME']=dict(scores=totals,tasks=tasks,independent_dual_question_audit='PASS');value=totals['Total'];unit='MME score';label='Accuracy + Accuracy+ total'
   writecsv('MME_per_task.csv',tasks)
  elif benchmark=='MMBench':
   collapsed,circular=analyze.circular_rows(ss,values);groups=defaultdict(list)
   for s,v in zip(ss,values):groups[s['circular_group']].append(v)
   count=sum(all(v) for v in groups.values());assert len(collapsed)==len(groups)==1176 and abs(circular.sum()-count)<1e-10
   value=100*count/1176;label='Circular group accuracy';details[benchmark]=dict(correct_groups=count,groups=1176,row_accuracy_percent=100*sum(values)/expected,complete_rotations_audited=True)
   writecsv('MMBench_circular_groups.csv',[dict(circular_group=k,rotations=len(v),all_correct=bool(all(v))) for k,v in sorted(groups.items())])
  elif benchmark=='OCRBench-v1':
   official=B/'featurize_new3090_c3v2_reaccess_sevenbench/ocr_official_LLaVA1_5.py';assert sha(official)==audit['OCR_official_source_sha256']
   source=official.read_text();tree=ast.parse(source)
   loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and 'data_type = data[i]["type"]' in ast.get_source_segment(source,n)]
   assert len(loops)==1
   data=[dict(type=s['category'],dataset_name=s['dataset_name'],answers=s['answer'],predict=records[s['key']]['generation']['text']) for s in ss]
   scope=dict(data=data);exec(compile(ast.Module(body=loops,type_ignores=[]),str(official),'exec'),scope)
   author=[r['result'] for r in data];assert author==values
   value=sum(values);unit='score / 1000';label='Official OCRBench v1';categories=[]
   for category in dict.fromkeys(s['category'] for s in ss):
    vv=[v for s,v in zip(ss,values) if s['category']==category];categories.append(dict(category=category,correct=sum(vv),questions=len(vv)))
   details[benchmark]=dict(correct=value,questions=1000,accuracy_percent=value/10,official_source_AST_independent_rescore='PASS',categories=categories)
   writecsv('OCRBench_categories.csv',categories)
  else:
   value=100*sum(values)/expected;details[benchmark]=dict(score_sum=sum(values),questions=expected)
   if benchmark=='TextVQA':label='Ten-answer VQA soft accuracy'
   if benchmark=='POPE':
    strategies={}
    for strategy in ['random','popular','adversarial']:
     indices=[i for i,s in enumerate(ss) if s['strategy']==strategy];strategies[strategy]=dict(questions=len(indices),correct=sum(values[i] for i in indices),accuracy_percent=100*sum(values[i] for i in indices)/len(indices))
    details['POPE'].update(strategies=strategies,macro_strategy_accuracy_percent=statistics.mean(x['accuracy_percent'] for x in strategies.values()),primary='All frozen 8910 rows accuracy; macro strategy also provided')
    writecsv('POPE_strategies.csv',[dict(strategy=k,**v) for k,v in strategies.items()])
 summary_rows.append(dict(benchmark=benchmark,status='COMPLETE' if complete else 'PARTIAL',completed=len(rs),expected=expected,metric=label,value=value,unit=unit,**{k:v for k,v in stats[benchmark].items() if k not in ['completed','expected']}))
writecsv('benchmark_summary.csv',summary_rows)
paired=[]
for s in samples:
 if s['key'] in records:
  r=records[s['key']];paired.append(dict(key=s['key'],benchmark=s['benchmark'],question_id=s['question_id'],image_sha256=s['image_sha256'],gold=json.dumps(s['answer'],ensure_ascii=False),prediction=r['generation']['text'],parsed=r['scoring']['parsed'],score=r['scoring']['score'],output_tokens=len(r['generation']['token_ids']),empty=r['generation']['empty'],truncated=r['generation']['truncated'],provenance=r['provenance']))
writecsv('predictions_scored.csv',paired)
eff=[]
for length in [32,128]:
 rr=[r for r in timings if r['length']==length]
 if rr:
  assert len({r['key'] for r in rr})==len(rr)
  for benchmark in ['ALL','GQA','TextVQA']:
   subset=[r for r in rr if benchmark=='ALL' or by[r['key']]['benchmark']==benchmark]
   if not subset:continue
   eff.append(dict(benchmark=benchmark,fixed_output_tokens=length,requests=len(subset),KV_live_MiB_mean=statistics.mean(r['result']['KV_live_MiB'] for r in subset),KV_storage_MiB_mean=statistics.mean(r['result']['KV_storage_MiB'] for r in subset),TPOT_ms_mean=statistics.mean(r['result']['decode_TPOT_ms'] for r in subset),TPOT_ms_median=statistics.median(r['result']['decode_TPOT_ms'] for r in subset),GPU_TPOT_ms_mean=statistics.mean(r['result']['decode_GPU_TPOT_ms'] for r in subset),visual_KV_residency_percent=0.0))
writecsv('efficiency_summary.csv',eff)
full=len(records)==36256 and len(timings)==60 and state['status']=='COMPLETE'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
for n,h in audit['frozen_sources'].items():assert sha(R/n)==h,n
for n,h in audit['fusion_sources'].items():assert sha(F/n)==h,n
summary=dict(status='COMPLETE' if full else 'PARTIAL',at_UTC=datetime.now(timezone.utc).isoformat(),condition=protocol['condition'],questions_completed=len(records),questions_expected=36256,stats=stats,benchmark_summary=summary_rows,details=details,efficiency=eff,efficiency_completed=len(timings),efficiency_expected=60,model_revision=protocol['revision'],code_commit=protocol['code_commit'],physical_smoke=read(O/'integration_smoke.json')['status'],inherited_numerical_smoke='PASS original unchanged thresholds, same GPU and executor',frozen_sources_unchanged=True,CPU_readback_independent_aggregation_audit='PASS',GPU_after=gpu,GPU_processes_after=processes,GPU_released=not bool(processes),no_GitHub_push=True)
save('score_summary.json',summary)
metadata=read(O/'run_metadata.json');metadata.update(status=summary['status'],questions_completed=len(records),timings_completed=len(timings),GPU_released=summary['GPU_released'],GPU_after=gpu,final_scoring_audit='PASS',finalized_at_UTC=summary['at_UTC']);save('run_metadata.json',metadata)
bybench={r['benchmark']:r['value'] for r in summary_rows};primary=next((r for r in eff if r['benchmark']=='ALL' and r['fixed_output_tokens']==128 and r['requests']==30),None)
def fmt(x):return '未完成' if x is None else f'{x:.4f}'
table=dict(Method='V2Drop-prefix+SVH',KV_MiB=None if primary is None else primary['KV_live_MiB_mean'],TPOT_ms=None if primary is None else primary['TPOT_ms_mean'],GQA=bybench['GQA'],SQA=bybench['ScienceQA-IMG'],TextVQA=bybench['TextVQA'],POPE=bybench['POPE'],MME=bybench['MME'],MMB=bybench['MMBench'],OCRBench=bybench['OCRBench-v1'])
writecsv('requested_table.csv',[table])
lines=['# V²Drop-prefix + SVH：七项完整质量评测与 KV/TPOT','',f"**状态：{summary['status']}**。质量预测 {len(records):,}/36,256；固定长度效率请求 {len(timings)}/60。复用 MME 2,374 条和 GQA100/TextVQA100 共 2,574 条；仅生成其余缺失的融合结果。",'',
 '| Method | KV (MiB) ↓ | TPOT (ms) ↓ | GQA ↑ | SQA ↑ | TextVQA ↑ | POPE ↑ | MME ↑ | MMB ↑ | OCRBench ↑ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
 '| V²Drop-prefix + SVH | '+' | '.join(fmt(table[k]) for k in ['KV_MiB','TPOT_ms','GQA','SQA','TextVQA','POPE','MME','MMB','OCRBench'])+' |','',
 'GQA/SQA/TextVQA/POPE/MMB 单位为 %；TextVQA 为原十答案 VQA Soft Accuracy。MMB 为 1,176 个完整 Circular 组全对的比例。MME 为 Perception+Cognition 的 Total Score（不是 %）；OCRBench 为官方 v1 的 /1000 累加分。POPE 主列为冻结 8,910 题逐题 Accuracy，三策略及等权平均另存 POPE_strategies.csv / score_summary.json。',
 'KV/TPOT 主列使用原 V²Drop 冻结的 30 个输入（GQA15 + TextVQA15）与固定 128 个输出 Token，另保留固定 32 Token 结果。每个长度预热 2 次；batch=1，单 GPU，同一计时方法。KV 按实际缓存 used_rows 和 dtype 计算 live bytes，排除预留容量；持久视觉 KV 为零。主表 TPOT=(墙钟 E2E−墙钟 TTFT)/(固定输出长度−1)，并单独保存 CUDA 逐 Decode 步 TPOT；两者均排除 Prefill。墙钟值包含真实执行、调度与 greedy/finite 检查。它不是七类数据所有请求的平均 TPOT；未混用旧 RTX 4090 计时数据。', '',
 '| Benchmark | Completed / Expected | New | Reused | Empty | Truncated | Score |','|---|---:|---:|---:|---:|---:|---:|']
for row in summary_rows:lines.append(f"| {row['benchmark']} | {row['completed']} / {row['expected']} | {row['new_generated']} | {row['reused']} | {row['empty']} | {row['truncated']} | {fmt(row['value'])} |")
lines+=['',f"真实环境：{state['GPU']}；PyTorch 2.6.0+cu124、Transformers 4.49.0、FP16/Eager、Greedy、EOS=2、batch=1。模型 `{protocol['model']}@{protocol['revision']}`；基线 Commit `{protocol['code_commit']}`。本轮新增源码及冻结核心的完整 SHA256 见 protocol.json；未修改 SVH/V²Drop 冻结核心、模型或旧结果。为在任何效率测量前对齐历史墙钟 TPOT，曾受控保存断点并重新加载一次；3,544 条已有预测完整保留，未重复生成。旧入口、受控中止记录及效率补充协议均保留。",'',
 '算法只复用官方 V²Drop L3/L17 的 Hidden L2-change Top-k 选择，实际物理执行 576→288→144，L20 前按 SVH 退出全部视觉 Hidden；不执行完整 V²Drop 的 L22 剪枝。P−1/Decode 全层无原始视觉 KV，L2–L5 答案 Attention 物理跳算，Norm/MLP/Residual 保留，文本未裁剪/重编号。该融合条件与完整 V²Drop 的剪枝预算不同。',
 '物理验收继承同 GPU、同执行器的四图严格数值门禁 PASS（门限不变）。本轮补充 SQA、POPE 三策略、完整 MMB Circular 组和 OCRBench 共 9 条真实输入的 Vision、物理形状、RoPE/位置、文本完整性、答案 Attention 跳算及缓存检查，均在正式预测中只计一次，详见 integration_smoke.json。',
 'GQA 使用已审计的官方对齐 Vicuna-v1 完整 System+USER+Image+Question+Short Answer、RGB/CLIP mean square-pad/CLIP336、max_new_tokens=32；逐题实际 Token IDs 与 FP16 张量哈希和历史官方对齐 Manifest 一致。不能与早期 USER-only/crop 的 GQA 结果作同协议直接差值。其余六项逐题核对冻结 Prompt/Token IDs/FP32 像素哈希（随后转 FP16）、Gold 与图像标识。',
 '生成上限：GQA32；MME/POPE/TextVQA/OCRBench128；SQA/MMB1024。质量生成遵循自然 EOS；固定长度只用于独立 KV/TPOT 测量。所有错误、空输出和截断原样保留。',
 '评分重新读取持久化预测，调用冻结 Scorer；GQA/POPE/SQA/MME 逐题与 TextVQA 十答案软评分独立逻辑复核；MME Accuracy/Accuracy+ 与 MMB 完整 Circular 组独立计数；OCRBench 另执行固定官方源码 AST 的原始评分循环，确认逐题一致。MMB 沿用既有 OFFLINE_REPRODUCTION/VLMEvalKit exact matcher，未使用外部 LLM Judge。',
 'MME 已知 landmark/03d5e3bfc958be38.jpg 字节版本差异涉及双题，两条均复用已完成正式结果，未替换或删题；旧/新 SHA 见 input_audit.json。图像完整性使用已恢复可信清单，未全量重新哈希或下载图片。',
 '预测使用 SQLite PRIMARY KEY + FULL 同步逐题事务，并保存 protocol binding、样本摘要、QID/Gold、输入张量绑定、原始输出与 L3/L17 选择索引；progress.json 可恢复，原始完整结果另压缩保存。不生成其他方法、不搜索层或比例、不扩大数据视图。',
 f"GPU 状态：{gpu}；活动计算进程：{processes or '无'}。服务器未关闭；未 GitHub Push。Windows 备份必须以真实 SFTP+SHA256 回执为准，见 windows_backup_receipt.json；无回执不得视作已备份。"]
if 'MME' in details:lines+=['',f"MME Perception={details['MME']['scores']['Perception']:.6f}，Cognition={details['MME']['scores']['Cognition']:.6f}，Total={details['MME']['scores']['Total']:.6f}。"]
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
dependencies={}
for name,source in {
 'fusion_engine_frozen.py':F/'fusion_engine.py',
 'fusion_smoke_helpers_frozen.py':F/'run_fusion.py',
 'v2drop_hf_adapter_frozen.py':V/'hf_adapter.py',
 'official_V2Drop_frozen.py':V/'official_sources/V2Drop.py',
 'm4c_evaluator_frozen.py':F/'scorer_source/llava/eval/m4c_evaluator.py',
 'inherited_fusion_protocol.json':F/'protocol.json',
 'inherited_strict_physical_smoke.json':F/'physical_smoke.json',
}.items():
 destination=O/name
 if destination.exists():assert sha(destination)==sha(source),name
 else:shutil.copyfile(source,destination)
 dependencies[name]=dict(source_path=str(source),sha256=sha(source),bytes=source.stat().st_size)
save('frozen_dependency_catalog.json',dict(status='VERIFIED',unchanged_archived_copies=dependencies,ATE_core_commit=protocol['code_commit'],ATE_core_hashes=protocol['frozen_sources'],official_V2Drop_commit='2a9431150f8ace3a1d9abcdc96a7d46acca597df'))
files=[p for p in O.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json','run.lock','windows_backup_receipt.json'] and not p.name.endswith(('-wal','-shm','.tmp'))]
save('artifact_manifest.json',dict(status=summary['status'],weights_images_datasets_excluded=True,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}))
print(json.dumps(dict(status=summary['status'],table=table,stats=stats,GPU_released=summary['GPU_released']),ensure_ascii=False))
