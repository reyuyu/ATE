"""CPU full-MME scorer audit and compact report; no GPU model or inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import csv,gzip,hashlib,json,sqlite3,subprocess,sys
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';O=B/'v2drop_prefix_svh_mme_full_3090';F=B/'v2drop_prefix_svh_accuracy_3090'
sys.path.insert(0,str(R))
from vlm_research.information_horizon_alignment import score as scorer,analyze
from vlm_research.mme_seed_quality.common import TASKS
def read(p):return json.loads(Path(p).read_text())
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def writecsv(n,rows):
 if rows:
  with (O/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def independently_score(samples,values):
 tasks=[]
 for task in sum(TASKS.values(),[]):
  rows=[(s,c) for s,c in zip(samples,values) if s['task']==task];pairs=defaultdict(list)
  for s,c in rows:pairs[s['pair_id']].append(c)
  assert all(len(p)==2 for p in pairs.values())
  correct=sum(c for s,c in rows);both=sum(sum(v)==2 for v in pairs.values())
  acc=100*correct/len(rows);plus=100*both/len(pairs)
  tasks.append(dict(task=task,questions=len(rows),pairs=len(pairs),correct_questions=correct,both_correct_pairs=both,accuracy=acc,accuracy_plus=plus,score=acc+plus))
 totals={group:sum(r['score'] for r in tasks if r['task'] in subtasks) for group,subtasks in dict(**TASKS,Total=sum(TASKS.values(),[])).items()}
 return totals,tasks
state=read(O/'progress.json');assert state['status']=='COMPLETE' and state['completed']==2374,'Do not publish a partial MME total'
protocol=read(O/'protocol.json');audit=read(O/'input_audit.json');samples=gz(O/'input_manifest.jsonl.gz');by={s['key']:s for s in samples};old={r['key']:r for r in gz(O/'SVH_reference_predictions.jsonl.gz')};records={}
with sqlite3.connect('file:'+str(O/'predictions.sqlite')+'?mode=ro',uri=True) as db:
 assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
 for key,payload,checksum in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(payload);s=by[key]
  assert digest(r)==checksum and r['binding']==digest(protocol) and r['sample_sha256']==digest(s)
  assert r['gold']==s['answer'] and r['same_input_binding_as_SVH']
  value=scorer.score(s,r['generation']['text']);assert value==r['scoring']
  for text,expected in [(r['generation']['text'],value),(old[key]['generation']['text'],old[key]['scoring'])]:
   low=text.lower();label=low if low in ['yes','no'] else 'yes' if 'yes' in low[:4] else 'no' if 'no' in low[:4] else 'other'
   assert expected['parsed']==label and expected['score']==float(label==s['answer'])
  assert [e['layer'] for e in r['pruning_events']]==[3,17] and [e['visual_after'] for e in r['pruning_events']]==[288,144]
  assert r['generation']['empty']==(not bool(r['generation']['text']))
  assert r['generation']['truncated']==(len(r['generation']['token_ids'])==128 and not r['generation']['eos'])
  assert scorer.score(s,old[key]['generation']['text'])==old[key]['scoring'];records[key]=r
assert len(records)==2374 and set(records)==set(by)
values=[records[s['key']]['scoring']['score'] for s in samples];oldvalues=[old[s['key']]['scoring']['score'] for s in samples]
totals,tasks=analyze.mme_score(samples,values);svh_totals,svh_tasks=analyze.mme_score(samples,oldvalues)
independent,independent_tasks=independently_score(samples,values);oldind,oldind_tasks=independently_score(samples,oldvalues)
assert totals.keys()==independent.keys() and all(abs(totals[k]-independent[k])<1e-9 for k in totals)
assert all(abs(svh_totals[k]-oldind[k])<1e-9 for k in svh_totals)
for a,b in zip(tasks,independent_tasks):assert a['task']==b['task'] and all(abs(a[k]-b[k])<1e-10 for k in ['accuracy','accuracy_plus','score'])
task_table=[]
for row,ref,detail in zip(tasks,svh_tasks,independent_tasks):task_table.append(dict(task=row['task'],questions=row['questions'],pairs=row['pairs'],correct_questions=detail['correct_questions'],both_correct_pairs=detail['both_correct_pairs'],accuracy=row['accuracy'],accuracy_plus=row['accuracy_plus'],MME_score=row['score'],SVH_reference_score=ref['score'],delta_vs_SVH=row['score']-ref['score']))
writecsv('per_task_scores.csv',task_table)
paired=[]
for s in samples:
 r=records[s['key']];ref=old[s['key']];paired.append(dict(key=s['key'],task=s['task'],pair_id=s['pair_id'],question_id=s['question_id'],gold=s['answer'],fusion_output=r['generation']['text'],fusion_parsed=r['scoring']['parsed'],fusion_score=r['scoring']['score'],SVH_output=ref['generation']['text'],SVH_score=ref['scoring']['score'],delta=r['scoring']['score']-ref['scoring']['score'],known_MME_image_difference=r['known_MME_image_difference']))
writecsv('paired_questions.csv',paired)
generations=[records[s['key']]['generation'] for s in samples]
changed=dict(correct_to_wrong=sum(a==0 and b==1 for a,b in zip(values,oldvalues)),wrong_to_correct=sum(a==1 and b==0 for a,b in zip(values,oldvalues)),same_input_bindings_verified=2374)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not processes
for name,h in audit['frozen_sources'].items():assert sha(R/name)==h,name
for name,h in audit['fusion_sources'].items():assert sha(F/name)==h,name
summary=dict(status='COMPLETE',at_UTC=datetime.now(timezone.utc).isoformat(),condition='V2Drop-prefix+SVH',benchmark='MME-paper',questions=2374,pairs=1187,
 scores=totals,SVH_reference_scores=svh_totals,delta_vs_SVH={k:totals[k]-svh_totals[k] for k in totals},
 tasks=task_table,row_accuracy_percent=100*sum(values)/2374,paired=changed,
 empty=sum(g['empty'] for g in generations),truncated=sum(g['truncated'] for g in generations),parse_failed=sum(records[s['key']]['scoring']['parse_failed'] for s in samples),
 model_loads=state['model_loads_this_process'],model_revision=protocol['revision'],code_commit=protocol['code_commit'],
 source_fusion_engine_sha256=audit['fusion_sources']['fusion_engine.py'],inherited_strict_numerical_smoke='PASS unchanged; same engine, same GPU',
 MME_integration_smoke=read(O/'MME_integration_smoke.json')['status'],first_pair_reused_in_complete_results=True,
 independent_MME_accuracy_and_accuracy_plus_audit='PASS',known_MME_image_difference=audit['known_MME_image_difference'],
 GPU_after=gpu,GPU_processes_after=processes,GPU_released=True,no_Native_V2Drop_or_SVH_generation=True,no_GitHub_push=True)
save('score_summary.json',summary)
lines=['# V²Drop-prefix + SVH：MME 全量正式评测','',
 '**状态：COMPLETE。** 完成 MME Paper 2,374/2,374 题、1,187/1,187 个同图双题组；14 类均完整。', '',
 '| 条件 | Perception | Cognition | Total |','|---|---:|---:|---:|',
 '| V²Drop-prefix + SVH | '+' | '.join(f'{totals[k]:.6f}' for k in ['Perception','Cognition','Total'])+' |',
 '| llava-SVH 历史同协议对照（直接复用） | '+' | '.join(f'{svh_totals[k]:.6f}' for k in ['Perception','Cognition','Total'])+' |',
 '| Fusion − SVH | '+' | '.join(f'{totals[k]-svh_totals[k]:+.6f}' for k in ['Perception','Cognition','Total'])+' |','',
 '分数单位为 MME Score：每个子任务 Accuracy + Accuracy+；Accuracy+ 要求同图两题全部答对，再按原任务分组累加。Total 不是百分比，也不是将所有问题简单平均。逐题 Yes/No 解析、子任务评分与总分均调用冻结原评分器，并独立用计数公式复核通过。',
 '',f"逐题 Accuracy（仅辅助）{summary['row_accuracy_percent']:.6f}%；空输出 {summary['empty']}；截断 {summary['truncated']}；Yes/No 解析失败 {summary['parse_failed']}，全部保留计分。",
 f"相对历史 SVH：正确→错误 {changed['correct_to_wrong']} 题，错误→正确 {changed['wrong_to_correct']} 题。全部 2,374 题的 Prompt、Token IDs、FP32 预处理像素哈希（随后转相同 FP16）、Gold 与图像标识均与历史存档逐题匹配，因此可作本协议下的配对比较；未重跑 SVH、Native 或 V²Drop。",'',
 '| Task | Questions | Pairs | Accuracy (%) | Accuracy+ (%) | MME Score | Δ vs SVH |','|---|---:|---:|---:|---:|---:|---:|']
for r in task_table:lines.append(f"| {r['task']} | {r['questions']} | {r['pairs']} | {r['accuracy']:.4f} | {r['accuracy_plus']:.4f} | {r['MME_score']:.4f} | {r['delta_vs_SVH']:+.4f} |")
lines+=['',
 f"真实环境：{gpu}；torch 2.6.0+cu124、Transformers 4.49.0、FP16/Eager、batch=1、greedy、EOS=2、max_new_tokens=128。模型 `{protocol['model']}@{protocol['revision']}`；冻结基线 Commit `{protocol['code_commit']}`；新增独立执行器文件 SHA256 见 protocol.json，未修改冻结核心或之前的融合实验结果。",'',
 '算法沿用上一轮已验收的 PrefixSVH，官方选择器只在 L3/L17 进行 576→288→144 物理剪枝，L20 前退出全部视觉 Hidden；P−1/Decode 全层不读取原始视觉 KV；L2–L5 答案 Attention 真实跳算，Norm/MLP/Residual 保留；持久视觉 KV 为零。不执行完整 V²Drop 的 L22 剪枝。',
 '上一轮同 GPU、同执行器的 4 图严格数值门禁全部 PASS，本轮未降低门限或重复 GQA/TextVQA。首个 MME 同图双题组追加真实图像、剪枝、位置/RoPE、Attention、缓存与计分检查，PASS；这两题直接计入 2,374 条正式结果，没有重复生成。模型只加载一次。',
 'MME 已知图片差异保留：landmark/03d5e3bfc958be38.jpg，涉及 :0/:1 两题。恢复后的 SHA 与历史 SVH 本次使用的数据一致，未替换图片、未删题；与采用另一图片字节版本的外部结果不能视为全部输入逐字节一致。具体旧/新哈希见 input_audit.json。',
 '原始预测、QID、Gold、Prompt/像素绑定和选择索引保存于 predictions.sqlite / predictions.jsonl.gz；逐题 FULL 同步事务与 progress.json 支持断点恢复。全部结果重新读取后计分，独立复核 1,187 个双题组及 14 类加总。',
 '没有下载模型/数据，没有运行其他 Benchmark、其他算法、层搜索或性能实验。模型进程已结束、GPU 已释放，服务器未关闭；没有 GitHub Push。Windows 真实 SFTP + SHA256 备份回执见 windows_backup_receipt.json。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
names=['report_zh.md','score_summary.json','per_task_scores.csv','paired_questions.csv','input_audit.json','protocol.json','progress.json','input_manifest.jsonl.gz','SVH_reference_predictions.jsonl.gz','MME_integration_smoke.json','predictions.sqlite','predictions.jsonl.gz','prepare_mme.py','run_mme.py','finalize_mme.py']
save('artifact_manifest.json',dict(files={n:dict(bytes=(O/n).stat().st_size,sha256=sha(O/n)) for n in names},weights_datasets_images_excluded=True))
print(json.dumps({k:summary[k] for k in ['status','questions','pairs','scores','SVH_reference_scores','delta_vs_SVH','empty','truncated','parse_failed','paired','GPU_after']},ensure_ascii=False))
