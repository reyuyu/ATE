"""Independent CPU scoring and three-condition input/KV audit, no GPU inference."""
import sys,json,gzip,csv,collections,datetime,subprocess,shutil,hashlib
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-svhr-efficient-attention-22875-20261113');B=O.parent/'qwen3vl-ocrbench-memory-efficient-attention-22875-20261113';L=O.parent/'ocrbench-ate-frozen-completion-22875-20261113';T=Path('/home/featurize/work/ATE/vlm_research/results/qwen3vl_ocrbench_svhr_efficient_attention');M=Path('/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct')
sys.path[:0]=[str(L),str(O/'sources'),str(O)]
import torch
from transformers import AutoTokenizer
from run_common import atomic,sha,score,RAW
assert not torch.cuda.is_initialized()
assert json.loads((O/'progress.json').read_text())['status']=='GENERATION_COMPLETE_AWAIT_CPU_AUDIT'
pid=int((O/'worker.pid').read_text());processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not any(x.split(',')[0].strip()==str(pid) for x in processes.splitlines())
atomic(O/'gpu_release.json',dict(status='RELEASED',worker_pid=pid,worker_absent=True,remaining_compute_processes=processes.splitlines(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip(),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),server_shutdown=False))
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');baseP=json.loads((B/'protocol.json').read_text())
assert P['base_protocol_sha256']==sha(B/'protocol.json')
protected=json.loads((O/'protected_sources.json').read_text());assert all(sha(p)==h for p,h in protected.items())
for f,h in P['new_source_sha256'].items():assert sha(O/f)==h
assert json.loads((B/'final_independent_audit.json').read_text())['status']=='PASS'
base_manifest=json.loads((B/'SOURCE_SHA256.json').read_text())
for name in ['Native_predictions_light.jsonl','Physical-Original_predictions_light.jsonl']:
 assert sha(B/name)==base_manifest[name]['sha256'],'Completed base prediction light changed after archive'
rows=json.loads((L/'results/qwen/rows.json').read_text());layouts=json.load(gzip.open(L/'results/qwen/cpu_layouts.json.gz','rt'));assert len(rows)==len(RAW)==len(layouts)==1000
tok=AutoTokenizer.from_pretrained(M,local_files_only=True);eos=json.loads((M/'generation_config.json').read_text())['eos_token_id'];eos={eos} if isinstance(eos,int) else set(eos)
base={c:{r['question_id']:r for r in map(json.loads,(B/(c+'_predictions_light.jsonl')).read_text().splitlines())} for c in ['Native','Physical-Original']}
assert all(len(x)==1000 for x in base.values())
files=sorted((O/'predictions').glob('*.json'));assert len(files)==1000
data={};audits=[];cats=collections.defaultdict(lambda:[0,0]);uuids=set();empty=trunc=EOS=0;peak=0;patterns=collections.Counter()
with (O/'SVH-R_predictions_light.jsonl').open('w',encoding='utf-8') as out,(O/'inputs_light.jsonl').open('w',encoding='utf-8') as inputsout:
 for i,f in enumerate(files):
  q=f'{i:04d}';r=json.loads(f.read_text());s=rows[i];g=r['generation'];l=layouts[q];a=g['physical_audit'];kv=g['kv'];count=g['generated_tokens'];n=l['input_length'];v=l['image_tokens'];nt=n-v
  assert r['question_id']==f.stem==s['question_id']==q and r['protocol_sha256']==pb
  assert r['input_layout_sha256']==P['all_layout_hashes'][q]==baseP['all_layout_hashes'][q]
  assert r['input_tensor_hashes']==l['input_tensor_hashes'] and r['image_sha256']==s['image_sha256'] and r['request_sha256']==s['request_SHA256']
  assert s['question']==RAW[i]['question'] and s['answer']==RAW[i]['answers'] and s['image_id']==RAW[i]['image_path']
  for b in base.values():
   assert b[q]['input_tensor_hashes']==r['input_tensor_hashes'] and b[q]['image_sha256']==r['image_sha256'] and b[q]['input_layout_sha256']==r['input_layout_sha256']
   expected_gold=RAW[i]['answers'] if isinstance(RAW[i]['answers'],list) else [RAW[i]['answers']]
   assert b[q]['gold']==expected_gold
  assert g['condition']=='SVH-R' and g['backend']=='SDPA_EFFICIENT_ATTENTION_ONLY' and g['finite_logits'] and g['independent_cache'] and g['initial_cache_none'] and g['natural_eos'] and not g['teacher_forced_reference']
  assert 0<count<=128==g['max_new_tokens'] and len(g['token_ids'])==count
  assert tok.decode(g['token_ids'],skip_special_tokens=True)==g['text']
  assert g['eos']==(g['token_ids'][-1] in eos) and g['truncated']==(not g['eos']) and g['empty']==(not g['text'].strip())
  assert g['request_cache_uuid'] not in uuids;uuids.add(g['request_cache_uuid'])
  def norm(x):
   x=str(x).strip().replace('\n',' ')
   return x.replace(' ','') if RAW[i]['dataset_name']=='HME100k' else x.lower()
  gold=RAW[i]['answers'];gold=gold if isinstance(gold,list) else [gold]
  independent=int(any(norm(x) in norm(g['text']) for x in gold));official=score(q,g['text'])['official_score'];assert independent==official==r['official_score']==r['independent_score']
  assert not a['physical_fallback'] and a['deepstack']==[0,1,2] and a['read_layers']==list(range(19,36))
  assert a['frozen_forward_and_SelectedKV_unchanged'] and a['capture_bridge_only'] and not a['dense_attention_scores_materialized']
  assert [x['layer'] for x in a['builder_layers']]==list(range(22,36)) and [x['layer'] for x in a['builder_checks']]==list(range(22,35))
  assert all(x['hidden_rows']==l['image_positions'][-1]+1 and x['image_rows']==v for x in a['builder_layers'])
  assert len(a['checks'])==30*count+6 and all(x['key_permissions_exact'] for x in a['checks']+a['builder_checks'])
  image_set=set(l['image_positions']);text_positions=[p for p in range(n) if p not in image_set]
  for observed in a['layers']:
   step=observed['step'];layer=observed['layer']
   positions=(list(range(n)) if layer<22 else text_positions) if step==0 else [n+step-1]
   assert observed['logical_positions']==positions and observed['hidden_rows']==len(positions)
  assert kv['persistent_image_bytes']==17*v*4096 and not kv['late_KV_not_consumed']
  assert len(kv['layers'])==36
  for z in kv['layers']:
   layer=z['layer'];skip=layer in range(1,7);visual=v if layer>=19 else 0;used=0 if skip else nt+visual+count-1;cap=0 if skip else nt+visual+127
   assert z['persistent_image_tokens']==visual and z['live_tokens']==used and z['capacity_tokens']==cap
   assert z['live_bytes']==used*4096 and z['capacity_bytes']==cap*4096
  skip=[z for z in a['attention_events'] if 'skipped_rows' in z];assert len(skip)==6*count
  assert all(z['layer'] in range(1,7) and z['qkv_calls']==(1 if z['step']==0 else 0) and z['o_proj_calls']==(1 if z['step']==0 else 0) for z in skip)
  assert all('role' in z for z in g['attention_calls'] if z['stage']=='Language')
  checks=collections.Counter(z['role'] for z in g['attention_calls'] if z['stage']=='Language');assert checks['VISUAL_KV_BUILDER']==13
  peak=max(peak,g['memory']['peak_allocated']);empty+=g['empty'];trunc+=g['truncated'];EOS+=g['eos'];cats[RAW[i]['type']][0]+=official;cats[RAW[i]['type']][1]+=1;patterns[g['text'].strip()]+=1
  light=dict(condition='SVH-R',question_id=q,annotation_id=RAW[i]['id'],image_sha256=r['image_sha256'],input_layout_sha256=r['input_layout_sha256'],input_tensor_hashes=r['input_tensor_hashes'],protocol_sha256=pb,model_revision=P['revision'],attention_backend=g['backend'],category=RAW[i]['type'],gold=gold,prediction=g['text'],token_ids=g['token_ids'],official_score=official,independent_score=independent,empty=g['empty'],truncated=g['truncated'],EOS=g['eos'],peak_allocated_GiB=g['memory']['peak_allocated']/2**30,visual_KV_live_MiB=kv['persistent_image_bytes']/2**20)
  out.write(json.dumps(light,ensure_ascii=False)+'\n');data[q]=light
  inputsout.write(json.dumps(dict(question_id=q,dataset_name=RAW[i]['dataset_name'],category=RAW[i]['type'],question=s['question'],request_body=s['body'],prompt=l['prompt'],image_id=s['image_id'],image_sha256=s['image_sha256'],input_length=n,image_tokens=v,image_grid_thw=l['image_grid_thw'],input_tensor_hashes=l['input_tensor_hashes'],input_layout_sha256=r['input_layout_sha256'],Gold_passed_to_model=False),ensure_ascii=False)+'\n')
  audits.append(dict(question_id=q,category=RAW[i]['type'],image_sha256=r['image_sha256'],input_layout_sha256=r['input_layout_sha256'],Native_score=base['Native'][q]['official_score'],Original_score=base['Physical-Original'][q]['official_score'],SVHR_score=official,SVHR_output=g['text'],SVHR_empty=g['empty'],SVHR_truncated=g['truncated'],SVHR_EOS=g['eos'],input_hashes_match_all_three=True,independent_score_agrees=True,physical_checks_PASS=True))
assert len(uuids)==1000 and len({x['image_sha256'] for x in data.values()})==930
all_data=dict(base,**{'SVH-R':data});summaries=[]
for c,d in all_data.items():
 summaries.append(dict(method=c,score_out_of_1000=sum(r['official_score'] for r in d.values()),coverage=len(d),expected=1000,empty_outputs=sum(r['empty'] for r in d.values()),truncated=sum(r['truncated'] for r in d.values()),natural_EOS=sum(r['EOS'] for r in d.values()),peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in d.values()),generation_source='reused_completed_new_backend' if c!='SVH-R' else 'new_SVHR_only'))
pairs=[]
for x,y in [('Physical-Original','Native'),('SVH-R','Native'),('SVH-R','Physical-Original')]:
 d1=all_data[x];d0=all_data[y];up=sum(d1[q]['official_score']>d0[q]['official_score'] for q in data);down=sum(d1[q]['official_score']<d0[q]['official_score'] for q in data)
 pairs.append(dict(comparison=x+' - '+y,score_delta=up-down,accuracy_delta_pp=(up-down)/10,improved=up,worsened=down,both_correct=sum(d1[q]['official_score']==d0[q]['official_score']==1 for q in data),both_wrong=sum(d1[q]['official_score']==d0[q]['official_score']==0 for q in data)))
categories=[]
for c,d in all_data.items():
 cc=collections.defaultdict(lambda:[0,0])
 for r in d.values():cc[r['category']][0]+=r['official_score'];cc[r['category']][1]+=1
 for category,(correct,count) in cc.items():categories.append(dict(method=c,category=category,score=correct,count=count))
for name,values in [('threeway_scores.csv',summaries),('paired_comparisons.csv',pairs),('threeway_categories.csv',categories),('per_question_audit.csv',audits)]:
 with (O/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,values[0].keys());w.writeheader();w.writerows(values)
for c in ['Native','Physical-Original']:shutil.copyfile(B/(c+'_predictions_light.jsonl'),O/(c+'_predictions_light.jsonl'))
gate=json.loads((O/'gate.json').read_text());num=json.loads((O/'numerical_comparison.json').read_text())
audit=dict(status='PASS',coverage_each=1000,unique_original_images=930,SVHR_scores_official_independent_equal=1000,SVHR_text_token_decode_exact=1000,matched_input_hashes_all_three=1000,independent_KV_per_request=True,SVHR_main_and_auxiliary_structure_all_PASS=True,protected_base_and_frozen_files_unchanged=len(protected),new_Native_or_Original_generation=0,summaries=summaries,paired=pairs,SVHR_physical_gate=gate['status'],SVHR_strict_numerical_gate=num['strict_gate'],base_Original_strict_numerical_gate='FAIL_UNCHANGED',CUDA_initialized=False,SVHR_top_repeated_outputs=patterns.most_common(8))
atomic(O/'final_independent_audit.json',audit)
lines=['# Qwen3-VL OCRBench1000：补充冻结 SVH-R 与三组对照','','状态：**COMPLETE_WITH_NUMERICAL_WARNING**。只新增 SVH-R；已完成的 Native/Physical-Original 原始预测和归档保持不变，未重复生成。','', '| 方法 | 得分 /1000 | 覆盖 | 空输出 | 截断 | 自然EOS | 峰值分配 GiB |','|---|---:|---:|---:|---:|---:|---:|']
for z in summaries:lines.append(f'| {z["method"]} | {z["score_out_of_1000"]} | {z["coverage"]}/1000 | {z["empty_outputs"]} | {z["truncated"]} | {z["natural_EOS"]} | {z["peak_allocated_GiB"]:.3f} |')
lines+=['','Native/Original 来自本轮内存高效 Attention 已完成实验；三组共用同一原1000题、930张图、Question/Gold/顺序、四项输入Tensor哈希、全分辨率Processor、Prompt、BF16与新Efficient后端。不是旧Eager结果，也不是混入其他分辨率的新协议。','','| 配对比较 | 分差 | Accuracy差 pp | 错→对 | 对→错 |','|---|---:|---:|---:|---:|']
for p in pairs:lines.append(f'| {p["comparison"]} | {p["score_delta"]:+d} | {p["accuracy_delta_pp"]:+.1f} | {p["improved"]} | {p["worsened"]} |')
lines+=['','## 冻结算法与真实物理检查','','Qwen SVH-R 使用0-based **L19–L35**回看，沿用原写入L0–L21和L22前视觉Hidden退出、L1–L6答案Attention跳算。它不是InternVL Late14或LLaVA L11–L24配置。主路径L22后只有文本Hidden；晚层视觉KV从实际L22输入的因果图像前缀构造：L22–L34共13个完整块，L35只计算Q/K/V与RoPE并截获KV，随后释放临时视觉Hidden。无第二次Vision/DeepStack，不把辅助分支开销省略成零。','','持久原图KV仅存在L19–L35共17层；其余13个非跳算层只存文本KV，6个跳算层无持久KV。KV Live和Storage Capacity分开审计，每题缓存独立。所有1000题实际Q/K长度、Key访问集合、视觉退出、KV合并位置与文本Token顺序检查通过。不是Dense Mask假物理压缩。','','原 Physical.forward、SelectedKV、Original.context 源码原样继承。唯一新适配：将冻结构造器原本仅拦截Eager的KV捕获钩子桥接至本轮SDPA调度，之后调用与Native/Original相同的Efficient-only算子。原权重、层集合、Mask逻辑值和图像预算未改变。GitHub原始physical.py/engine.py与服务器快照逐字节哈希核验相同。','','## Smoke与数值限制','',f'0308、0601、0738三题完整自然EOS Smoke保留并直接计入1000题，另对0308做一次同前缀冻结Functional-R诊断，不计入正式分数。SVH-R Gate为 **{gate["status"]}**，原严格阈值max_abs≤0.5、mean_abs≤0.02；本次严格数值门禁为 **{num["strict_gate"]}**，短题全部Greedy Token一致。逐步误差见numerical_comparison.json。', '', '此前Original对Dense-F的严格数值FAIL保留不变；其38分严重退化不能因本次回看结果而隐藏或改写。结构通过/Greedy短样本一致不证明全局数值等价；没有全量Functional-R或Eager对照，不能将所有质量变化归因于某一单因素。', '', '0738保留4961×7016原图，grid=[1,304,214]，16264个LM视觉Token，总长16307；没有降分辨率、跳题或失败补零。三组所用实际Attention为aten::_scaled_dot_product_efficient_attention / CUTLASS，Math/Flash/Cudnn后端禁用。Smoke保存真实CUDA算子证据。','','## 协议与可追溯性','',f'- 模型：{P["model"]}@{P["revision"]}，复用本地已校验权重。', '- RTX3090 24GB；PyTorch2.6.0+cu124、Transformers4.57.6；BF16、Batch1、Greedy、自然EOS、上限128、logits_to_keep=1；原单线程与TF32关闭。',f'- 冻结SVH-R源码Commit：{P["source_commit"]}。本次隔离适配与执行源码SHA见protocol.json和SOURCE_SHA256.json；工作区HEAD e281754ef17b3accc09d6938b5b9f020048b2aec没有修改。', '- 1000题官方OCRBench v1评分与独立注释逻辑逐题完全一致，Token IDs重新解码与保存文本一致；全部错误/空输出/截断原样保留。', '- 逐题轻量预测、Gold、输入哈希、类别分数和配对表都保存；完整Attention/KV审计、日志与Gate Logits保存在本地独立目录及压缩归档。', '- 输入和Gold分离，Gold未传入模型。主请求每题独立KV；三次Smoke直接复用，仅997条后续正式生成，无Native/Original重跑。', '', '## 保存与资源','', '模型、图片、环境及实际运行输出在本地ext4盘；持续盘仅保存轻量证据和压缩checkpoint/归档。新SVH-R目录与此前Base目录独立，旧完整预测和archive的SHA不变。', '', 'GPU实验进程已退出，GPU释放证据见gpu_release.json。Windows实传与逐文件SHA校验见windows_backup_receipt.json；GitHub提交由后续独立分支发布回执确认，报告生成时不虚构Commit。未修改main，未关闭服务器，未启动其他Benchmark或效率实验。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (O/'report_zh.md').open('a',encoding='utf-8') as f:
 f.write('\n检查覆盖说明：Gate检查全部文本Query和固定的视觉Query；正式逐题使用固定的文本/视觉Query子集验证可见Key集合，并核对实际Q/K长度与完整持久KV，不声称逐题比较全量Attention矩阵或全局数值等价。Gate保存的全部主路径绝对位置索引已在CPU复核。耗时包含验收审计，未作为标准效率或TPOT实验成绩。\n')
meta=json.loads((O/'run_metadata.json').read_text());meta.update(status='COMPLETE_WITH_NUMERICAL_WARNING',CPU_final_audit='PASS',SVHR_score=summaries[2]['score_out_of_1000'],new_formal_generations=997,reused_SVHR_smoke=3,diagnostic_generations=1,GPU_released=True);atomic(O/'run_metadata.json',meta)
atomic(O/'progress.json',dict(status='COMPLETE_WITH_NUMERICAL_WARNING',completed=1000,expected=1000,CPU_final_audit='PASS',GPU_released=True,score=summaries[2]['score_out_of_1000']))
for p in O.iterdir():
 if p.is_file() and p.stat().st_size<16*2**20:shutil.copyfile(p,T/p.name)
assert not torch.cuda.is_initialized();print(json.dumps(audit,ensure_ascii=False));print('SVHR_CPU_FINAL_AUDIT_PASS_GPU_RELEASED')
