"""CPU audits, compact paired tables, compressed full evidence and report."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1')
from pathlib import Path
from datetime import datetime,timezone
import sys,json,gzip,hashlib,sqlite3,csv,io,statistics,subprocess,shutil,platform,math
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_quick_reproduction';D=Path('/home/featurize/data/ate-recovery');sys.path[:0]=[str(O),str(R)]
import torch,transformers
from transformers import AutoTokenizer,AutoConfig
from vlm_research.information_horizon_alignment import score as scorer
from vlm_research.gqa_testdev_c3.common import prediction

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def gzsave(n,rows):
 raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode();target=O/n
 if target.exists():assert gzip.decompress(target.read_bytes())==raw,n
 else:target.write_bytes(gzip.compress(raw,mtime=0))
def csvsave(n,rows):
 if not rows:return
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(name):
 out=[]
 for key,raw,h in db.execute('SELECT key,payload,sha256 FROM '+name):
  r=json.loads(raw);assert digest(r)==h;out.append(r)
 return out

state=read(O/'progress.json');p=read(O/'protocol.json');config=read(O/'execution_config.json')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Wait for GPU release'
for name,h in config['sources'].items():assert sha(O/name)==h
for name,h in p['frozen_ATE_sources'].items():assert sha(R/name)==h
assert sha(O/'protocol.json')==config['protocol_sha256'] and sha(O/'input_manifest.jsonl.gz')==p['manifest_sha256']
with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:samples={s['key']:s for s in map(json.loads,f)}
db=sqlite3.connect('file:'+str(O/'results.sqlite')+'?mode=ro',uri=True);pred=table('predictions');timing=table('timing');counts=table('counts');db.close()
scorer.AUTHOR=D/'sources/Information-Horizon';e=scorer.text_evaluator();tok=AutoTokenizer.from_pretrained(p['model_path'],local_files_only=True)
light=[];accuracy=[];audit={'status':'PASS','records':len(pred),'independent_score_mismatches':0,'input_binding_errors':0,'token_decode_errors':0,'CPU_only':True}
for r in pred:
 s=samples[r['key']];g=r['generation'];assert r['sample_sha256']==s['sample_sha256'] and r['input_hashes']==s['input_hashes']
 text=tok.decode(g['token_ids'],skip_special_tokens=True).strip();assert text==g['text'] and g['all_logits_finite'] and g['vision_forward_calls']==1
 if s['benchmark']=='GQA':
  parsed=text
  while parsed.endswith('.'):parsed=parsed[:-1]
  parsed=parsed.lower();score=float(parsed==s['answer']);assert prediction(text)==parsed
 else:
  parsed=e.answer_processor(text);answers=[e.answer_processor(a) for a in s['answer']];n=answers.count(parsed)
  score=sum(min((n-int(a==parsed))/3,1) for a in answers)/10;assert abs(score-scorer.score(s,text)['score'])<1e-12
 assert abs(score-r['score'])<1e-12 and r['parsed']==parsed
 light.append({'key':s['key'],'benchmark':s['benchmark'],'question_id':s['question_id'],'condition':r['condition'],'image_sha256':s['image_sha256'],'prompt_sha256':s['prompt_sha256'],
               'gold':s['answer'],'raw_output':text,'token_ids':g['token_ids'],'parsed':parsed,'score':score,'empty':g['empty'],'truncated':g['truncated'],'sample_sha256':s['sample_sha256']})
for bench in ('GQA','TextVQA'):
 base=None
 for method in ('Native','V2Drop'):
  rr=[r for r in light if r['benchmark']==bench and r['condition']==method];assert len({r['key'] for r in rr})==len(rr)
  value=100*sum(r['score'] for r in rr)/len(rr) if rr else None
  if method=='Native':base=value
  accuracy.append({'benchmark':bench,'condition':method,'questions':len(rr),'score_sum':sum(r['score'] for r in rr),'accuracy_percent':value,
                   'delta_Native_pp':None if value is None or base is None else value-base,'empty':sum(r['empty'] for r in rr),'truncated':sum(r['truncated'] for r in rr),
                   'metric':'Official exact-match Accuracy' if bench=='GQA' else 'Ten-answer VQA soft Accuracy'})
csvsave('accuracy_summary.csv',accuracy);gzsave('predictions_full.jsonl.gz',pred)
(O/'predictions_light.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in light))
(O/'inputs_light.jsonl').write_text(''.join(json.dumps({k:s[k] for k in ['key','question_id','benchmark','selection_rank','selection_hash','image_id','image_sha256','prompt','prompt_sha256','answer','input_view','input_hashes','sample_sha256']},ensure_ascii=False,separators=(',',':'))+'\n' for s in samples.values()))
cfg=AutoConfig.from_pretrained(p['model_path'],local_files_only=True);lc=cfg.text_config;vc=cfg.vision_config
d=lc.hidden_size;inner=lc.intermediate_size;vocab=lc.vocab_size;head=d//lc.num_attention_heads;kdim=lc.num_key_value_heads*head
vision_tokens=(vc.image_size//vc.patch_size)**2+1;vd=vc.hidden_size;vi=vc.intermediate_size
expected_vision=(2*3*vc.patch_size**2*vd*(vision_tokens-1)+vc.num_hidden_layers*(8*vision_tokens*vd**2+4*vision_tokens*vd*vi+4*vision_tokens**2*vd)
                 +2*(vision_tokens-1)*(vd*d+d*d))
countmap={(r['key'],r['condition']):r for r in counts};flat=[];layouts=[];rawtime=[];flop_audits=[]
for r in timing:
 key,method,length=r['key'],r['condition'],r['length'];g=r['result'];assert len(g['token_ids'])==length and g['counter_enabled'] is False
 assert r['input_hashes']==samples[key]['input_hashes'] and g['all_logits_finite']
 if (key,method) not in countmap:continue
 counted=countmap[key,method]['result'];assert counted['token_ids'][:length]==g['token_ids']
 c=counted['counter_checkpoints'][str(length)];assert c['full_request_flops']==c['prefill_flops']+c['decode_flops'] and c['prefill_flops']==c['vision_flops']+c['language_prefill_flops']
 expected_language=2*d*vocab*length+head*(len(samples[key]['input_ids'])+length-1)
 for x in c['shape_audit']:
  hidden=x['prefill']['hidden']+length-1
  expected_language+=2*d*(d*x['q_projection_tokens']+kdim*(x['k_projection_tokens']+x['v_projection_tokens'])+d*x['total_Q'])+6*d*inner*hidden+4*d*x['total_QK']
 flop_audits.append({'key':key,'condition':method,'length':length,'observed_vision':c['vision_flops'],'expected_vision':expected_vision,
                     'observed_language':c['full_request_flops']-c['vision_flops'],'expected_language':expected_language})
 assert expected_language==c['full_request_flops']-c['vision_flops'],'Independent language FLOPs differ'
 assert expected_vision==c['vision_flops'],'Independent vision/projector FLOPs differ'
 assert sum(c['operator_flops'].values())==c['full_request_flops']
 kv=g['cache'];assert kv['live_bytes']==c['cache']['live_bytes'] and kv['visual_live_bytes']==c['cache']['visual_live_bytes']
 layout=json.loads(json.dumps(c['cache']))
 image_positions={i for i,v in enumerate(samples[key]['input_ids']) if v==32000}
 for current,real in zip(layout['layers'],kv['layers']):
  assert current['used_tokens']==real['used_tokens'] and len(current['original_positions'])==current['used_tokens']
  observed_visual=sum(pos in image_positions for pos in current['original_positions'])
  assert observed_visual==current['visual_tokens']==real['visual_tokens']
  if current['used_tokens']:assert current['visual_live_bytes']==observed_visual*(current['live_bytes']/current['used_tokens'])
  current['storage_bytes']=real['storage_bytes'];current['capacity_tokens']=real['capacity_tokens']
 layout['storage_bytes']=kv['storage_bytes']
 layouts.append({'key':key,'condition':method,'length':length,'cache':layout,'shapes':c['shape_audit'],
                 'source':'Actual counter prefix shapes/positions, with storage capacity from the independent matching-length timing request'})
 row={'key':key,'benchmark':r['benchmark'],'condition':method,'length':length,'full_request_TFLOPs':c['full_request_flops']/1e12,
      'prefill_TFLOPs':c['prefill_flops']/1e12,'vision_TFLOPs':c['vision_flops']/1e12,'decode_TFLOPs':c['decode_flops']/1e12,
      'persistent_KV_live_MiB':kv['live_bytes']/2**20,'KV_storage_MiB':kv['storage_bytes']/2**20,'visual_KV_residency_percent':kv['visual_KV_residency_percent'],
      'TTFT_ms':g['ttft_ms'],'GPU_prefill_ms':g['gpu_prefill_ms'],'Decode_TPOT_ms':(g['E2E_ms']-g['ttft_ms'])/(length-1),'Decode_GPU_TPOT_ms':g['decode_TPOT_ms'],'E2E_ms':g['E2E_ms'],
      'Peak_GPU_allocated_MiB':g['peak_allocated_MiB'],'Peak_GPU_reserved_MiB':g['peak_reserved_MiB'],
      'selection_hidden_gather_GPU_ms':sum(v['selection_and_hidden_gather_gpu_ms'] for v in g.get('selection_overhead_gpu_ms',[]))}
 assert 0<row['TTFT_ms']<row['E2E_ms'] and row['Decode_TPOT_ms']>0
 assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
 flat.append(row);rawtime.append(dict(row,step_gpu_ms=g['step_gpu_ms'],sample_sha256=r['sample_sha256'],input_hashes=r['input_hashes']))
csvsave('efficiency_per_input.csv',flat);gzsave('raw_timings.jsonl.gz',rawtime);gzsave('cache_layouts_and_shapes.jsonl.gz',layouts);gzsave('operator_counts_full.jsonl.gz',counts)
csvsave('independent_FLOPs_audit.csv',flop_audits)
metrics=[k for k in flat[0] if k not in ('key','benchmark','condition','length')] if flat else []
summaries=[]
for scope in ('ALL30','GQA','TextVQA'):
 for length in (32,128):
  for method in state.get('executed_efficiency_methods',[]):
   rr=[r for r in flat if r['length']==length and r['condition']==method and (scope=='ALL30' or r['benchmark']==scope)]
   if not rr:continue
   out={'scope':scope,'length':length,'condition':method,'inputs':len(rr)}
   for metric in metrics:out[metric]=statistics.mean(r[metric] for r in rr);out[metric+'_median']=statistics.median(r[metric] for r in rr)
   summaries.append(out)
csvsave('efficiency_summary.csv',summaries)
layer_rows=[]
for scope in ('GQA','TextVQA'):
 for method in state.get('executed_efficiency_methods',[]):
  ll=[r for r in layouts if samples[r['key']]['benchmark']==scope and r['condition']==method and r['length']==128]
  if not ll:continue
  for layer in range(32):
   row={'benchmark':scope,'condition':method,'layer':layer,'inputs':len(ll),
        'visual_hidden_prefill':statistics.mean(r['shapes'][layer]['prefill']['hidden']-(len(samples[r['key']]['input_ids'])-576) for r in ll),
        'persistent_visual_KV':statistics.mean(r['cache']['layers'][layer]['visual_tokens'] for r in ll),
        'mean_prefill_hidden_rows':statistics.mean(r['shapes'][layer]['prefill']['hidden'] for r in ll),
        'mean_prefill_attention_Q':statistics.mean(r['shapes'][layer]['prefill']['attention_Q'] for r in ll),
        'mean_prefill_attention_K':statistics.mean(r['shapes'][layer]['prefill']['attention_K'] for r in ll),
        'mean_last_decode_attention_K':statistics.mean(r['shapes'][layer]['last_decode']['attention_K'] for r in ll)}
   layer_rows.append(row)
csvsave('per_layer_summary.csv',layer_rows)
complete=state['status']=='GENERATIONS_AND_MEASUREMENTS_COMPLETE' and len(pred)==400 and len(timing)==state['efficiency_expected'] and len(counts)==state['counter_expected'] and len(flat)==len(timing)
if complete:
 assert {(r['key'],r['condition']) for r in pred}=={(k,m) for k in samples for m in ('Native','V2Drop')}
 assert {(r['key'],r['condition'],r['length']) for r in flat}=={(k,m,t) for k in p['efficiency_keys'] for m in state['executed_efficiency_methods'] for t in (32,128)}
 assert {(r['key'],r['condition']) for r in counts}=={(k,m) for k in p['efficiency_keys'] for m in state['executed_efficiency_methods']}
smoke=read(O/'smoke_summary.json') if (O/'smoke_summary.json').exists() else {'status':'NOT_VALIDATED'}
status='PASS' if complete and smoke['status']=='PASS' else 'PARTIAL' if pred or timing else 'BLOCKED'
audit['independent_FLOPs_cases']=len(flop_audits);audit['vision_and_language_FLOPs_mismatches']=0;save('independent_audit.json',audit)
frozen_dir=O/'frozen_scoring_sources';frozen_dir.mkdir(exist_ok=True);original=scorer.AUTHOR/'llava/eval/m4c_evaluator.py';copied=frozen_dir/'m4c_evaluator.py'
if copied.exists():assert sha(copied)==sha(original)
else:shutil.copy2(original,copied)
scorer_provenance={'TextVQA_external_evaluator':{'path':str(original),'sha256':sha(original),'artifact':str(copied.relative_to(O))},
                   'ATE_scoring_wrapper_sha256':sha(R/'vlm_research/information_horizon_alignment/score.py'),'GQA_scoring_wrapper_sha256':sha(R/'vlm_research/gqa_testdev_c3/common.py')}
save('scorer_provenance.json',scorer_provenance)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
metadata={'REPRODUCTION_STATUS':status,'finished_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(O/'protocol.json'),'execution_config_sha256':sha(O/'execution_config.json'),
          'accuracy':accuracy,'efficiency':[r for r in summaries if r['scope']=='ALL30'],'smoke':smoke,'execution':state,'independent_audit':audit,'GPU_after_exit':gpu,'scorer_provenance':scorer_provenance,'python':platform.python_version(),
          'software':{'torch':torch.__version__,'transformers':transformers.__version__,'CUDA_runtime':torch.version.cuda},
          'complete_official_benchmark_reproduction':False,'noninferiority_claim':False,'no_shutdown_performed':True,'other_experiments_started':False,
          'limitations':['Exploratory hash-selected100 per benchmark; not full official benchmark scores or statistical noninferiority.',
          'HF converted weights + Transformers4.49 eager carrier, not original4.37 native LLaVA environment; algorithm kernel reused from pinned official AST.',
          'FLOPs cover actual matrix products and convolutions, MAC=2; elementwise, norm, softmax, top-k/index/sort excluded. All their runtime remains included in latency.',
          'Selection timing includes L2/top-k/hidden gather; mask/RoPE gather overhead is included in TTFT/E2E but not separately isolated.',
          'Counter128 provides the exact first32 prefix; greedy IDs are checked against separate32/128 timing runs. Actual matching-length KV storage comes from timing runs.',
          'GPU-resident preprocessed inputs; disk I/O, image decoding, tokenizer and H2D excluded. Vision+projector and all real decode included. TTFT and E2E are wall times; reported TPOT=(E2E-TTFT)/(T-1), with CUDA step TPOT also retained separately.',
          'Two warmup requests per method; one timing observation per input/length, balanced method and length order; means/medians are descriptive, without a latency confidence claim.']}
save('run_metadata.json',metadata)
lines=['# V²Drop 快速复现','',f'**REPRODUCTION_STATUS：{status}**','',
 '固定官方Commit `2a9431150f8ace3a1d9abcdc96a7d46acca597df`；模型 `llava-hf/llava-1.5-7b-hf@'+p['revision']+'`，RTX4090 / FP16 / Eager / Transformers4.49 / batch1。',
 f"服务器Git HEAD：`{p['code_commit']}`；实际执行的冻结ATE源码以protocol.json中的逐文件SHA256为准，发布分支继承既有证据Commit `71a5762cd8ccb8cc99efe9b6b0267d8abef70c46`。Python {platform.python_version()} / Torch {torch.__version__} / CUDA {torch.version.cuda}。所有200题当前图像起点恰好为35；动态起点适配另经起点5/35的CPU检查，不声称这些真实输入存在起点错误。",
 '官方L2差分、top-k、L3/L17/L22、0.5/0.5/0和至少1 Token均保持不变；从固定原始源码AST直接提取选择代码，仅把4处起点35替换成真实图像起点。采用HF层前钩子收缩Hidden、Mask、原始位置和共享RoPE，使用原生DynamicCache的逐层不同长度，保留较早层KV；没有修改任何ATE冻结核心。',
 '官方环境为Torch2.1.2/Transformers4.37.2，README要求修补旧RoPE长度。当前HF4.49原生位置驱动RoPE保持原位置；不下载第二套模型、不安装新环境。',
 (f"Smoke：{smoke['status']}。4张真实图片，禁用剪枝时与Native Token及Logits完全一致；官方未改动固定35选择参考与动态起点选择完全一致；576→288→144→1、文本保留、RoPE、因果Mask和真实Decode/KV检查通过。CPU另测起点5/35的文本保护。" if smoke['status']=='PASS' else f"Smoke：{smoke['status']}；未通过时停止后续大量生成，见失败记录。"),'',
 '| Benchmark | 条件 | 题数 | Accuracy | 相对Native | 空输出 | 截断 |','|---|---|---:|---:|---:|---:|---:|']
for r in accuracy:lines.append(f"| {r['benchmark']} | {r['condition']} | {r['questions']} | {r['accuracy_percent']:.4f}% | {r['delta_Native_pp']:+.4f} pp | {r['empty']} | {r['truncated']} |" if r['accuracy_percent'] is not None else f"| {r['benchmark']} | {r['condition']} | 0 | 未完成 | — | — | — |")
lines+=['','TextVQA为Paper-OCR十答案VQA软分，不是普通二值准确率。每组同一图片、Prompt、Gold、输入张量哈希与Scorer；GQA100 Native与此前同4090/同官方协议完整结果逐Token一致。仅探索性样本，未复现官方完整成绩，也未证明统计非劣效。','',
 '效率样本：同一200题中的GQA前15+TextVQA前15（按固定SHA排名）；每方法先32/128各预热一次。每输入/长度计时一次，方法顺序与长度顺序均交错；下表为30输入算术均值。固定32/128输出包含1次Prefill+31/127次Decode，忽略EOS，和正常Accuracy生成分开。',
 'Prefill列包含Vision+Projector，Full=Prefill+Decode；统一Torch实际算子计数（矩阵乘/卷积，MAC=2），并保存每层真实Q/K/V形状和持久KV布局。另按真实逐层投影Token数、Q×K及模型维度独立复算全部Language FLOPs，按CLIP与Projector维度复算Vision，逐项精确一致。计数独立运行128步，其前32步与单独32步计时的全部Token一致；计数开销未混入计时。元素操作/Norm/Softmax/排序/索引不计入FLOPs，运行时间全部计入。','']
for length in (32,128):
 lines += ['',f'## 固定{length}输出Token','', '| 方法 | Full TFLOPs | Prefill TFLOPs | Decode TFLOPs | KV Live MiB | 视觉KV% | TTFT ms | TPOT ms | E2E ms | Peak MiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in summaries:
   if r['scope']=='ALL30' and r['length']==length:lines.append(f"| {r['condition']} | {r['full_request_TFLOPs']:.4f} | {r['prefill_TFLOPs']:.4f} | {r['decode_TFLOPs']:.4f} | {r['persistent_KV_live_MiB']:.2f} | {r['visual_KV_residency_percent']:.4f} | {r['TTFT_ms']:.3f} | {r['Decode_TPOT_ms']:.3f} | {r['E2E_ms']:.3f} | {r['Peak_GPU_allocated_MiB']:.1f} |")
if complete:
 cells={(r['condition'],r['length']):r for r in summaries if r['scope']=='ALL30'}
 a,b=cells['Native',128],cells['V2Drop',128]
 percent=lambda new,old:100*(new/old-1)
 lines+=['',f"本轮128输出Token时，V²Drop相对Native的Full FLOPs减少{-percent(b['full_request_TFLOPs'],a['full_request_TFLOPs']):.2f}%，Prefill FLOPs减少{-percent(b['prefill_TFLOPs'],a['prefill_TFLOPs']):.2f}%，Decode FLOPs仅减少{-percent(b['decode_TFLOPs'],a['decode_TFLOPs']):.2f}%。TTFT均值下降{-percent(b['TTFT_ms'],a['TTFT_ms']):.2f}%，但E2E均值增加{percent(b['E2E_ms'],a['E2E_ms']):.2f}%；32输出Token时E2E均值下降{-percent(cells['V2Drop',32]['E2E_ms'],cells['Native',32]['E2E_ms']):.2f}%。本轮未观察到明显完整请求加速；这些是一次固定样本测量的描述统计。选择及Hidden gather的GPU时间均值为32输出{cells['V2Drop',32]['selection_hidden_gather_GPU_ms']:.3f} ms、128输出{b['selection_hidden_gather_GPU_ms']:.3f} ms。"]
lines+=['','V²Drop逐层持久视觉KV为L0–2:576，L3–16:288，L17–21:144，L22–31:1；合计6490，Native18432，实际Residency约35.21%。最后只剩1个视觉Hidden不表示历史层KV只剩1个。C3的视觉KV为0；Reaccess只在L11–24保留，共14层、43.75%。',
 'Token选择额外时间在efficiency_summary.csv的selection_hidden_gather_GPU_ms；其余位置/Mask收缩包含在TTFT/E2E中，未单独隔离。TTFT、E2E为墙钟时间；主表TPOT=(E2E−TTFT)/(输出Token数−1)，CUDA逐步TPOT另列保存。Peak列为PyTorch峰值allocated显存（含模型），reserved另存CSV。GPU输入已预处理，计时不含图片读取、Tokenizer及H2D；包含真实Vision、投影、Prefill、Decode、冷请求控制器与KV构建。',
 'ATE效率两组直接调用冻结MiddleWriter，未改层集合或算法；四方法同一输入、H1最后位置LM Head、同循环、同计数器，适合本轮同硬件效率对照。此次没有新增ATE Accuracy实验，不能将200题V²Drop分数与ATE全量成绩直接配对。',
 f"GPU进程已结束：`{gpu}`。模型加载次数：{state.get('model_loads')}；未启动其他研究任务、未关闭服务器。",'',
 '完整输入Manifest、预测、逐层布局与算子计数压缩保留本地并备份Windows；GitHub仅发布轻量源码、来源与许可证、Smoke、200题双条件预测、配置、统计及原始计时。windows_backup_receipt.json随结果发布；github_sync_receipt.json与completion_receipt.json在归档同步阶段生成并保存于服务器和Windows目录，实际Commit链接由最终汇报提供。','',
 '[固定官方源码](https://github.com/xuyang-liu16/V2Drop/blob/2a9431150f8ace3a1d9abcdc96a7d46acca597df/llava/model/language_model/V2Drop.py)。']
if state.get('failure'):lines+=['','失败记录：`'+state['failure']['error']+'`；没有通过修改剪枝层/比例/模型/Backend追求通过。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
print('FINALIZED',status,json.dumps(accuracy,ensure_ascii=False),flush=True)
