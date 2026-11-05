"""CPU-only evidence extraction after frozen numerical gate failed; no GPU work."""
import pathlib,json,hashlib,csv,collections,datetime,subprocess,os,re
W=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer');O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer';P=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_physical_transfer')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def atomic(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def table(name,rows,fields):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
from vlm_research.qwen3vl_c3v2_physical.worker import score
m=read(O/'sample_manifest.json');smap={s['id']:s for s in m['samples']};bindings=read(O/'input_bindings.json');src=read(O/'source_manifest.json')
for p,h in src['files'].items():assert sha(W/p)==h,p
assert sha(O/'sample_manifest.json')==read(O/'protocol.json')['data_manifest_sha256']
assert sha(O/'input_bindings.json')==read(O/'protocol.json')['input_bindings_sha256']
hist=collections.defaultdict(set)
def walk(x,hint):
 if isinstance(x,dict):
  im=x.get('image_id',x.get('imageId'));p=str(x.get('image_path',''))+' '+str(x.get('dataset',''))+' '+hint
  dataset='GQA' if ('GQA' in p or 'gqa' in p.lower()) else 'VQAv2' if ('coco' in p.lower() or 'vqa_v2' in p.lower() or 'vqav2' in p.lower()) else None
  if im is not None and str(im).isdigit() and dataset:hist[dataset].add(str(int(im)))
  for v in x.values():walk(v,hint)
 elif isinstance(x,list):
  for v in x:walk(v,hint)
for source in read(O/'historical_manifest_sources.json'):
 p=pathlib.Path(source['path']);assert sha(p)==source['sha256'];walk(read(p),str(p))
id_overlap={d:sorted({s['image_id'] for s in m['samples'] if s['dataset']==d and str(int(s['image_id'])) in hist[d]}) for d in ['GQA','VQAv2']}
audit=read(O/'image_overlap_audit.json');audit['postfreeze_image_id_audit']=dict(historical_id_counts={d:len(v) for d,v in hist.items()},overlap_image_ids=id_overlap,note='Dataset labels inferred conservatively from manifest path and image path; SHA exclusion frozen pre-prediction; this audit never changes sample membership')
atomic(O/'image_overlap_audit.json',audit)
rows=[];physical=[];shape_rows=[];cases=[]
for rank in range(2):
 for p in sorted((O/f'shard{rank}/smoke').glob('*.json')):
  ck=read(p);s=ck['sample'];r=ck['result'];a=r['audit'];layout=bindings[s['id']]
  assert r['eos'] and not r['truncated'] and not r['empty'] and not r['physical_fallback']
  assert len([e for e in a['attention_events'] if e.get('residual_exact')])==36*len(r['token_ids'])
  assert a['deepstack']==[0,1,2]
  assert all(e['forbidden_probability']==0 for e in a['probability'])
  for layer in range(36):
   shape=[q for q in a['projection_shapes'] if q['layer']==layer]
   if r['condition'] in ['C','B'] and layer in range(1,7):
    assert all(not(q['step']>0 and q['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV']) for q in shape)
    assert [q['input'][1] for q in shape if q['step']==0 and q['op']=='q_proj']==[layout['input_length']-1]
    assert a['kv']['layers'][layer]['capacity_bytes']==0
   for step in range(len(r['token_ids'])):
    assert len([q for q in shape if q['step']==step and q['op']=='down_proj'])==1
  for q in a['layers']:
   expect=layout['input_length']-layout['image_tokens'] if r['condition'] in ['C','P'] and q['layer']>=22 else layout['input_length']
   assert q['hidden_rows']==(expect if q['step']==0 else 1)
   if q['step']==0 and r['condition'] in ['C','P'] and q['layer']>=22:assert q['logical_positions']==[i for i in range(layout['input_length']) if i not in set(layout['image_positions'])]
  rows.append(dict(phase='FAILED_SMOKE_ONLY',rank=rank,question_id=s['question_id'],dataset=s['dataset'],image_id=s['image_id'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=r['condition'],answer=r['text'],score=score(s,r['text']),token_ids=json.dumps(r['token_ids']),eos=r['eos'],truncated=r['truncated'],empty=r['empty'],formal_sample=False))
  physical.append(dict(rank=rank,question_id=s['question_id'],condition=r['condition'],input_rows=layout['input_length'],image_rows=layout['image_tokens'],L21_rows=next(q['hidden_rows'] for q in a['layers'] if q['step']==0 and q['layer']==21),L22_rows=next(q['hidden_rows'] for q in a['layers'] if q['step']==0 and q['layer']==22),language_matrix_gflops=sum(q['flops'] for q in a['projection_shapes'])/1e9,kv_live_bytes=a['kv']['live_bytes'],kv_capacity_bytes=a['kv']['capacity_bytes'],attention_skipped_decode_layers=sum(x['capacity_bytes']==0 for x in a['kv']['layers']),forbidden_probability_max=max(x['forbidden_probability'] for x in a['probability']),formal_performance=False))
  for q in a['projection_shapes']:shape_rows.append(dict(rank=rank,condition=r['condition'],question_id=s['question_id'],**{k:json.dumps(v) if isinstance(v,list) else v for k,v in q.items()}))
table('per_question.csv',rows,list(rows[0]))
table('smoke_physical_execution.csv',physical,list(physical[0]))
table('actual_matrix_shapes.csv',shape_rows,['rank','condition','question_id','layer','step','op','input','output','query','key','value','flops'])
table('accuracy_summary.csv',[dict(dataset=d,condition=c,n=0,accuracy='',status='NOT_RUN_NUMERICAL_GATE_FAILED') for d in ['GQA','VQAv2'] for c in ['N','F','P','B','C']],['dataset','condition','n','accuracy','status'])
table('paired_comparisons.csv',[],['dataset','comparison','n','delta_pp','ci95_low','ci95_high','multiplicity','status'])
table('latency_flops_kv.csv',[dict(dataset=d,condition=c,status='NOT_RUN_NUMERICAL_GATE_FAILED',e2e_ms='',speedup='',matrix_flops='',kv_live_bytes='',kv_capacity_bytes='') for d in ['GQA','VQAv2'] for c in ['N','F','P','B','C']],['dataset','condition','status','e2e_ms','speedup','matrix_flops','kv_live_bytes','kv_capacity_bytes'])
table('negative_cases.csv',[dict(kind='NUMERICAL_VALIDATION_FAILURE',rank=i,question_id='06905471',condition_pair='F_vs_C',step=0,max_abs=.4453125,mean_abs=.08348099887371063,mean_abs_limit=.02,top1_equal=True) for i in range(2)],['kind','rank','question_id','condition_pair','step','max_abs','mean_abs','mean_abs_limit','top1_equal'])
atomic(O/'physical_execution_audit.json',dict(status='OBSERVED_ON_ONE_SMOKE_IMAGE_ONLY_NOT_ACCEPTED_ENGINE',distinct_images=1,gpu_replicates=2,actual_hidden_exit=True,actual_qkv_attention_output_skip=True,skip_layers=list(range(1,7)),all36_mlp_residual_checks=True,deepstack_injections=[0,1,2],original_logical_positions_preserved=True,compact_text_kv=True,skipped_layer_kv_absent=True,not_proven=['numerical equivalence','long-input smoke','pilot quality','formal quality','speedup','cross-model algorithm viability'],data=physical,flops_scope='Actual language linear GEMMs + QK/AV only, excludes vision encoder, LM head and non-matmul operations. Not full-model FLOPs or hardware counters. Shapes captured under audit, never used as latency benchmark.'))
gate=read(O/'STOP.json')
atomic(O/'gpu_smoke_validation.json',dict(status='FAILED_STOPPED',source_commit=src['commit'],requested_cases_per_gpu=3,completed_accepted_cases_per_gpu=0,first_case='GQA__06905471',checks_reached=dict(native_noop='Strict assertions completed before F/P/B/C loop on both GPUs; vectors not archived, code/control-flow evidence only',native_hf='Strict generation token and effective-logit assertions completed before F/P/B/C loop on both GPUs; vectors not archived',physical_shapes='All six nonnative conditions saved for first image on both GPUs'),failed_gate=dict(comparison='F_vs_C',step=0,max_abs=.4453125,max_abs_limit=.5,mean_abs=.08348099887371063,mean_abs_limit=.02,top1_equal=True,complete_token_ids_F=[2152,151645],complete_token_ids_C=[2152,151645]),not_reached=['remaining two smoke cases per GPU','B-vs-F and P-vs-W22 numerical checks','pilot','formal quality','performance'],thresholds_changed=False,failure=gate))
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert 'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6' not in procs and 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185' not in procs
atomic(O/'gpu_release_audit.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_query=gpus,process_query=procs,released=True,new_gpu_tasks_after_failure=0,other_processes_killed=False))
atomic(O/'completion.json',dict(status='STOPPED_NUMERICAL_VALIDATION_FAILED',experiment_complete=False,delivery_complete=False,frozen_questions=collections.Counter(s['dataset'] for s in m['samples']),formal_generations_completed=0,formal_generations_planned=len(m['samples'])*5,saved_final_attempt_smoke_checkpoints=len(rows),unique_smoke_questions=1,gpu_smoke_accepted=0,pilot_completed=False,quality_completed=False,performance_completed=False,bootstrap_run=False,reason='F-vs-C mean absolute effective-logit error exceeds frozen gate on both GPUs',no_threshold_relaxation=True,no_quality_or_speed_claim=True,source_commit=src['commit']))
# Copy only small logs and reference reports; never copy models, images or arrays.
for p in P.glob('*.log'):
 if p.stat().st_size<1_000_000:
  target=O/'logs'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
for p in P.glob('*_reference.md'):
 target=O/'historical_reports'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())
print('CPU audit PASS',len(rows),'saved checkpoints; ID overlaps:',{d:len(x) for d,x in id_overlap.items()},flush=True)
