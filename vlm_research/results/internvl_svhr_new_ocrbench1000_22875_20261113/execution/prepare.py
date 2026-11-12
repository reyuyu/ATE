"""Freeze the user-named Late14 configuration and1000 original OCRBench inputs."""
import sys,json,shutil,gzip,zipfile
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-ocrbench1000-22875-20261113')
L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
H=Path('/home/featurize/ate-experiments/internvl3-ocrbench-late14-22875-20261113')
sys.path.insert(0,str(L));from run_common import atomic,sha,RAW,D,gpu_idle,score
assert not (O/'protocol.json').exists();print('GPU_BEFORE',gpu_idle())
old=json.loads((H/'protocol.json').read_text());rows=json.loads((L/'results/internvl3/frozen_rows.json').read_text());paths=json.loads((L/'results/internvl3/input_paths.json').read_text())
assert len(rows)==len(RAW)==1000
assert sha(D/'OCRBench.json')=='df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864'
for i,r in enumerate(rows):
 raw=RAW[i];assert r['question_id']==f'{i:04d}' and r['official_id']==raw['id'] and r['question']==raw['question'] and r['gold']==raw['answers'] and r['category']==raw['type']
 assert r['max_new_tokens']==128 and len(r['roles']['image'])==256 and len(r['input_ids'])==r['input_tokens']
 assert Path(paths[r['image_sha256']]).is_file()
assert len({r['image_sha256'] for r in rows})==930
inherited={};inherited_sha={}
for f in sorted((H/'predictions/Late14_L14_27').glob('*.json')):
 x=json.loads(f.read_text());q=x['question_id'];r=rows[int(q)]
 assert x['protocol_sha256']==sha(H/'protocol.json') and x['generation']['reaccess_layers']==list(range(14,28)) and x['generation']['cutoff']==28
 assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
 assert score(q,x['generation']['text'])['official_score']==x['official_score']==x['independent_score']
 inherited[q]=x;inherited_sha[q]=dict(path=str(f),sha256=sha(f))
assert set(inherited)==set(old['ids']) and len(inherited)==100
source=old['source_SHA256']
for p,h in source.items():assert sha(L/'late9_sources'/p)==h,p
config=dict(schema_version=1,configuration_name='interVL-SVH-R-new',backbone='OpenGVLab/InternVL3-8B',model_revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9',
 algorithm='SVH-R terminal Late14',internal_engine_condition='Late14_L14_27',layer_indexing='0-based',reaccess_layers=list(range(14,28)),
 R_image_permission_boundary=20,skip_answer_attention=[2,3,4,5],visual_hidden_exit='after_L27_before_final_norm',cutoff=28,
 physical_persistent_visual_KV=True,independent_cache_per_request=True,dtype='bfloat16',attention='eager',torch='2.6.0+cu124',transformers='4.49.0',batch_size=1,
 image_protocol='ATE_SVH_1TILE_SCORER_ALIGNED',image_shape=[1,3,448,448],visual_tokens=256,greedy=True,max_new_tokens=128,eos_token_id=151645,
 frozen_core_path='vlm_research/internvl3_svh_r_terminal_windows_textvqa/engine.py',frozen_core_sha256=source['vlm_research/internvl3_svh_r_terminal_windows_textvqa/engine.py'],
 numerical_qualification='EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; historical strict failures remain; naming does not certify global numerical equivalence',
 physical_scope='Selective physical persistent visual KV; all28 prefill blocks still process visual Hidden, transient zero scratch preserves reduction width; no efficiency claim',
 naming_authority='User explicitly named the frozen Late14 configuration interVL-SVH-R-new; spelling retained verbatim',
 predecessor_commit='2cff899aa0dfc325da69e9fc97eafc0fdbb2db20')
atomic(O/'configuration.json',config)
atomic(O/'frozen_rows.json',rows);atomic(O/'input_paths.json',paths);atomic(O/'inherited_late14_100.json',inherited)
shutil.copy2(L/'results/internvl3/inputs_verified.json',O/'expected_bindings.json')
smokecats=['Regular Text Recognition','Doc-oriented VQA','Key Information Extraction','Handwritten Mathematical Expression Recognition']
smoke_ids=[next(r['question_id'] for r in rows if r['category']==c and r['question_id'] not in inherited) for c in smokecats]
protocol=dict(status='FROZEN_BEFORE_MISSING900_GENERATION',configuration_name=config['configuration_name'],configuration_sha256=sha(O/'configuration.json'),
 model=config['backbone'],revision=config['model_revision'],benchmark='OCRBench v1',expected=1000,unique_images=930,max_new_tokens=128,greedy=True,
 input_protocol=config['image_protocol'],inherited_commit=config['predecessor_commit'],inherited_count=100,inherited_source_SHA256=inherited_sha,
 new_generation_limit=900,reaccess_layers=list(range(14,28)),cutoff=28,R_image_permission_boundary=20,skip_answer_attention=[2,3,4,5],
 source_SHA256=source,rows_sha256=sha(O/'frozen_rows.json'),bindings_sha256=sha(O/'expected_bindings.json'),inherited_records_sha256=sha(O/'inherited_late14_100.json'),
 annotation_sha256=sha(D/'OCRBench.json'),smoke_ids=smoke_ids,baseline_commit='a4fb52e3aa4d57385bd1108832c58a28471ecea3',
 scorer='Unchanged official OCRBench v1 substring scoring, separately independently recomputed',numerical_status=config['numerical_qualification'],
 hard_stop='OOM, nonfinite, input/prompt/token/processor mismatch, wrong layer execution, KV/RoPE/position/skip/hidden exit errors; preserve all partial outputs',
 recovery='Atomic per-question local records; compressed persistent checkpoint every50 new outputs; validate protocol before resuming; never regenerate inherited/completed IDs',
 selection_note='Complete original1000 questions in original order. Previous100 were used to explore windows; full1000 includes them and is not an independent holdout.',
 no_other_experiments=True)
atomic(O/'protocol.json',protocol)
atomic(O/'cpu_audit.json',dict(status='PASS',questions=1000,images=930,inherited=100,new_missing=900,source_files=len(source),weights_redownloaded=False,weights_rehashed=False,full_image_SHA_repeated=False,existing_image_receipts='Prior all930 verified; actual loaded images and tensors rechecked against frozen bindings per request'))
W=O/'publication';REL='vlm_research/results/internvl_svhr_new_ocrbench1000_22875_20261113';P=W/REL;P.mkdir(parents=True,exist_ok=True)
C=W/'vlm_research/configs/interVL-SVH-R-new.json';C.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/'configuration.json',C)
for f in ['protocol.json','cpu_audit.json']:shutil.copy2(O/f,P/f)
(P/'README_zh.md').write_text('# interVL-SVH-R-new：OCRBench v1全集\n\n用户命名的配置：InternVL3-8B，Late14=L14–L27（0-based），其余冻结ATE协议不变。配置见../../configs/interVL-SVH-R-new.json。\n\n预注册状态：PARTIAL，100/1000题继承上一轮完整预测；仅授权生成缺失900题。未填写或推测全量分数。先做4道未完成题的Physical结构Smoke，失败即停止。\n\n该命名与完整覆盖不代表已通过历史全局数值等价门禁，EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER继续保留。沿用单448图块/256视觉Token的ATE协议，不混入官方动态24图块协议。\n',encoding='utf-8')
with zipfile.ZipFile(O/'preregistration.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for f in sorted(W.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(W)))
atomic(O/'preregistration_receipt.json',dict(bytes=(O/'preregistration.zip').stat().st_size,sha256=sha(O/'preregistration.zip')))
print('PREPARED_1000_REUSE100_MISSING900',json.dumps(dict(config=config['configuration_name'],smoke_ids=smoke_ids,config_sha256=sha(O/'configuration.json'))))
