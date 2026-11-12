"""CPU-only binding of the unchanged 5ee2 Native@32 inputs, model and source."""
from pathlib import Path
import os,sys,json,gzip,hashlib,subprocess,shutil
R=Path('/home/featurize/work/ATE');U=R/'vlm_research/results/unified_e2e32_4090'
O=R/'vlm_research/results/native_e2e32_3090_recheck'
M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
RESTORE=Path('/home/featurize/work/ate-restores/3090-16473-20261110')
COMMIT='5ee2a0dbe78df336229b55fed14a4a6d0d880bed';REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
UUID='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27'
sys.path[:0]=[str(R),str(U/'code'),str(R/'vlm_research/external_baselines/fastv/compatibility'),str(R/'vlm_research/external_baselines/v2drop/compatibility'),str(R/'vlm_research/external_baselines/e2e32_compatibility')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(name,obj):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def prepare():
 os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',USE_TF='0',USE_FLAX='0')
 import torch,transformers
 from transformers import AutoProcessor,AutoConfig
 from vlm_research.information_horizon_alignment.engine import paper_pixels,image_ids
 from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
 assert not torch.cuda.is_initialized() and torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 old=read(U/'protocol.json');sources=dict(old['frozen_source_SHA256'])
 for name,h in old['runner_source_SHA256'].items():sources[str(U/'code'/name)]=h
 for path,h in sources.items():assert sha(path)==h,path
 refs_files=['protocol.json','input_verification.json','input_manifest.jsonl.gz','historical_token_references.json','question_positions.json','efficiency_per_input.csv','environment.json','drift_summary.json','formal_requests.jsonl.gz']
 refs={}
 for name in refs_files:
  path=U/name;blob=subprocess.check_output(['git','-C',str(R),'show',COMMIT+':'+str(path.relative_to(R))]);assert hashlib.sha256(blob).hexdigest()==sha(path),name;refs[name]=sha(path)
 for name in old['runner_source_SHA256']:
  path=U/'code'/name;blob=subprocess.check_output(['git','-C',str(R),'show',COMMIT+':'+str(path.relative_to(R))]);assert hashlib.sha256(blob).hexdigest()==sha(path)
 receipt=read(M/'.ate_revision.json');assert receipt['status']=='VERIFIED' and receipt['revision']==REV and receipt['all_LFS_SHA256_verified']
 for name,entry in receipt['files'].items():assert (M/name).stat().st_size==entry['bytes'],name
 rows=lines(U/'input_manifest.jsonl.gz');assert [s['key'] for s in rows]==old['keys'] and len(rows)==30
 assert sum(s['benchmark']=='GQA' for s in rows)==sum(s['benchmark']=='TextVQA' for s in rows)==15
 tokenrefs=read(U/'historical_token_references.json');historical={s['key']:s['generation']['token_ids'] for s in lines(U/'formal_requests.jsonl.gz') if s['method']=='Native'}
 assert len(historical)==30
 for s in rows:assert len(historical[s['key']])==32 and historical[s['key']]==tokenrefs[s['key']+'|Native']
 paths=read(RESTORE/'runtime_paths.json')['datasets'];catalog={}
 for b,view in [('GQA','GQA-testdev-balanced'),('TextVQA','TextVQA-OCR')]:catalog[b]={str(s['question_id']):s for s in map(json.loads,Path(paths[view]['inputs']).read_text().splitlines())}
 proc=AutoProcessor.from_pretrained(M,local_files_only=True);cfg=AutoConfig.from_pretrained(M,local_files_only=True)
 proc.patch_size=cfg.vision_config.patch_size;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy=cfg.vision_feature_select_strategy;proc.tokenizer.padding_side='left'
 oldcheck={s['key']:s for s in read(U/'input_verification.json')['records']};mapping={};checks=[]
 for s in rows:
  mapped=catalog[s['benchmark']][str(s['question_id'])];path=Path(mapped['image_path']);assert mapped['question']==s['question'] and sha(path)==s['image_sha256']
  ids=[]
  for token in image_ids(s['prompt'],proc.tokenizer,-200):ids.extend([32000]*576 if token==-200 else [token])
  assert ids==s['input_ids'] and hashlib.sha256(s['prompt'].encode()).hexdigest()==s['prompt_sha256']
  raw=dict(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),pixel_values=paper_pixels(path,proc.image_processor).half())
  hashes={k:tensor_hash(v) for k,v in raw.items()};assert hashes==s['input_hashes']==oldcheck[s['key']]['input_hashes']
  mapping[s['key']]=str(path);checks.append(dict(key=s['key'],question_id=s['question_id'],sample_sha256=s['sample_sha256'],image_sha256=s['image_sha256'],prompt_sha256=s['prompt_sha256'],input_hashes=hashes,original_image_path=s['image_path'],mapped_image_path=str(path),historical_token_ids=historical[s['key']]))
 protocol=dict(status='FROZEN',task='Native Full fixed32 only; three independent sessions; no other methods, natural EOS or quality evaluation',reference_commit=COMMIT,model=old['model'],revision=REV,model_path=str(M),model_receipt_sha256=sha(M/'.ate_revision.json'),GPU_UUID=UUID,torch='2.6.0+cu124',transformers='4.49.0',precision='FP16',attention='eager',batch=1,greedy=True,use_cache=True,logits_to_keep=1,generated_tokens=32,ignore_eos=True,prefill_calls=1,real_decode_calls=31,sessions=3,formal_per_session=30,formal_expected=90,keys=old['keys'],probe_keys=old['smoke_keys'],input_manifest_SHA256=refs['input_manifest.jsonl.gz'],input_path_mapping_SHA256=digest(mapping),historical_native_token_SHA256=digest(historical),sources=sources,reference_files_SHA256=refs,model_validation='Reuse current-instance download-stream full LFS SHA receipt; verify revision, receipt SHA and all 17 file sizes; no redundant full-weight scan.',timing=old['timing'],included=old['included'],excluded=old['excluded'],runtime='Call frozen e2e32_runtime.generate(a,layout,Native,smoke=False) unchanged for all formal/warmup/probes. No algorithm constructor is invoked. No per-step EOS extraction/stop is added.',no_op='Frozen identity_attention wraps only Native forward; smoke must match original Native full32 tokens and first logits exactly.',warmup={'minimum_seconds':180,'schedule':'Complete cycles of both frozen probe inputs; retain all warmup requests, excluded from formal.'},drift={'threshold':.02,'baseline_rounds':6,'after_each_inputs':5,'consecutive_windows':2,'metrics':['E2E_ms','TPOT_ms'],'definition':'Each window is the mean of the same two fixed probes; compare each metric to the mean of six post-warmup baseline rounds. Pause all remaining work when either metric exceeds 2% absolute drift for two consecutive windows. No rebaselining or replacing failed sessions. New baseline only at the next planned session after prior session passes. Baseline is pre-session check; after30 is post-session check.'},resume='Immutable SQLite (session,key) unique formal records plus protocol/input/model/source hashes. A process interrupted within a session is preserved as incomplete and is not silently combined with a reloaded-model session; report BLOCKED. Completed sessions can be reused without duplicate requests.',GPU_settings_changes=False,GitHub_push=False,Windows_backup=False)
 protocol['entry_sources_SHA256']={p.name:sha(p) for p in (O/'code').glob('*.py') if p.name!='analyze_native.py'}
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol,'Refuse changing frozen Native recheck protocol'
 else:
  save('protocol.json',protocol);save('input_path_mapping.json',mapping);save('historical_native_tokens.json',historical)
  for name in refs_files:
   dest=O/'frozen_reference'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(U/name,dest)
 save('input_verification.json',dict(status='PASS',inputs=30,GQA=15,TextVQA=15,original_IDs_and_order_exact=True,Prompt_OCR_preprocessing_exact=True,all_three_input_tensor_SHA256_exact=True,CUDA_initialized=torch.cuda.is_initialized(),records=checks))
 assert not torch.cuda.is_initialized();print('CPU_INPUT_SOURCE_MODEL_CHECKS_PASS 30 exact frozen inputs')
if __name__=='__main__':prepare()
