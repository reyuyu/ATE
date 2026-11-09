"""Read-only CPU acceptance of restored pinned resources, never GPU inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
from collections import Counter
import ast,gzip,hashlib,json,platform,struct,subprocess,sys
import torch,transformers,torchvision
from transformers import AutoConfig,AutoProcessor
from PIL import Image
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';O=Path(os.environ.get('ATE_RECOVERY_OUT',str(B/'featurize_29535_environment_restore')));V=B/'v2drop_quick_reproduction'
M=Path(os.environ.get('ATE_MODEL_DIR','/home/featurize/ate-local-models/llava-1.5-7b-hf'));D=Path('/home/featurize/data/ate-recovery')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def th(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
assert torch.__version__=='2.6.0+cu124' and torchvision.__version__=='0.21.0+cu124' and transformers.__version__=='4.49.0'
assert torch.version.cuda=='12.4' and not torch.cuda.is_initialized()
torch.set_num_threads(1)
assert subprocess.check_output(['findmnt','-T',str(M),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
assert subprocess.check_output(['findmnt','-T',str(D.resolve()),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
receipt=read(O/'model_integrity.json');assert receipt['status']=='VERIFIED' and receipt['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
# Download already hashed every byte against pinned LFS/git metadata. Inspect only
# safetensors headers here; do not reread 14GB or deserialize model weights.
index=read(M/'model.safetensors.index.json');headers={};total=0
for shard in sorted(set(index['weight_map'].values())):
 path=M/shard
 with path.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert 0<n<10**7;header=json.loads(f.read(n))
 tensors={k:v for k,v in header.items() if k!='__metadata__'}
 assert set(tensors)=={k for k,v in index['weight_map'].items() if v==shard}
 end=0
 for key,t in sorted(tensors.items(),key=lambda kv:kv[1]['data_offsets'][0]):
  lo,hi=t['data_offsets'];assert lo==end and hi>=lo
  assert t['dtype']=='F16';count=1
  for d in t['shape']:count*=d
  assert hi-lo==count*2;total+=hi-lo;end=hi
 assert path.stat().st_size==8+n+end
 headers[shard]={'tensor_count':len(tensors),'all_dtype':'F16','payload_bytes':end,'file_bytes':path.stat().st_size}
assert total==index['metadata']['total_size']
cfg=AutoConfig.from_pretrained(M,local_files_only=True)
assert cfg.model_type=='llava' and cfg.vision_config.image_size==336 and cfg.vision_config.patch_size==14 and cfg.image_token_index==32000
assert cfg.text_config.num_hidden_layers==32
proc=AutoProcessor.from_pretrained(M,local_files_only=True)
proc.patch_size=cfg.vision_config.patch_size;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy=cfg.vision_feature_select_strategy
# Execute the two unchanged frozen preprocessing helpers in an isolated CPU namespace.
source=(R/'vlm_research/information_horizon_alignment/engine.py').read_text();tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('image_ids','paper_pixels')]
assert len(functions)==2
ns={'Image':Image};exec(compile(ast.Module(body=functions,type_ignores=[]),'<unchanged frozen preprocessing>','exec'),ns)
p=read(V/'protocol.json');assert sha(V/'input_manifest.jsonl.gz')==p['manifest_sha256']
for rel,h in p['frozen_ATE_sources'].items():assert sha(R/rel)==h,rel
windows=read(O/'windows_archive_checksums.json')
for name,entry in windows['files'].items():assert (V/name).stat().st_size==entry['bytes'] and sha(V/name)==entry['sha256'],name
data_receipt=read(B/'featurize_29535_environment_restore/dataset_restore.json');assert data_receipt['status']=='PASS'
current_data={}
for view in ['GQA-testdev-balanced','TextVQA-OCR']:
 expected=data_receipt['datasets'][view];dp=D/'normalized'/view
 assert sha(dp/'inputs.jsonl')==expected['inputs_sha256'] and sha(dp/'answers.json')==expected['answers_sha256']
 rr=[json.loads(l) for l in (dp/'inputs.jsonl').read_text().splitlines()];gold=read(dp/'answers.json')
 assert len(rr)==expected['questions'] and {r['question_id'] for r in rr}==set(gold)
 current_data[view]={'rows':{r['question_id']:r for r in rr},'gold':gold}
cohort=gz(V/'input_manifest.jsonl.gz');assert len(cohort)==200 and Counter(s['benchmark'] for s in cohort)=={'GQA':100,'TextVQA':100}
assert len(p['efficiency_keys'])==30 and set(p['efficiency_keys'])<{s['key'] for s in cohort}
pred=[json.loads(l) for l in (V/'predictions_light.jsonl').read_text().splitlines()]
assert len(pred)==400 and len({(r['key'],r['condition']) for r in pred})==400
assert Counter((r['condition'],r['benchmark']) for r in pred)=={('Native','GQA'):100,('Native','TextVQA'):100,('V2Drop','GQA'):100,('V2Drop','TextVQA'):100}
text_old={r['key']:r for r in gz(B/'featurize_new3090_c3v2_evaluation/predictions.jsonl.gz') if r.get('benchmark')=='TextVQA' and r['key'] in {s['key'] for s in cohort}}
cases=[];image_checked={}
for s in cohort:
 view='GQA-testdev-balanced' if s['benchmark']=='GQA' else 'TextVQA-OCR';current=current_data[view]
 c=current['rows'][s['question_id']]
 assert s['answer']==current['gold'][s['question_id']] and s['question']==c['question'] and s['image_id']==c['image_id']
 if s['benchmark']=='TextVQA':assert c['paper_ocr'] and c['prompt']==s['prompt'] and len(s['answer'])==10
 path=Path(s['image_path']);assert path.is_file() and path.resolve().is_relative_to(D.resolve())
 if str(path) not in image_checked:image_checked[str(path)]=sha(path)
 assert image_checked[str(path)]==s['image_sha256']
 ids=[]
 for t in ns['image_ids'](s['prompt'],proc.tokenizer,-200):ids.extend([32000]*576 if t==-200 else [t])
 assert ids==s['input_ids'] and th(torch.tensor([ids],dtype=torch.long))==s['input_hashes']['input_ids']
 assert th(torch.ones((1,len(ids)),dtype=torch.long))==s['input_hashes']['attention_mask']
 pixels=ns['paper_pixels'](path,proc.image_processor);assert list(pixels.shape)==[1,3,336,336]
 assert th(pixels.half())==s['input_hashes']['pixel_values'],s['key']+' FP16 pixel hash'
 if s['benchmark']=='TextVQA':assert th(pixels)==text_old[s['key']]['pixel_values_sha256'],s['key']+' historical FP32 preprocessing hash'
 cases.append(dict(key=s['key'],question_id=s['question_id'],image_sha256=s['image_sha256'],input_ids_sha256=s['input_hashes']['input_ids'],pixel_values_fp16_sha256=s['input_hashes']['pixel_values'],status='PASS'))
 print('CPU_INPUTS_VERIFIED',len(cases),flush=True) if len(cases)%50==0 else None
assert not torch.cuda.is_initialized()
version=dict(python=platform.python_version(),executable=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,transformers=transformers.__version__,CUDA_build=torch.version.cuda,CUDA_initialized=False)
save('resource_acceptance.json',dict(status='PASS',runtime=version,model_revision=receipt['revision'],model_path=str(M),model_files=len(receipt['files']),weight_headers=headers,total_weight_payload_bytes=total,
 model_deserialized=False,GPU_inference_run=False,FP16_Eager_GPU_execution='NOT_VALIDATED_THIS_TASK',image_processor_config=proc.image_processor.to_dict(),
 model_config=dict(layers=32,image_size=336,patch_size=14,image_token_index=32000),data_path=str(D),physical_data_path=str(D.resolve()),
 datasets={v:dict(questions=len(d['rows']),Gold_mapping='PASS',frozen_input_SHA256='PASS') for v,d in current_data.items()},
 fixed_accuracy_cohort=200,unique_selected_images_SHA256_verified=len(image_checked),input_tensor_hashes_verified=200,
 historical_TextVQA_FP32_and_current_FP16_pixel_hashes_verified=100,archived_Native_V2_predictions=400,windows_archive_files_matched=len(windows['files']),missing_required_files=[],cases=cases,
 previous_data_receipt=str(B/'featurize_29535_environment_restore/dataset_restore.json'),priority_data_restored_from_Windows=True,no_frozen_source_modified=True))
(O/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('CPU_RESOURCE_ACCEPTANCE_PASS',json.dumps(version),flush=True)
