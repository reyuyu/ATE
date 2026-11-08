"""Fifteen fixed C3 cases on the replacement GPU; never a formal cohort."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
 CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,hashlib,sys,subprocess,importlib.util,gc,traceback
import torch,transformers
from PIL import Image
REPO=Path('/home/featurize/work/ATE');O=REPO/'vlm_research/results/featurize_4090_c3v2_handoff';D=Path('/home/featurize/data/ate-recovery')
sys.path[:0]=[str(REPO),str(REPO/'vlm_research/results/featurize_new3090_recovery/scripts')]
import ate_data
from vlm_research.information_horizon_alignment import engine as frozen,score as scorer
from vlm_research.information_horizon_alignment.common import digest,tensor_sha
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3 import engine as gqa_engine,common as gqa_common
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=O/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def cmd(*a):return subprocess.check_output(a,cwd=REPO,text=True).strip()
def now():return datetime.now(timezone.utc).isoformat()
spec=importlib.util.spec_from_file_location('original_c3_harness',O.parent/'featurize_new3090_c3v2_evaluation/c3_runner.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.OUT=O
def prepare(a,s):
 if s['benchmark']!='GQA':return frozen.prepare(a,dict(prompt=s['prompt'],user_text=s['user_text'],image_path=s['image_path']))
 with Image.open(s['image_path']) as im:
  prompt=a.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
  x=a.processor(images=im.convert('RGB'),text=prompt,return_tensors='pt')
 ids=gqa_common.expand(s);assert x['input_ids'][0].tolist()==ids
 a.inputs=x.to('cuda:0',dtype=torch.float16)
 image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000)
 relay=tuple(range(image[-1]+1,len(ids)-1))
 return Layout(len(ids),image,text,relay,tuple(s['question_positions']),20),dict(prompt=prompt,input_ids=ids,
  input_hashes={k:tensor_hash(v) for k,v in a.inputs.items()},expanded_length=len(ids))
def run():
 plan=read(O/'comparison_cases.json');model=read(O/'model_download.json');env=read(O/'environment.json');data=read(O/'dataset_restore.json')
 assert model['status']==env['status']==data['status']=='PASS'
 assert model['revision']==plan['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
 assert D.resolve()==Path('/home/featurize/ate-local-data-4090/ate-recovery')
 assert Path(model['path']).resolve()==Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
 assert not Path('/home/featurize/work/ate-data-4090').exists()
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 assert cmd('git','rev-parse','HEAD')==plan['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
 for name,h in plan['frozen_source_hashes'].items():assert sha(REPO/name)==h,name
 oldmeta=read(O.parent/'featurize_new3090_c3v2_evaluation/run_metadata.json')
 assert sha(spec.origin)==oldmeta['runner_sha256']
 for name,info in model['files'].items():assert (Path(model['path'])/name).stat().st_size==info['bytes']
 for b in ['GQA','MME','ScienceQA-IMG','POPE','TextVQA','MMBench']:
  view='GQA-testdev-balanced' if b=='GQA' else helper.VIEWS[b]
  inputs={r['question_id']:r for r in ate_data.iter_inputs(view)};gold=ate_data.load_answers(view)
  for case in plan['cases']:
   s=case['sample']
   if s['benchmark']!=b:continue
   q=s.get('normalized_question_id',s['question_id']);row=inputs[q]
   assert row['question']==s['question'] and row['image_id']==s['image_id'] and gold[q]==s['answer']
   assert sha(s['image_path'])==s['image_sha256'] and case['reference_3090']['gold']==s['answer']
   if b!='GQA':assert row['prompt']==s['prompt'] and row['user_text']==s['user_text']
   if b=='TextVQA':assert row['paper_ocr'] and 'Reference OCR token:' in s['prompt']
   if b=='POPE':assert row['strategy']==s['strategy']
 scorer.AUTHOR=D/'sources/Information-Horizon';scorer.text_evaluator();scorer.matcher()
 r={'status':'RUNNING','started_at':now(),'condition':'Original Physical C3-V2, no reaccess','formal_benchmark':False,
  'code_commit':plan['code_commit'],'model_revision':plan['revision'],'model_path':model['path'],'model_loads':0,
  'dtype':'float16','attention':'eager','greedy':True,'batch_size':1,'cases':[],'expected':15,
  'environment':env,'local_storage_only':True,'reference_sources':sorted({c['reference_commit'] for c in plan['cases']})}
 adapter=None;phase='GPU_LOAD'
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090'
  assert torch.cuda.mem_get_info()[0]>20*2**30
  r['gpu']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.free','--format=csv,noheader')
  torch.cuda.reset_peak_memory_stats();loader.MODELS=dict(loader.MODELS,llava=Path(model['path']))
  r['model_loads']+=1;save('comparison_results.json',r);print('LOADING_SINGLE_FP16_EAGER_MODEL',flush=True)
  adapter=frozen.load('B')
  assert {str(p.device) for p in adapter.model.parameters()}=={'cuda:0'} and {p.dtype for p in adapter.model.parameters()}=={torch.float16}
  assert adapter.model.config.text_config._attn_implementation==adapter.model.config.vision_config._attn_implementation=='eager'
  phase='PHYSICAL_SMOKE'
  physical=helper.smoke(adapter,[c['sample'] for c in plan['cases'] if c['sample']['benchmark']!='GQA'])
  compact={'status':physical['status'],'cases':[{k:c[k] for k in ['benchmark','key','status','numerical_equivalence','rope_exact','visual_hidden_removed_before_L20','answer_attention_physically_skipped','compact_kv_original_positions_exact','blocked_image_probability_zero']} for c in physical['cases']]}
  save('physical_smoke_summary.json',compact);r['physical_smoke_status']='PASS'
  phase='FIXED_CASE_COMPARISON'
  for case in plan['cases']:
   s=case['sample'];ref=case['reference_3090'];layout,binding=prepare(adapter,s)
   if s['benchmark']=='GQA':
    assert binding['input_hashes']==ref['input_binding']['input_hashes']
    assert digest(binding['input_ids'])==ref['input_binding']['prompt_ids_sha256']
    generation,*_=gqa_engine.generate(adapter,layout,'C3-V2',s['max_new_tokens'])
    parsed=gqa_common.prediction(generation['text']);scoring={'score':float(parsed==s['answer']),'parsed':parsed}
   else:
    assert digest(binding['prompt'])==ref['prompt_sha256'] and digest(binding['input_ids'])==ref['input_ids_sha256']
    assert binding['pixel_values_sha256']==ref['pixel_values_sha256']
    generation,_=frozen.hf_generate(adapter,layout,'Physical-C3',s['max_new_tokens'])
    scoring=scorer.score(s,generation['text'])
   matches={'input_binding_exact':True,'token_ids_exact':generation['token_ids']==ref['generation']['token_ids'],
    'raw_text_exact':generation['text']==ref['generation']['text'],'score_and_parsed_exact':scoring==ref['scoring']}
   oldlogit=ref['generation'].get('first_logits_sha256')
   record={'key':s['key'],'benchmark':s['benchmark'],'question_id':s['question_id'],'image_id':s['image_id'],
    'gold':s['answer'],'max_new_tokens':s['max_new_tokens'],'reference_3090_generation':ref['generation'],
    'generation_4090':generation,'reference_scoring':ref['scoring'],'scoring_4090':scoring,'matches':matches,
    'first_logits_bit_exact':generation['first_logits_sha256']==oldlogit if oldlogit else None,
    'first_logits_note':'Diagnostic only; output-token/input/score equality defines this hardware migration check.',
    'pair_id':s.get('pair_id'),'circular_group':s.get('circular_group'),'strategy':s.get('strategy')}
   r['cases'].append(record);save('comparison_results.json',r)
   print('CASE',s['key'],'3090='+repr(ref['generation']['text']),'4090='+repr(generation['text']),json.dumps(matches),flush=True)
   adapter.inputs={}
  r['completed']=len(r['cases']);r['mismatches']=[c['key'] for c in r['cases'] if not all(c['matches'].values())]
  groups=defaultdict(list);pairs=defaultdict(list)
  for c in r['cases']:
   if c['benchmark']=='MMBench':groups[c['circular_group']].append(c)
   if c['benchmark']=='MME':pairs[c['pair_id']].append(c)
  assert len(groups)==1 and len(next(iter(groups.values())))==4 and len(pairs)==1 and len(next(iter(pairs.values())))==2
  r['MMBench_complete_circular_group']={str(k):{'rows':len(v),'all_correct_3090':all(c['reference_scoring']['score']==1 for c in v),'all_correct_4090':all(c['scoring_4090']['score']==1 for c in v)} for k,v in groups.items()}
  r['MME_complete_pair']={str(k):{'rows':len(v),'accuracy_3090':sum(c['reference_scoring']['score'] for c in v)/2,'accuracy_4090':sum(c['scoring_4090']['score'] for c in v)/2,'both_correct_3090':all(c['reference_scoring']['score']==1 for c in v),'both_correct_4090':all(c['scoring_4090']['score']==1 for c in v)} for k,v in pairs.items()}
  r['status']='PASS' if r['completed']==15 and not r['mismatches'] else 'PARTIAL_DIFFERENCE'
 except BaseException as e:
  r['status']='BLOCKED';r['failure']={'phase':phase,'error':repr(e),'traceback':traceback.format_exc()[-8000:]}
  print('STOPPED',phase,traceback.format_exc(),flush=True)
 finally:
  if torch.cuda.is_initialized():
   r['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;r['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
   adapter=None;gc.collect();torch.cuda.empty_cache();r['release']={'allocated_mib':torch.cuda.memory_allocated()/2**20,'reserved_mib':torch.cuda.memory_reserved()/2**20,'process_exit_releases_context':True}
  r['finished_at']=now();save('comparison_results.json',r)
  print('FINAL_COMPARISON',r['status'],len(r['cases']),flush=True)
 if r['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':run()
