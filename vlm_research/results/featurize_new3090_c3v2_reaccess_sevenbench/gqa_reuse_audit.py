import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import sys,json,gzip,sqlite3,hashlib,csv,subprocess
from pathlib import Path
from collections import Counter
import torch
from PIL import Image
from transformers import AutoProcessor,AutoConfig
R=Path('/home/featurize/work/ATE');D=Path('/home/featurize/data/ate-recovery');REC=R/'vlm_research/results/featurize_new3090_recovery'
OUT=R/'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench';H=REC/'history/c3v2-gqa-testdev-balanced-12578/vlm_research/results/c3v2_gqa_testdev_balanced_12578';DB=D/'sources/gqa-reaccess-history';M=D/'models/llava-1.5-7b-hf'
sys.path[:0]=[str(R),str(REC/'scripts')]
from vlm_research.gqa_testdev_c3.common import digest,expand,prediction,score
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
import ate_data
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
torch.set_num_threads(1)
freeze=json.loads((H/'execution_freeze.json').read_text());binding=digest(freeze)
assert binding=='c51570bee80c823f0778148b4c0cf92b598cc735a073deac999b2b86b726663d'
source_checks={}
for old,h in freeze['sources'].items():
 rel='vlm_research/'+old.split('/vlm_research/',1)[1];p=R/rel
 if p.is_file():actual=sha(p)
 else:actual=hashlib.sha256(subprocess.check_output(['git','show','57e37ce024b1b2adf9fa24fc24aabec32a1fed08:'+rel],cwd=R)).hexdigest()
 assert actual==h,(rel,actual,h)
 source_checks[rel]=h
accept=json.loads((REC/'model_acceptance.json').read_text());weight_checks={}
for old,h in freeze['weights'].items():
 name=Path(old).name;p=M/name
 assert p.is_file()
 if name.endswith('.safetensors'):
  assert accept['files'][name]['sha256']==h and accept['files'][name]['bytes']==p.stat().st_size
 else:assert sha(p)==h
 weight_checks[name]=h
assert sha(H/'sample_manifest.jsonl.gz')==freeze['artifacts']['sample_manifest.jsonl.gz']
with gzip.open(H/'sample_manifest.jsonl.gz','rt') as f:oldrows=[json.loads(l) for l in f]
old={s['question_id']:s for s in oldrows};assert len(old)==len(oldrows)==12578
rows={s['question_id']:s for s in ate_data.iter_inputs('GQA-testdev-balanced')};gold=ate_data.load_answers('GQA-testdev-balanced');assert set(rows)==set(old)==set(gold)
with (DB/'per_question.csv').open() as f:csvrows={(r['question_id'],r['condition']):r for r in csv.DictReader(f)}
records={};condition_counts=Counter();checks=0
for rank in [0,1]:
 db=sqlite3.connect((DB/f'worker{rank}/results.sqlite').as_uri()+'?mode=ro',uri=True)
 assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
 assert db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0]==binding
 for qid,label,payload,h in db.execute('SELECT qid,condition,payload,sha256 FROM results'):
  r=json.loads(payload);s=old[qid];assert digest(r)==h and r['binding']==binding and r['sample_sha256']==digest(s)
  assert r['question_id']==qid and r['condition']==label and r['image_sha256']==s['image_sha256']
  assert r['score']==score(s,r['generation']['text'])
  c=csvrows[qid,label];assert c['text']==r['generation']['text'] and c['prediction']==prediction(r['generation']['text']) and float(c['accuracy'])==r['score']
  assert json.loads(c['token_ids'])==r['generation']['token_ids']
  assert r['generation']['all_effective_logits_finite'] and r['generation']['independent_cache']
  condition_counts[label]+=1;checks+=1
  if label=='C3-V2+L11-L24':
   assert qid not in records;records[qid]=dict(record=r,record_sha256=h)
 db.close()
assert condition_counts=={'Native':12578,'C3-V2':12578,'C3-V2+L11-L24':12578}
processor=AutoProcessor.from_pretrained(M,local_files_only=True);config=AutoConfig.from_pretrained(M,local_files_only=True)
processor.patch_size=config.vision_config.patch_size;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy=config.vision_feature_select_strategy;processor.tokenizer.padding_side='left'
pixels={};mapped=[]
for i,s in enumerate(oldrows):
 q=s['question_id'];current=rows[q];r=records[q]['record']
 assert current['question']==s['question'] and current['image_id']==s['image_id'] and gold[q]==s['answer']
 assert s['prompt_text']==s['question']+'\nAnswer the question using a single word or phrase.'
 prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
 ids=processor.tokenizer(prompt)['input_ids'];assert ids==s['compact_prompt_ids'],q
 assert digest(prompt)==s['rendered_prompt_sha256'],(q,list(s))
 expanded=expand(s);assert digest(expanded)==s['expanded_prompt_ids_sha256']
 if s['image_id'] not in pixels:
  with Image.open(current['image_path']) as im:
   inp=processor(images=im.convert('RGB'),text=prompt,return_tensors='pt')
  assert inp['input_ids'][0].tolist()==expanded
  pixels[s['image_id']]=tensor_hash(inp['pixel_values'].to(torch.float16))
 assert r['input_binding']['input_hashes']=={'input_ids':tensor_hash(torch.tensor([expanded],dtype=torch.long)),
  'attention_mask':tensor_hash(torch.ones((1,len(expanded)),dtype=torch.long)),'pixel_values':pixels[s['image_id']]},q
 assert r['input_binding']['prompt_ids_sha256']==digest(expanded)
 mapped.append(dict(s,image_path=current['image_path'],key='GQA:'+q,benchmark='GQA',input_view='GQA-testdev-balanced',
  original_sample_sha256=digest(s),normalized_question_id=q,author_question_id=q,max_new_tokens=32,historical_record_sha256=records[q]['record_sha256']))
 if (i+1)%2000==0:print('CPU_REUSE_INPUT_CHECK',i+1,flush=True)
assert len(pixels)==398 and not torch.cuda.is_initialized()
with gzip.GzipFile(filename=str(OUT/'gqa_mapped_manifest.jsonl.gz'),mode='wb',mtime=0) as f:
 for s in mapped:f.write((json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n').encode())
with gzip.GzipFile(filename=str(OUT/'gqa_verified_records.jsonl.gz'),mode='wb',mtime=0) as f:
 for q in sorted(records):f.write((json.dumps(records[q],ensure_ascii=False,separators=(',',':'))+'\n').encode())
save('gqa_reuse_audit.json',dict(status='PASS',source_commit='57e37ce024b1b2adf9fa24fc24aabec32a1fed08',historical_binding=binding,
 source_sha256=source_checks,weights_sha256=weight_checks,model_weight_hash_basis='Already accepted shard hashes; no repeat large weight hashing.',
 records_validated=checks,reaccess_records=12578,conditions=dict(condition_counts),all_record_and_manifest_bindings_verified=True,
 all_questions_gold_and_exact_input_tensor_hashes_verified=True,distinct_preprocessed_images=398,input_preprocessing='Original HF crop and USER chat template without paper Vicuna system, unchanged GQA protocol.',
 max_new_tokens=32,score=sum(x['record']['score'] for x in records.values()),empty=sum(x['record']['generation']['empty'] for x in records.values()),
 cuda_initialized=False,runtime_patch_difference='Historical Python 3.11.15 vs current 3.11.8; identical torch2.6.0+cu124/tfm4.49.0/model FP16/Eager and exact input tensors. Driver/GPU instance identity differs.',
 raw_database_gzip_sha256={f'worker{i}':sha(DB/f'worker{i}/results.sqlite.gz') for i in [0,1]},manifest_sha256=sha(H/'sample_manifest.jsonl.gz')))
print('GQA_REUSE_PASS',len(mapped),sum(x['record']['score'] for x in records.values()),flush=True)
