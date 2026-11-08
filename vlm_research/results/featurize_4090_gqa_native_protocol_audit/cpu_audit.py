"""Read-only audit of every archived Native answer; freeze a blind stratified cohort."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from collections import Counter,defaultdict
from types import SimpleNamespace
import json,gzip,hashlib,csv,sys,shutil,subprocess,statistics
import torch
from PIL import Image
from transformers import AutoProcessor,AutoConfig,LlamaTokenizer,CLIPImageProcessor
from protocol_helpers import O,D,M,S,official_functions,official_prompt,official_ids,historical_prompt,configure,make_inputs
R=Path('/home/featurize/work/ATE');sys.path[:0]=[str(R),str(R/'vlm_research/results/featurize_new3090_recovery/scripts')]
import ate_data
from vlm_research.gqa_testdev_c3.common import prediction as frozen_prediction,score as frozen_score,digest,expand
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
H=R/'vlm_research/results/featurize_new3090_recovery/history/c3v2-gqa-testdev-balanced-12578/vlm_research/results/c3v2_gqa_testdev_balanced_12578'
H7=R/'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench'
BUNDLE=D/'restored_bundles/ate3090-gqa-testdev-balanced'
SUPPORT=D/'audit_support/gqa_native_protocol';SUPPORT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
torch.set_num_threads(1)
sources=read(S/'source_manifest.json')
for name,info in sources['sources'].items():assert sha(S/name)==info['sha256']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='e281754ef17b3accc09d6938b5b9f020048b2aec'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
with gzip.open(H/'sample_manifest.jsonl.gz','rt') as f:history=[json.loads(l) for l in f]
with gzip.open(H7/'gqa_historical_controls.jsonl.gz','rt') as f:controls=[json.loads(l) for l in f]
assert len(history)==len(controls)==12578
old={s['question_id']:s for s in history};pred={c['question_id']:c for c in controls}
rawpath=BUNDLE/'source_annotations/gqa/testdev_balanced_questions.json';raw=read(rawpath)
current={r['question_id']:r for r in ate_data.iter_inputs('GQA-testdev-balanced')};gold=ate_data.load_answers('GQA-testdev-balanced')
assert set(old)==set(pred)==set(raw)==set(current)==set(gold) and len(set(old))==12578
protocol=read(H/'protocol.json');expected_raw=next(h for p,h in protocol['annotations'].items() if p.endswith('/testdev_balanced_questions.json'))
assert sha(rawpath)==expected_raw
processor=configure(AutoProcessor.from_pretrained(M,local_files_only=True),AutoConfig.from_pretrained(M,local_files_only=True))
rows=[];saved_scores=Counter();caps=[];empty=[];norm_changes=[];extra_vocab=[];official_input=SUPPORT/'historical_native_raw_predictions.jsonl'
with official_input.open('w') as f:
 for q in sorted(old):
  s=old[q];c=pred[q];a=c['controls']['Native'];g=raw[q]
  assert s['question']==current[q]['question']==g['question']
  assert s['answer']==gold[q]==c['gold']==g['answer'] and s['image_id']==current[q]['image_id']==c['image_id']==g['imageId']
  assert g['isBalanced'] and s['split']=='testdev_balanced' and c['source_commit']=='57e37ce024b1b2adf9fa24fc24aabec32a1fed08'
  ids=a['token_ids'];assert 0<len(ids)<=32
  assert processor.tokenizer.decode(ids,skip_special_tokens=True).strip()==a['text'],q
  # Independent implementation of the official conversion: strip only trailing
  # periods, then lowercase, with no substring, synonym or VQA soft match.
  manual=a['text']
  while manual.endswith('.'):manual=manual[:-1]
  manual=manual.lower();correct=int(manual==g['answer'])
  assert manual==a['parsed']==frozen_prediction(a['text']) and correct==a['score']==frozen_score(s,a['text']),q
  eos=ids[-1]==2;truncated=len(ids)==32 and not eos
  if truncated:caps.append(q)
  if not a['text']:empty.append(q)
  if manual!=a['text']:norm_changes.append(q)
  if any(i>=32000 for i in ids):extra_vocab.append(q)
  rows.append({'question_id':q,'image_id':s['image_id'],'structural':g['types']['structural'],'semantic':g['types']['semantic'],
   'gold':g['answer'],'raw_output':a['text'],'normalized_prediction':manual,'saved_score':a['score'],'recomputed_score':correct,
   'generated_tokens':len(ids),'eos':eos,'truncated_at_32':truncated,'empty':not bool(a['text'])})
  saved_scores[g['types']['structural']]+=correct
  f.write(json.dumps({'question_id':q,'text':a['text']},ensure_ascii=False)+'\n')
official_converted=SUPPORT/'historical_native_official_converted.json'
subprocess.run([sys.executable,str(S/'release_convert_gqa.py'),'--src',str(official_input),'--dst',str(official_converted)],check=True)
converted=read(official_converted);assert len(converted)==12578 and all(x['prediction']==pred[x['questionId']]['controls']['Native']['parsed'] for x in converted)
official_eval=subprocess.run([sys.executable,str(S/'GQA_eval_llava.py'),'--tier','testdev_balanced','--questions',str(rawpath),'--predictions',str(official_converted)],text=True,capture_output=True)
(O/'official_eval.log').write_text(official_eval.stdout+'\n'+official_eval.stderr);assert official_eval.returncode==0,official_eval.stderr[-2000:]
csvwrite('native_per_question_audit.csv',rows)
counts=Counter(x['structural'] for x in rows);types=[]
for kind,n in sorted(counts.items()):types.append({'structural':kind,'questions':n,'correct':saved_scores[kind],'accuracy_percent':100*saved_scores[kind]/n})
csvwrite('native_by_type.csv',types)
byimage=defaultdict(list)
for s in history:byimage[s['image_id']].append(s)
assert len(byimage)==398
image_manifest=read(BUNDLE/'image_sha256.json');inventory=[]
clip=CLIPImageProcessor.from_json_file(S/'CLIP_processor.json');conv,functions=official_functions()
original=read(S/'original_model_api.json');official_token_info=next(x for x in original['siblings'] if x['rfilename']=='tokenizer.model')
assert official_token_info['lfs']['sha256']==sha(M/'tokenizer.model')
token_dir=SUPPORT/'official_tokenizer';token_dir.mkdir(exist_ok=True)
for name in ['tokenizer_config.json','special_tokens_map.json']:shutil.copy2(S/('original_'+name),token_dir/name)
if not (token_dir/'tokenizer.model').exists():os.link(M/'tokenizer.model',token_dir/'tokenizer.model')
official_tokenizer=LlamaTokenizer.from_pretrained(token_dir,local_files_only=True,legacy=True)
different_pixels=0;different_fast_tokens=0;prompt_examples=[]
for i,(image_id,samples) in enumerate(sorted(byimage.items())):
 s=samples[0];s=dict(s,image_path=current[s['question_id']]['image_path'])
 p=Path(s['image_path']);assert p.is_file()
 # Whole portable ZIP hash and full image audit already accepted; read dimensions
 # and preprocess the 398 distinct images without repeating every image SHA.
 with Image.open(p) as im:width,height=im.size;mode=im.mode;rgb=im.convert('RGB')
 crop=processor.image_processor(rgb,return_tensors='pt')['pixel_values']
 pad=functions.process_images([rgb],processor.image_processor,SimpleNamespace(image_aspect_ratio='pad'))
 refpad=functions.process_images([rgb],clip,SimpleNamespace(image_aspect_ratio='pad'))
 assert torch.equal(pad,refpad),'HF core CLIP parameters differ from official'
 crop_hash=tensor_hash(crop.to(torch.float16));assert crop_hash==pred[s['question_id']]['controls']['Native']['input_binding']['input_hashes']['pixel_values']
 differs=not torch.equal(crop,pad);different_pixels+=differs
 inventory.append({'image_id':image_id,'width':width,'height':height,'source_mode':mode,'crop_pad_pixels_differ':differs,'historical_crop_fp16_hash_match':True,
  'current_image_sha256_from_accepted_manifest':s['image_sha256'],'crop_fp16_sha256':crop_hash,'pad_fp16_sha256':tensor_hash(pad.to(torch.float16))})
 if (i+1)%100==0:print('CPU_IMAGE_PROTOCOL_CHECK',i+1,398,flush=True)
for s in history:
 q=s['question_id'];hp=historical_prompt(s,processor);op=official_prompt(s,conv)
 assert digest(hp)==s['rendered_prompt_sha256'] and processor.tokenizer(hp)['input_ids']==s['compact_prompt_ids']
 oi=official_ids(op,official_tokenizer,functions)
 hf=processor.tokenizer(op)['input_ids'];hf=[v for t in hf for v in ([32000]*576 if t==32000 else [t])]
 different_fast_tokens+=oi!=hf
 if len(prompt_examples)<2:prompt_examples.append({'question_id':q,'historical_prompt':hp,'official_prompt':op,'historical_compact_ids':s['compact_prompt_ids'],'official_compact_ids':functions.tokenizer_image_token(op,official_tokenizer),'official_prompt_hf_fast_ids_differ':oi!=hf,'historical_expanded_length':len(expand(s)),'official_expanded_length':len(oi)})
csvwrite('image_protocol_audit.csv',inventory)
# Proportional structural strata; allocation and ranking never consult predictions,
# Gold or observed accuracy. Exactly 200 distinct fixed question IDs.
seed='ATE-GQA-official-protocol-audit-v1-20261108';N=200
alloc={k:int(N*v/12578) for k,v in counts.items()}
for k in sorted(counts,key=lambda k:(-(N*counts[k]/12578-alloc[k]),k))[:N-sum(alloc.values())]:alloc[k]+=1
selected=[]
for kind in sorted(counts):
 pool=[s for s in history if s['task']==kind]
 ranked=sorted(pool,key=lambda s:hashlib.sha256((seed+'|'+s['question_id']).encode()).hexdigest())
 for s in ranked[:alloc[kind]]:selected.append(dict(s,image_path=current[s['question_id']]['image_path']))
assert len(selected)==len({s['question_id'] for s in selected})==200
selected=sorted(selected,key=lambda s:hashlib.sha256((seed+'|order|'+s['question_id']).encode()).hexdigest())
smoke=[]
for kind in ['query','verify','choose']:
 smoke.append(next(s['question_id'] for s in selected if s['task']==kind))
freeze={'status':'FROZEN_BEFORE_GPU','seed':seed,'selection':'Structural-proportional largest-remainder allocation; SHA256 rank by seed|qid; no use of outputs or correctness.',
 'counts':alloc,'sample_ids':[s['question_id'] for s in selected],'smoke_ids':smoke,'samples':selected,
 'historical_native':{s['question_id']:pred[s['question_id']]['controls']['Native'] for s in selected},
 'historical_controls_sha256':sha(H7/'gqa_historical_controls.jsonl.gz'),'historical_manifest_sha256':sha(H/'sample_manifest.jsonl.gz'),
 'historical_commit':'57e37ce024b1b2adf9fa24fc24aabec32a1fed08','model_revision':protocol['revision'],
 'conditions':{'A':'Official Vicuna-v1 prompt/tokenization only; historical crop','B':'Official square-pad preprocessing only; historical HF USER prompt/IDs','C':'Official prompt/tokenization and square-pad preprocessing'},
 'other_generation':'Frozen HF Native FP16/Eager, greedy, EOS2, max_new_tokens32; no other backend, search or full benchmark.',
 'sources':{str(Path('official_sources')/n):v['sha256'] for n,v in sources['sources'].items()},
 'helper_sha256':sha(O/'protocol_helpers.py'),'frozen_gqa_engine_sha256':sha(R/'vlm_research/gqa_testdev_c3/engine.py')}
save('gpu_sample_plan.json',freeze)
csvwrite('fixed_sample_ids.csv',[{'question_id':s['question_id'],'structural':s['task'],'semantic':s['types']['semantic'],'image_id':s['image_id']} for s in selected])
total=sum(r['recomputed_score'] for r in rows)
summary={'status':'PASS','CUDA_initialized':torch.cuda.is_initialized(),'questions':len(rows),'correct':total,'accuracy_percent':100*total/len(rows),
 'official_model_zoo_reference_percent':62.0,'difference_to_rounded_reference_pp':100*total/len(rows)-62.0,
 'missing_predictions':0,'duplicate_predictions':len(controls)-len(pred),'question_image_Gold_split_mismatches':0,'saved_score_mismatches':0,
 'official_conversion_predictions_exact':True,'official_evaluator_returncode':official_eval.returncode,'official_evaluator_stdout':official_eval.stdout[-10000:],
 'empty_outputs':len(empty),'empty_question_ids':empty,'truncated_at_32':len(caps),'truncated_question_ids':caps,
 'max_observed_generated_tokens':max(r['generated_tokens'] for r in rows),'token_count_histogram':dict(Counter(r['generated_tokens'] for r in rows)),
 'normalization_changes_output':len(norm_changes),'generated_added_vocab_ids':len(extra_vocab),
 'theoretical_max_gain_from_only_extending_truncated_outputs_pp':100*len(caps)/12578,
 'raw_official_annotation_sha256':sha(rawpath),'raw_annotation_matches_historical_official_source_hash':True,
 'distinct_images':398,'missing_images':0,'current_crop_FP16_pixel_hashes_matching_historical_Native':398,
 'images_with_crop_pad_pixel_difference':different_pixels,'official_CLIP_pad_equals_current_core_CLIP_after_external_pad':True,
 'official_tokenizer_sentencepiece_SHA256_same_as_HF':True,'official_slow_vs_HF_fast_tokenization_differences_for_official_prompt':different_fast_tokens,
 'prompt_examples':prompt_examples,'structural_counts':dict(counts),'sample_counts':alloc,
 'model_weight_identity':'UNRESOLVED: same architecture/tokenizer/conversion lineage, but no native-vs-HF tensor audit or native-backend parity run; no alternative weight download.',
 'historical_image_provenance':protocol['image_caveat'],'historical_GQA_eval_script_SHA256':next(h for p,h in protocol['annotations'].items() if p.endswith('/eval_llava_official.py')),
 'downloaded_official_GQA_eval_script_SHA256':sha(S/'GQA_eval_llava.py')}
assert not summary['CUDA_initialized'];save('cpu_audit_summary.json',summary)
print('CPU_NATIVE_AUDIT_PASS',total,len(rows),summary['accuracy_percent'],'EMPTY',len(empty),'TRUNCATED',len(caps),'PAD_DIFF_IMAGES',different_pixels,'PROMPT_TOKENIZER_DIFF',different_fast_tokens,flush=True)
