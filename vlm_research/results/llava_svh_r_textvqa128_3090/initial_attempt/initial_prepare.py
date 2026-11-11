from common import *
from PIL import Image
import torch,transformers,platform,subprocess
from transformers import AutoProcessor
from cpu_tests import test
def main():
 assert platform.python_version()=='3.11.8' and torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 assert not torch.cuda.is_initialized()
 source=read('/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090/results/sample_manifest.json');assert len(source['Main'])==128 and len(source['Smoke'])==3
 receipt=read(MODEL/'.ate_revision.json');assert receipt['status']=='VERIFIED' and receipt['revision']==REVISION
 checked=[]
 for name,x in receipt['files'].items():
  p=MODEL/name
  assert x['status']=='VERIFIED' and p.is_file() and p.stat().st_size==x['bytes'],name
  if p.suffix!='.safetensors':
   raw=p.read_bytes()
   got=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() if x['hash_algorithm']=='git-blob-SHA1' else hashlib.sha256(raw).hexdigest()
   assert got==x['hash'],name
  checked.append(dict(name=name,bytes=p.stat().st_size,receipt_hash=x['hash'],method='trusted completed download SHA receipt + current size; no repeated weight hashing' if p.suffix=='.safetensors' else 'independent current small-file content hash'))
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
 manifest={'source_model_dataset':'InternVL3 previous B128, questions/images/Gold/order only; LLaVA tokenizer and preprocessing intentionally model-specific','source_commit':'5219e469d8b65f90f84669b29126ad573b6bfae9','source_manifest_sha256':sha('/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090/results/sample_manifest.json'),'Main':[],'Smoke':[]}
 bindings={};seen=set()
 for phase in ('Main','Smoke'):
  for old in source[phase]:
   qid=str(old['question_id']);assert qid not in seen;seen.add(qid)
   path=Path('/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr/images/textvqa')/(old['image_id']+'.jpg')
   assert sha(path)==old['image_sha256'] and path.stat().st_size==old['image_bytes']
   short='Answer the question using a single word or phrase.'
   request=old['question']+'\n'+short
   prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':request}]}],add_generation_prompt=True)
   with Image.open(path) as im:inputs=processor(images=im.convert('RGB'),text=prompt,return_tensors='pt')
   inputs={k:v.to(dtype=torch.float16) if v.is_floating_point() else v for k,v in inputs.items()}
   assert set(inputs)=={'input_ids','attention_mask','pixel_values'} and inputs['pixel_values'].shape==(1,3,336,336)
   assert int((inputs['input_ids']==32000).sum())==576 and inputs['attention_mask'].bool().all()
   ids=inputs['input_ids'][0].tolist();image=[i for i,x in enumerate(ids) if x==32000];r=list(range(image[-1]+1,len(ids)-1))
   expanded=prompt.replace(processor.image_token,processor.image_token*576)
   offsets=processor.tokenizer(expanded,return_offsets_mapping=True);assert offsets['input_ids']==ids
   begin=expanded.index(old['question']);end=begin+len(old['question']);question=[i for i,(a,b) in enumerate(offsets['offset_mapping']) if b>begin and a<end]
   assert question and set(question)<=set(r)
   sample={k:old[k] for k in ('question_id','question','image_id','image_sha256','image_bytes','gold')}
   assert len(sample['gold'])==10
   sample.update(phase=phase,image_path=str(path),prompt=prompt,request=request,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_tokens=len(ids),image=image,text=[i for i,v in enumerate(ids) if v!=32000],relay=r,question_positions=question,original_internvl_sample_sha256=old['sample_sha256'])
   b=dict(tensors={k:dict(sha256=th(v),dtype=str(v.dtype),shape=list(v.shape)) for k,v in inputs.items()},image_sha256=old['image_sha256'],prompt_sha256=sample['prompt_sha256'],qid=qid)
   sample['input_binding']=digest(b);bindings[qid]=b
   # Model input file contains no answers or Gold.
   torch.save(inputs,RUN/'prepared'/(qid+'.pt'));manifest[phase].append(sample)
 assert [x['question_id'] for x in manifest['Main']]==[x['question_id'] for x in source['Main']]
 assert not torch.cuda.is_initialized()
 write(OUT/'sample_manifest.json',manifest);write(OUT/'input_bindings.json',bindings);write(OUT/'model_manifest.json',dict(model='llava-hf/llava-1.5-7b-hf',revision=REVISION,path=str(MODEL),receipt_sha256=sha(MODEL/'.ate_revision.json'),checked_files=checked))
 files={str(p.relative_to(CODE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(CODE.rglob('*.py'))}
 write(OUT/'source_manifest.json',dict(frozen_algorithm_commit='5ee2a0dbe78df336229b55fed14a4a6d0d880bed',new_parent_commit='5219e469d8b65f90f84669b29126ad573b6bfae9',files=files))
 protocol=dict(status='FROZEN_BEFORE_GPU',model='llava-hf/llava-1.5-7b-hf',revision=REVISION,model_path=str(MODEL),GPU_UUID=GPU,Python='3.11.8',Torch='2.6.0+cu124',Transformers='4.49.0',precision='FP16',attention='eager',batch=1,greedy=True,use_cache=True,logits_to_keep=1,max_new_tokens=32,EOS='native eos_token_id=2; normal stop',image_only=True,external_OCR=False,prompt='pinned HF native USER/image/ASSISTANT chat; question newline Answer the question using a single word or phrase.',image_processor='pinned HF CLIP shortest-edge resize336 and center crop336; RGB; no new square-pad',visual_tokens=576,model_loads=1,formal_conditions=['Native','SVH-R'],formal_requests=256,questions_per_condition=128,smoke_ids=[x['question_id'] for x in manifest['Smoke']],Native='unmodified HF4.49 complete32-layer full visual access',SVH_R=dict(answer_keep_layers=list(KEEP),visual_Hidden_exit_before_layer=25,R_to_image_forbidden_from_layer=20,answer_attention_physical_skip=[2,3,4,5],cache='physical selective original image KV only L11..L24; L2..L5 absent; preserve logical RoPE positions'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'),input_binding_sha256=sha(OUT/'input_bindings.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),scorer='official TextVQA ten-answer leave-one-out EvalAI softscore + independent recomputation',scorer_sha256=sha(RUN/'references/textvqa_eval.py'),smoke_gate=dict(Native_NoOp='strict torch.equal valid last logits and complete tokens',Dense_Physical='allclose with frozen historical llava logit atol0.10 rtol0.003, hidden relativeRMSE<=0.01 and cosine error<=1e-4; full greedy tokens equal on3 smoke',minimum_true_decode_steps=8),hard_stop='nonfinite/OOM/source or input mismatch/structural KV RoPE mask error/cache pollution or hook leak/cap breach',quality_outputs='retain and score all incorrect/repeated/truncated answers, no deletion or tuning',bootstrap=dict(clusters='Image SHA paired',repetitions=20000,seed=20261112,CI='ordinary unadjusted95%'),scope='same small set within-model Native vs SVH-R; no speed, no GQA/OCRBench; not independent holdout; cross-model prompt/precision/processor differ',GitHub_branch='codex/llava-svh-r-textvqa128-3090')
 write(OUT/'protocol.json',protocol);write(OUT/'cpu_tests.json',test())
 write(OUT/'cpu_input_validation.json',dict(status='PASS',main=128,smoke=3,unique_image_SHA=len({x['image_sha256'] for x in manifest['Main']}),IDs_order_images_questions_Gold_equal=True,input_files_contain_only_model_tensors=True,processor='pinned LLaVA native',CUDA_initialized=torch.cuda.is_initialized()))
 print(json.dumps(dict(status='CPU_PREPARED_AND_TESTED',source_files=len(files),formal=256,Main=128,Smoke=3,protocol_sha256=sha(OUT/'protocol.json'))))
if __name__=='__main__':main()
