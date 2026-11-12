import json,shutil,os
from run_common import *
O=L/'results/llava';cpu=json.loads((O/'cpu_bindings.json').read_text());rows={r['question_id']:r for r in json.loads((O/'rows.json').read_text())};old=json.loads((L/'history/fastv_native1000.json').read_text());rp=json.loads((L/'history/fastv_protocol.json').read_text())
assert rp['revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9' and rp['OCRBench']['max_new_tokens']==128 and rp['environment']['precision']=='FP16' and rp['environment']['attention']=='eager'
assert rp['environment']['torch']=='2.6.0+cu124' and rp['environment']['transformers']=='4.49.0'
reused={};differences=[];current=load_records(O/'predictions/Native')
for r in old:
 q=r['question_id'];g=r['generation'];c=cpu[q]
 assert r['prompt']==c['prompt'] and r['input_binding']['input_hashes']==c['input_hashes'] and r['input_binding']['image_sha256']==c['image_sha256'] and r['gold']==rows[q]['answer']
 assert g['all_logits_finite'] and not g['fixed_output_length']
 ss=score(q,g['text']);assert ss['official_score']==r['score']
 reused[q]=dict(model='LLaVA-1.5-7B',method='Native',question_id=q,generation={k:g[k] for k in ['text','token_ids','eos','truncated']},image_sha256=c['image_sha256'],input_hashes=c['input_hashes'],**ss,origin='REUSED_VERIFIED_WINDOWS_FASTV_NATIVE_FULL',original_record_sha256=digest(r),source='history/fastv_native1000.json')
 reused[q]['generation']['empty']=not g['text'].strip()
 if q in current:
  cg=current[q]['generation']
  if cg['token_ids']!=g['token_ids']:differences.append(dict(question_id=q,historical_text=g['text'],duplicate_text=cg['text']))
assert len(reused)==1000 and sum(r['official_score'] for r in reused.values())==301
atomic(O/'verified_full_native_reuse.json',reused)
assert not (O/'excluded_duplicate_generations').exists();(O/'excluded_duplicate_generations').mkdir()
shutil.move(O/'predictions/Native',O/'excluded_duplicate_generations/Native')
atomic(O/'historical_inventory_correction.json',dict(status='CORRECTED_AND_DUPLICATES_EXCLUDED',error='Initial historical search missed the Windows FastV full Native archive; repeat generations were incorrectly started before discovering it. This violates the requested no-repeat rule.',historical_native_coverage=1000,historical_native_score=301,all_1000_prompt_gold_image_tensor_hashes_identical=True,excluded_duplicate_native_generations=len(current),excluded_question_ids=sorted(current),duplicate_token_changes=len(differences),duplicate_differences=differences,action='Interrupted first GPU process; retain all repeated raw files outside accepted predictions; use the historical 1000 only; continue ONLY missing Physical-C3 IDs in a separate process.',initial_smoke_inventory_claim_superseded=True,additional_llava_model_load_required=True))
if (O/'failure.json').exists():shutil.move(O/'failure.json',O/'intentional_interruption_attempt1.json')
if (O/'environment.json').exists():shutil.copy2(O/'environment.json',O/'environment_attempt1.json')
src=(L/'llava_run.py').read_text();src=src.replace("records={m:load_records(O/'predictions'/m) for m in P['conditions']}","records={m:load_records(O/'predictions'/m) for m in ['Physical-C3']}")
src=src.replace("complete={m:len(records[m])+len(reuse[m]) for m in records}","complete=dict(Native=1000,**{m:len(records[m])+len(reuse[m]) for m in records})")
src=src.replace("model_loads=1","model_loads=2")
src=src.replace("for m in P['conditions']:","for m in ['Physical-C3']:")
src=src.replace("atomic(O/'smoke.json'","atomic(O/'continuation_smoke.json'")
src=src.replace(" for s in rows:\n", " missing=[s['question_id'] for s in rows if s['question_id'] not in records['Physical-C3'] and s['question_id'] not in reuse['Physical-C3']]\n if missing:infer(missing[0],'Physical-C3',True)\n for s in rows:\n")
src=src.replace("  current=dict(question_id=q,method=m)","  assert m=='Physical-C3','Native GPU generation forbidden after historical inventory correction'\n  current=dict(question_id=q,method=m)")
assert src.count("for m in ['Physical-C3']")==3
(L/'llava_c3_continue.py').write_text(src)
atomic(O/'continuation_protocol.json',dict(reason='Correct initial incomplete historical inventory; Native 1000 fully reused, no further Native inference.',original_protocol_sha256=sha(O/'protocol.json'),wrapper_SHA256=sha(L/'llava_c3_continue.py'),only_condition='Physical-C3',missing_ids=[s['question_id'] for s in rows.values() if not (O/'predictions/Physical-C3'/f"{s['question_id']}.json").exists() and s['question_id'] not in json.loads((O/'verified_reuse.json').read_text())['Physical-C3']],scientific_parameters_unchanged=True))
print('HISTORY_CORRECTED_NATIVE_REUSE1000_EXCLUDED',len(current),'C3_MISSING',len(json.loads((O/'continuation_protocol.json').read_text())['missing_ids']))
