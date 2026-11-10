"""Publish a paused, auditable snapshot; never mutate predictions, inputs or scoring."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import csv,gzip,hashlib,json,subprocess
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110');W=L/'worktree';O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';P=O/'protocol_review';P.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Paused single worker required'
pause=read(O/'protocol_review_pause.json');assert pause['status']=='PAUSED_FOR_PROTOCOL_REVIEW_GPU_RELEASED'
m=read(O/'input_manifest.json');diagnosis=read(O/'prompt_and_pause_diagnosis.json');grouped=defaultdict(list)
for s in m['rows']:grouped[s['benchmark'],s['protocol_label']].append(s)
with gzip.open(O/'per_question.csv.gz','rt',encoding='utf-8',newline='') as f:observed=list(csv.DictReader(f))
counts=defaultdict(lambda:defaultdict(set))
for r in observed:counts[r['benchmark'],r['protocol_label']][r['condition']].add(r['key'])
matrix=[];inventory=[]
for (b,label),rows in grouped.items():
 c=counts[b,label];pairs=c['Native']&c['Physical-Original'];matrix.append(dict(benchmark=b,frozen_label=label,expected_questions=len(rows),Native_records=len(c['Native']),Physical_records=len(c['Physical-Original']),complete_pairs=len(pairs),max_new_tokens=rows[0]['max_new_tokens'],processor='Frozen Qwen model default RGB processor; no per-benchmark pixel budget',protocol_claim='Dataset and scorer provenance checked; full Qwen author run recipe NOT validated'))
 inventory.append(dict(benchmark=b,frozen_label=label,examples=[{k:r[k] for k in ['key','question_id','question','body','answer','image_sha256','max_new_tokens','source_inputs_SHA256','source_Gold_SHA256','request_SHA256'] if k in r} for r in rows[:2]]))
table(P/'protocol_matrix.csv',matrix);dump(P/'prompt_examples.json',inventory)
official=grouped['POPE','OFFICIAL_PROTOCOL'];paper={}
for s in grouped['POPE','ATE_PAPER_PROTOCOL']:paper.setdefault((s['image_sha256'],s['question'],s.get('strategy')),s)
n=diagnosis['progress']['completed_questions'];cases=[]
for s in official[max(0,n-40):n]:
 t=paper.get((s['image_sha256'],s['question'],s.get('strategy')))
 if t is None:continue
 assert s['answer']==t['answer']
 for v in [s,t]:
  for condition in ['Native','Physical-Original']:
   index=read(O/'benchmark_predictions/POPE'/v['protocol_label']/(hashlib.sha256(v['key'].encode()).hexdigest()+'__'+condition+'.json'));rawpath=O/index['request_path'];raw=read(rawpath);r=raw['result'];text=r['text'];words=text.partition('.')[0].replace(',','').split(' ');parsed='no' if any(x in words for x in ['No','not','no']) else 'yes'
   assert index['scoring']['score']==float(parsed==s['answer']),(s['question_id'],condition)
   cases.append(dict(question_id=s['question_id'],paper_question_id=t['question_id'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],protocol_label=v['protocol_label'],condition=condition,body=v['body'],request_SHA256=v['request_SHA256'],generation_record_file_SHA256=sha(rawpath),generation_binding=raw['binding'],input_ids=raw['input_layout']['input_ids'],input_tensor_hashes=raw['input_layout']['input_tensor_hashes'],raw_text=text,token_ids=r['token_ids'],generated_tokens=r['generated_tokens'],truncated=r['truncated'],empty=r['empty'],timing=r['timing'],author_word_rule_parsed=parsed,independent_score=float(parsed==s['answer']),saved_initial_scoring=index['scoring']))
assert len(cases)==diagnosis['matched_question_pairs']*4
with (P/'pope_matched_prompt_cases.jsonl').open('w',encoding='utf-8') as f:
 for r in cases:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
table(P/'pope_matched_prompt_summary.csv',diagnosis['matched_prompt_statistics'])
dump(P/'publication_authorization.json',dict(at_UTC=datetime.now(timezone.utc).isoformat(),authorization='Latest human message explicitly requests synchronizing evidence to GitHub and explaining suspected protocol issues',scope='Publish light evidence on independent codex/qwen3vl-c3-3090-official-eval branch only; no main changes, PR or new GPU run',supersedes_prior_no_push_for_this_evidence_only=True,frozen_protocol_no_push_field_preserved_as_historical_record=True))
failed_namespace=dict(status='CPU_PRECHECK_FAILED_NOT_ACTIVATED',reason='Private-parent prototype asserted findmnt ext4 and failed for /home/featurize/work; no GPU process started and no host mount changed',failing_source='namespace_storage_launcher.py',line=19,not_a_validated_fix=True)
dump(P/'namespace_fix_attempt.json',failed_namespace)
dump(P/'snapshot.json',dict(status='PARTIAL_PAUSED_FOR_PROTOCOL_REVIEW',at_UTC=datetime.now(timezone.utc).isoformat(),GPU_processes='',source_commit_before_evidence=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),source_manifest_SHA256=sha(O/'source_manifest.json'),protocol_SHA256=sha(O/'protocol.json'),input_manifest_SHA256=sha(O/'input_manifest.json'),model_manifest_SHA256=sha(O/'model_manifest.json'),scored_prediction_records=len(observed),view_matrix=matrix,formal_efficiency_started=False,prompt_cohort_question_pairs=diagnosis['matched_question_pairs'],prompt_cohort_requests=len(cases),protocol_claim='Full official Qwen evaluation recipe NOT validated; frozen OFFICIAL_PROTOCOL labels retained verbatim for provenance'))
print('REVIEW_EVIDENCE_BUILT',json.dumps(dict(records=len(observed),paired_prompt_cases=len(cases),matrix=matrix),ensure_ascii=False),flush=True)
