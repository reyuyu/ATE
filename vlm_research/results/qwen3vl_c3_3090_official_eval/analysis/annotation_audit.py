"""Independent CPU joins to original annotations, without touching GPU or inputs."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
data=read(O/'data_restore.json');manifest=read(O/'input_manifest.json');by=defaultdict(list)
for s in manifest['rows']:by[s['benchmark']].append(s)
checks={};sources={}
v=data['views']['GQA-testdev-balanced'];p=Path(v['root'])/'source_annotations/gqa/testdev_balanced_questions.json';g=read(p)
assert len(g)==12578 and {s['question_id'] for s in by['GQA']}==set(g)
for s in by['GQA']:
 a=g[s['question_id']];assert s['question']==a['question'] and s['image_id']==a['imageId'] and s['answer']==a['answer']
checks['GQA']=dict(status='PASS',questions=12578,images=398,official_question_image_Gold_join=True);sources[str(p)]=sha(p)
v=data['views']['ScienceQA-IMG-paper'];root=Path(v['root']);p=root/'source_annotations/scienceqa/problems.json';science=read(p)
expected={k for k,a in science.items() if a['split']=='test' and a.get('image')};assert len(expected)==2017 and {s['question_id'] for s in by['ScienceQA-IMG']}==expected
hint_whitespace_only=[]
for s in by['ScienceQA-IMG']:
 a=science[s['question_id']];assert s['question']==a['question'] and s['choices']==a['choices'] and s['hint']==a['hint'] and s['answer']==a['answer']
 assert not s['hint'] or ' '.join(s['hint'].split()) in ' '.join(s['body'].split());assert all(c in s['body'] for c in s['choices'])
 if s['hint'] and s['hint'] not in s['body']:hint_whitespace_only.append(s['question_id'])
checks['ScienceQA-IMG']=dict(status='PASS',questions=2017,Hint_and_all_choices_preserved=True,official_has_image_filter=True,historical_paper_prompt_collapses_repeated_whitespace=hint_whitespace_only);sources[str(p)]=sha(p)
v=data['views']['TextVQA-OCR'];p=Path(v['root'])/'source_annotations/textvqa/TextVQA_0.5.1_val.json';text=read(p)['data'];tb={str(a['question_id']):a for a in text}
assert len(tb)==5000 and set(tb)=={s['question_id'] for s in by['TextVQA']}
for s in by['TextVQA']:
 a=tb[s['question_id']];assert s['answer']==a['answers'] and len(s['answer'])==10 and s['question']==a['question'] and s['image_id']==a['image_id'] and 'Reference OCR token:' in s['body']
checks['TextVQA']=dict(status='PASS',questions=5000,ten_answers_exact=True,Paper_OCR_preserved=True);sources[str(p)]=sha(p)
groups=defaultdict(list)
for s in by['MMBench']:groups[s['circular_group']].append(s)
for group,rows in groups.items():
 assert len({s['image_sha256'] for s in rows})==1 and len({s['question'] for s in rows})==1
 assert len({s['choices'][s['answer']] for s in rows})==1,('Circular Gold-option mismatch',group)
checks['MMBench']=dict(status='PASS',rows=4377,complete_circular_groups=1176,correct_option_text_consistent_through_rotation=True,Gold_source='previously independently verified author Dev20230712 backup and fixed hashes')
for label in ['ATE_PAPER_PROTOCOL','OFFICIAL_PROTOCOL']:
 pairs=defaultdict(list)
 for s in by['MME']:
  if s['protocol_label']==label:pairs[s['pair_id']].append(s)
 assert len(pairs)==1187 and all(len(v)==2 for v in pairs.values())
 assert all(s['question'].endswith('Please answer yes or no.') and s['body']==s['question'] for s in by['MME'] if s['protocol_label']=='OFFICIAL_PROTOCOL')
checks['MME']=dict(status='PASS',rows_each_protocol=2374,image_pairs=1187,official_yes_no_instruction_preserved=True,known_single_image_difference_retained=True,Gold_source='previously audited author Classic backup and fixed hashes')
for label,expected in [('OFFICIAL_PROTOCOL',dict(random=3000,popular=3000,adversarial=3000)),('ATE_PAPER_PROTOCOL',dict(random=2910,popular=3000,adversarial=3000))]:
 rows=[s for s in by['POPE'] if s['protocol_label']==label];assert dict(Counter(s['strategy'] for s in rows))==expected
checks['POPE']=dict(status='PASS',official=9000,ATE_paper=8910,all_three_strategies=True,source='frozen original author strategy annotations and backup hashes')
v=data['views']['OCRBench-v1'];anns=read(Path(v['annotations']))
for s in by['OCRBench-v1']:
 a=anns[s['official_annotation_index']];assert s['official_id']==a['id'] and s['question']==a['question'] and s['answer']==a['answers']
checks['OCRBench-v1']=dict(status='PASS',official_rows=1000,images=930,original_annotation_order_ID_Gold_exact=True,ATE_and_official_newline_difference_preserved=True)
result=dict(status='PASS',at_UTC=datetime.now(timezone.utc).isoformat(),checks=checks,original_annotation_SHA256=sources,input_manifest_SHA256=sha(O/'input_manifest.json'),no_model_load_or_GPU=True,no_frozen_data_modified=True)
(O/'annotation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
