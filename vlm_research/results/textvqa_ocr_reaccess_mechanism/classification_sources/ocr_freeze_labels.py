import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,csv,hashlib,datetime,collections
W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-ocr-reaccess-mechanism');O=W/'vlm_research/results/textvqa_ocr_reaccess_mechanism';P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_ocr_reaccess_mechanism')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rr):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
pre=json.loads((O/'preclassification_freeze.json').read_text());assert sha(O/'classification_protocol.md')==pre['protocol_sha256']
blind=json.loads((O/'blind_inputs.json').read_text());ann={r['question_id']:r for r in json.loads((O/'aligned_ocr_annotations.json').read_text())};labels=list(csv.DictReader((P/'blind_labels.tsv').open(),delimiter='\t'));assert len(labels)==300 and {int(r['index']) for r in labels}==set(range(1,301))
names=dict(D='OCR-Direct',R='OCR-Reasoning',N='Non-OCR Visual',A='Ambiguous');out=[];ev=[]
for b,r in zip(blind,labels):
 assert int(r['index'])==b['index'] and r['category'] in names and r['evidence'];a=ann[b['question_id']]
 out.append(dict(question_id=b['question_id'],image_id=b['image_id'],image_sha256=b['image_sha256'],question=b['question'],category=names[r['category']],confidence=r['confidence'],image_inspected=True,question_inspected=True,gold_or_predictions_used=False,classifier='Codex multimodal visual review of frozen input-only contact sheets; not human gold',needs_independent_human_review=True,review_priority='high' if r['category'] in ['A','R'] or r['confidence']!='high' else 'normal'))
 words=[x['text'] for x in a['TextOCR_words']];legible=[x for x in words if x!='.']
 ev.append(dict(question_id=b['question_id'],image_id=b['image_id'],image_path=b['image_path'],image_sha256=b['image_sha256'],category=names[r['category']],rationale=r['evidence'],blind_sheet=(b['index']-1)//12+1,image_view='Frozen original bytes rendered at preserved aspect ratio for input-only review; OCR metadata supplementary',TextOCR_matched=a['TextOCR_matched'],OCR_word_count=len(words),OCR_legible_count=len(legible),official_OCR_text=json.dumps(words,ensure_ascii=False),OCR_annotation_ids=json.dumps([x['annotation_id'] for x in a['TextOCR_words']]),OCR_provenance='Official TextOCR0.1 val, exact ID/stem/dimensions matched; dot means illegible/non-English',label_limit='Single AI review; classification uncertainty retained; no human inter-rater reliability'))
csvwrite(O/'ocr_labels.csv',out);csvwrite(O/'classification_evidence.csv',ev);csvwrite(O/'human_review_queue.csv',[r for r in out if r['review_priority']=='high'])
write(O/'classification_freeze.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels_sha256=sha(O/'ocr_labels.csv'),evidence_sha256=sha(O/'classification_evidence.csv'),blind_input_sha256=sha(O/'blind_inputs.json'),protocol_sha256=sha(O/'classification_protocol.md'),aligned_OCR_sha256=sha(O/'aligned_ocr_annotations.json'),counts=dict(collections.Counter(r['category'] for r in out)),n=300,image_inspected=300,source_labels_sha256=sha(P/'blind_labels.tsv'),frozen_before_per_question_effect_analysis=True,no_outcome_label_revision_permitted=True,not_independent_human_truth=True))
print(json.dumps(dict(status='CLASSIFICATION_FROZEN',counts=dict(collections.Counter(r['category'] for r in out)),labels_sha256=sha(O/'ocr_labels.csv')),ensure_ascii=False))
