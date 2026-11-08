"""Match the official prepared question file and summarize trusted image provenance."""
from pathlib import Path
import json,gzip,subprocess,hashlib
from protocol_helpers import O,D,M
R=Path('/home/featurize/work/ATE');H=R/'vlm_research/results/featurize_new3090_recovery/history/c3v2-gqa-testdev-balanced-12578/vlm_research/results/c3v2_gqa_testdev_balanced_12578'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
path=D/'audit_support/gqa_native_protocol/official_llava_gqa_testdev_balanced.jsonl'
prepared=[json.loads(l) for l in path.read_text().splitlines()];official={str(x['question_id']):x for x in prepared}
with gzip.open(H/'sample_manifest.jsonl.gz','rt') as f:hist=[json.loads(l) for l in f]
assert len(prepared)==len(official)==len(hist)==12578 and set(official)=={s['question_id'] for s in hist}
for s in hist:
 row=official[s['question_id']]
 assert row['text']==s['prompt_text'],(s['question_id'],row['text'],s['prompt_text'])
 assert Path(row['image']).stem==s['image_id'],(s['question_id'],row['image'],s['image_id'])
save('official_prepared_input_audit.json',{'status':'VERIFIED','questions':12578,'question_ids_exact':True,'question_text_and_single_word_suffix_exact':True,'image_ids_exact':True,
 'source':'Official LLaVA docs eval.zip, extracted only gqa/llava_gqa_testdev_balanced.jsonl via HTTP Range',
 'sha256':sha(path),'official_zip_member_metadata':'official_eval_zip_members.json','official_zip_7B_predictions_present':False,
 'official_62_score':'Model Zoo 7B row, not recomputed from 13B predictions. Actual7B output archive not present in official eval.zip.'})
source='vlm_research/results/c3v2_gqa_testdev_balanced_12578/source_dataset_audit.json'
raw=subprocess.check_output(['git','show','57e37ce024b1b2adf9fa24fc24aabec32a1fed08:'+source],cwd=R)
(O/'historical_image_provenance.json').write_bytes(raw)
print('OFFICIAL_PREPARED_INPUTS_EXACT',len(hist),'HISTORICAL_IMAGE_PROVENANCE',raw.decode()[:14000],flush=True)
