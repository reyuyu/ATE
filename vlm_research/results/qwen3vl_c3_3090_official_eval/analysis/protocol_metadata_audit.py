"""Read-only OCR prompt/source audit; preserve frozen manifest and all predictions."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,zipfile

O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
C=Path('/home/featurize/work/ate-qwen3090-control-29535')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 manifest=read(O/'input_manifest.json');data=read(O/'data_restore.json');v=data['views']['OCRBench-v1']
 entry=next(x for x in read(C/'datasets_restore_plan.json')['packages'] if x['kind']=='OCRBench')
 archive=next(Path(x['path']) for x in data['archives'] if x['name']==entry['name'])
 source=Path(v['path']).read_bytes();assert sha(source)==v['inputs_SHA256']
 with zipfile.ZipFile(archive) as z:
  names=[n for n in z.namelist() if n.endswith('/inputs.jsonl')];assert len(names)==1
  original=z.read(names[0]);assert original==source,'Restored input differs from trusted backup member'
 rows=[json.loads(x) for x in source.decode().splitlines()];by=defaultdict(dict)
 for s in manifest['rows']:
  if s['benchmark']=='OCRBench-v1':by[s['question_id']][s['protocol_label']]=s
 assert len(rows)==len(by)==1000
 examples=[]
 for r in rows:
  a=by[r['question_id']]['ATE_PAPER_PROTOCOL'];b=by[r['question_id']]['OFFICIAL_PROTOCOL']
  assert a['body']==r['user_text'] and a['source_inputs_SHA256']==v['inputs_SHA256']
  assert a['body']==b['body']==r['question']+'\nAnswer the question using a single word or phrase.'
  assert a['request_SHA256']==b['request_SHA256']
  assert '\\n' not in a['body'] and '\n' in a['body']
  if len(examples)<2:examples.append(dict(question_id=r['question_id'],ATE_body_repr=repr(a['body']),OFFICIAL_body_repr=repr(b['body']),request_SHA256=a['request_SHA256']))
 result=dict(status='PROTOCOL_DESCRIPTION_CORRECTION_INPUTS_UNCHANGED',at_UTC=datetime.now(timezone.utc).isoformat(),benchmark='OCRBench-v1',affected_field='input_manifest.json.protocol_differences OCR newline description',original_description='Restored ATE user_text contains literal backslash-n; official uses a real newline',verified_correction='All 1000 restored ATE and official bodies already use the same real newline and are byte-identical; no literal backslash-n exists in these user_text fields',trusted_backup_archive=str(archive),archive_expected_SHA256=entry['sha256'],archive_integrity_provenance='Previously verified full archive SHA256 in data_restore; not rehashed in this audit',source_member=names[0],source_member_SHA256=sha(original),restored_source_exact_backup_member=True,source_input_SHA256_matches_frozen_manifest=True,question_pairs=1000,identical_prompt_pairs=1000,identical_request_pairs=1000,official_prompt_matches_author_real_newline=1000,model_input_changed=False,scorer_changed=False,algorithm_changed=False,frozen_manifest_changed=False,predictions_changed=False,request_sharing_valid=True,GPU_rerun_required=False,examples=examples)
 (O/'protocol_description_correction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('OCR_PROTOCOL_DESCRIPTION_ONLY_CORRECTED',json.dumps({k:result[k] for k in ['status','source_member_SHA256','question_pairs','identical_prompt_pairs','model_input_changed','GPU_rerun_required']}),flush=True)
if __name__=='__main__':main()
