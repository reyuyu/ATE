"""CPU-only freeze of the existing complete MME Paper view for prefix+SVH."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,subprocess
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';F=B/'v2drop_prefix_svh_accuracy_3090';T=B/'featurize_new3090_c3v2_evaluation';O=B/'v2drop_prefix_svh_mme_full_3090';D=Path('/home/featurize/data/ate-recovery/normalized/MME-paper')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
assert not O.exists(),'Preserve an existing MME evaluation directory'
O.mkdir()
assert read(F/'score_summary.json')['status']=='COMPLETE' and read(F/'physical_smoke.json')['status']=='PASS'
fp=read(F/'protocol.json')
for name,h in fp['frozen_sources'].items():assert sha(R/name)==h,name
for name,h in fp['sources'].items():assert sha(F/name)==h,name
metadata=read(T/'run_metadata.json');cfg=metadata['config'];expected=cfg['inputs']['MME']
assert sha(D/'inputs.jsonl')==expected['inputs_sha256'] and sha(D/'answers.json')==expected['answers_sha256']
current=[json.loads(l) for l in (D/'inputs.jsonl').read_text().splitlines()];byid={r['question_id']:r for r in current};gold=read(D/'answers.json')
samples=[s for s in gz(T/'mapped_manifest.jsonl.gz') if s['benchmark']=='MME'];old={r['key']:r for r in gz(T/'predictions.jsonl.gz') if r['benchmark']=='MME'}
assert len(samples)==len(current)==len(byid)==len(gold)==len(old)==2374
pairs=defaultdict(list)
for s in samples:
 c=byid[s['question_id']];r=old[s['key']]
 assert all(c[k]==s[k] for k in ['question_id','image_id','image_path','question','prompt','user_text'])
 assert s['answer']==gold[s['question_id']]==r['gold'] and s['max_new_tokens']==128 and s['input_view']=='MME-paper'
 assert s['image_sha256']==r['image_sha256'] and Path(s['image_path']).is_file()
 pairs[s['pair_id']].append(s)
assert len(pairs)==1187 and all(len(p)==2 and len({s['image_id'] for s in p})==1 and {s['answer'] for s in p}=={'yes','no'} for p in pairs.values())
assert cfg['model_revision']==fp['revision'] and cfg['dtype']=='float16' and cfg['attention']=='eager' and cfg['greedy'] and cfg['num_beams']==1
with gzip.GzipFile(filename=str(O/'input_manifest.jsonl.gz'),mode='wb',mtime=0) as f:
 for s in samples:f.write((json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n').encode())
with gzip.GzipFile(filename=str(O/'SVH_reference_predictions.jsonl.gz'),mode='wb',mtime=0) as f:
 for s in samples:f.write((json.dumps(old[s['key']],ensure_ascii=False,separators=(',',':'))+'\n').encode())
known=[dict(key=s['key'],image_id=s['image_id'],historical_image_sha256=s.get('historical_image_sha256'),recovered_image_sha256=s['image_sha256']) for s in samples if s.get('known_recovered_image_version_difference')]
assert len(known)==2
for s in samples[:2]:assert sha(s['image_path'])==s['image_sha256']
assert sha(samples[next(i for i,s in enumerate(samples) if s.get('known_recovered_image_version_difference'))]['image_path'])==known[0]['recovered_image_sha256']
save('input_audit.json',dict(status='PASS',questions=2374,pairs=1187,tasks=dict(Counter(s['task'] for s in samples)),
 exact_current_frozen_input_and_Gold_SHA256=True,source_mapped_manifest_sha256=sha(T/'mapped_manifest.jsonl.gz'),
 current_inputs_sha256=sha(D/'inputs.jsonl'),current_Gold_sha256=sha(D/'answers.json'),
 mapped_manifest_sha256=sha(O/'input_manifest.jsonl.gz'),SVH_reference_sha256=sha(O/'SVH_reference_predictions.jsonl.gz'),
 known_MME_image_difference=known,data_not_replaced=True,all_image_paths_present=True,
 image_hash_policy='Reuse verified restored archive manifests, check the paired smoke image and known changed image; no full image rehash',
 source_physical_smoke_sha256=sha(F/'physical_smoke.json'),source_fusion_protocol_sha256=sha(F/'protocol.json'),
 previous_strict_GPU_numerical_gate='PASS unchanged, same engine/GPU; no repeated GQA/TextVQA generation',
 frozen_sources=fp['frozen_sources'],fusion_sources={n:sha(F/n) for n in ['fusion_engine.py','run_fusion.py']},
 model=fp['model'],revision=fp['revision'],code_commit=fp['code_commit'],old_SVH_config=cfg))
print('MME_CPU_FREEZE_PASS',len(samples),'questions',len(pairs),'pairs',len(known),'known SHA-difference questions')
