"""Read-only CPU investigation of the recorded Native historical output mismatch."""
from pathlib import Path
import json,gzip,sqlite3,hashlib
B=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_repeatability');D=B/'session1';U=B.parent/'unified_e2e32_4090'
db=sqlite3.connect('file:'+str(D/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True);rows=[json.loads(p) for (p,) in db.execute('SELECT payload FROM records ORDER BY id')];db.close()
bad=[r for r in rows if not r['history_acceptance']['accepted']];assert len(bad)==1
r=bad[0];assert r['method']=='Native' and r['key']=='TextVQA:38508' and r['formal_index']==151
with gzip.open(U/'formal_requests.jsonl.gz','rt') as f:h=next(x for x in map(json.loads,f) if x['key']==r['key'] and x['method']==r['method'])
g=r['generation'];old=h['generation'];policy=r['history_acceptance'];assert policy['first_difference_0based']==11 and policy['current_EOS_positions_0based'][0]==policy['historical_EOS_positions_0based'][0]==3
from tokenizers import Tokenizer
tok=Tokenizer.from_file('/home/featurize/ate-local-models/llava-1.5-7b-hf/tokenizer.json')
fields=['layer','used_tokens','capacity_tokens','visual_tokens','live_bytes','storage_bytes','K_shape','V_shape','dtype','original_positions_sha256']
layout=all({k:x.get(k) for k in fields}=={k:y.get(k) for k in fields} for x,y in zip(g['cache']['layers'],old['cache']['layers']))
assert layout and r['input_hashes']==h['input_hashes'] and r['sample_sha256']==h['sample_sha256']
sources=json.loads((D/'source_manifest.json').read_text())['frozen_sources']
for p,v in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==v
native=[x for x in rows if x['method']=='Native'];formal=[x for x in rows if x['category']=='formal']
out=dict(status='CPU_AUDIT_COMPLETE',GPU_requests_added=0,stop_reason='Mandatory Native full32 historical equality failed; performance warnings did not stop the run',request_id=r['request_id'],formal_index=r['formal_index'],input_index=r['input_index'],key=r['key'],method=r['method'],policy=policy,current_text=tok.decode(g['token_ids'],skip_special_tokens=False),historical_text=tok.decode(old['token_ids'],skip_special_tokens=False),through_first_EOS_inclusive_exact=g['token_ids'][:4]==old['token_ids'][:4],difference_only_after_first_EOS=True,input_tensor_hashes_exact=True,sample_SHA256_exact=True,input_hashes=r['input_hashes'],KV_shapes_lengths_positions_dtypes_equal=layout,current_KV_live_bytes=g['cache']['live_bytes'],historical_KV_live_bytes=old['cache']['live_bytes'],all_logits_finite=g['all_logits_finite'],prefill_calls=g['prefill_calls'],real_decode_calls=g['real_decode_calls'],current_metrics={k:g[k] for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']},historical_metrics={k:old[k] for k in ['E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms']},formal_records_retained=len(formal),formal_records_passing_history_gate=sum(x['history_acceptance']['accepted'] for x in formal),Native_requests=len(native),Native_exact=sum(x['historical_tokens_exact'] for x in native),Native_formal=len([x for x in formal if x['method']=='Native']),frozen_source_files_verified=len(sources),logits_or_KV_numerical_comparison='Unavailable: formal records store finite checks, argmax IDs, layouts and timings; full per-step logits and numerical KV contents were not saved.',causal_root='Unresolved; post-EOS divergence does not alone establish an algorithm implementation error or answer quality loss.',authorization_boundary='Latest user still requires Native full32 equality. No automatic relaxation, second session or GPU rerun after this hard failure.')
(B/'comparison/hard_failure_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['policy','input_hashes']},ensure_ascii=False,indent=2))
