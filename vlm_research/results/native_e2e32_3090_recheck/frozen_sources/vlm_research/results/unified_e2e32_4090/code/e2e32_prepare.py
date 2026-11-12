"""CPU-only freeze: same 30 historical inputs; eight requested methods, 32 tokens."""
from pathlib import Path
import json,gzip,hashlib,os,subprocess,shutil
R=Path('/home/featurize/work/ATE')
EXT=R/'vlm_research/external_baselines'
OLD=R/'vlm_research/results/v2drop_quick_reproduction'
FAST=R/'vlm_research/results/fastv_k3_192_kv_4090_20261109'
CAL=FAST/'native_calibration'
COMP=EXT/'fastv/compatibility'
SOURCE=EXT/'fastv/official_source/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py'
VCOMP=EXT/'v2drop/compatibility'
VSOURCE=EXT/'v2drop/official_source/V2Drop.py'
NEWCOMP=EXT/'e2e32_compatibility'
OUT=R/'vlm_research/results/unified_e2e32_4090'
M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
UUID='GPU-724fa9c3-d33a-df3f-e349-533cb4634a53'
METHODS=['Native','FastV','SparseVLM','V2Drop','ApET','DUET','SVH','SVH-R']
LABELS=dict(zip(METHODS,['Native Full','FastV','SparseVLM','V²Drop','ApET','DUET-VLM (C+all)','llava-SVH','llava-SVH-R']))
BLOCKED={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def lines(p):
 with gzip.open(p,'rt') as f:return list(map(json.loads,f))
def verify_sources():
 p=read(OLD/'protocol.json')
 for name,h in p['frozen_ATE_sources'].items():assert sha(R/name)==h,name
 assert sha(OLD/'runtime.py')==read(OLD/'execution_config.json')['sources']['runtime.py']
 e=read(FAST/'environment.json')
 for name in ('fastv_runtime.py','fastv_hf_adapter.py'):assert sha(COMP/name)==e['sources'][name],name
 assert sha(SOURCE)=='48dfd9351aafede82ccc15bbb772571293a0669e0b7051572dc344c8a40e0726'
 assert sha(VSOURCE)==sha(OLD/'official_sources/V2Drop.py')=='e39ed8d99c3e1f23bd156067bfd28179827771df3556631b6daed4427a7b8b61'
 assert sha(VCOMP/'hf_adapter.py')==sha(OLD/'hf_adapter.py')
 return p
def prepare(freeze=True):
 OUT.mkdir(parents=True,exist_ok=True);p=verify_sources();keys=p['efficiency_keys'];assert keys==read(FAST/'protocol.json')['efficiency']['keys'] and len(keys)==30
 original={s['key']:s for s in lines(OLD/'input_manifest.jsonl.gz')};new=lines(FAST/'efficiency_input_manifest.jsonl.gz')
 assert [s['key'] for s in new]==keys
 rows=[original[k] for k in keys];checked={}
 for a,b in zip(rows,new):
  assert a==b and a['input_ids'].count(32000)==576
  if a['image_path'] not in checked:checked[a['image_path']]=sha(a['image_path'])
  assert checked[a['image_path']]==a['image_sha256'],a['key']
 qs={str(s['question_id']):s['question_positions'] for s in lines(OLD.parent/'featurize_4090_gqa_official_aligned_threeway/input_manifest.jsonl.gz')}
 questions={s['key']:qs[str(s['question_id'])] for s in rows if s['benchmark']=='GQA'}
 refs={}
 for s in lines(FAST/'efficiency_full.jsonl.gz'):
  if s['condition'] in ('Native','FastV'):refs[s['key']+'|'+s['condition']]=s['generation']['token_ids'][:32]
 smoke_keys=[next(k for k in keys if k.startswith(b+':')) for b in ('GQA','TextVQA')]
 with gzip.open(OLD/'operator_counts_full.jsonl.gz','rt') as f:
  for s in map(json.loads,f):
   if s['key'] in smoke_keys and s['condition'] in ('C3-V2','C3-V2+L11-L24','V2Drop'):
    method={'C3-V2':'SVH','C3-V2+L11-L24':'SVH-R','V2Drop':'V2Drop'}[s['condition']]
    refs[s['key']+'|'+method]=s['result']['token_ids'][:32]
 sources={str(R/name):h for name,h in p['frozen_ATE_sources'].items()}
 for path in (OLD/'runtime.py',COMP/'fastv_hf_adapter.py',SOURCE,VCOMP/'hf_adapter.py',VSOURCE,NEWCOMP/'e2e32_official_compat.py'):sources[str(path)]=sha(path)
 for folder,relative in [('apet','llava/model/llava_arch.py'),('apet','llava/model/modeling_llama_x.py'),('apet','llava/model/utils.py'),('duet_vlm','visionzip/utils.py'),('duet_vlm','visionzip/clip_encoder_cw.py'),('duet_vlm','llava/model/modeling_llama_pdrop.py'),('sparsevlm_icml2025','llava/model/language_model/modelling_sparse_llama.py'),('sparsevlm_icml2025','llava/model/language_model/utils.py'),('sparsevlm_icml2025','llava/model/language_model/score.py')]:
  path=EXT/folder/'official_source'/relative;sources[str(path)]=sha(path)
 code={f.name:sha(f) for f in sorted((OUT/'code').glob('*.py')) if f.name!='e2e32_analyze.py'}
 receipts={folder:read(EXT/folder/'source_receipt.json') for folder in ('sparsevlm_icml2025','apet','duet_vlm')}
 protocol=dict(status='FROZEN',task='Unified E2E@32 efficiency only; zero quality benchmark requests',requested_methods=METHODS,method_labels=LABELS,preblocked=BLOCKED,
  model='llava-hf/llava-1.5-7b-hf',revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',model_path=str(M),GPU_UUID=UUID,
  torch='2.6.0+cu124',transformers='4.49.0',precision='FP16',attention='eager',batch=1,greedy=True,use_cache=True,logits_to_keep=1,
  keys=keys,requested_formal_requests=240,formal_requests_per_audited_method=30,generated_tokens=32,prefill_calls=1,real_decode_calls=31,ignore_EOS=True,
  smoke_keys=smoke_keys,smoke_trace_steps=[0,1,2,3,4,5,6,7,8,31],equal_warmup_per_method=2,
  schedule='For each frozen input rotate the surviving methods list by input index modulo its length; no mixing of hardware UUID or input sets.',
  drift={'keys':smoke_keys,'baseline_rounds':3,'after_each_inputs':5,'threshold_fraction':0.05,'metrics':['E2E_ms','TPOT_ms'],'comparison':'Mean of same two probe inputs vs mean of three baseline rounds; >5% absolute relative change pauses before further formal requests. Probe records excluded from formal data.'},
  timing='CUDA synchronize then start before per-request controller/position/mask setup; TTFT after Vision+Projector+Prefill+argmax+finite enqueue+sync; E2E after token 32 and necessary controller teardown+sync. TPOT=(E2E-TTFT)/31.',
  excluded='File reading, tokenizer/processor, H2D, SHA verification, model loading, warmup, final KV audit, CPU token extraction, telemetry and output I/O',
  included='Vision, projector, full Prefill, 31 real cached Decode, selection/clustering/pruning, necessary algorithm setup/teardown, greedy argmax and common finite enqueue',
  events='Identical 32 pairs created before timing; reported event mean over 31 Decode calls, no per-step CPU extraction or synchronize',
  cache='Real terminal persistent CUDA K/V tensors after 32 outputs: 31 generated input tokens cached. Layer shape, used extent, live/storage bytes audited; merged visual tokens counted as visual representations, original identity is not falsely assigned.',
  algorithms={'Native':'576 visual tokens, 32 layers; calibrated unmodified HF 4.49 forward',
   'FastV':'Pinned official static KV selector K3 R2/3; 576 -> 192 entering L3; original gathered RoPE',
   'V2Drop':'Byte-identical historical adapter/selector: entering L3/L17/L22, 576 -> 288 -> 144 -> 1; NOT paper 192-token quality operating point',
   'SVH':'Frozen MiddleWriter cutoff20, zero persistent visual KV, answer Attention physically bypassed L2..5',
   'SVH-R':'Frozen MiddleWriter cutoff25, visual KV only L11..24, same L2..5 bypass',
   'ApET':'Official unchanged AST encode_images, FPS, both token_merging functions, layer_prune; vision 288, basis10, 96 entering L16; compact RoPE; average192',
   'DUET':'Official unchanged AST CLIP Attention/EncoderLayer/local clustering/pdrop_rank_drop; dominant300 incl CLS + contextual7, cluster_width4; 307 -> 153 entering L16 -> 0 entering L24. Current source ranks ALL post-image text (C+all), not paper main C last-token ranking. Average191.75; no trained weights.',
   'SparseVLM':'ICML v1.5 nominal192 schedule [300,200,110], prune AFTER zero-based L2/L6/L15, official recycling adds 9/4/3 clustered tokens: real visual layouts 576 (L0..2),309 (L3..6),204 (L7..15),113 (L16..31), mean206.5. Official scaled_dot_product_attention is actually explicit eager QK, FP16 softmax, AV; this exact function and original ranking/pruning/merging AST are called unchanged. Bridge bypasses broken old sdpa-only class wiring and deprecated APIs without using fused SDPA/Flash or changing core math. Compact RoPE and per-layer real cache Decode.'},
  official_source_receipts=receipts,FastV_commit='d1659729b5bf1be225e99ee15783deeea80f63b1',old_evidence_commit='0c94eaa451a6000e2a7e0bec5570278a1de12292',
  frozen_source_SHA256=sources,runner_source_SHA256=code,images_SHA256_verified=len(checked),manifest_rows_equal=True,
  compatibility='Official AST compiled unchanged and called on proxies bridging legacy rotary and attributes; same frozen modules/weights. New-method independent compact-position HF decoder loop. BridgeNative no-op must match stock Native logits and tokens before use. Installed packages and frozen algorithm sources unchanged.',
  resume='SQLite FULL synchronous immutable payload+SHA, same pinned UUID/protocol/source/input. Every process repeats smoke/warmup and drift baseline; session IDs retained. Drift failures stop the process and require investigation; no automatic rebaselining.',
  stop_on='OOM/nonfinite/input or source mismatch/invalid algorithm KV or positions; drift over5%; other GPU process; GPU UUID mismatch',quality_requests=0,GitHub_writes=False,GPU_settings_changes=False)
 if freeze and (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'Frozen protocol or source changed'
 elif freeze:
  save('protocol.json',protocol);save('historical_token_references.json',refs);save('question_positions.json',questions)
  with gzip.open(OUT/'input_manifest.jsonl.gz','wt') as f:
   for s in rows:f.write(json.dumps(s,ensure_ascii=False)+'\n')
  save('method_status.json',BLOCKED)
 return rows,protocol,questions,refs
