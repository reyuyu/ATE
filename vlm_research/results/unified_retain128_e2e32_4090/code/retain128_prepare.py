"""Retain128 freeze; reads previous 30 inputs and code without rewriting history."""
from pathlib import Path
import sys,json,gzip,hashlib,os,shutil,subprocess,struct,math
R=Path('/home/featurize/work/ATE');E=R/'vlm_research/results/unified_e2e32_4090'
sys.path.append(str(E/'code'))
import e2e32_prepare as old
EXT=old.EXT;OLD=old.OLD;FAST=old.FAST;COMP=old.COMP;SOURCE=old.SOURCE;VCOMP=old.VCOMP;VSOURCE=old.VSOURCE;M=old.M;UUID=old.UUID
OLDNEWCOMP=old.NEWCOMP;NEWCOMP=EXT/'retain128_compatibility';OUT=R/'vlm_research/results/unified_retain128_e2e32_4090'
METHODS=['Native','SparseVLM','V2Drop','ApET','DUET']
REQUESTED=['Native','FastV','SparseVLM','V2Drop','ApET','DUET']
LABELS=dict(Native='Native Full',FastV='FastV-128',SparseVLM='SparseVLM-128',V2Drop='V²Drop-128',ApET='ApET-128',DUET='DUET-VLM-128 (C+all)')
BLOCKED={'FastV':dict(status='BLOCKED_EVIDENCE',reason='Pinned official FastV has configurable K/R and no uniquely documented average128 preset. No K/R was guessed or solved to force average128; no deep128 count presented as average128.')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
digest=old.digest;lines=old.lines
def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def expected_layout(method):
 return {'Native':[576]*32,'SparseVLM':[576]*3+[312]*4+[117]*9+[39]*16,'V2Drop':[576]*3+[161]*14+[40]*5+[1]*10,'ApET':[192]*16+[64]*16,'DUET':[205]*16+[102]*8+[0]*8}[method]
def verify_sources():
 p=read(E/'protocol.json');assert sha(E/'protocol.json')=='376e92f324de7249be25dd6c40c1644a4f62d32d05f82be8b9820f721480811b'
 for path,h in p['frozen_source_SHA256'].items():assert sha(path)==h,path
 for name,h in p['runner_source_SHA256'].items():assert sha(E/'code'/name)==h,name
 env=read(E/'environment.json')
 for r in env['sources'].values():assert sha(r['path'])==r['sha256'],r['path']
 if (OUT/'protocol.json').exists():
  for path,h in read(OUT/'protocol.json')['source_SHA256'].items():assert sha(path)==h,path
 return p
def prepare(freeze=True):
 OUT.mkdir(parents=True,exist_ok=True);p=verify_sources();rows,_,questions,refs=old.prepare(freeze=False)
 assert len(rows)==30 and [s['key'] for s in rows]==p['keys']
 sources=dict(p['frozen_source_SHA256'])
 for n,h in p['runner_source_SHA256'].items():sources[str(E/'code'/n)]=h
 for r in read(E/'environment.json')['sources'].values():sources[r['path']]=r['sha256']
 for f in (OUT/'code').glob('retain128_*.py'):
  if f.name not in ('retain128_analyze.py','retain128_publish.py'):sources[str(f)]=sha(f)
 sources[str(NEWCOMP/'retain128_adapter.py')]=sha(NEWCOMP/'retain128_adapter.py')
 for rel in ['sparsevlm_icml2025/official_source/README.md','sparsevlm_icml2025/official_source/llava/model/language_model/score.py','apet/official_source/scripts/llava1_6/textvqa.sh','apet/official_source/scripts/llava1_5/mme.sh','duet_vlm/official_source/README.md','v2drop/official_source/source_manifest.json','fastv/official_source/README.md']:
  sources[str(EXT/rel)]=sha(EXT/rel)
 audit={
  'Native':dict(status='CONFIRMED',parameters={'visual_tokens':576},paper_point=True),
  'FastV':BLOCKED['FastV'],
  'SparseVLM':dict(status='CONFIRMED',parameters={'retained_tokens':128,'topk_schedule':[303,110,36],'prune_after_layers':[2,6,15],'recycling':True},paper_point=True,evidence='Pinned official score.py retained_tokens=128, unchanged official recycling AST. Nominal budget ignores extra recycling and a layer convention offset; true average is audited.'),
  'V2Drop':dict(status='CONFIRMED',parameters={'prune_before_layers':[3,17,22],'keep_ratios':[.28,.25,0],'floor':True,'minimum_retained':1},paper_point=True,evidence='https://arxiv.org/html/2509.01552v2#A1 Appendix A remove72%,75%,100%. Only published parameters replace demo .5/.5; original norm/topk/floor/min1 retained.'),
  'ApET':dict(status='BUDGET_CONFIRMED_EXACT_QUALITY_POINT_UNCONFIRMED',parameters={'vision':192,'LLM':64,'prune_before_layer':16,'basis_tokens':10,'selection':'FPS'},paper_point=None,evidence='Official released scripts/llava1_6/textvqa.sh explicitly uses vision192 and L16 keep64; 16 layers each gives average128 for the unchanged 32-layer LLaVA backbone. LLaVA1.5 release scripts explicitly provide 288->96 and96->32, the same3:1 allocation. Paper Table1 reports128 but does not independently specify its allocation. This is a released-parameter128 budget transfer; do NOT claim proven identity to the published LLaVA1.5 quality row.'),
  'DUET':dict(status='CONFIRMED_BUDGET_C_PLUS_ALL',parameters={'dominant_including_CLS':170,'contextual':35,'cluster_width':4,'layer_list':[16,24],'ratios':[1,.5,0],'variant':'C+all','trained_weights':False},paper_point=False,evidence='https://arxiv.org/html/2602.18846v1 Appendix C Table12:170+35 for128. Released pdrop_rank_drop inference defaults to ALL post-image queries. Paper C last-token path only commented out; unchanged official C+all used explicitly.')}
 for m in METHODS:audit[m].update(expected_visual_by_layer=expected_layout(m),expected_actual_average=sum(expected_layout(m))/32,physical=True,use_cache=True)
 protocol=dict(status='FROZEN',task='Retain128 unified E2E@32 efficiency only',previous_commit='5ee2a0dbe78df336229b55fed14a4a6d0d880bed',GPU_UUID=UUID,
  model=p['model'],revision=p['revision'],model_path=str(M),torch=p['torch'],transformers=p['transformers'],precision='FP16',attention='eager',batch=1,greedy=True,use_cache=True,logits_to_keep=1,
  requested_methods=REQUESTED,admitted_methods=METHODS,preblocked=BLOCKED,method_labels=LABELS,keys=p['keys'],requested_formal_max=180,expected_admitted_formal=150,
  generated_tokens=32,prefill_calls=1,real_decode_calls=31,ignore_EOS=True,smoke_keys=p['smoke_keys'],drift=p['drift'],equal_warmup_per_method=2,
  timing=p['timing'],excluded=p['excluded'],included=p['included'],events=p['events'],cache=p['cache'],schedule=p['schedule'],configuration_audit=audit,
  official_source_receipts=p['official_source_receipts'],FastV_commit=p['FastV_commit'],V2Drop_commit='2a9431150f8ace3a1d9abcdc96a7d46acca597df',source_SHA256=sources,
  source_code_changes='Independent parameter-only subclass, new runtime copy, no installed package/frozen SVH or Retain192 file edits. V2Drop two ratios now paper128. Sparse wrapper parameter128/table loaded from source. ApET released192/64 parameters. DUET170/35. Selection/merging/pruning official core unchanged.',
  resume=p['resume'],stop_on=p['stop_on'],quality_requests=0,OCRBench_requests=0,Windows_backups=False,GitHub_upload_authorized=True,GPU_settings_changes=False,
  cross_run192_comparison='Same physical GPU UUID and inputs, different measurement sessions; use contemporaneous Native-normalized ratio and disclose cross-session uncertainty. Earlier V2Drop demo parameters were NOT paper192. DUET remains C+all.')
 if freeze and (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'Frozen protocol changed'
 elif freeze:
  save('protocol.json',protocol);save('configuration_audit.json',audit);save('source_manifest.json',dict(files=sources,official_commits={**{k:v['commit'] for k,v in p['official_source_receipts'].items()},'FastV':p['FastV_commit'],'V2Drop':protocol['V2Drop_commit']},efficiency_reference_commit=protocol['previous_commit']))
  save('historical_token_references.json',{k:v for k,v in refs.items() if k.endswith('|Native')});save('question_positions.json',questions)
  with gzip.open(OUT/'input_manifest.jsonl.gz','wt') as f:
   for s in rows:f.write(json.dumps(s,ensure_ascii=False)+'\n')
  save('method_status.json',BLOCKED)
 return rows,protocol,questions,{k:v for k,v in refs.items() if k.endswith('|Native')}
