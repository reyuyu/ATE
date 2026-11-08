"""CPU-only source/evidence preservation and GPU release receipt, after exit."""
from pathlib import Path
import json,hashlib,shutil,subprocess,importlib.metadata,importlib.util,sys
from datetime import datetime,timezone
from e2e32_prepare import *
def copy(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert src.read_bytes()==dest.read_bytes(),str(dest)
 else:shutil.copyfile(src,dest)
def main():
 p=read(OUT/'protocol.json');state=read(OUT/'progress.json')
 assert state['phase']=='COMPLETE' and state['status'] in ('PASS','PARTIAL')
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
 assert not active,'GPU not released: '+active
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();assert UUID in gpu
 save('gpu_release.json',dict(status='RELEASED',at=datetime.now(timezone.utc).isoformat(),GPU=gpu,compute_processes=active,no_further_GPU_work=True))
 # Source release archives and manifests are permanently on the work disk.
 packages=[]
 for name in ('sparsevlm_icml2025','apet','duet_vlm'):
  receipt=read(EXT/name/'source_receipt.json');commit=receipt['commit'];archive=Path('/home/featurize/ate-local-data/source_cache')/(name+'-'+commit+'.tar.gz')
  assert sha(archive)==receipt['archive_sha256'] and archive.stat().st_size==receipt['archive_bytes']
  persistent=EXT/'official_source_archives'/archive.name;copy(archive,persistent)
  manifest=read(EXT/name/'official_source_manifest.json')
  for relative,info in manifest.items():
   src=EXT/name/'official_source'/relative;assert src.stat().st_size==info['bytes'] and sha(src)==info['sha256'],str(src)
  for basename in ('source_receipt.json','official_source_manifest.json'):copy(EXT/name/basename,OUT/'official_sources'/name/basename)
  packages.append(dict(method=name,commit=commit,path=str(persistent),bytes=persistent.stat().st_size,sha256=sha(persistent),verified_source_files=len(manifest)))
 save('official_source_packages.json',dict(packages=packages))
 # Thin method-specific entries retain all adaptation code in external_baselines.
 for method,name in [('SparseVLM','sparsevlm_icml2025'),('ApET','apet'),('DUET','duet_vlm')]:
  dest=EXT/name/'compatibility';dest.mkdir(exist_ok=True)
  text='''from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'e2e32_compatibility'))
from e2e32_official_compat import OfficialCompat
def controller(adapter, layout, trace=False):
    return OfficialCompat(adapter, layout, METHOD, trace=trace)
'''.replace('METHOD',repr(method))
  (dest/'adapter.py').write_text(text)
  recipe=dict(method=method,algorithm=p['algorithms'][method],shared_adapter=str(NEWCOMP/'e2e32_official_compat.py'),shared_adapter_sha256=sha(NEWCOMP/'e2e32_official_compat.py'),core_commit=read(EXT/name/'source_receipt.json')['commit'],runtime_result_path=str(OUT),fixed_GPU_UUID=UUID,output_tokens=32)
  (dest/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n')
  (dest/'README.md').write_text('# Frozen HF 4.49 compatibility\n\n'+p['algorithms'][method]+'\n\nOfficial source is unchanged. Independent adapter uses AST-loaded original core functions. See recipe.json and the unified result protocol. No trained weights or quality benchmark generated.\n')
  for x in dest.iterdir():
   if x.is_file():copy(x,OUT/'code/external_compatibility'/name/x.name)
 for name in ('e2e32_preparation_1.log','e2e32_formal_1.log','e2e32_formal_2.log'):
  src=Path('/home/featurize/ate-local-data')/name
  if src.exists():copy(src,OUT/'logs'/name)
 # The only installed dependency was the missing official SparseVLM pin.
 origin=Path(importlib.util.find_spec('einops').origin)
 save('dependency_receipt.json',dict(added_package='einops',version=importlib.metadata.version('einops'),import_path=str(origin),source_sha256=sha(origin),reason='Missing module at CPU import; installed only official SparseVLM pyproject pin einops==0.6.1 (42KB wheel). No Torch/Transformers reinstall.',python=sys.executable))
 fast_commit=subprocess.check_output(['git','-C',str(EXT/'fastv/official_source'),'rev-parse','HEAD'],text=True).strip();assert fast_commit==p['FastV_commit']
 save('algorithm_equivalence_audit.json',dict(status='PASS',FastV_commit=fast_commit,official_archives_verified=True,frozen_source_files_verified=True,
  ApET_vision_bridge='Uses the identical CLIP hidden_states[-2] including CLS before official encode_images removes CLS. Does not retain/stack auxiliary CLIP attention outputs discarded by encode_images; no ranking/merging uses them. Frozen Eager CLIP computes identical image features. This removes unused legacy auxiliary export, not algorithmic selection.',
  SparseVLM='Official SDPA-named function is manual eager matmul/softmax/matmul; executed AST unchanged in independent API bridge. Explicit causal mask from original CPU helper and necessary transfers remain counted. Original broken class wiring not edited. Pruning block compiled unchanged. Actual post-merge sequence and cache lengths are audited.',
  DUET='Official inference function with use_salient_tokens=False ranks all post-image text, C+all; not the last-token C variant. Complete source remains unmodified; legacy rotary wrapper uses the same frozen HF rotary module, exact compact position conventions.',
  timing='One shared external generation lifecycle; independent algorithm controllers have necessary implementation overhead included. New method decoder API compatibility is independent, and BridgeNative stock logits/tokens exact. This does not assert cross-implementation kernel-level identity.',
  no_FastV_or_SVH_source_changes=True,no_quality_Benchmarks=True,no_GitHub_write=True))
 # Snapshot the actual imported HF Native core and byte-identical ATE core too.
 # This writes archival copies only; imports still use the original frozen files.
 environment=read(OUT/'environment.json');native={}
 for key in ('installed_llama','installed_llava','installed_cache'):
  src=Path(environment['sources'][key]['path']);dest=EXT/'native_hf/official_source'/src.name
  copy(src,dest);copy(src,OUT/'code/native_hf_snapshot'/src.name);native[src.name]=dict(original_path=str(src),sha256=sha(src))
 clip=Path(environment['sources']['installed_llava']['path']).parents[2]/'models/clip/modeling_clip.py'
 copy(clip,EXT/'native_hf/official_source'/clip.name);copy(clip,OUT/'code/native_hf_snapshot'/clip.name);native[clip.name]=dict(original_path=str(clip),sha256=sha(clip))
 native_receipt=dict(transformers_version='4.49.0',provenance='Actual imported package core; byte-identical archival copies, not replacements',files=native)
 (EXT/'native_hf/source_receipt.json').write_text(json.dumps(native_receipt,indent=2)+'\n')
 ate_sources=read(OLD/'protocol.json')['frozen_ATE_sources']
 for relative,h in ate_sources.items():
  assert sha(R/relative)==h;copy(R/relative,EXT/'ate_svh/frozen_sources'/relative)
 (EXT/'ate_svh/source_receipt.json').write_text(json.dumps(dict(evidence_commit='0c94eaa451a6000e2a7e0bec5570278a1de12292',frozen_sources=ate_sources,original_sources_unchanged=True),indent=2)+'\n')
 copy(EXT/'native_hf/source_receipt.json',OUT/'official_sources/native_hf/source_receipt.json')
 copy(EXT/'ate_svh/source_receipt.json',OUT/'official_sources/ate_svh/source_receipt.json')
 print(json.dumps(dict(status='PRESERVED',packages=packages,GPU=gpu),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
