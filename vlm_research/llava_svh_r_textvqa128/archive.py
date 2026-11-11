from common import *
import shutil,subprocess,gzip
S=read(RUN/'state.json');ROOT=Path(S['root']);DEST=ROOT/S['result'];PKG=ROOT/S['pkg']
def main():
 assert read(OUT/'validation.json')['status']=='PASS' and read(OUT/'run_completion.json')['formal_records']==256
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();assert not apps,apps
 verify_sources();DEST.mkdir(parents=True,exist_ok=True);PKG.mkdir(parents=True,exist_ok=True)
 for p in OUT.rglob('*'):
  if p.is_file():
   target=DEST/p.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 for p in CODE.rglob('*.py'):
  target=PKG/p.relative_to(CODE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 # Score reference is standalone, unchanged official implementation.
 (PKG/'references').mkdir(exist_ok=True);shutil.copyfile(RUN/'references/textvqa_eval.py',PKG/'references/textvqa_eval.py')
 raw=DEST/'server_only';raw.mkdir(exist_ok=True)
 src=sqlite3.connect('file:'+str(RUN/'predictions.sqlite')+'?mode=ro',uri=True);target=sqlite3.connect(raw/'predictions.sqlite');src.backup(target);target.close();src.close()
 target=sqlite3.connect('file:'+str(raw/'predictions.sqlite')+'?mode=ro',uri=True);assert target.execute('pragma integrity_check').fetchone()[0]=='ok';assert target.execute('select count(*) from predictions').fetchone()[0]==256;target.close()
 logs=raw/'logs';logs.mkdir(exist_ok=True)
 for p in (RUN/'logs').glob('*'):
  if p.is_file() and p.name!='publish.log':shutil.copyfile(p,logs/p.name)
 shutil.copyfile(RUN/'state.json',raw/'state.json')
 shutil.copyfile(RUN/'initial_attempt_empty_predictions.sqlite',raw/'initial_attempt_empty_predictions.sqlite')
 # Independent full initial diagnostic is small and retained for audit.
 (PKG/'README.md').write_text('''# LLaVA-SVH-R / Native same TextVQA128 image-only
Model: llava-hf/llava-1.5-7b-hf @ b234b804b114d9e37bb655e11cbbb5f5e971b7a9. FP16/Eager, TF4.49, Torch2.6+cu124.
The 26 frozen_sources files are byte-identical dependencies extracted from 5ee2a0dbe78df336229b55fed14a4a6d0d880bed. No frozen algorithm was edited.
Native / SVH-R each128 requests, same original sample IDs/order/images/Gold as the InternVL B128; model-specific prompt and 336 processor preserved. No OCR text, no speed experiment.
SVH-R: answer KEEP L11..L24, image Hidden exit entering L25, R->Image forbidden fromL20, answer Attention skip L2..L5 and real compact selective KV. Original P-1 prefill K/V for skip layers still computed; future Decode QKV all absent.
Use current results/protocol.json and source_manifest.json as authoritative as-run settings. initial_attempt preserves the original failed nine-step numerical assertion; v2 assesses numeric thresholds only before native EOS, with9 synthetic steps for finite/cache/position structure. Post-EOS numeric failure is retained; no claim of fixed-length numerical PASS.
Entry points (use separate empty local SSD RUN and verified local model/images, never overwrite results):
1. Copy this directory to RUN/code and references/textvqa_eval.py to RUN/references. Export ATE_LLAVA128_RUN=RUN, same CUDA_VISIBLE_DEVICES UUID, CPU threads1, CUBLAS_WORKSPACE_CONFIG=:4096:8, HF_HUB_OFFLINE=1.
2. prepare.py reproduces CPU inputs from the pinned previous sample_manifest and local original images; documented absolute paths must be bound to verified local resources. CPU preparation contains no GPU inference. Source sample manifest is distributed in the parent commit and its SHA is recorded.
3. python -B RUN/code/run.py runs native-EOS correctness smoke and exactly256 paired formal generations with checksum-bound resumable SQLite. Full checkpoint stays on server, not GitHub.
4. python -B RUN/code/analyze.py independently checks records, official and leave-one-out scoring, actual KV, paired20000 image-cluster bootstrap.
5. archive.py and publish.py only persist/verify this experiment's allowlist on its own branch. They never run inference.
Source_manifest freezes only inference/preparation files; analyze/diagnose/archive/publish post-code is separately SHA-bound by analysis_source_manifest.json. Frozen official scorer normalization is shared by two independently written score aggregations.
Models, images, prepared Tensor files, dependencies and all credentials are excluded from Git.
''')
 post={}
 frozen=read(OUT/'source_manifest.json')['files']
 for p in CODE.glob('*.py'):
  if p.name not in frozen:post[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
 write(DEST/'analysis_source_manifest.json',dict(frozen_inference_manifest_sha256=sha(OUT/'source_manifest.json'),post_processing_files=post))
 light=sorted([p for base in (DEST,PKG) for p in base.rglob('*') if p.is_file() and 'server_only' not in p.parts and p.name not in ('artifact_manifest.json','SHA256SUMS')])
 rawfiles=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw.rglob('*')) if p.is_file() and p.name!='git_publish_receipt.json']
 files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in light]
 write(DEST/'artifact_manifest.json',dict(status='COMPLETE_REQUESTED_QUALITY_SCOPE_WITH_POST_EOS_NUMERICAL_LIMITATION',GPU_UUID=GPU,model_revision=REVISION,GitHub_branch=S['branch'],parent_commit=S['parent'],source_commit=S['source_commit'],protocol_sha256=sha(OUT/'protocol.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'),GitHub_files=files,server_only_files=rawfiles,self_index_exclusions=['artifact_manifest.json (self hash)','SHA256SUMS (self hash)'],excluded_resources=['model weights','images','datasets','prepared tensors','caches','credentials'],publish_receipt='server_only/git_publish_receipt.json contains final Commit and remote readback after publication, excluded to avoid self-reference'))
 light.append(DEST/'artifact_manifest.json')
 (DEST/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(light)))
 light.append(DEST/'SHA256SUMS')
 files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(light)]
 write(RUN/'publish_plan.json',dict(files=files))
 print(json.dumps(dict(status='ARCHIVED',files=len(files),bytes=sum(x['bytes'] for x in files),largest=max(x['bytes'] for x in files),server_only=len(rawfiles))))
if __name__=='__main__':main()
