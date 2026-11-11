from common import *
import shutil,subprocess,gzip
S=read(RUN/'state.json');ROOT=Path(S['root']);DEST=ROOT/S['result'];PKG=ROOT/S['pkg']
def main():
 assert read(OUT/'validation.json')['status']=='PASS_REQUESTED_NATURAL_EOS_QUALITY' and read(OUT/'run_completion.json')['formal_records']==128
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
 target=sqlite3.connect('file:'+str(raw/'predictions.sqlite')+'?mode=ro',uri=True);assert target.execute('pragma integrity_check').fetchone()[0]=='ok';assert target.execute('select count(*) from predictions').fetchone()[0]==128;target.close()
 logs=raw/'logs';logs.mkdir(exist_ok=True)
 for p in (RUN/'logs').glob('*'):
  if p.is_file() and p.name!='publish.log':shutil.copyfile(p,logs/p.name)
 shutil.copyfile(RUN/'state.json',raw/'state.json')
 parent=Path(S['parent_run'])
 src=sqlite3.connect('file:'+str(parent/'predictions.sqlite')+'?mode=ro',uri=True);target=sqlite3.connect(raw/'reused_parent_predictions.sqlite');src.backup(target);target.close();src.close()
 check=sqlite3.connect('file:'+str(raw/'reused_parent_predictions.sqlite')+'?mode=ro',uri=True);assert check.execute('pragma integrity_check').fetchone()[0]=='ok' and check.execute('select count(*) from predictions').fetchone()[0]==256;check.close()
 (PKG/'README.md').write_text("""# LLaVA-SVH-R L0-L25 on frozen TextVQA128
Exact original 128 TextVQA Image-only inputs; official ten-answer soft scoring, same GPU.
Model llava-hf/llava-1.5-7b-hf @ b234b804b114d9e37bb655e11cbbb5f5e971b7a9, FP16/Eager, Torch2.6+cu124, Transformers4.49, greedy, Batch1, normalEOS max32.
User-requested reaccess L0..L25, while answer Attention L2..L5 stays physically skipped. Effective KEEP: L0,L1,L6..L25 (22 layers). R->Image allowed L0..L19 only. Physical visual Hidden exits beforeL26. Only22 layers persist original image KV, skip layers have no persistent KV; original logical positions retained.
26 frozen core files and TerminalMiddleWriter math are unchanged. Configuration, Observer cutoff and CPU tests are separate as-run files. CPU tests strictly regress original cutoff25 and verify actual cutoff26 deletion.
prepare.py verifies exact original sample/input manifests,131 original images and prepared CPU tensors, prior256 checkpoint checksums and sameGPU, model trusted receipt; it freezes source/protocol before any GPU.
run.py loads the model once, executes3 outside-input structural/naturalEOS numerical Smoke,3 exact-token Native reuse calibration checks, then exactly128 new L0-L25 generations. Native and middle14 each128 are reused from177c631521d5c9d0436768e11e529ec492c2cbc8, never relabeled as new.
Inherited v2 numeric scope is explicit: same0.10/0.003 normalEOS valid Logit thresholds and hidden relativeRMSE0.01/cosineerror1e-4;9 forced steps afterEOS are structure/finite tests only and all numeric differences retained. Native/No-op remains strict torch.equal. No threshold was changed in response to results.
Set ATE_LLAVA128_RUN to a new isolated localSSD RUN with verified local model/images/prepared tensors, source/sample/input/model manifests and state.json. Copy this package to RUN/code, scorer to RUN/references. See protocol/source manifests for actual paths, CPU threads1, offline HF and CUDA UUID. Do not resume old protocol/source mismatches or overwrite existing results.
analyze.py independently verifies384 records (128 new+256 reused), all32layer KV/positions, official TextVQA scores versus leave-one-out calculation and paired20000 imageSHA-cluster Bootstrap. Wrong/empty/repeated/truncated answers are retained.
This same128 set was explored repeatedly; it is not independent confirmation or the full5000 benchmark. No speed/OCRBench/GQA/InternVL or continued singleton scanning.
Complete SQLite and logs remain under server_only; lightweight code/CSV/protocol/report/checksums are uploaded on this own branch. No model,images,prepared tensors,keys,or credentials are in Git.
""")
 post={}
 frozen=read(OUT/'source_manifest.json')['files']
 for p in CODE.glob('*.py'):
  if p.name not in frozen:post[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
 write(DEST/'analysis_source_manifest.json',dict(frozen_inference_manifest_sha256=sha(OUT/'source_manifest.json'),post_processing_files=post))
 light=sorted([p for base in (DEST,PKG) for p in base.rglob('*') if p.is_file() and 'server_only' not in p.parts and p.name not in ('artifact_manifest.json','SHA256SUMS')])
 rawfiles=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw.rglob('*')) if p.is_file() and p.name!='git_publish_receipt.json']
 files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in light]
 write(DEST/'artifact_manifest.json',dict(status='COMPLETE_REQUESTED_NATURAL_EOS_QUALITY',GPU_UUID=GPU,model_revision=REVISION,GitHub_branch=S['branch'],parent_commit=S['parent'],source_commit=S['source_commit'],protocol_sha256=sha(OUT/'protocol.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'),GitHub_files=files,server_only_files=rawfiles,self_index_exclusions=['artifact_manifest.json (self hash)','SHA256SUMS (self hash)'],excluded_resources=['model weights','images','datasets','prepared tensors','caches','credentials'],publish_receipt='server_only/git_publish_receipt.json contains final Commit and remote readback after publication, excluded to avoid self-reference'))
 light.append(DEST/'artifact_manifest.json')
 (DEST/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(light)))
 light.append(DEST/'SHA256SUMS')
 files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(light)]
 write(RUN/'publish_plan.json',dict(files=files))
 print(json.dumps(dict(status='ARCHIVED',files=len(files),bytes=sum(x['bytes'] for x in files),largest=max(x['bytes'] for x in files),server_only=len(rawfiles))))
if __name__=='__main__':main()
