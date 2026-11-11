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
 ref=Path(S['zero25_run'])
 src=sqlite3.connect('file:'+str(ref/'predictions.sqlite')+'?mode=ro',uri=True);target=sqlite3.connect(raw/'reused_L0_L25_predictions.sqlite');src.backup(target);target.close();src.close()
 check=sqlite3.connect('file:'+str(raw/'reused_L0_L25_predictions.sqlite')+'?mode=ro',uri=True);assert check.execute('pragma integrity_check').fetchone()[0]=='ok' and check.execute('select count(*) from predictions').fetchone()[0]==128;check.close()
 (PKG/'README.md').write_text("""# LLaVA-SVH-R L10-L25 on frozen TextVQA128
128 original TextVQA image-only questions, official ten-answer consensus soft scoring. No OCR text.
Model llava-hf/llava-1.5-7b-hf @ b234b804b114d9e37bb655e11cbbb5f5e971b7a9, FP16/Eager, Torch2.6+cu124, Transformers4.49, Batch1, greedy, max32, normalEOS, use_cache and logits_to_keep1.
P-1 and real Decode image-key reaccess onlyL10..L25 (16 layers). Answer Attention L2..L5 remains physically skipped. R->Image onlyL0..L19; visual Hidden physically exits beforeL26; original RoPE logical positions and text paths remain unchanged. Actual compact visual KV persists only in16 allowed layers. No frozen core algorithm was edited.
Same original128 ID/order/imageSHA/question/Gold and per-model prompt/CLIP336 bindings. prepared files contain only input_ids,attention_mask,pixel_values, never Gold.
Sources include unchanged26 frozen core modules from5ee2a0dbe78df336229b55fed14a4a6d0d880bed and byte-identical TerminalMiddleWriter. New source/protocol manifests were frozen beforeGPU. prepare.py validates131 image/tensor bindings, prior Native/middle256 and L0-L25 128 original records/source manifests on the sameGPU. CPU checks strict legacycut25 regression and newcut26 actual physical deletion, KV and original positions.
run.py loads once, runs3 outside-input Native/No-op and Dense/Physical numerical/structural smokes,3 Native exact full-token reuse calibration checks, then exactly128 newL10-L25 generations. Source/protocol/input SHA-bound SQLite supports strict resume. It never continues the paused singleton sweep.
Inherited v2 numerical scope: unchanged0.10/0.003 logits tolerance on normalEOS valid outputs, hidden relativeRMSE<=0.01/cosineerror<=1e-4.9 forced steps afterEOS only check structure/finite and retain all numeric differences; Native/No-op remains strict torch.equal. No inference or tolerance changed based on quality results.
analyze.py verifies512 records:128 new plus Native128/middle14 128 from177c631521d5c9d0436768e11e529ec492c2cbc8 and previousL0-L25 128 fromf7db13807e810c8dfd6cb5db0c18329d171c0745. All reused data preserve original protocol/source/checksum/Commit and match current input/GPU. It independently recomputes official scoring vs leave-one-out, every actual32layer cache layout, and paired imageSHA-cluster20000 bootstrap. All wrong/empty/truncated outputs count.
Set ATE_LLAVA128_RUN to a new isolated SSD RUN with state.json, original verified model/images/prepared CPU tensors and manifests; copy this package to RUN/code and scorer to RUN/references. Use CPUthreads1,CUBLAS_WORKSPACE_CONFIG=:4096:8,offlineHF and the recorded singleGPU. See protocol/sample/input/source/model manifests. Never resume across changedSHA or overwrite original results.
archive.py preserves the full new128 checkpoint and original384 reused records with logs under persistent server_only. Git only includes lightweight code,CSV,report,protocol,validation and SHA lists. publish.py uses an experiment allowlist on its codex branch and independently verifies every remote immutable-Commit byte. No main edits, forcepush or PR.
Same128 questions have been explored repeatedly; intervals are ordinary unadjusted exploratory95%, not independent confirmation or full5000 accuracy. PreviousL0-L25 also exits beforeL26, while originalmiddle14 exits beforeL25; do not treat all three as differing only in layer membership. No efficiency, OCRBench,GQA,InternVL or additional layer search. Model/images/prepared/caches/credentials are excluded.
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
