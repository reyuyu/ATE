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
 # Independent full initial diagnostic is small and retained for audit.
 (PKG/'README.md').write_text("# LLaVA-SVH-R late11 L21..L31 / same frozen TextVQA128\nModel: llava-hf/llava-1.5-7b-hf @ b234b804b114d9e37bb655e11cbbb5f5e971b7a9. FP16/Eager, Torch2.6+cu124, TF4.49, greedy, cap32, normalEOS.\nThis supplement generates exactly128 late11 predictions, reuses256 Native and original middle14 records from immutable parent177c631521d5c9d0436768e11e529ec492c2cbc8 on the same GPU UUID and model inputs. 3 deterministically selected shortest/median/longest Native calibration reruns must reproduce prior complete tokens/text before reuse.\nUnchanged26 frozen_sources come from5ee2a0dbe78df336229b55fed14a4a6d0d880bed. Independent engine.TerminalMiddleWriter copies the unchanged parent forward math and adds terminal physical Hidden exit afterL31 before final tokenwise norm when cutoff32. CPU cutoff25 regression is torch.equal. No frozen module is modified.\nLate11 config: P-1/Decode original image KV permitted onlyL21..L31; L2..L5 answer Attention skipped, R->Image forbidden fromL20. All32 language blocks retain image Hidden. No Prefill block savings or latency claimed. Actual cache lengths, original RoPE positions, norm/MLP and projection call counts are smoke-audited.\nInference entry: python -B RUN/code/run.py with ATE_LLAVA128_RUN=RUN, current authorized CUDA_VISIBLE_DEVICES UUID, CPU threads1, CUBLAS_WORKSPACE_CONFIG=:4096:8 and offline HF. Use localSSD model and prepared Tensor paths from state.json; never overwrite the result directory.\nprepare.py binds and verifies the exact parent sample_manifest, input_bindings, 131 image/tensor hashes and256 original SQLite records. It reuses original prepared tensors through an SSD-local link, without image/model downloads. It creates a new frozen protocol and source manifest before GPU work.\ncpu_tests.py uses the official tiny32layer Llama to check terminal exit, original positions, actual compact KV, skip path and oldcutoff25 strict regression.\nrun.py executes 3 outside-input Native/NoOp and Dense/Physical Smoke, 3 Native reuse calibrations, then only the128 new late11 quality generations. Source/protocol/input hashes protect resumable SQLite. Wrong/empty/truncated answers remain scored. No speed, GQA, OCRBench or other windows.\nThe inherited v2 numeric policy is frozen before this run: unchanged0.10/0.003 logits thresholds on valid naturalEOS outputs, strict hidden thresholds, 9 fixed synthetic steps for structure/finite, retaining postEOS numerical failures separately. This is a normalEOS quality result, not a fixed-length numerical certification.\nanalyze.py independently verifies384 records (256 reused +128 new), official ten-answer softscoring and independent leave-one-out formula, actual32layer cache and paired20000-image-cluster bootstrap. Gold is only used for scoring, not model Tensor inputs.\narchive.py and publish.py persist and independently verify the current allowlist only on this branch. Full SQLite and logs remain under server_only; SHA256SUMS and artifact_manifest explicitly distinguish GitHub and server-only files. Model/images/prepared tensors/private credentials never enter Git.\n")
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
