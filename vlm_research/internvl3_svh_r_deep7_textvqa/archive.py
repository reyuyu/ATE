"""Persist complete follow-up evidence and freeze an explicit lightweight Git allowlist."""
from .common import *
import shutil,gzip,subprocess
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-deep7-textvqa-3090");REL=Path("vlm_research/results/internvl3_svh_r_deep7_textvqa_3090");DEST=ROOT/REL
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
def gz(src,dst):
 with open(src,"rb") as f,open(dst,"wb") as raw:
  with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as g:shutil.copyfileobj(f,g)
 with gzip.open(dst,"rb") as f:assert hashlib.sha256(f.read()).hexdigest()==sha(src)
def main():
 assert read(OUT/"validation.json")["status"]=="PASS_COMPLETENESS_STRUCTURE_SCORING"
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
 DEST.mkdir(parents=True,exist_ok=True);server=DEST/"server_only";server.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.name!="physical_cutoff_audit.json":copy(p,DEST/p.name)
 copy(OUT/"physical_cutoff_audit.json",server/"physical_cutoff_audit.json");gz(OUT/"physical_cutoff_audit.json",DEST/"physical_cutoff_audit.full.json.gz")
 write(DEST/"physical_cutoff_audit.json",dict(status="PASS",keep=list(range(20,27)),cutoff=27,R_image_boundary=20,uncompressed_sha256=sha(OUT/"physical_cutoff_audit.json"),complete_evidence="physical_cutoff_audit.full.json.gz",compressed_sha256=sha(DEST/"physical_cutoff_audit.full.json.gz"),uncompressed_server_path=str(server/"physical_cutoff_audit.json")))
 src=sqlite3.connect("file:"+str(RUN/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True);dst=sqlite3.connect(server/"predictions.sqlite");src.backup(dst);assert dst.execute("pragma integrity_check").fetchone()==("ok",) and dst.execute("select count(*) from predictions").fetchone()[0]==944;dst.close();src.close()
 for p in (RUN/"logs").rglob("*"):
  if p.is_file():copy(p,server/"logs"/p.relative_to(RUN/"logs"))
 copy(RUN/"setup.json",DEST/"git_setup_validation.json")
 ref=RUN/"references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py";copy(ref,DEST/"references/textvqa_eval.py")
 for p in (RUN/"code"/PKG).glob("*.py"):copy(p,ROOT/PKG/p.name)
 frozen=read(OUT/"source_manifest.json")["runtime_source_sha256"]
 for p,h in frozen.items():
  assert sha(RUN/"code"/p)==h
  if p.startswith(PKG+"/"):assert sha(ROOT/p)==h
  else:assert hashlib.sha256(subprocess.check_output(["git","-C",str(ROOT),"show",PARENT_COMMIT+":"+p])).hexdigest()==h
 write(DEST/"analysis_source_manifest.json",dict(note="Post-GPU CPU-only analysis/report/archive/publish; frozen runtime unchanged",source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/PKG).glob("*.py") if str(p.relative_to(ROOT)) not in frozen}))
 readme="""# Deep7 L20-L26 follow-up

New user-requested fixed window only: keep=L20..L26, cutoff=27.
R->Image permissions still stop at L20. Original logical RoPE positions retained.
This is an exploratory follow-up on the same frozen historical B128, NOT an independent Holdout.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED remains.

## Counts and sources

128 new formal quality generations. No new speed, GQA, OCRBench, training or other windows.
816 real prior records reused: Main128 x6 (Native/R0/R6/R14/Deep6/Deep8) + Exit27_NoLateRead x48.
Parent Commit c5a1036abccc1ecf2d77d1b187bde55d64b8231a.
Deep6/Deep8/Exit27 and the new Deep7 used the same current GPU; four older baselines have the documented historical GPU limitation.
Actual GPU GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27.
Same fixed model OpenGVLab/InternVL3-8B revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
BF16/Eager, Torch2.6.0+cu124, TF4.49.0, batch1 greedy normalEOS cap32.
Image-only single448tile256visual; same native Chat Template and scorer-aligned short instruction.
Prepared inputs contain no Gold or OCR text; all 131 cached input tensor bindings rechecked on CPU.

## Locations

RUN=/home/featurize/ate-runs/internvl3-svh-r-deep7-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-deep7-textvqa-3090
Old verified prepared tensors=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Isolated timm0.9.16=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps
Model=/home/featurize/ate-local-models/InternVL3-8B
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
All runtime weights/images/prepared tensors remain on local SSD, no Windows transfer.

## CPU-only audit

Set ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN,
PYTHONPATH=RUN/code:/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_deep7_textvqa.analyze then .report.
Place the uploaded references/textvqa_eval.py at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py.
The primary official scorer and independent ten-answer leave-one-out scoring match on every saved record.

## Reproduction source

The parameterized engine is inherited byte-for-byte from the parent. No algorithm repair or new attention path.
source_manifest.json binds all70 as-run Python files, frozen BEFORE GPU; four new post-processing files are separately bound.
CPU tiny official28layer Qwen2 float64 checks 1prefill+8decode.
GPU Smoke has Native/No-op strict equality plus3 Dense/Physical short/medium/long tests.
run.py loads one model and permits only128 NEW formal records. Fresh independent caches.
A new GPU run would require separate authorization and a NEW output directory/protocol; do not relaunch this completed run.

## Evidence

per_question.csv includes all944 saved answers, Gold, scores, Token IDs and raw record SHA256.
kv_layer_audit.csv includes944x28 real cache layouts and storage bytes.
physical_cutoff_audit.full.json.gz is lossless full shape/position/probability evidence; rawJSON stays server_only.
Whole-image-SHA paired bootstrap5000, ordinary95%CI, no multiplicity correction or independent-holdout claim.
No Deep7 E2E@32 or KV@32 was measured; normalEOS KV is explicitly labeled.

server_only/predictions.sqlite is a complete SQLite backup (944 rows), never a partial WAL.
Raw checkpoints, audit and execution logs remain on persistent work. No model/image/cache/key upload.
artifact_manifest.json indexes uploaded light and server-only evidence, except itself and SHA256SUMS.
SHA256SUMS includes the artifact manifest but excludes itself.
The post-push mutable receipt server_only/git_publish_receipt.json is excluded from the hashed self-index to avoid Commit recursion.
GitHub readback verifies every uploaded file using immutable Commit HTTPS bytes plus SSH branch head.
No main changes, force push, PR, extra benchmark or automatic server shutdown. GPU released after generation.
"""
 (DEST/"README.md").write_text(readme);(ROOT/PKG/"README.md").write_text(readme)
 files=sorted([p for p in DEST.rglob("*") if p.is_file() and "server_only" not in p.relative_to(DEST).parts and p.name not in ("artifact_manifest.json","SHA256SUMS")]+[p for p in (ROOT/PKG).glob("*") if p.is_file()])
 raw=sorted(p for p in server.rglob("*") if p.is_file() and p.name!="git_publish_receipt.json")
 def item(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
 write(DEST/"artifact_manifest.json",dict(status="COMPLETE_EXPLORATORY_WITH_NUMERICAL_WAIVER",GitHub_branch="codex/internvl3-svh-r-deep7-textvqa-3090",parent_source_commit=PARENT_COMMIT,model_revision=read(OUT/"protocol.json")["revision"],GPU_UUID=GPU,protocol_sha256=sha(OUT/"protocol.json"),input_manifest_sha256=sha(OUT/"sample_manifest.json"),GitHub_files=[item(p) for p in files],server_only_files=[item(p) for p in raw],self_index_exclusions=["artifact_manifest.json","SHA256SUMS","server_only/git_publish_receipt.json"],publish_receipt="server_only/git_publish_receipt.json generated after successful push",excluded_resources=["weights","images","prepared .pt tensors","dependencies","keys"]))
 files.append(DEST/"artifact_manifest.json");(DEST/"SHA256SUMS").write_text("".join(sha(p)+"  "+str(p.relative_to(ROOT))+"\n" for p in sorted(files)));files.append(DEST/"SHA256SUMS")
 plan=dict(status="READY",branch="codex/internvl3-svh-r-deep7-textvqa-3090",files=[item(p) for p in sorted(files)],server_only_files=[item(p) for p in raw],persistent_root=str(ROOT),result=str(REL));write(RUN/"publish_plan.json",plan)
 assert all(sha(ROOT/x["path"])==x["sha256"] for x in plan["files"])
 print(json.dumps(dict(status="PERSISTED_VERIFIED",files=len(files),bytes=sum(x["bytes"] for x in plan["files"]),raw_files=len(raw))),flush=True)
if __name__=="__main__":main()
