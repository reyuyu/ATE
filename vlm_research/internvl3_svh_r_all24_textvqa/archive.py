"""Persist complete follow-up evidence and freeze an explicit lightweight Git allowlist."""
from .common import *
import shutil,gzip,subprocess
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-all24-textvqa-3090");REL=Path("vlm_research/results/internvl3_svh_r_all24_textvqa_3090");DEST=ROOT/REL
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
 write(DEST/"physical_cutoff_audit.json",dict(status="PASS",configs=CONFIGS,R_image_boundary=20,uncompressed_sha256=sha(OUT/"physical_cutoff_audit.json"),complete_evidence="physical_cutoff_audit.full.json.gz",compressed_sha256=sha(DEST/"physical_cutoff_audit.full.json.gz"),uncompressed_server_path=str(server/"physical_cutoff_audit.json")))
 src=sqlite3.connect("file:"+str(RUN/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True);dst=sqlite3.connect(server/"predictions.sqlite");src.backup(dst);assert dst.execute("pragma integrity_check").fetchone()==("ok",) and dst.execute("select count(*) from predictions").fetchone()[0]==1792;dst.close();src.close()
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
 readme="""# All24 visual reaccess except L2-L5

Exactly one user-requested configuration: KEEP=L0,L1,L6..L27 (24 layers),cutoff28.
Same frozen128 historical TextVQA Image-only questions/order.
128 NEW formal quality answers plus1664 previous Main predictions reused,1792 total.
No new timing/control/other windows/GQA/OCRBench/training/Windows transfer.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED preserved.

## Meaning and source

P-1 and real Decode may read originalImageKV in every nonSKIP layer.
L2-L5 answerAttention/QKV/QK/AV/o_proj are physically skipped, Norm/MLP/Residual retained.
R->Image still prohibited fromL20. This condition retains these two SVH restrictions relative to Native.
VisualHidden exists in all28 language blocks, exits AFTER L27 before tokenwise finalNorm.
There is NO Prefill language-block visual-row saving.
Actual originalImageKV persists in24 KEEP layers,256 visualKV each; SKIP cache empty.
Original RoPE/logical positions retained. Inherited engine is byte-identical to parent, only KEEP differs.
Parent Commit 1407275cbdffca5d4a23cbf8b8d233002e2840dd, all92 inherited Python files preserved; all96 as-run files frozen BEFORE GPU.
analysis_source_manifest.json binds postprocessing separately.
CPU official28layerQwen2 Float64,1prefill+8decode; independent cache/position/storage/mask/cleanup checks.
GPU3short/medium/long Dense/Physical Smoke and Native/No-op strict validLogits check precede128 formal answers.
Small-sample numerical observations do not lift the prior global numerical waiver.

## Model, input and provenance

OpenGVLab/InternVL3-8B Revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
BF16/Eager,Torch2.6.0+cu124,TF4.49.0,Python3.11.8;batch1 greedy cap32,naturalEOS,logits_to_keep1.
Single448tile256visual,nativeChat,scorer-aligned short instruction,no OCR/Gold auxiliary input.
All131 prepared native token/pixel/prompt/roles bindings CPU-verified again.
GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, idle before loading, same GPU as previous physical windows.
Native/R0/R6/R14 original predictions still historical oldGPU, explicitly labeled.
Repeatedly used128 historical questions,NOT independent Holdout/noninferiority confirmation.
Official TextVQA ten-answer scores independently rederived via leave-one-out calculation for all1792 records.
Paired whole-image-SHA bootstrap5000,ordinary95%CI,uncorrected multiple comparisons.
24-layer and8/9-layer conditions have DIFFERENT budgets.

## Runtime and CPU audit

RUN=/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-all24-textvqa-3090
PREPARED=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
ExtraDeps=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps (isolated timm0.9.16)
Model=/home/featurize/ate-local-models/InternVL3-8B
All heavy resources localSSD; full raw evidence persisted on work.

Set ATE_DEEP_RUN=RUN,ATE_SEVEN_RUN=RUN,PYTHONPATH=RUN/code:ExtraDeps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_all24_textvqa.analyze then .report for CPU-only reaudit.
Official reference at RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py;
identical uploaded reference in references/textvqa_eval.py.
A fresh GPU run needs separate authorization and new protocol/output/source bindings; do not rerun completed inference for reporting.

## Evidence

per_question.csv contains all1792 rawanswers/Gold/Tokens/scores/source recordSHAs.
paired_soft_changes.csv has13 comparisons; per_question_changes.csv has every paired change/harm.
kv_layer_audit.csv has1792x28 actualTensor shape/dtype/length/Live/Storage layouts.
physical_cutoff_audit.full.json.gz is lossless fullSmoke shapes/positions/probabilities; rawJSON stays server_only.
NormalEOS KV means are NOT fixed32Token KV@32; no new speed numbers are claimed.
server_only/predictions.sqlite is a complete1792-row SQLite backup,not partialWAL.
Raw audits/logs remain persistent; no weight/image/preparedtensor/dependency/cache/key upload.
artifact_manifest.json inventories files except itself/SHA256SUMS/post-push receipt.
SHA256SUMS covers the artifactmanifest and excludes itself.
server_only/git_publish_receipt.json records immutable GitHub HTTPS per-file bytes/SHA and SSH remote branchhead agreement.
No main change/forcepush/PR/old result overwrite/automatic next experiment or server shutdown. GPU released.
"""
 (DEST/"README.md").write_text(readme);(ROOT/PKG/"README.md").write_text(readme)
 files=sorted([p for p in DEST.rglob("*") if p.is_file() and "server_only" not in p.relative_to(DEST).parts and p.name not in ("artifact_manifest.json","SHA256SUMS")]+[p for p in (ROOT/PKG).glob("*") if p.is_file()])
 raw=sorted(p for p in server.rglob("*") if p.is_file() and p.name!="git_publish_receipt.json")
 def item(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
 write(DEST/"artifact_manifest.json",dict(status="COMPLETE_EXPLORATORY_WITH_NUMERICAL_WAIVER",GitHub_branch="codex/internvl3-svh-r-all24-textvqa-3090",parent_source_commit=PARENT_COMMIT,model_revision=read(OUT/"protocol.json")["revision"],GPU_UUID=GPU,protocol_sha256=sha(OUT/"protocol.json"),input_manifest_sha256=sha(OUT/"sample_manifest.json"),GitHub_files=[item(p) for p in files],server_only_files=[item(p) for p in raw],self_index_exclusions=["artifact_manifest.json","SHA256SUMS","server_only/git_publish_receipt.json"],publish_receipt="server_only/git_publish_receipt.json generated after successful push",excluded_resources=["weights","images","prepared .pt tensors","dependencies","keys"]))
 files.append(DEST/"artifact_manifest.json");(DEST/"SHA256SUMS").write_text("".join(sha(p)+"  "+str(p.relative_to(ROOT))+"\n" for p in sorted(files)));files.append(DEST/"SHA256SUMS")
 plan=dict(status="READY",branch="codex/internvl3-svh-r-all24-textvqa-3090",files=[item(p) for p in sorted(files)],server_only_files=[item(p) for p in raw],persistent_root=str(ROOT),result=str(REL));write(RUN/"publish_plan.json",plan)
 assert all(sha(ROOT/x["path"])==x["sha256"] for x in plan["files"])
 print(json.dumps(dict(status="PERSISTED_VERIFIED",files=len(files),bytes=sum(x["bytes"] for x in plan["files"]),raw_files=len(raw))),flush=True)
if __name__=="__main__":main()
