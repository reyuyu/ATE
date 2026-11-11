"""Persist complete follow-up evidence and freeze an explicit lightweight Git allowlist."""
from .common import *
import shutil,gzip,subprocess
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-eight-shift-textvqa-3090");REL=Path("vlm_research/results/internvl3_svh_r_eight_shift_textvqa_3090");DEST=ROOT/REL
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
 src=sqlite3.connect("file:"+str(RUN/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True);dst=sqlite3.connect(server/"predictions.sqlite");src.backup(dst);assert dst.execute("pragma integrity_check").fetchone()==("ok",) and dst.execute("select count(*) from predictions").fetchone()[0]==1280;dst.close();src.close()
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
 readme="""# Three eight-layer TextVQA follow-ups

Only user-requested windows L18-L25/cutoff26, L17-L24/cutoff25, L16-L23/cutoff24.
Same frozen128 historical TextVQA Image-only questions/order, no new sampling.
384 new formal quality answers,896 previous Main answers reused, total1280 records.
No new timing, extra control, other window, GQA, OCRBench, training or Windows transfer.
EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER; original NUMERICAL_EQUIVALENCE_FAILED remains.

## Sources and input

Parent Commit 1260af2e5e7a8a29d5b9cdbb06e06f106ce48d37, with74 inherited Python files retained byte-for-byte.
source_manifest.json freezes all78 as-run files BEFORE GPU; analysis_source_manifest.json binds post-processing separately.
Generic physical engine is unchanged; only the requested keep/cutoff parameters differ.
R->Image still blocked fromL20; answer image reads only KEEP; answer Attention/QKV skipped inL2-L5.
Original RoPE/logical positions unchanged. Cutoff removes all256 visual Hidden rows; no token-level selection.
Persistent visualKV only in8 KEEP layers, zeroSKIP cache, original-position zeroScratch transient/masked.
Official TextVQA ten-answer scoring independently recomputed for all1280 saved records.
Original Native/R0/R6/R14 quality is historical oldGPU; Deep6/7/8 and all new windows share the current GPU.
All comparisons are exploratory historical same-question comparisons, NOT independent Holdout or global layer optimum.
Whole-image-SHA paired bootstrap5000, ordinary95%CI, uncorrected multiple comparisons.

## Local runtime

RUN=/home/featurize/ate-runs/internvl3-svh-r-eight-shift-textvqa-3090
ROOT=/home/featurize/work/ATE-internvl3-svh-r-eight-shift-textvqa-3090
PREPARED=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared
Python=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
ExtraDeps=/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/extra_deps (isolated timm0.9.16)
Model=/home/featurize/ate-local-models/InternVL3-8B
Model revision853e3a797a661694b1b8ece0cb72dc2b23e3dac9.
GPU=GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27, verified idle before load.
BF16/Eager, Torch2.6.0+cu124 TF4.49.0, batch1 greedy naturalEOS max32.
NativeChat, single448tile256visual, no OCR or Gold in model inputs.
All131 cached native tokenized/pixel/prompt/roles input bindings were rechecked on CPU.

## Reaudit without GPU

Set ATE_DEEP_RUN=RUN, ATE_SEVEN_RUN=RUN, PYTHONPATH=RUN/code:ExtraDeps,
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.
Run Python -B -m vlm_research.internvl3_svh_r_eight_shift_textvqa.analyze then .report.
Official reference is RUN/references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py,
and the same bytes are uploaded in references/textvqa_eval.py.
A genuinely new GPU run needs separate authorization and NEW output directory/source/protocol bindings.
cpu_tests.py precedes run.py. Do not rerun the completed GPU generation just to obtain a report.

## Evidence and storage

per_question.csv contains all1280 answers/Gold/TokenIDs/scores/recordSHAs.
per_question_changes.csv has all24 pairwise comparisons and harm cases.
kv_layer_audit.csv contains1280x28 actual tensor shape/dtype/length/live/storage layouts.
physical_cutoff_audit.full.json.gz is lossless fullSmoke evidence; rawJSON stays server_only.
operator_input_rows.csv and attention_probability_audit.csv expose actual execution and permission checks.
CPU float64 uses official28layerQwen2,1prefill+8decode;3short/medium/longGPU Smoke each condition.
All normalEOS KV numbers are labeled accordingly, NOT fixed32-token KV@32 or latency.
No speed has been measured for the new windows.

server_only/predictions.sqlite is a full1280-row SQLite backup, not partialWAL.
Complete raw audit/logs stay persistent; models/images/preparedtensors/dependencies/keys are not uploaded.
artifact_manifest.json inventories lightGitHub and server-only evidence except itself/SHA256SUMS/post-push receipt.
SHA256SUMS includes the artifactmanifest and excludes itself.
server_only/git_publish_receipt.json is post-push mutable to avoid Commit/self-hash recursion.
Every uploaded file is read back via fixedCommit GitHub HTTPS and matched by SHA256; SSH verifies branchhead.
No main change, forcepush, PR, server shutdown or automatic next experiment. GPU released.
"""
 (DEST/"README.md").write_text(readme);(ROOT/PKG/"README.md").write_text(readme)
 files=sorted([p for p in DEST.rglob("*") if p.is_file() and "server_only" not in p.relative_to(DEST).parts and p.name not in ("artifact_manifest.json","SHA256SUMS")]+[p for p in (ROOT/PKG).glob("*") if p.is_file()])
 raw=sorted(p for p in server.rglob("*") if p.is_file() and p.name!="git_publish_receipt.json")
 def item(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
 write(DEST/"artifact_manifest.json",dict(status="COMPLETE_EXPLORATORY_WITH_NUMERICAL_WAIVER",GitHub_branch="codex/internvl3-svh-r-eight-shift-textvqa-3090",parent_source_commit=PARENT_COMMIT,model_revision=read(OUT/"protocol.json")["revision"],GPU_UUID=GPU,protocol_sha256=sha(OUT/"protocol.json"),input_manifest_sha256=sha(OUT/"sample_manifest.json"),GitHub_files=[item(p) for p in files],server_only_files=[item(p) for p in raw],self_index_exclusions=["artifact_manifest.json","SHA256SUMS","server_only/git_publish_receipt.json"],publish_receipt="server_only/git_publish_receipt.json generated after successful push",excluded_resources=["weights","images","prepared .pt tensors","dependencies","keys"]))
 files.append(DEST/"artifact_manifest.json");(DEST/"SHA256SUMS").write_text("".join(sha(p)+"  "+str(p.relative_to(ROOT))+"\n" for p in sorted(files)));files.append(DEST/"SHA256SUMS")
 plan=dict(status="READY",branch="codex/internvl3-svh-r-eight-shift-textvqa-3090",files=[item(p) for p in sorted(files)],server_only_files=[item(p) for p in raw],persistent_root=str(ROOT),result=str(REL));write(RUN/"publish_plan.json",plan)
 assert all(sha(ROOT/x["path"])==x["sha256"] for x in plan["files"])
 print(json.dumps(dict(status="PERSISTED_VERIFIED",files=len(files),bytes=sum(x["bytes"] for x in plan["files"]),raw_files=len(raw))),flush=True)
if __name__=="__main__":main()
