"""Persist lightweight evidence and exact raw checkpoints without Windows transfer."""
import json,hashlib,shutil,gzip,subprocess,os,time,sqlite3
from pathlib import Path
RUN=Path("/home/featurize/ate-runs/internvl3-svh-r14-full-middle-reaccess-textvqa-3090")
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r14-full-middle-reaccess-textvqa-3090")
REL=Path("vlm_research/results/internvl3_svh_r14_full_middle_reaccess_textvqa_3090")
PKG=Path("vlm_research/internvl3_svh_r14_full_middle_reaccess_textvqa")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def main():
 out=RUN/"results";dest=ROOT/REL;dest.mkdir(parents=True,exist_ok=True)
 validation=json.load(open(out/"validation.json"));assert validation["status"]=="COMPLETE_EXPLORATORY" and validation["GPU_released"]
 # Fresh exact persistent checkpoint backup, never copy a live WAL piecemeal.
 server=dest/"server_only";server.mkdir(exist_ok=True)
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");backup=sqlite3.connect(server/"predictions.sqlite");db.backup(backup);backup.close();db.close()
 for p in (RUN/"logs").glob("*.log"):shutil.copy2(p,server/p.name)
 
 shutil.copy2(RUN/"prepared/input_bindings.json",server/"input_bindings.json")
 for name in ["reused_A_records.json","reused_native_B_smoke.json"]:shutil.copy2(RUN/"prepared"/name,server/name)
 # CPU image tensors remain on local SSD, raw image datasets stay at their original source; not copied into work/Git.
 files=[];uncompressed_only=[]
 for p in sorted(out.iterdir()):
  if not p.is_file() or p.name in ["smoke_checks_partial.json"]:continue
  if p.name=="kv_layout_per_layer.csv" or (p.name=="smoke_structure_audits.json" and p.stat().st_size>2*1024*1024):
   target=dest/(p.name+".gz")
   with p.open("rb") as src,gzip.open(target,"wb",compresslevel=6) as z:shutil.copyfileobj(src,z)
   assert gzip.decompress(target.read_bytes())==p.read_bytes()
   files.append(target);uncompressed_only.append({"SSD":str(p),"sha256":sha(p),"GitHub_compressed":str(target.relative_to(ROOT)),"decompressed_sha256":sha(p)})
  else:shutil.copy2(p,dest/p.name);files.append(dest/p.name)
 pkg=ROOT/PKG;pkg.mkdir(parents=True,exist_ok=True)
 for p in (RUN/"code"/PKG).glob("*.py"):shutil.copy2(p,pkg/p.name);files.append(pkg/p.name)
 for p in (RUN/"code"/PKG).glob("*.md"):shutil.copy2(p,pkg/p.name);files.append(pkg/p.name)
 # Verify each unchanged runtime dependency against the actual inherited Git blob.
 frozen=json.load(open(out/"source_manifest.json"))["runtime_source_sha256"];deps=[]
 for rel,h in frozen.items():
  if rel.startswith(str(PKG)+"/"):continue
  content=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:"+rel],timeout=90)
  assert hashlib.sha256(content).hexdigest()==h,("DEPENDENCY_GIT_MISMATCH",rel)
  deps.append({"path":rel,"sha256":h,"source":"inherited parent Git commit"})
 write(dest/"dependency_source_validation.json",{"status":"PASS","dependencies":deps})
 files.append(dest/"dependency_source_validation.json")
 source_final={"as_run_runtime_source_manifest_sha256":sha(out/"source_manifest.json"),"all_new_source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in files if str(p).endswith(".py")},"original_frozen_runtime_source_unchanged":True}
 write(dest/"analysis_source_manifest.json",source_final);files.append(dest/"analysis_source_manifest.json")
 inventory=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in files]
 serverfiles=[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(server.iterdir()) if p.is_file()]
 write(dest/"artifact_manifest.json",{"status":validation["status"],"GitHub_files":inventory,"server_only":serverfiles,"SSD_only_CPU_image_tensors":{"path":str(RUN/"prepared"),"not_uploaded":True},"compressed_artifacts":uncompressed_only,"GitHub_commit_receipt":"server_only/github_delivery_receipt.json written after remote verification; avoids recursive commit SHA","no_weights_images_or_secrets_uploaded":True})
 files.append(dest/"artifact_manifest.json")
 (dest/"SHA256SUMS").write_text("".join(sha(p)+"  "+str(p.relative_to(ROOT))+"\n" for p in sorted(files)))
 files.append(dest/"SHA256SUMS")
 # Small allowlist consumed by publish.py; nothing else can be staged.
 write(RUN/"publish_allowlist.json",[str(p.relative_to(ROOT)) for p in files])
 assert all(p.stat().st_size<5*1024*1024 for p in files),[(str(p),p.stat().st_size) for p in files if p.stat().st_size>=5*1024*1024]
 print("PERSISTED",len(files),sum(p.stat().st_size for p in files),"server_raw",sum(p.stat().st_size for p in server.iterdir() if p.is_file()),flush=True)
if __name__=="__main__":main()
