"""Storage-only checkpoint spooling. Never imports torch or changes experiment code."""
import os,signal,time,json,hashlib,shutil
from pathlib import Path
W=Path("/home/featurize/work/ATE-internvl3-figure3-3000")
O=W/"vlm_research/results/internvl3_visual_writing_figure3_3000"
SSD=Path("/home/featurize/ate-runs/internvl3-figure3-3000/checkpoints")
PERSIST=O/"checkpoints_persistent"
PID=7056
def h(b):return hashlib.sha256(b).hexdigest()
def worker_alive():
    p=Path("/proc")/str(PID)/"cmdline"
    return p.exists() and b"vlm_research.internvl3_visual_writing_figure3_3000.run" in p.read_bytes()
assert worker_alive()
assert not SSD.exists() and not PERSIST.exists() and not (O/"checkpoints").is_symlink()
paused=False
try:
    for trial in range(30):
        os.kill(PID,signal.SIGSTOP);paused=True;time.sleep(.15)
        open_tmp=[]
        for f in (Path("/proc")/str(PID)/"fd").iterdir():
            try:
                n=os.readlink(f)
                if str(O/"checkpoints") in n and n.endswith(".tmp"):open_tmp.append(n)
            except FileNotFoundError:pass
        if not open_tmp and not list((O/"checkpoints").glob("*.tmp")):break
        os.kill(PID,signal.SIGCONT);paused=False;time.sleep(.25)
    else:raise RuntimeError("No safe closed-file checkpoint boundary found")
    shutil.copytree(O/"checkpoints",SSD)
    hashes={}
    for f in (O/"checkpoints").glob("*.json"):
        b=f.read_bytes();assert h(b)==h((SSD/f.name).read_bytes());hashes[f.name]=h(b)
    (O/"checkpoints").rename(PERSIST)
    (O/"checkpoints").symlink_to(SSD,target_is_directory=True)
    assert (O/"checkpoints").resolve()==SSD
    report=dict(phase="SPOOLING",storage_only=True,source_and_protocol_unchanged=True,
                gpu_worker_PID=PID,local_spool=str(SSD),persistent_mirror=str(PERSIST),
                moved_files=len(hashes),migration_SHA256=hashes,protocol_sha256=h((O/"protocol.json").read_bytes()))
    (O/"io_layout.json").write_text(json.dumps(report,indent=2))
finally:
    if paused:os.kill(PID,signal.SIGCONT)
print("SSD_CHECKPOINT_SPOOL_ACTIVE",len(hashes),flush=True)
copied={};last_partial={};start=time.time()
while True:
    alive=worker_alive()
    for f in sorted(SSD.glob("*.json")):
        b=f.read_bytes();r=json.loads(b);d=h(b)
        if copied.get(f.name)==d:continue
        if alive and not r["complete"] and time.time()-last_partial.get(f.name,0)<60:continue
        p=PERSIST/f.name;tmp=p.with_suffix(".mirror.tmp")
        tmp.write_bytes(b);tmp.replace(p)
        assert h(p.read_bytes())==d
        copied[f.name]=d;last_partial[f.name]=time.time()
    status=dict(phase="SPOOLING" if alive else "FINAL_MIRROR",persisted_files=len(copied),
                complete_files=sum(json.loads(f.read_text())["complete"] for f in SSD.glob("*.json")),
                elapsed_seconds=time.time()-start,time=time.time())
    p=O/"persistence_progress.json";tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(status));tmp.replace(p)
    if not alive:break
    time.sleep(10)
all_hashes={}
for f in SSD.glob("*.json"):
    b=f.read_bytes();assert h(b)==h((PERSIST/f.name).read_bytes());all_hashes[f.name]=h(b)
# Worker is gone, no open checkpoint descriptor remains: restore ordinary persistent directory.
assert not worker_alive() and (O/"checkpoints").is_symlink()
(O/"checkpoints").unlink();PERSIST.rename(O/"checkpoints")
(O/"persistence_complete.json").write_text(json.dumps(dict(passed=True,files=len(all_hashes),
    checkpoint_directory=str(O/"checkpoints"),all_files_SHA256=all_hashes,
    original_protocol_unchanged=True,local_spool_retained=str(SSD)),indent=2))
print("ALL_CHECKPOINTS_PERSISTED_SHA256_PASS",len(all_hashes),flush=True)
