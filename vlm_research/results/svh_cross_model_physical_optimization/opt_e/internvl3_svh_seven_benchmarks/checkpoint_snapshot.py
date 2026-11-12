"""Consistent online SQLite snapshot to persistent storage; never copies a live WAL database."""
from .common import *
import sqlite3,shutil
def main(bench):
    assert bench in BENCHES
    source=RUN/"checkpoints"/(bench+".sqlite");folder=RUN/"snapshots";folder.mkdir(exist_ok=True)
    dest=folder/(bench+".sqlite");temp=folder/(bench+".partial.sqlite")
    if temp.exists():temp.unlink()
    src=sqlite3.connect("file:"+str(source)+"?mode=ro",uri=True);dst=sqlite3.connect(temp)
    src.backup(dst,pages=128,sleep=.01);dst.close();src.close()
    check=sqlite3.connect(temp)
    assert check.execute("PRAGMA quick_check").fetchone()[0]=="ok"
    n=check.execute("SELECT count(*) FROM predictions").fetchone()[0]
    metadata=dict(check.execute("SELECT key,value FROM metadata"));check.close()
    assert metadata["protocol_sha256"]==sha(OUT/"protocol.json")
    os.replace(temp,dest);h=sha(dest)
    target=PERSIST/"server_snapshots"/(bench+".sqlite");target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_suffix(".copying");shutil.copyfile(dest,part);assert sha(part)==h;os.replace(part,target)
    entry={"benchmark":bench,"status":"VERIFIED_PARTIAL_SNAPSHOT","generation_rows":n,"ssd_path":str(dest),"persistent_path":str(target),"sha256":h,"bytes":dest.stat().st_size,"protocol_sha256":metadata["protocol_sha256"],"uploaded":False,"snapshot_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"reason":"Consistent resumable online SQLite snapshot, stored server-only."}
    receipt=OUT/"checkpoint_snapshot_manifest.json";d=readj(receipt) if receipt.exists() else {};d[bench]=entry;writej(receipt,d)
    writej(PERSIST/"checkpoint_snapshot_manifest.json",d)
    print(canonical(entry),flush=True)
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--benchmark",required=True);args=a.parse_args();main(args.benchmark)
