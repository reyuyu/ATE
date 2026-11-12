"""Restore existing server archives to SSD. No remote data or model downloads."""
from .common import *
import shutil,zipfile
SOURCE_DATA=Path("/home/featurize/work/ate-qwen3090-data/benchmarks")
DATA=Path("/home/featurize/data/ate-seven-benchmarks-3090")
def restore():
    DATA.mkdir(exist_ok=True,parents=True);receipts=[]
    for name in ["ate3090-scienceqa-img.zip","ate3090-pope.zip","ate3090-mme.zip","ate3090-mmbench.zip","ocrbench_windows_restore.zip"]:
        src=SOURCE_DATA/"backup_packages"/name;dst=DATA/"archives"/name
        done=DATA/("restored_"+name+".json")
        if done.exists():receipts.append(readj(done));continue
        dst.parent.mkdir(exist_ok=True)
        expected=readj(SOURCE_DATA/".restore_state"/(name+".json"))["SHA256"]
        if not dst.exists():
            h=hashlib.sha256()
            with src.open("rb") as a,dst.with_suffix(".part").open("wb") as b:
                while True:
                    chunk=a.read(8*1024**2)
                    if not chunk:break
                    h.update(chunk);b.write(chunk)
            if h.hexdigest()!=expected:raise RuntimeError("ARCHIVE_COPY_HASH "+name)
            os.replace(dst.with_suffix(".part"),dst)
        else:
            assert sha(dst)==expected
        target=DATA/"restored"
        with zipfile.ZipFile(dst) as z:
            for item in z.infolist():
                path=(target/item.filename).resolve()
                if not path.is_relative_to(target.resolve()):raise RuntimeError("UNSAFE_ARCHIVE_PATH")
            z.extractall(target)
        rec={"source":str(src),"archive":str(dst),"sha256":expected,"files":len(z.infolist()),"status":"PASS"}
        writej(done,rec);receipts.append(rec);print("RESTORED",name,flush=True)
    writej(OUT/"restore_receipts.json",receipts)
if __name__=="__main__":restore()
