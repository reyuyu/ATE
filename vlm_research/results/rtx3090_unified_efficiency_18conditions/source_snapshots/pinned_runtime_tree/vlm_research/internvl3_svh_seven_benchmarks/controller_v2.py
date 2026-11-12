"""Sequential v2 resume controller: only GQA/TextVQA/OCRBench; never reruns inherited results."""
from .common import *
from .stop_policy_v2 import *
import subprocess
PY="/home/featurize/.venvs/ate-frozen/bin/python"
CPU_PY="/environment/miniconda3/bin/python"
MODULE="vlm_research.internvl3_svh_seven_benchmarks."
def env(gpu=False):
    return dict(os.environ,ATE_SEVEN_RUN=str(RUN),ATE_SEVEN_ROOT=str(ROOT),PYTHONPATH=str(RUN/"code"),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=GPU if gpu else "",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="4")
def alive(pid):
    try:return b"internvl3_svh_seven_benchmarks.resume_worker_v2" in (Path("/proc")/str(pid)/"cmdline").read_bytes()
    except FileNotFoundError:return False
def cpu(module,*args,label=None):
    log=RUN/("publish_"+str(label)+".log") if module=="publish_v2" else None
    if log:
        with log.open("w") as f:
            x=subprocess.run([CPU_PY,"-u","-m",MODULE+module,*args],cwd=RUN,env=env(),stdout=f,stderr=subprocess.STDOUT)
        print(log.read_text()[-2000:],flush=True)
    else:
        x=subprocess.run([CPU_PY,"-u","-m",MODULE+module,*args],cwd=RUN,env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(x.stdout[-4000:],flush=True)
    if x.returncode:raise RuntimeError("CPU_STAGE_FAILED "+module+" "+str(x.returncode))
def main():
    assert readj(OUT/"resume_validation_v2.json")["status"]=="PASS";check_sources()
    for bench in TARGETS:
        if (RUN/"STOP").exists():return
        done=RUN/"stages"/(bench+".v2.json")
        if done.exists():continue
        active=RUN/"active_worker_v2.json"
        adopted=False
        if active.exists():
            a=readj(active)
            if a["benchmark"]==bench:
                adopted=True
                while alive(a["pid"]):
                    if (RUN/"STOP").exists():return
                    time.sleep(5)
        status=RUN/"status"/(bench+".json")
        current=readj(status) if status.exists() else {"status":"READY_RESUME_V2"}
        if not adopted and current["status"] in ("READY_RESUME_V2","NOT_STARTED"):
            gpu_idle();f=(RUN/(bench+".worker_v2.log")).open("a")
            x=subprocess.Popen([PY,"-u","-m",MODULE+"resume_worker_v2","--benchmark",bench],cwd=RUN,env=env(True),stdout=f,stderr=subprocess.STDOUT)
            writej(active,{"benchmark":bench,"pid":x.pid});print("STARTED_V2",bench,x.pid,flush=True);x.wait();f.close()
        current=readj(status)
        if current["status"] in ("FULL_GENERATION","LOADING_MODEL"):raise RuntimeError("STALE_OR_CRASHED_WORKER "+bench)
        gpu_idle();cpu("analyze_v2","--benchmark",bench)
        cpu("publish_v2","--label",bench+"-v2",label=bench)
        receipt=readj(RUN/"publication.json")
        writej(done,{"benchmark":bench,"state":current["status"],"publication":receipt})
        print("STAGE_DONE_V2",bench,current["status"],flush=True)
    cpu("analyze_v2")
    gpu_idle();cpu("publish_v2","--label","v2-all-available-complete",label="final")
    writej(RUN/"completion_v2.json",{"status":readj(OUT/"validation.json")["status"],"publication":readj(RUN/"publication.json"),"GPU_released":True})
    print("V2_ALL_AVAILABLE_COMPLETE",flush=True)
if __name__=="__main__":main()
