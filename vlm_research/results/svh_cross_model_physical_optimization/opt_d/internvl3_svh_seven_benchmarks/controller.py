"""Sequential per-benchmark worker -> independent CPU analysis -> verified persistent/GitHub publication."""
from .common import *
import argparse
PY="/home/featurize/.venvs/ate-frozen/bin/python"
CPU_PY="/environment/miniconda3/bin/python"
PKG="vlm_research.internvl3_svh_seven_benchmarks."
def env(gpu=False):
    return dict(os.environ,PYTHONPATH=str(RUN/"code"),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=GPU if gpu else "",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="4")
def alive(pid):
    p=Path("/proc")/str(pid)/"cmdline"
    try:return b"internvl3_svh_seven_benchmarks.worker" in p.read_bytes()
    except FileNotFoundError:return False
def cpu(module,*args):
    x=subprocess.run([CPU_PY,"-u","-m",PKG+module,*args],cwd=RUN,env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(x.stdout,flush=True)
    if x.returncode:raise RuntimeError("CPU_STAGE_FAILED "+module+" "+str(x.returncode))
def main():
    (RUN/"stages").mkdir(exist_ok=True);datasets=readj(OUT/"dataset_manifest.json")
    for bench in BENCHES:
        if (RUN/"STOP").exists():break
        done=RUN/"stages"/(bench+".json")
        if done.exists():continue
        active=RUN/"active_worker.json"
        if active.exists():
            a=readj(active)
            if a["benchmark"]==bench:
                while alive(a["pid"]):
                    if (RUN/"STOP").exists():return
                    time.sleep(5)
        current=readj(RUN/"status"/(bench+".json"))
        if datasets[bench]["status"]=="CPU_DATA_PASS" and current["status"] in ("READY_PREFLIGHT","NOT_STARTED"):
            gpu_idle();f=(RUN/(bench+".worker.log")).open("a")
            p=subprocess.Popen([PY,"-u","-m",PKG+"worker","--benchmark",bench],cwd=RUN,env=env(True),stdout=f,stderr=subprocess.STDOUT)
            writej(active,{"benchmark":bench,"pid":p.pid});print("STARTED",bench,p.pid,flush=True)
            p.wait();f.close()
        current=readj(RUN/"status"/(bench+".json"))
        if current["status"] in ("FULL_GENERATION","LOADING_MODEL","PREFLIGHT"):
            raise RuntimeError("Unexpected stale/active worker state "+bench)
        gpu_idle()
        cpu("analyze","--benchmark",bench)
        cpu("publish","--label",bench+"-"+current["status"].lower())
        receipt=readj(RUN/"publication.json")
        writej(done,{"benchmark":bench,"state":current["status"],"publication":receipt})
        print("STAGE_DONE",bench,current["status"],flush=True)
    cpu("analyze")
    gpu_idle()
    cpu("publish","--label","final")
    writej(RUN/"completion.json",{"status":readj(OUT/"validation.json")["status"],"publication":readj(RUN/"publication.json"),"gpu_released":True})
    print("ALL_AVAILABLE_BENCHMARKS_FINISHED",flush=True)
if __name__=="__main__":main()
