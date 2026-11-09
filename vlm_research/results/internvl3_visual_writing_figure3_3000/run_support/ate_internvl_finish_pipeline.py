"""Finish already-authorized CPU analysis after the existing GPU worker and disk mirror."""
import subprocess,time,json,os
from pathlib import Path
W=Path("/home/featurize/work/ATE-internvl3-figure3-3000")
O=W/"vlm_research/results/internvl3_visual_writing_figure3_3000"
PY="/home/featurize/.venvs/ate-frozen/bin/python"
pid=7056
def alive():
    p=Path("/proc")/str(pid)/"cmdline"
    return p.exists() and b"vlm_research.internvl3_visual_writing_figure3_3000.run" in p.read_bytes()
while alive():time.sleep(15)
if not (O/"generation_complete.json").exists():
    raise RuntimeError("GPU generation did not complete. Preserve all results and stop; no automatic restart.")
while not (O/"persistence_complete.json").exists():
    if not Path("/proc/7378/cmdline").exists():raise RuntimeError("Persistence mirror exited without PASS")
    time.sleep(10)
assert json.loads((O/"persistence_complete.json").read_text())["passed"]
env=dict(os.environ,OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",CUDA_VISIBLE_DEVICES="")
steps=[
 [PY,"-u","-m","vlm_research.internvl3_visual_writing_figure3_3000.analyze"],
 [PY,str(O/"plot_figure3.py"),"--directory",str(O)],
 [PY,"-u","-m","vlm_research.internvl3_visual_writing_figure3_3000.finalize"]]
for command in steps:
    print("CPU_PHASE",command[-1],flush=True)
    subprocess.run(command,cwd=W,env=env,check=True)
print("ALL_AUTHORIZED_WORK_COMPLETE_REVIEW_PENDING",flush=True)
