"""Single-tile quality evaluation utilities; no model or physical algorithm changes."""
import os,json,hashlib,csv,time,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ.get("ATE_SEVEN_ROOT","/home/featurize/work/ATE-internvl3-svh-seven-benchmarks-3090"))
RUN=Path(os.environ.get("ATE_SEVEN_RUN","/home/featurize/ate-runs/internvl3-svh-seven-benchmarks-3090"))
OUT=RUN/"results"
PERSIST=ROOT/"vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090"
MODEL=Path("/home/featurize/ate-local-models/InternVL3-8B")
GPU="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e"
REV="853e3a797a661694b1b8ece0cb72dc2b23e3dac9"
SOURCE="e288eee271b7aeb64e5df59783d3201bb2a6b313"
BENCHES=["GQA","ScienceQA-IMG","TextVQA","POPE","MME","MMBench","OCRBench"]
CAPS={"GQA":16,"ScienceQA-IMG":16,"TextVQA":32,"POPE":8,"MME":16,"MMBench":16,"OCRBench":128}
EXPECTED={"GQA":12578,"ScienceQA-IMG":2017,"TextVQA":5000,"POPE":9000,"MME":2374,"MMBench":4377,"OCRBench":1000}
METHODS=["native","physical_d"]
def sha(x):
    if isinstance(x,Path): return hashlib.sha256(x.read_bytes()).hexdigest()
    if isinstance(x,str): x=x.encode()
    return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def bind(x):return sha(canonical(x))
def writej(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(t,p)
def readj(p):return json.loads(Path(p).read_text())
def loadrows(p):return [json.loads(x) for x in Path(p).open() if x.strip()]
def csvwrite(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def state(b,status,**kw):
    writej(RUN/"status"/(b+".json"),{"benchmark":b,"status":status,"updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kw})
def task_user(b,r):
    q=r["question"];hint=r.get("hint","")
    if b in ("ScienceQA-IMG","MMBench"):
        choices=r["choices"]
        if isinstance(choices,list):choices=dict(zip("ABCDE",choices))
        # Full original hint and choices, with real newlines; no gold or rationale.
        prefix=("Context: "+hint+"\n") if hint else ""
        return prefix+q+"\n"+"\n".join(k+". "+v for k,v in choices.items())+"\nAnswer with the option's letter from the given choices directly."
    if b=="POPE":return q+"\nAnswer yes or no only."
    if b=="MME":return q+"\nAnswer yes or no only."
    if b=="TextVQA":return q+"\nAnswer with the text visible in the image, using a word or short phrase. Do not explain."
    if b=="OCRBench":return q+"\nGive only the requested text or answer. Do not explain."
    return q+"\nAnswer the question using a single word or phrase."
def request_text(b,r):return "<image>\n"+task_user(b,r)
def gpu_idle():
    q=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.total,driver_version","--format=csv,noheader"],text=True)
    if GPU not in q or "3090" not in q:raise RuntimeError("GPU_UUID_MISMATCH "+q)
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True).strip()
    if processes:raise RuntimeError("GPU_NOT_IDLE "+processes)
    return {"gpu":q.strip(),"compute_processes":processes}
def tensor_sha(t):
    # CPU representation only for input evidence, never inside timed region (no timing in this task).
    return sha(t.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
def source_fingerprints():
    p=ROOT/"vlm_research/internvl3_svh_physical_d_pilot/engine.py"
    q=ROOT/"vlm_research/internvl3_visual_writing_figure3_3000/engine.py"
    return {str(p.relative_to(ROOT)):sha(p),str(q.relative_to(ROOT)):sha(q)}
