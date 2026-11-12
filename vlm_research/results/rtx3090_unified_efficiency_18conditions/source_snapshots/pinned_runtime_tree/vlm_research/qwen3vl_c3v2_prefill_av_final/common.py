import pathlib,json,hashlib,os,datetime,shutil
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_c3v2_prefill_av_final'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_prefill_av_final')
PARENT=ROOT/'vlm_research/results/qwen3vl_c3v2_l22_kernel_diagnosis'
OLD_RAW=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_l22_kernel_diagnosis')
MODEL=pathlib.Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
ORIGINAL='GQA__06905471'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def check():
 assert shutil.disk_usage('/data').free>20*2**30,'Stop: disk free below20GiB'
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(read(OUT/'protocol.json')['GPU_cutoff_UTC'])
