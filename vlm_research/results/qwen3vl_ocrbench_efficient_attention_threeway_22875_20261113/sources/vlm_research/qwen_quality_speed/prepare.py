"""CPU-only, deterministic sample/input/protocol freeze before any new prediction."""
import sys,collections,shutil,subprocess,os
from .common import *

def main():
 import torch
 torch.set_num_threads(1)
 sys.path.insert(0,'/data/lk/ATE-resources/experiments/qwen3vl_c3v2_physical_transfer/cpu_tools')
 import pyarrow.parquet as pq
 from transformers import AutoProcessor
 from ..qwen3vl_early_windows.experiment import prepare
 assert git('merge-base','HEAD',BASE)==BASE
 assert shutil.disk_usage('/data').free>20*2**30
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);os.chmod(PRIVATE,0o700)
 source=Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_val_balanced.parquet')
 rows=pq.read_table(source).to_pylist();pool=[];images={};total_images=set()
 for r in rows:
  iid=r['imageId'];total_images.add(iid);ip=Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')/('image_'+iid+'.jpg')
  if not ip.exists():continue
  if iid not in images:images[iid]=sha(ip)
  pool.append(dict(question_id=r['id'],image_id=iid,image_path=str(ip),image_sha256=images[iid],question=r['question'],gold=r['answer'],task=r['types']['semantic'],types=r['types']))
 selected=sorted(pool,key=lambda r:digest([SEED,'quality',r['question_id'],r['image_sha256']]))[:1000]
 assert len(selected)==1000 and len({r['question_id'] for r in selected})==1000
 groups=collections.defaultdict(list)
 for r in selected:groups[r['image_sha256']].append(r)
 loads=[0,0]
 for h in sorted(groups,key=lambda h:(-len(groups[h]),h)):
  rank=min(range(2),key=lambda i:(loads[i],digest([h,i])))
  for r in groups[h]:r['rank']=rank
  loads[rank]+=len(groups[h])
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 bindings={}
 for i,r in enumerate(selected):
  x,l=prepare(processor,r);bindings[r['question_id']]=l
  if i%50==0:print('INPUT_FREEZE',i,flush=True)
  del x
 smoke=[];bench=[]
 for rank in range(2):
  ss=sorted([r for r in selected if r['rank']==rank],key=lambda r:(bindings[r['question_id']]['input_length'],r['question_id']))
  smoke.extend(ss[round(i*(len(ss)-1)/9)]['question_id'] for i in range(10))
  # One question per image, length-spanning deterministic 24 per worker.
  seen=set();unique=[]
  for r in ss:
   if r['image_sha256'] not in seen:unique.append(r);seen.add(r['image_sha256'])
  bench.extend(unique[round(i*(len(unique)-1)/23)]['question_id'] for i in range(24))
 assert len(set(smoke))==20 and len(set(bench))==48
 assert len({next(r for r in selected if r['question_id']==q)['image_sha256'] for q in bench})==48
 write(OUT/'sample_manifest.json',dict(samples=selected,smoke_question_ids=smoke,benchmark_question_ids=bench,input_bindings=bindings,selection=dict(seed=SEED,rule='SHA256 canonical JSON [seed,quality,qid,image_sha], take first 1000; no answers or outputs used',annotation_path=str(source),annotation_sha256=sha(source),official_questions=len(rows),official_images=len(total_images),local_eligible_questions=len(pool),local_eligible_images=len(images),selected_images=len(groups),shard_questions=loads,benchmark_selection='24 distinct images per worker, evenly spaced input-length ranks; 48 total',smoke_selection='10 evenly spaced input-length ranks per worker, including shortest/longest')))
 comps=[['Physical-Original','Native'],['Dense-F','Native'],['Physical-Original','Dense-F'],['Physical-Repaired','Dense-F'],['Physical-Repaired','Physical-Original'],['Physical-Repaired','Native']]
 write(OUT/'protocol.json',dict(status='FROZEN_BEFORE_NEW_GPU_PREDICTIONS',at=now(),base_commit=BASE,model_revision=MODEL.name,dtype='bfloat16',attention='eager',transformers='4.57.6',seed=SEED,conditions=CONDITIONS,quality_n=1000,smoke_n=20,benchmark_n=48,max_new_tokens=32,greedy=True,logits_to_keep=1,batch_size=1,independent_cache=True,training=False,kernel_search=False,old_strict_gate_required=False,stop_rules=['OOM','NaN/Inf','invalid cache/positions/mask','unclean hooks','unexpected source/input hash'],bootstrap=dict(replicates=20000,seed=SEED,unit='Image SHA256 cluster',confidence=.95,question_weighted=True,comparisons=comps,exploratory=True),benchmark=dict(regimes=['natural_eos_max32','fixed16_including_prefill_prediction'],warmups_per_condition_regime=2,measured_repeats=5,condition_order='cyclic Latin rotation by question SHA and repeat; same image all conditions same GPU',timing='synchronized request wall includes H2D, vision, execution/cache/context setup, autoregressive token choice and EOS; excludes image decode/processor/model loading and post-timing text decode',gpu_events='vision; first forward (TTFT GPU); prefill language = first forward minus vision; incremental Decode',primary_controlled_speed='Native/condition fixed16 ratio of summed per-question repeat-median E2E; paired image bootstrap',natural_length_caveat=True,peak='PyTorch peak allocated and reserved; model baseline plus incremental; not whole process nvidia-smi memory',cache='actual live and allocated KV separately; repaired transient zero K/V not persistent',audit_probes_disabled=True),gpu_uuids=UUIDS,sample_manifest_sha256=sha(OUT/'sample_manifest.json'),condition_definitions=dict(Native='unaltered native model',**{'Dense-F':'dense functional early L1-L6 P-1/Decode attention zero; P-1/Decode image hard-mask all layers; R image masked L22-L35','Physical-Original':'frozen original Execution C: genuine L1-L6 skip, L22 image hidden exit, compact persistent text KV, no numerical repair','Physical-Repaired':'same physical C with frozen early dead key, Decode image-zero columns, late Prefill image-zero columns and fixed cuBLASLt four projections L22-L35; read-only probes disabled, math unchanged'})))
 # Full model/processor asset hash freeze, not just metadata.
 assets={str(p.relative_to(MODEL)):sha(p) for p in MODEL.rglob('*') if p.is_file()}
 write(OUT/'model_manifest.json',dict(path=str(MODEL),revision=MODEL.name,files=assets))
 write(OUT/'preflight.json',dict(at=now(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv']).decode(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']).decode(),disk_free=shutil.disk_usage('/data').free,main_before=git('rev-parse','main'),base=BASE,authorization='User explicitly authorized up to two idle GPUs in this task'))
 print('FROZEN',len(selected),'images',len(groups),'length range',min(x['input_length'] for x in bindings.values()),max(x['input_length'] for x in bindings.values()),flush=True)
if __name__=='__main__':main()
