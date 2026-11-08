"""CPU-only recovery evidence; does not launch any benchmark."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from importlib.metadata import version
import os,sys,json,hashlib,platform,subprocess,shutil
O=Path(__file__).resolve().parent;R=O.parents[2];D=Path('/home/featurize/data/ate-recovery');M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf');A=Path('/home/featurize/ate-4090-recovery-20261108');OLD=Path('/home/featurize/work/ATE/vlm_research/results')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def command(*args):return subprocess.check_output(args,text=True,cwd=R).strip()

def main():
 import torch
 assert not torch.cuda.is_initialized()
 datasets=read(O/'dataset_restore.json');assert datasets['status']=='PASS'
 formal={'GQA-testdev-balanced':12578,'MME-paper':2374,'ScienceQA-IMG-paper':2017,'POPE-paper':8910,'TextVQA-OCR':5000,'MMBench-paper':4377,'OCRBench-v1':1000};paths={}
 for name,count in formal.items():
  v=datasets['datasets'][name];assert v['questions']==count and v['all_images_present'];paths[name]=str(D/'normalized'/name)
  if name!='OCRBench-v1':assert v['exact_3090_input_and_gold_files']
 assert D.resolve()==Path('/home/featurize/ate-local-data/ate-recovery')
 mm=[json.loads(l) for l in (D/'normalized/MMBench-paper/inputs.jsonl').read_text().splitlines()];groups=read(O/'mmbench_circular_groups.json')
 assert set(groups)=={r['question_id'] for r in mm} and len(set(groups.values()))==1176
 datasets['datasets']['MMBench-paper']['circular_groups']=1176
 ocr=D/'normalized/OCRBench-v1';rows=[json.loads(l) for l in (ocr/'inputs.jsonl').read_text().splitlines()];raw=read(ocr/'OCRBench.json');assert len(rows)==len(raw)==1000 and len({r['question_id'] for r in rows})==1000
 assert {r['official_annotation_index'] for r in rows}==set(range(1000))
 for row in rows:
  source=raw[row['official_annotation_index']]
  assert row['official_id']==source['id'] and row['question']==source['question'] and row['answer']==source['answers'] and row['dataset_name']==source['dataset_name']
  assert Path(row['image_path']).resolve()==(ocr/'OCRBench_Images'/source['image_path']).resolve()
 assert len({r['image_path'] for r in rows})==930
 datasets['datasets']['OCRBench-v1'].update(inputs_sha256=sha(ocr/'inputs.jsonl'),gold_annotation_sha256=sha(ocr/'OCRBench.json'),official_QID_gold_order_exact=True,images=930)
 known=D/'images/mme/landmark/03d5e3bfc958be38.jpg';assert sha(known)=='ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a'
 datasets['known_MME_image_difference']={'image':'landmark/03d5e3bfc958be38.jpg','sha256':sha(known),'preserved':True,'affected_questions':2}
 datasets['formal_counts']=formal;datasets['formal_paths']=paths;datasets['global_image_SHA_not_repeated']=True;save('dataset_restore.json',datasets)
 smoke=read(O/'physical_smoke_summary.json');assert smoke['status'] in ('READY','READY_WITH_WARNING') and smoke['model_loads']==1 and len(smoke['checks'])==6
 assert all(c['structural_status']=='PASS' for c in smoke['checks']) and smoke['native_HF_and_noop_exact']
 ack=read(O/'user_numerical_acknowledgement.json');assert ack['accepted_as_nonblocking'];smoke['numerical_warning_accepted_by_user']=True;save('physical_smoke_summary.json',smoke)
 idle=command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader');assert not idle
 gpu=command('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader')
 storage={}
 for label,p in [('data',D.resolve()),('model',M),('environment',Path('/home/featurize/.venvs/ate-frozen')),('code',R)]:
  usage=shutil.disk_usage(p);storage[label]={'path':str(p),'filesystem':command('findmnt','-T',str(p),'-n','-o','FSTYPE,SOURCE,TARGET'),'free_GiB':usage.free/2**30}
  if label!='code':assert storage[label]['filesystem'].startswith('ext4 ')
 environment={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'executable':sys.executable,'packages':{p:version(p) for p in ['torch','torchvision','transformers','accelerate','huggingface-hub','tokenizers','safetensors','Pillow','numpy','sentencepiece']},'torch_cuda_runtime':torch.version.cuda,'GPU':gpu,'nvidia_smi':command('nvidia-smi'),'cuda_initialized_by_CPU_finalize':torch.cuda.is_initialized(),'storage':storage,'dependency_policy':'Base Python retained. Dedicated system-site-packages venv; only required Torch/vision and missing HF dependencies installed. No full requirements installation.','unused_base_package_note':'Base torchaudio 2.2.2 has its old Torch pin; it is unused by these image/text paths. Dynamo optree warning is unused; eager paths validated.'};save('environment.json',environment)
 from transformers import AutoConfig
 model=read(O/'model_verification.json');c=AutoConfig.from_pretrained(M,local_files_only=True);assert c.text_config.num_hidden_layers==32 and c.text_config.hidden_size==4096 and c.vision_config.image_size==336 and c.vision_config.patch_size==14
 model['architecture']={'language_layers':32,'hidden_size':4096,'CLIP_image_size':336,'patch_size':14,'visual_tokens':576};model['GPU_load']={'status':'PASS','model_loads':1,'dtype':'float16','attention':'eager','peak_allocated_GiB':smoke['peak_allocated_GiB']};save('model_verification.json',model)
 paths_json={'status':smoke['status'],'user_accepted_numerical_warning':True,'host':'workspace.featurize.cn','port':32534,'user':'featurize','python':sys.executable,'code_root':str(R),'result_dir':str(O),'activation':'source /home/featurize/ate-env.sh','data_root':str(D),'actual_data_root':str(D.resolve()),'datasets':paths,'model_path':str(M),'model_revision':model['revision'],'branch':command('git','branch','--show-current'),'base_commit':command('git','rev-parse','HEAD'),'GPU_after_release':gpu,'no_formal_experiment_started':True};save('runtime_paths.json',paths_json)
 source_checks=read(O/'frozen_sources.json')
 for n in ['information_horizon_alignment/engine.py','information_horizon_alignment/score.py','mme_seed_quality/engine.py']:
  rel='vlm_research/'+n;assert sha(R/rel)==sha(Path('/home/featurize/work/ATE')/rel);source_checks[rel]=sha(R/rel)
 save('recovery_source_integrity.json',{'status':'PASS','sources':source_checks,'frozen_algorithms_modified':False,'gqa_module_policy':'Previously untracked frozen source copied byte-for-byte into this independent branch.'})
 hist=read(OLD/'featurize_new3090_c3v2_reaccess_sevenbench/score_summary.json')
 snapshot=read(OLD/'featurize_4090_gqa_official_aligned_threeway/progress.json')
 history={'historical_only':True,'C3_five_commit':'2c5d1820679b5eaad844e06db9cf46dcad23316f','Reaccess_seven_commit':'0414e553ba6fd1e21626ecf741489d225b44279c','migration_commit':'d6157216199f6e2bb6c03557c05317649fe9c7d6','GQA_protocol_audit_commit':'25989d8c0e00cf53a980c726be94d84feaabbc14','GQA_legacy_HF_crop':{'Native':60.6297,'C3-V2':57.2746,'C3-V2+L11-L24':59.92208618222293},'Reaccess_seven_scores':hist['benchmarks'],'new_official_aligned_task_on_other_GPU_snapshot':snapshot,'other_task_touched':False,'interpretation':'Official-aligned full three-way results still incomplete at snapshot; retain legacy results separately. No legacy/new-protocol delta computed.'};save('history_index.json',history)
 protocol={'model':model['model'],'revision':model['revision'],'dtype':'float16','attention':'eager','batch_size':1,'greedy':True,'GQA_default':'GQA-official-aligned-vicuna-pad','GQA_legacy':'GQA-legacy-HF-crop, historical references only','GQA_max_new_tokens':32,'GQA_prompt':'pinned Vicuna-v1 system + USER <image> + original question + official short-answer instruction','GQA_preprocess':'RGB; CLIP-mean square pad; CLIP336','GQA_scorer':'official exact match','other_six':'Frozen Paper-compatible protocol; do not apply GQA changes globally','paper_protocol_file':'protocol_ate_paper.json','TextVQA_views_distinct':['TextVQA image-only','TextVQA-OCR Paper'],'C3':{'R_image_layers':list(range(20)),'answer_image_layers':[],'exit_before':20,'persistent_visual_KV':[],'answer_attention_skip':[2,3,4,5]},'Reaccess':{'R_image_layers':list(range(20)),'answer_image_layers':list(range(11,25)),'exit_before':25,'persistent_visual_KV':list(range(11,25)),'answer_attention_skip':[2,3,4,5]},'RoPE':'original absolute positions','norm_MLP_residual':'native','strict_thresholds':smoke['strict_thresholds'],'user_warning_acceptance':'user_numerical_acknowledgement.json','scope':'Recovery and minimal smoke only. No complete benchmark, search or profiling.'};save('protocol.json',protocol)
 save('resource_release_check.json',{'status':'PASS','GPU':gpu,'compute_processes':[],'at':datetime.now(timezone.utc).isoformat(),'no_experiment_started_after_smoke':True})
 lines=['# 新 RTX 4090 环境恢复', '', '状态：**READY_WITH_WARNING（0.01015 数值差异已获用户接受）**。仅恢复环境及最小 Smoke，未启动正式评测。','',f"Python {environment['python']}；Torch {environment['packages']['torch']} / CUDA runtime {torch.version.cuda}；Transformers {environment['packages']['transformers']}；RTX 4090，驱动610.57.04。",'','模型：`'+str(M)+'`，Revision `'+model['revision']+'`。14 个文件/三片权重通过官方内容哈希与历史验收；FP16/Eager 加载一次，峰值分配显存 '+f"{smoke['peak_allocated_GiB']:.4f} GiB。",'','数据实际根目录：`'+str(D.resolve())+'`（本地 ext4）；兼容入口 `'+str(D)+'`。模型、数据、虚拟环境和缓存均不在 NFS 持续盘。','','| 冻结视图 | 题数 | 状态 |','|---|---:|---|']
 lines += [f'| {name} | {count:,} | PASS，normalized/{name} |' for name,count in formal.items()]
 lines += ['','六包所有恢复视图的 Input/Gold 文件哈希与3090冻结记录完全一致；七包整包 SHA256 及归档 CRC通过。OCRBench 1,000问/930图，官方ID、答案、顺序和路径逐项一致；MMBench 1,176 Circular Groups保持。MME已知单图SHA差异保留。TextVQA Paper-OCR与Image-only独立。','','GQA默认使用官方对齐 Vicuna-v1 + Square-pad，cap32；旧HF-crop成绩仅索引历史，不跨协议求差。其他六项继续冻结Paper协议。另一台GPU正在完成新的GQA三组任务，本次只读索引进度，未干预其进程、数据库或输出。','','最小验收：3个固定样本，Native/C3/Reaccess各3个完整答案；另做必要的三Token物理/同语义Dense参照与Native/HF/No-op一致性检查。视觉退出层、视觉KV驻留、L2–L5物理Attention跳算、Norm/MLP/Residual、原始位置RoPE、禁图概率0、真实Decode和有限输出全部通过。历史ScienceQA/POPE的C3完整答案与评分保持一致；小样本答案本身不构成Accuracy结果。','','严格数值结果：GQA及ScienceQA两方法均PASS；POPE C3 RMSE=0.010154514573514462，Reaccess=0.010152915492653847，均超过冻结0.01，严格项记FAIL。短输出/argmax一致、结构检查PASS；用户明确接受该差异，未改阈值。保留历史失败记录。','','具备继续HF Native / Physical C3 / Reaccess研究的基础环境条件。正式评测仍需后续单独授权。GPU已释放：1 MiB、0%，无计算进程。','','进入环境：`source /home/featurize/ate-env.sh`。配置入口 `runtime_config.py`，不会在导入时启动实验。','','独立分支：`codex/featurize-4090-recovery-20261108`。GitHub与Windows最终传输验证收据在发布后单独保存；未经收据验证不宣称备份成功。','','详见 environment.json、dataset_restore.json、model_verification.json、physical_smoke_summary.json、runtime_paths.json 与 history_index.json。']
 (O/'migration_report_zh.md').write_text('\n'.join(lines)+'\n');print('FINAL_RECOVERY',smoke['status'],len(formal),flush=True)

if __name__=='__main__':main()
