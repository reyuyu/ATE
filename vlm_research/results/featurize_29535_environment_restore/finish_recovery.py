"""Finalize evidence for the authorized CPU-only resource restoration."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shutil
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_29535_environment_restore'
def read(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read('resource_acceptance.json');d=read('dataset_restore.json');h=read('history_archive_reuse.json');m=read('model_integrity.json')
assert a['status']==d['status']==h['status']=='PASS' and m['status']=='VERIFIED'
assert not a['runtime']['CUDA_initialized'] and not a['model_deserialized'] and not a['GPU_inference_run']
assert all(read(n+'_job.json')['status']=='PASS' for n in ('model','environment'))
tracked=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=R,text=True).strip();assert not tracked
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
lock=(O/'requirements.lock.txt').read_text().splitlines();previous=(O/'previous_requirements.lock.txt').read_text().splitlines()
assert {x.lower() for x in lock}=={x.lower() for x in previous},'Environment differs from archived version lock'
result={'status':'PASS','scope':'GQA + TextVQA Paper-OCR + pinned local model + frozen CPU software environment',
 'endpoint':'workspace.featurize.cn:29535','finished_at_UTC':datetime.now(timezone.utc).isoformat(),
 'model':m['model'],'revision':m['revision'],'model_path':m['model_path'],'model_files':len(m['files']),'model_file_hashes':'ALL_PINNED_LFS_SHA256_OR_GIT_BLOB_HASHES_VERIFIED',
 'data_path':d['canonical_data_root'],'physical_data_path':d['physical_local_data_root'],'priority_datasets':{'GQA':12578,'TextVQA-Paper-OCR':5000},
 'priority_images':{'GQA':398,'TextVQA':3166},'historical_files_verified_reused':len(h['files']),'historical_prediction_transfer_bytes':0,
 'dataset_transfer_bytes':d['total_transfer_bytes'],'runtime':a['runtime'],'software_lock_identical_to_previous_instance':True,
 'selected_cohort':a['fixed_accuracy_cohort'],'input_hashes_verified':a['input_tensor_hashes_verified'],'missing_required_files':a['missing_required_files'],
 'other_datasets_deferred_not_required_this_scope':['MME','ScienceQA-IMG','POPE','MMBench','OCRBench-v1'],
 'Git_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'tracked_changes_after':tracked,
 'GPU':gpu,'GPU_compute_processes':apps,'GPU_inference_run':False,'FP16_Eager_GPU_execution':'NOT_VALIDATED_THIS_TASK',
 'local_disk_free_GiB':round(shutil.disk_usage('/home/featurize').free/2**30,2),'shutdown_executed':False,'GitHub_push_this_restore':False}
(O/'recovery_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(O/'recovery_progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
text=f'''# RTX 3090 新实例 29535：ATE 环境恢复

**状态：PASS（本轮优先恢复范围）**。未运行模型推理、Benchmark、融合实验或 GPU 验证。

| 资源 | 位置 / 状态 |
|---|---|
| 模型 | `{m['model_path']}` |
| 固定 Revision | `{m['revision']}` |
| 数据稳定入口 | `{d['canonical_data_root']}` |
| 数据物理位置 | `{d['physical_local_data_root']}` |
| Python | `{a['runtime']['executable']}` |
| 软件 | PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124 / Transformers 4.49.0 |
| GQA | 12,578 题，398 图 |
| TextVQA Paper-OCR | 5,000 题，3,166 图 |

模型、数据、虚拟环境实际位于本地 ext4 盘；持续盘只保留原有代码、历史结果及本次轻量恢复证据。
模型 17 个文件按固定提交清单验证：所有三个权重分片均匹配 LFS SHA256，Tokenizer、Processor 与配置匹配 Git blob 哈希。
下载支持已完成文件复用与部分文件续传；HTTP 部分响应触发续传的历史状态保留在下载日志中，最终以完整哈希为准。
CPU 仅检查 safetensors 的头、张量映射、FP16 dtype 和文件长度，不反序列化权重。

从真实 Windows `D:\\ATE-dataset-backups-3090` 主动 SFTP 上传两个缺失 ZIP，共 {d['total_transfer_bytes']:,} 字节。
GQA 包完整上传校验通过后复用。TextVQA 从已校验的 69,468,160 字节前缀继续，以四个互不重叠的范围上传，分片与最终完整 ZIP 的 SHA256 均验证。
ZIP 解压检查 CRC，并沿用已有可信图片哈希清单；没有重新扫描所有图片 SHA256。
恢复后的 GQA、TextVQA Image-only / Paper-OCR 三个独立输入视图及 Gold 文件与历史冻结 SHA256 逐字节一致。
全部 Question ID/Gold 集合和图片路径完整；Paper-OCR 视图保留原 OCR 文本及十答案，未换成 Image-only。

持续盘上 23 个历史预测/配置/报告文件与本机可信归档逐文件 SHA256 一致，全部直接复用，没有重传或重新生成预测。
固定 GQA100+TextVQA100 的 CPU 验收重建 200 题的原 Token IDs、Attention Mask、FP16 Pixel Values 哈希，并核对选中图片 SHA256；
TextVQA 历史 FP32 预处理哈希也逐题对应。Native/V²Drop 400 条历史预测完整保留。没有执行 452 项回归或 GPU 模型加载。

Python 环境复用原实例完整版本锁安装，`pip check` 通过，最终 `pip freeze` 与原锁定版本集合相同。
FP16 权重、Eager 所需软件和输入协议已准备；真实 FP16/Eager GPU Prefill/Decode 在本任务中为 **NOT_VALIDATED**。
冻结源码 Git HEAD `{result['Git_HEAD']}`；tracked diff 为空，旧研究输出没有覆盖。

本轮必需文件缺失：无。MME、ScienceQA、POPE、MMBench、OCRBench 的图片数据按用户“按需恢复”要求暂未传输；
其 Windows 原备份和持续盘历史结果均保留，不能将本轮状态误称为七项数据全部可运行。

GPU 当前：{gpu}；活动计算进程：{apps or '无'}。CUDA 在 CPU 验收中未初始化，服务器未关闭。
本地盘剩余 {result['local_disk_free_GiB']:.2f} GiB。
下次恢复步骤见 `restore_next_gpu_zh.md`；本机归档和真实 SHA256 回执在后续备份完成后生成。
'''
(O/'report_zh.md').write_text(text)
print('CPU_RECOVERY_COMPLETE_PASS',flush=True)
