"""Freeze this SVH-R supplement; the completed base archive is read-only."""
import sys,json,tarfile,hashlib,shutil,os,datetime
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-svhr-efficient-attention-22875-20261113');B=O.parent/'qwen3vl-ocrbench-memory-efficient-attention-22875-20261113';L=O.parent/'ocrbench-ate-frozen-completion-22875-20261113';T=Path('/home/featurize/work/ATE/vlm_research/results/qwen3vl_ocrbench_svhr_efficient_attention')
sys.path[:0]=[str(L)]
from run_common import atomic
assert json.loads((O/'final_independent_audit.json').read_text())['status']=='PASS' and json.loads((O/'gpu_release.json').read_text())['status']=='RELEASED'
R=O/'frozen_references';R.mkdir(exist_ok=True)
H=Path('/home/featurize/work/ATE/vlm_research/results/qwen_svhr_physical_overhead_audit');controller=H/'source_snapshots/home/featurize/ate-experiments/rtx3090-18conditions-20261113/vlm_research/results/rtx3090_unified_efficiency_18conditions/code/cross_runtime.py'
source_manifest=json.loads((H/'source_manifest.json').read_text());entry=next(x for x in source_manifest['files'] if x['path'].endswith('/code/cross_runtime.py'))
assert hashlib.sha256(controller.read_bytes()).hexdigest()==entry['sha256']=='14a19c24b33f2693d7cd1e69d4d09562b69fda5b6ba8343109fc81ebcb67460c'
shutil.copyfile(controller,R/'prior_cross_runtime.py');shutil.copyfile(H/'source_manifest.json',R/'prior_audit_source_manifest.json')
for p in [L/'run_common.py',L/'qwen_frozen_benchmark_inputs.py',L/'results/qwen/rows.json',L/'results/qwen/cpu_layouts.json.gz',Path('/home/featurize/ate-local-data/bundles/OCRBench-v1/OCRBench.json')]:shutil.copyfile(p,R/p.name)
for f in ['archive_receipt.json','windows_backup_receipt.json','final_report_zh.md','gate_decision.json','numerical_comparison.json','gpu_backend_probe.json','svh_physical_gate.json','new_physical_source_binding.json','new_native_source_binding.json','acceptance_summary.json','prediction_pattern_audit.json','model_and_scorer_binding.json','physical_engine.py','runtime.py','formal_run.py','native_gate_run.py','svh_stage.py','cpu_prepare.py','physical_cpu_test.py','compact_math_test.py']:
 shutil.copyfile(B/f,R/('base_'+f))
shutil.copytree(B/'frozen_dependencies',R/'base_frozen_dependencies',dirs_exist_ok=True)
(O/'README_REPRODUCE_zh.md').write_text('''# 冻结 Qwen SVH-R OCRBench1000 复核说明

本目录独立于已完成的Native/Original实验；不改历史目录。原始模型/图片不在GitHub或结果包内，需按既有模型Revision与930原图SHA恢复到本地盘。

正式协议见protocol.json。read_layers=19..35；主视觉Hidden在L22前退出；L1..6答案Attention跳算；辅助图像前缀构造L22..34完整块与L35仅QKV。不得替换为InternVL/LLaVA层集合。

执行器svhr_adapter.py只继承冻结Physical并桥接其KV捕获钩子。sources/中36项基础依赖复用已完成Base实验的7e18895a6ccd57c4cb3aaedd5d512789e7467b59源码快照，新增SVH-R physical.py/engine.py来自GitHub f21254f7c9d371b0f2cd48cf25a8ac81cc9a2760并逐字节核验。两处Original核心SHA相同：a719b590975e6282ad983fcd27b80f2e031c302caf47eaf54d50e1f3e36cca77。L19..35配置来自此前overhead audit已冻结cross_runtime.py，SHA 14a19c24b33f2693d7cd1e69d4d09562b69fda5b6ba8343109fc81ebcb67460c，与旧source_manifest.json一致；不声称这个Controller路径已在f212提交。

完整归档在本地/持续盘与Windows备份，含predictions/1000个原始JSON、物理KV/Mask执行审计、Gate Logits、源文件、冻结输入布局和原注释。GitHub仅放轻量逐题预测、输入/Gold/哈希、CSV、报告与必要源码，勿上传模型或图片。

便携CPU复核：在此目录执行 `python -B verify_light_cpu.py .`，只需Python标准库，无需Torch、模型、图片或网络。它对三份轻量预测共3000条逐题复算OCRBench v1得分，核对相同Gold和全部输入哈希；结果写入light_verification_result.json。Gold保存在预测文件中，inputs_light.jsonl不含Gold。

完整CPU独立复核：按本说明中的服务器布局恢复后，在既有qwen3vl-torch260-tf4576环境执行 `python -B final_audit.py`；它只读取预测/Tokenizer/原注释，不加载GPU模型，另外核对Token解码、真实KV与物理执行审计。脚本包含首次终态保护；再次审计应在独立副本上调整状态/路径，禁止改写原证据。SOURCE_SHA256.json对应完整服务器归档；GitHub只保留可核验的轻量子集。

GPU脚本 `run.py` 使用逐题原子保存、fresh KV及断点读取；已完成1000题，禁止直接重跑。重新复现需新的独立目录，保持protocol/所有输入Tensor哈希不变。依赖版本Torch2.6.0+cu124、Transformers4.57.6、BF16，单线程TF32关闭；只启用SDPA_EFFICIENT_ATTENTION，禁用MATH/FLASH/CUDNN。

严格数值门禁FAIL和对应警告完整保留。Score不是Eager完整对照、不是数值等价证明，三组同一输入/新后端才可配对比较。既有base预测和archive SHA均未改变。

归档与Windows真实SHA核验分别见archive_receipt.json、windows_backup_receipt.json；GitHub真实Commit以发布后独立回执为准。GPU释放见gpu_release.json，未关闭服务器。
''',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
files=sorted(p for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp' and p.name not in ['SOURCE_SHA256.json','archive_receipt.json','windows_backup_receipt.json','github_publication_receipt.json'])
manifest={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};atomic(O/'SOURCE_SHA256.json',manifest)
archive=O.parent/(O.name+'-results.tar.gz');assert not archive.exists()
tmp=archive.with_suffix('.tmp')
with tarfile.open(tmp,'w:gz',compresslevel=3) as tar:
 for p in files+[O/'SOURCE_SHA256.json']:tar.add(p,arcname=str(Path(O.name)/p.relative_to(O)),recursive=False)
os.replace(tmp,archive);h=sha(archive);dest=T/archive.name;temp=dest.with_suffix('.tmp');shutil.copyfile(archive,temp);assert sha(temp)==h;os.replace(temp,dest)
receipt=dict(status='SERVER_ARCHIVE_VERIFIED',local_archive=str(archive),persistent_archive=str(dest),bytes=archive.stat().st_size,sha256=h,files=len(files)+1,persistent_copy_sha256_equal=True,contains_model_weights=False,contains_dataset_images=False,contains_credentials=False,Windows_backup='PENDING_ACTUAL_TRANSFER',base_archive_receipt=json.loads((B/'archive_receipt.json').read_text()),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
atomic(O/'archive_receipt.json',receipt)
for f in ['archive_receipt.json','SOURCE_SHA256.json','README_REPRODUCE_zh.md']:shutil.copyfile(O/f,T/f)
print(json.dumps({k:v for k,v in receipt.items() if k!='base_archive_receipt'}))
