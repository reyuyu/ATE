"""Render a truthful compact migration report after the GPU process exits."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,subprocess,hashlib,ast
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_4090_c3v2_handoff'
def read(n):return json.loads((O/n).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
results=read('comparison_results.json');d=read('dataset_restore.json');e=read('environment.json');m=read('model_download.json');plan=read('comparison_cases.json')
assert cmd('git','rev-parse','HEAD')==plan['code_commit'] and not cmd('git','status','--porcelain','--untracked-files=no')
assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')
assert d['status']==e['status']==m['status']=='PASS'
gpu=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.free,utilization.gpu','--format=csv,noheader')
paths=read('runtime_paths.json');paths.update(status=results['status'],dataset_restore_status=d['status'],comparison_completed=len(results['cases']),GPU_after_process_exit=gpu,finished_at=datetime.now(timezone.utc).isoformat())
save('runtime_paths.json',paths)
m['storage_repair']='Initial task-owned NFS staging failed during a shard read. Final model was downloaded and validated on local ext4. All task-owned persistent data staging was then removed; local_disk_migration.json records the transfer. No model protocol or weights changed.'
save('model_download.json',m)
summary=read('physical_smoke_summary.json') if (O/'physical_smoke_summary.json').exists() else {'status':'NOT_VALIDATED','cases':[]}
reused=[]
if results['status']=='BLOCKED' and results['failure']['phase']=='PHYSICAL_SMOKE':
 raw=read('physical_smoke.json');refs={c['sample']['key']:c['reference_3090'] for c in plan['cases']}
 fields=['benchmark','key','status','numerical_equivalence','rope_exact','visual_hidden_removed_before_L20','answer_attention_physically_skipped','compact_kv_original_positions_exact','blocked_image_probability_zero']
 summary={'status':'BLOCKED','passed_cases':[dict((k,c[k]) for k in fields) for c in raw['cases']],
  'failed_benchmark':'POPE','failed_key':'POPE:10000001','failure_phase':'same-algorithm physical versus dense short numerical gate',
  'numerical_failure':ast.literal_eval(results['failure']['error'].removeprefix('AssertionError(').removesuffix(')')),
  'not_run':['TextVQA physical smoke','MMBench physical smoke','Dedicated fifteen-case comparison'],
  'unchanged_frozen_tolerances':True,'no_further_GPU_work':True}
 save('physical_smoke_summary.json',summary)
 for c in raw['cases']:
  a=c['actual'];ref=refs[c['key']]
  assert a['eos'] and not a['truncated'],'Do not compare an incomplete prefix as a full answer'
  reused.append({'key':c['key'],'source':'Reuse completed physical-Smoke output; no new generation',
   'original_smoke_cap':3,'stopped_on_EOS':True,
   'output_4090':{k:a[k] for k in ['text','token_ids','eos','truncated','first_logits_sha256']},
   'output_3090':ref['generation'],'scoring_4090':c['scoring'],'scoring_3090':ref['scoring'],
   'matches':{'tokens':a['token_ids']==ref['generation']['token_ids'],'text':a['text']==ref['generation']['text'],'parsed_and_score':c['scoring']==ref['scoring']},
   'input_tensor_comparison':'Dedicated cross-hardware input-tensor comparison was not reached; frozen normalized input/Gold file hashes and selected image hashes were verified before model load.'})
 save('completed_physical_case_comparison.json',{'status':'PARTIAL','compared':len(reused),'cases':reused,'whole_migration_status':'BLOCKED','no_new_generation':True})
lines=['# RTX 4090 原始 Physical C3-V2 迁移验收','',f"**状态：{results['status']}**。本轮仅恢复数据、模型、环境并进行固定案例迁移核对；不是正式 Benchmark 成绩。",'',
 f"代码：`{plan['code_commit']}`。模型：`llava-hf/llava-1.5-7b-hf@{plan['revision']}`。",
 f"环境：Python {e['python']}；Torch {e['packages']['torch']} / CUDA {e['cuda_runtime']}；Transformers {e['packages']['transformers']}；FP16、Eager、greedy、batch=1。",
 f"GPU：{results.get('gpu','not loaded')}。模型加载 {results['model_loads']} 次；峰值 allocated {results.get('peak_allocated_mib',0)/1024:.3f} GiB，reserved {results.get('peak_reserved_mib',0)/1024:.3f} GiB。",'',
 '数据、模型、venv、wheel 缓存均在实例本地 ext4。持续盘只保留代码、报告及小型结果。','',
 '- 实际数据：`/home/featurize/ate-local-data-4090/ate-recovery`。',
 '- 实际模型：`/home/featurize/ate-local-models/llava-1.5-7b-hf`。',
 '- Python：`/home/featurize/.venvs/ate4090-frozen/bin/python`。',
 '- 原路径 `/home/featurize/data/ate-recovery` 通过本地盘软链接保留，冻结数据与 Prompt 不变。','',
 '数据来自已经验证的 Windows `D:\\ATE-dataset-backups-3090`，没有重新下载数据集。六份 portable ZIP 和 OCRBench 迁移包均核对整体 SHA256；七类数据均恢复。六类数据的 14 个 normalized 视图均包含输入和 Gold，合计 28 个文件与 3090 验收 SHA256 完全一致，所有图片映射均存在。OCRBench 保留官方 1,000 题、930 张图及原始清单。模型 14 个文件经过官方固定 revision 内容校验，三份权重 shard 与 3090 SHA256 完全相同。','',
 '| 冻结输入视图 | 恢复题数 |', '|---|---:|']
for view in ['MME-paper','ScienceQA-IMG-paper','POPE-paper','TextVQA-OCR','MMBench-paper','GQA-testdev-balanced','OCRBench-v1']:
 lines.append(f"| {view} | {d['datasets'][view]['questions']} |")
lines+=['',f"物理 Smoke：**{summary['status']}**（原计划五类各 1 个样本，短 decode；复用原冻结物理检查）。检查实际图像输入、L20 前视觉 Hidden 移除、L2–L5 答案 Attention 物理跳算、P−1/Decode 图像访问禁令、紧凑 KV 原位置、RoPE、输出有限性与同语义 dense 短数值参考。未放宽误差门槛。",'',
 f"独立固定案例对比：{len(results['cases'])}/15 完成。物理门禁失败时，后续对比不启动。"]
if results['cases']:lines+=['','| 样本 key | 3090 原输出 | 4090 原输出 | 输入 / token / 文本 / 评分全同 |', '|---|---|---|---|']
def cell(x):return str(x).replace('|','\\|').replace('\n','<br>')
for c in results['cases']:
 lines.append(f"| {cell(c['key'])} | {cell(c['reference_3090_generation']['text'])} | {cell(c['generation_4090']['text'])} | {'是' if all(c['matches'].values()) else '否：'+str(c['matches'])} |")
if reused:
 lines+=['', '**停止原因**：第三个物理样本 POPE:10000001 在 physical 与同语义 dense C3 的数值参考比较中，hidden_relative_rmse=0.010154514573514462，超过冻结阈值 0.01。最大 logits 绝对差 0.015625、argmax 一致、hidden_cosine_distance=0.000030875205993652344；但门禁仍为失败。MME、ScienceQA 两例物理检查通过，TextVQA/MMBench 物理检查及独立 15 案例循环均未执行。没有 OOM 或已报告的 NaN。',
  '这是 4090 上物理实现与同语义 dense 短参考的数值门禁失败，不能据此断言 3090/4090 的答案已经不同。需要定位数值差异的来源；本轮不改变层号、误差阈值、FP16 或 Backend，也不追加 GPU 试跑。',
  '', '已把两个成功物理 Smoke 的原输出离线对照旧 3090 预测；两者均提前遇到 EOS，短 cap 没有截断答案，不产生额外 GPU 生成：','', '| 已保存的物理样本 | 3090 | 4090 | token / 文本 / 评分一致 |','|---|---|---|---|']
 for c in reused:lines.append(f"| {c['key']} | {c['output_3090']['text']} | {c['output_4090']['text']} | {'是' if all(c['matches'].values()) else '否'} |")
 lines+=['', '这两例的 frozen 输入/Gold 文件和图片已校验；独立跨硬件输入 tensor 对比循环尚未开始。不能用这两例替代六类/15 例完整验收。当前结论是数据与模型恢复 PASS、物理迁移门禁 BLOCKED，尚不具备直接开展正式 C3 质量评测的验收结论。']
diag=[c for c in results['cases'] if c['first_logits_bit_exact'] is not None]
lines+=['',f"独立对比进度与失败记录保存在 comparison_results.json；完整计划输入绑定和历史来源保存在 comparison_cases.json。输出差异列表（仅适用于已经执行的独立对比）：{results.get('mismatches',[])}。",'',
 '首步 logits 逐位哈希仅作跨硬件诊断，未执行的独立对比不记作相同；本轮没有将不同 GPU 的 FP16 中间量逐位相同作为迁移标准。','',
 '冻结计划保留 MME 同图双题；本次图片 OCR/0001.jpg 不涉及已知 SHA 差异 landmark/03d5e3bfc958be38.jpg，未替换数据。计划的 TextVQA 为 Paper-OCR 十答案软评分，POPE 保留三策略及 8,910 题论文视图，MMBench 为完整 4 行 Circular 组，GQA 为原历史 C3-V2 入口与答案规范化。未执行的对比与组评分不记为通过。',
 'OCRBench 已恢复全部数据；由于没有此前原始 C3-V2 对照，本轮没有额外生成 OCRBench，也没有启动任何正式全量或回看变体实验。','',
 '模型进程已退出；退出后 GPU：`'+gpu+'`。没有后台模型任务。',
 'Windows 独立归档目录：`D:\\ATE-dataset-backups-3090\\ATE-results\\4090-c3v2-handoff`。GitHub 仅同步本轮轻量证据，具体提交与逐文件 SHA256 验证见 github_sync_receipt.json。']
if results.get('failure'):lines+=['','完整原始异常与 traceback 已保存在 comparison_results.json 和 gpu_smoke.log。未修改冻结代码或误差门槛。']
(O/'report_zh.md').write_text(('\n'.join(lines)+'\n').replace('\\\\','\\'))
names=['report_zh.md','comparison_results.json','comparison_cases.json','physical_smoke_summary.json','physical_smoke.json','completed_physical_case_comparison.json','gpu_smoke.log','environment.json','model_download.json','dataset_restore.json','runtime_paths.json','local_disk_migration.json','compare_c3.py','restore_datasets.py','download_model.py','activate_4090.sh','finalize_handoff.py']
artifacts={}
for name in names:
 p=O/name
 if p.is_file():artifacts[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
save('artifact_manifest.json',{'status':results['status'],'files':artifacts})
print('FINAL_REPORT_READY',results['status'],len(results['cases']),gpu,flush=True)
