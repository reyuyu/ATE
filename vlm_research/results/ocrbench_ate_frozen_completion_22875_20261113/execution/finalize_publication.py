"""Package completed/blocked results; never performs model inference."""
import sys,json,gzip,shutil,hashlib,subprocess,datetime,collections,csv
from pathlib import Path
from run_common import *
OUT=L/'publication/vlm_research/results/ocrbench_ate_frozen_completion_22875_20261113'
S=json.loads((OUT/'score_summary.json').read_text());table=S['results'];stats={}
def gzcopy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 with Path(src).open('rb') as fi,gzip.open(dst,'wb') as fo:shutil.copyfileobj(fi,fo)
for name in ['internvl3','llava','qwen']:
 R=L/'results'/name;status=json.loads((R/'progress.json').read_text())
 assert status['status'] in ['COMPLETE_PENDING_INDEPENDENT_AUDIT','BLOCKED'],(name,status)
 stats[name]=status
 for fn in ['smoke_independent_check.json','inputs_verified.json','token_decode_audit.json','environment_cpu_audit.json','historical_inventory_correction.json','continuation_protocol.json','continuation_smoke.json','partial_smoke.json','environment_attempt1.json','intentional_interruption_attempt1.json']:
  if (R/fn).exists():shutil.copy2(R/fn,OUT/'protocols'/name/fn)
 (OUT/'inputs').mkdir(exist_ok=True)
 if name=='internvl3':shutil.copy2(R/'inputs_verified.json',OUT/'inputs/internvl3_bindings.json')
 if name=='llava':gzcopy(R/'cpu_bindings.json',OUT/'inputs/llava_cpu_bindings.json.gz')
 if name=='qwen':shutil.copy2(R/'cpu_layouts.json.gz',OUT/'inputs/qwen_cpu_layouts.json.gz')
for src,dst in [(L/'history/llava_ocr_protocol.json','llava_original_ocr_protocol.json'),(L/'history/llava_paper_protocol.json','llava_original_paper_protocol.json'),(L/'history/llava_native20_protocol.json','llava_verified_partial_original_protocol.json'),(L/'history/fastv_protocol.json','verified_full_native_original_protocol.json'),(L/'history/fastv_windows_integrity.json','verified_full_native_windows_SHA256.json')]:
 if src.exists():shutil.copy2(src,OUT/'protocols'/dst)
shutil.copy2(A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py',OUT/'official_ocrbench_v1_example.py')
shutil.copy2(L/'source_audit.json',OUT/'source_audit.json')
shutil.copy2(L/'input_request_audit.json',OUT/'input_request_audit.json')
(OUT/'execution').mkdir(exist_ok=True)
for fn in ['run_common.py','internvl_prepare.py','internvl_run.py','llava_prepare.py','llava_run.py','llava_c3_continue.py','correct_history.py','qwen_prepare.py','qwen_run.py','audit_collect.py','check_physical.py','verify_tokens.py','finalize_publication.py']:
 shutil.copy2(L/fn,OUT/'execution'/fn)
shutil.copy2(L/'verify_cpu.py',OUT/'verify_cpu.py')
(OUT/'logs').mkdir(exist_ok=True)
for n in ['internvl','llava','llava_c3_continue','qwen']:
 p=L/'control'/f'{n}_run.log'
 if p.exists():
  lines=p.read_text(errors='replace').splitlines();(OUT/'logs'/f'{n}_run.log').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for n in ['qwen_local_integrity.json']:
 if (L/'control'/n).exists():shutil.copy2(L/'control'/n,OUT/'protocols'/n)
proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not proc,proc
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
atomic(OUT/'gpu_release.json',dict(status='RELEASED',compute_apps=[],gpu=gpu,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
failures={n:json.loads((L/'results'/n/'failure.json').read_text()) for n in stats if (L/'results'/n/'failure.json').exists()}
correction=json.loads((L/'results/llava/historical_inventory_correction.json').read_text());repeats=correction['excluded_duplicate_native_generations']
meta=dict(status='PARTIAL',all_conditions_full_coverage=all(s['complete'] for s in table),execution_compliance='FAILED_NO_REPEAT_REQUIREMENT',dataset='OCRBench v1',questions=1000,unique_original_images=930,annotation_sha256=S['dataset_annotation_SHA256'],source_workspace_commit=subprocess.check_output(['git','-C','/home/featurize/work/ATE','rev-parse','HEAD'],text=True).strip(),frozen_source_commits={n:json.loads((L/'results'/n/'protocol.json').read_text())['source_commit'] for n in stats},gpu=gpu,model_loads=dict(internvl3=1,llava=2,qwen=1),sequential_gpu_only=True,official_24_tile_protocol_used=False,completed_generations_repeated=repeats,excluded_duplicate_generations=repeats,accepted_new_generated_total=sum(s['new_generated'] for s in table),actual_successful_GPU_generations=sum(s['new_generated'] for s in table)+repeats,reused_total=sum(s['reused'] for s in table),independent_scores_match=True,failures=failures,all_models_images_env_on_local_disk=True,no_frozen_source_modification=True)
atomic(OUT/'run_metadata.json',meta)
with (OUT/'images_930_SHA256.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['image_id','sha256']);seen={r['image_id']:r['image_sha256'] for r in json.loads((L/'results/internvl3/frozen_rows.json').read_text())};w.writerows(sorted(seen.items()))
report=['# OCRBench v1：原冻结 ATE 协议缺失条件补齐', '',f"状态：**{meta['status']}**。存在历史检索遗漏造成的重复计算，已纠正、保留并排除重复输出；详见下文。",'', '| 模型 | 方法 | 得分 /1000 | 覆盖 | 接受的新生成 | 历史复用 | 空输出 | 达生成上限 |','|---|---|---:|---:|---:|---:|---:|---:|']
for s in table:
 score=str(s['score_0_1000']) if s['complete'] else '未完成，不报正式分数'
 report.append(f"| {s['model']} | {s['method']} | {score} | {s['coverage']}/1000 | {s['new_generated']} | {s['reused']} | {s['empty']} | {s['truncated']} |")
report+=['','## 输入及协议','', '1000 道题按原始索引 0000–0999；原始官方 ID 存在重复，未以它覆盖索引 ID。930 张唯一原图均按冻结 SHA256 核验，Question、Gold、原顺序一致。Gold 只供离线评分，不加入模型输入。所有条件独立新建 KV；Smoke 成功输出计入正式预测，未重复生成。', '', '| 模型 | 冻结 Revision | dtype / Transformers | 图像及 Prompt | 生成 |', '|---|---|---|---|---|', '| LLaVA-1.5-7B | b234b804b114d9e37bb655e11cbbb5f5e971b7a9 | FP16 / 4.49.0 | RGB，CLIP mean 补正方形，336；原 Vicuna-v1、单词/短语指令；576 视觉 Token | Eager，greedy，batch=1，最多128 Token |', '| Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | BF16 / 4.57.6 | 冻结默认 RGB Processor，无像素/Token 缩减；原 Qwen Chat Template 和 body 后缀 | Eager，greedy，batch=1，最多128 Token |', '| InternVL3-8B | 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 | BF16 / 4.49.0 | **原 ATE 单图块 448×448 /256视觉 Token**，internvl2_5 template，原简短答案指令 | Eager，greedy，batch=1，最多128 Token |','', '**没有混入官方动态 max_num=24、缩略图新协议结果；本表不是该新协议的官方复现成绩，也不直接证明与论文分数的差距原因。** 所有环境使用 PyTorch 2.6.0+cu124，模型、图片、环境及生成目录位于本地 ext4。', '', '## 复用与物理检查','', '- InternVL Native751、Physical-D32 的各1000条原始输出全部复用并重评分；Late9 复用0053、0963、0988三条，仅新生成997条。四题 Smoke 检查视觉映射、L28前 Hidden 退出、L19–27选择性视觉 KV、答案跳算、原位置和有限性。', '- InternVL Late9 **保留原 `EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER`**。本次结构 Smoke 不消除历史严格数值门禁失败，也不构成新的 Native/Dense 等价性证明。', '- LLaVA SVH-R 原1000条直接复用；另一历史归档虽计划20题，实际已完成且经逐题输入/实现核验可复用的是 Native、SVH各10条；其余各990条才生成。四题 Smoke 的 L20前视觉 Hidden 退出、L2–L5答案 Attention 跳算、原文本位置、禁止答案视觉访问和最终无原始视觉 KV 均独立检查。', '- LLaVA 附加 CPU 检查器首版把 compact `query_positions` 当作原 RoPE位置，引发断言；已改查 `visual_audit.original_query_positions`，四题通过。冻结执行器、参数、已有预测均未更改，未重跑样本。', '- Qwen 的 ATE/official 两个旧 Manifest 视图对 OCRBench 的1000个请求哈希完全一致，因此不会为相同请求重复生成。Native 与 Physical-Original 使用同一输入，原模型默认预处理保留不变；最长 Smoke 为0738，16307个总Token /16264个图像Token。','']
if failures:
 report+=['## 阻塞及失败','']
 for n,e in failures.items():report += [f"- **{n} BLOCKED**：{e.get('error_type')}；请求 `{e.get('current',e.get('question_id'))}`。{e.get('error','')}" ]
 report+=['','出现 OOM 后停止该模型，不跳题、不补零、不降分辨率、不换 Backend；已成功输出保留，缺失请求仍为缺失。部分正确数只保存在统计文件中，不当作 /1000 正式成绩。Qwen Physical-Original 的历史数值门禁/探索性标记同样保留；本次没有运行 Dense-F，也未宣称新的数值等价性通过。','']
report+=['## 独立评分和证据','', '每条保存原始文本、输出 Token IDs、Question ID、图像 SHA、输入绑定和来源。官方 OCRBench v1 example.py 的固定评分循环与单独实现的标准化/子串匹配逐题交叉核对，分母固定1000；HME100k 保留大小写并去空格，其他集合转小写。所有已保存得分一致，未删空输出或截断。', '', '`predictions/*.jsonl.gz` 可直接 CPU 复算；`input_manifest.jsonl`、`images_930_SHA256.csv` 和 `inputs/` 记录输入/Gold/图像及处理后张量绑定；`protocols/` 保存冻结来源、源文件 SHA、环境、Smoke 和失败。`execution/` 仅本次隔离入口，冻结核心按指定 Commit 读取，未覆盖原仓库。', '', '执行 `python verify_cpu.py` 可核验文件 SHA、1000题映射、930图像哈希清单和全部已有预测的两套评分。完整逐题 GPU 审计及缓存布局另外压缩归档，模型与原图不会上传 GitHub。', '',f"GPU 已释放：`{gpu}`；未关闭服务器。",'', 'Windows及持续盘备份的实际传输/SHA回执见 `backup_receipt.json`（提交前完成并核验）。']
report=[s.replace('Smoke 成功输出计入正式预测，未重复生成。','缺失条件的 Smoke 成功输出计入正式预测；被误重跑的 Native Smoke 随其他重复输出一并排除。').replace('- LLaVA SVH-R 原1000条直接复用；另一历史归档虽计划20题，实际已完成且经逐题输入/实现核验可复用的是 Native、SVH各10条；其余各990条才生成。','- LLaVA SVH-R 原1000条直接复用。补查 Windows FastV 归档发现 Native 已完整1000条，输入哈希/Gold/协议逐项完全一致，正式表全部复用该历史301分。Physical-C3 复用另一归档已完成的10条，本次仅接受其余990条。') for s in report]
report+=['','## 执行失误与纠正','',f'前期历史检索遗漏了 Windows FastV Native 全量归档，误启动并完成 **{repeats}条重复 Native 生成**，违反了用户的严禁重复要求。发现后立即暂停并终止首个 LLaVA 进程，恢复/核验历史1000条，后续仅补 Physical-C3 的65条缺失请求，因此 LLaVA 额外加载一次。', '', '重复原始文件全部保留在完整归档 `results/llava/excluded_duplicate_generations/Native/`，未删除、未覆盖历史预测，也未以其替换历史301分或加入正式分母。机器可读的历史检索纠正、排除ID及初始Smoke中已失效的“未重复”库存判断见 `protocols/llava/historical_inventory_correction.json`。本轮不能宣称满足零重复执行要求；已完成条件的输入/评分正确性与该执行失误分开报告。']
(OUT/'report_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
readme='''# CPU复核与恢复说明

本目录仅补原冻结ATE协议OCRBench，不是InternVL动态24图块新协议。

1. 使用标准Python 3.11+运行 `python verify_cpu.py`，无需GPU、模型或原图。
2. 冻结核心分别从 `protocols/{llava,qwen,internvl3}/protocol.json` 的 source_commit 及 source_SHA256 精确取回；不可用当前 main 中相似文件替代。
3. execution/ 下入口的 run_common.py 记录本次服务器资源根目录；恢复到其他实例时仅调整隔离入口资源路径，先CPU比对 inputs/ 处理后输入哈希，再允许GPU。不要改旧Manifest/核心算法。
4. InternVL已完整1000题，不再生成；LLaVA已完成条件同理。Qwen若BLOCKED，只能在原协议允许的足够显存资源上恢复缺失请求；更换分辨率或Attention Backend属于新协议，须另行授权与标记。
5. 每题独立KV，原始输出逐题原子持久化。只允许恢复缺失ID，完整记录必须验证原protocol哈希后复用；不得把Smoke成功样本再跑一遍。

源分支：
- LLaVA: codex/featurize-new3090-c3v2-reaccess-sevenbench @0414e553ba6fd1e21626ecf741489d225b44279c
- Qwen: codex/qwen3vl-c3-3090-official-eval @7e18895a6ccd57c4cb3aaedd5d512789e7467b59
- InternVL: codex/internvl3-svhr-late9-sevenbench-2x3090 @c60a9919112a3cf5bec8c494a533cba56a97f29b
- InternVL旧预测审计归档: 93ff60f839644fabf81f202d699bbc0638484f42
- LLaVA经输入验证的10+10条补充归档: 5ee2a0dbe78df336229b55fed14a4a6d0d880bed

新结果全部位于独立目录，历史输出保持只读。
'''
(OUT/'README_REPRODUCE_zh.md').write_text(readme,encoding='utf-8')
print('PUBLICATION_STAGED',str(OUT),'files',sum(p.is_file() for p in OUT.rglob('*')),'bytes',sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
