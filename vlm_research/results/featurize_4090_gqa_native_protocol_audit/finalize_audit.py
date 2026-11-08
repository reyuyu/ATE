"""Independent CPU scoring, paired statistics and bounded conclusions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,csv,hashlib,subprocess,math
import numpy as np
from protocol_helpers import O,D,M
R=Path('/home/featurize/work/ATE')
def read(n):return json.loads((O/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def writecsv(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cpu=read('cpu_audit_summary.json');g=read('gpu_progress.json');p=read('gpu_sample_plan.json');prepared=read('official_prepared_input_audit.json')
assert cpu['status']=='PASS' and g['status']=='COMPLETE' and prepared['status']=='VERIFIED'
assert g['sample_plan_sha256']==sha(O/'gpu_sample_plan.json') and g['cpu_audit_sha256']==sha(O/'cpu_audit_summary.json')
assert g['model_loads']==1 and g['completed_control_generations']==600 and all(x['tokens_and_text_exact'] for x in g['native_smoke'])
rows=[json.loads(l) for l in (O/'gpu_predictions.jsonl').read_text().splitlines()];samples={s['question_id']:s for s in p['samples']}
assert len(rows)==600 and len({(r['question_id'],r['condition']) for r in rows})==600
lookup={(r['question_id'],r['condition']):r for r in rows};baseline=sum(x['score'] for x in p['historical_native'].values());N=200
for r in rows:
 q=r['question_id'];assert q in samples and r['gold']==samples[q]['answer'] and r['sample_plan_sha256']==g['sample_plan_sha256']
 text=r['generation']['text']
 while text.endswith('.'):text=text[:-1]
 text=text.lower();assert text==r['parsed'] and int(text==r['gold'])==r['score']
 assert r['delta_correct']==r['score']-r['historical_native_score'] and r['answer_changed']==(text!=r['historical_native_parsed'])
 assert r['generation']['all_effective_logits_finite'] and r['vision_forward_calls']==1
for q in samples:
 a,b,c=[lookup[q,k] for k in 'ABC'];ref=p['historical_native'][q]
 assert a['input_hashes']['pixel_values']==ref['input_binding']['input_hashes']['pixel_values']
 assert b['input_hashes']['input_ids']==ref['input_binding']['input_hashes']['input_ids']
 assert a['input_hashes']['input_ids']==c['input_hashes']['input_ids'] and b['input_hashes']['pixel_values']==c['input_hashes']['pixel_values']
ids=p['sample_ids'];images=sorted({samples[q]['image_id'] for q in ids});summary=[];rng=np.random.default_rng(20261108)
bootstrap_indices=rng.integers(0,len(images),size=(4000,len(images)))
for condition in 'ABC':
 rr=[lookup[q,condition] for q in ids];sums=defaultdict(int);counts=Counter()
 for r in rr:sums[r['image_id']]+=r['delta_correct'];counts[r['image_id']]+=1
 bs=np.array([sums[x] for x in images]);bn=np.array([counts[x] for x in images]);distribution=100*bs[bootstrap_indices].sum(axis=1)/bn[bootstrap_indices].sum(axis=1)
 ci=np.percentile(distribution,[2.5,97.5]).tolist();correct=sum(r['score'] for r in rr)
 summary.append({'condition':condition,'questions':N,'correct':correct,'accuracy_percent':100*correct/N,'historical_correct':baseline,'historical_accuracy_percent':100*baseline/N,
  'paired_delta_pp':100*(correct-baseline)/N,'answers_changed':sum(r['answer_changed'] for r in rr),'improved':sum(r['delta_correct']==1 for r in rr),'worsened':sum(r['delta_correct']==-1 for r in rr),
  'changed_but_score_unchanged':sum(r['answer_changed'] and r['delta_correct']==0 for r in rr),'truncated':sum(r['generation']['truncated'] for r in rr),'empty':sum(r['generation']['empty'] for r in rr),
  'image_cluster_bootstrap95_low_pp':ci[0],'image_cluster_bootstrap95_high_pp':ci[1]})
writecsv('gpu_summary.csv',summary)
paired=[]
for q in ids:
 s=samples[q];ref=p['historical_native'][q];r={'question_id':q,'image_id':s['image_id'],'structural':s['task'],'semantic':s['types']['semantic'],'question':s['question'],'gold':s['answer'],
  'historical_text':ref['text'],'historical_parsed':ref['parsed'],'historical_score':ref['score']}
 for condition in 'ABC':
  x=lookup[q,condition];r.update({condition+'_text':x['generation']['text'],condition+'_parsed':x['parsed'],condition+'_score':x['score'],condition+'_delta':x['delta_correct'],condition+'_answer_changed':x['answer_changed']})
 paired.append(r)
writecsv('paired_predictions.csv',paired)
changed=[]
for r in rows:
 if r['answer_changed']:changed.append({k:r[k] for k in ['question_id','condition','image_id','structural','question','gold','historical_native_text','historical_native_parsed','historical_native_score','parsed','score','delta_correct']})
writecsv('changed_answer_cases.csv',changed)
bytype=[]
for condition in 'ABC':
 for kind in sorted(p['counts']):
  rr=[r for r in rows if r['condition']==condition and r['structural']==kind];b=sum(r['historical_native_score'] for r in rr);c=sum(r['score'] for r in rr)
  bytype.append({'condition':condition,'structural':kind,'questions':len(rr),'historical_correct':b,'correct':c,'paired_delta_pp':100*(c-b)/len(rr),'answers_changed':sum(r['answer_changed'] for r in rr),'improved':sum(r['delta_correct']==1 for r in rr),'worsened':sum(r['delta_correct']==-1 for r in rr)})
writecsv('gpu_by_type.csv',bytype)
provenance=read('historical_image_provenance.json');inv={r['image_id']:r for r in provenance['images']};assert len(inv)==398
image_rows=list(csv.DictReader((O/'image_protocol_audit.csv').open()))
assert {r['image_id'] for r in image_rows}==set(inv)
assert all(r['current_image_sha256_from_accepted_manifest']==inv[r['image_id']]['sha256'] for r in image_rows)
assert all((D/'images/gqa'/f'{i}.jpg').stat().st_size==v['bytes'] for i,v in inv.items())
image_summary={'status':'VERIFIED_WITH_PRIOR_PROVENANCE_LIMIT','images':398,'current_FP16_crop_pixel_hash_matches':398,'official_ZIP_member_size_CRC_verified_in_historical_audit':398,
 'direct_Stanford_byte_comparisons_in_historical_audit':sum(v['direct_stanford_bytes_also_verified'] for v in inv.values()),
 'current_byte_sizes_and_recorded_SHA_mapping_match':398,'missing_images':0,'crop_pad_different_images':cpu['images_with_crop_pad_pixel_difference'],
 'image_sizes_checked':True,'no_complete_image_download_or_repeat_all_image_SHA':True,
 'limit':'Not all398 current files independently re-downloaded from Stanford; trusted prior officialZIP CRC/size checks plus80 direct byte comparisons, verified migration archive and identical historical model pixels.'}
save('image_provenance_summary.json',image_summary)
model_source=read('official_sources/source_manifest.json');orig=read('official_sources/original_config.json')
protocol_rows=[
 {'item':'公开参考成绩','official':'LLaVA-1.5 7B full_ft-1e:62.0%;13B:63.3%','ATE':'Native 7626/12578=60.6296709%','status':'VERIFIED','implication':'官方62.0是四舍五入数；不假定精确正确题数或用13B脚本默认配置替代7B。'},
 {'item':'数据集与计分分母','official':'GQA testdev_balanced,12578,每题等权exact-match','ATE':'同12578题；所有qid/question/imageId/Gold/split匹配','status':'VERIFIED','implication':'官方eval.zip实际prepared文件qid、题干及短答后缀、imageId逐题一致；rawGold文件SHA与历史官方源一致。'},
 {'item':'Prompt','official':'vicuna_v1 System + USER:<image> + question/short suffix + ASSISTANT:','ATE':'USER:<image> + 完全相同question/short suffix + ASSISTANT:，没有System','status':'VERIFIED','implication':'System带来30个额外文本token；图像标记由-200映射到HF32000并展开576个视觉位置，含义一致。'},
 {'item':'Tokenizer','official':'原生SentencePiece，BOS1/EOS2','ATE':'同SentencePiece文件SHA；HF fast；所有12578官方Prompt token IDs与slow官方算法一致','status':'VERIFIED','implication':'没有发现分词器为额外差异来源。'},
 {'item':'预处理','official':'RGB→按CLIP mean颜色补方形→CLIP336 bicubic/normalize','ATE':'RGB→短边336 resize→center crop336；不先pad','status':'VERIFIED','implication':'389/398图片pixel values不同；先外部pad后当前CLIP核心输出与官方CLIP处理器逐像素相同。'},
 {'item':'图片版本','official':'Stanford GQA images.zip对应398张','ATE':'398张完整；旧官方ZIP CRC/size全匹配、80张Stanford直接byte匹配；当前像素哈希全部匹配历史Native','status':'VERIFIED_WITH_LIMIT','implication':'未发现缺图或错图；不是398张均重新向官方做独立SHA下载验证。'},
 {'item':'生成与停止','official':'greedy/beam1/FP16/EOS2；2023release cap128','ATE':'greedy/beam1/FP16/EOS2/Eager，cap32','status':'VERIFIED','implication':'历史最长6token(含EOS)、0截断；cap32不是本次历史差距原因。对照仍固定32。'},
 {'item':'Scorer','official':"输出strip后，text.rstrip('.').lower()；与官方answer完全相等；balanced等权",'ATE':'同转换、同分母及exact-match；官方完整脚本离线复算一致','status':'VERIFIED','implication':'无大小写/句号/分母或评分实现问题；不做同义词宽松匹配。'},
 {'item':'模型/权重/Backend','official':'原生LlavaLlamaForCausalLM；Vicuna7B32层4096；CLIP-L/14-336；原始词表32000','ATE':'HF LlavaForConditionalGeneration；同架构/视觉层-2/投影结构；词表扩至32064；Transformers4.49/Eager','status':'UNRESOLVED','implication':'同源转换模型，但未独立逐tensor对照官方原生权重或做原生backend数值对齐；历史无新增词表token输出，不把全部差异归因Prompt。'}]
save('protocol_differences.json',{'official_release_commit':model_source['llava_release_commit'],'ATE_code_commit':g['code_commit'],'HF_revision':g['model_revision'],'original_model_metadata_revision':model_source['original_model_revision'],'rows':protocol_rows})
writecsv('protocol_comparison.csv',protocol_rows)
freeze={k:v for k,v in p.items() if k not in {'samples','historical_native','sources'}}
freeze.update(sample_plan_sha256=sha(O/'gpu_sample_plan.json'),GPU_generations=600,Native_smoke_generations=3,source_manifest_sha256=sha(O/'official_sources/source_manifest.json'))
save('sample_freeze_summary.json',freeze)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
conclusion={'audit_status':'COMPLETE','causal_attribution_status':'LIKELY_PARTIAL_EXPLANATION_NOT_FULLY_ESTABLISHED','verified':['Official7B62.0 reference distinct from13B63.3','Same12578 prepared inputs and officialGold/score denominator','HistoricalNative7626correct;0missing/empty/truncated/scoring mismatch','SystemPrompt differs and389/398crop/pad pixels differ','3Native smoke exact;600ABC completed once','CombinedC127/200 vs121/200;10improvements4regressions'],
 'likely':'Prompt与square-pad联合协议偏差是最有实证支持的部分原因；小样本存在联合正向变化，单独Prompt/预处理没有净提升。',
 'unresolved':['完整1.370329pp中各因素占比','原生与HF权重逐tensor及Backend数值完全一致性','官方7B12578逐题预测及精确运行配置；官方eval.zip仅有13B预测','200题与有限Native smoke不足以排除全部跨GPU浮点argmax变化或小样本波动'],
 'minimum_next_validation':'优先取得官方7B同split逐题预测与精确运行配置，先CPU比对具体错误集合；若拿不到，下一轮需原生LLaVA backend与HF在同一固定小样本上对齐权重与逐token输出，才能隔离转换/Backend残余差异。要量化完整1.37pp仍需更大、预先授权的配对证据，本轮不启动。',
 'GPU_after_exit':gpu_after,'model_loads':1,'formal_benchmark_rerun':False,'frozen_history_modified':False,'C3_gate_modified':False,
 'GPU_summary':summary,'image_clusters_in_sample':len(images),'bootstrap':'4000 image-cluster resamples, seed20261108; exploratory95%percentile interval; not a full-benchmark causal estimate.',
 'finished_at':datetime.now(timezone.utc).isoformat()}
save('audit_conclusion.json',conclusion)
lines=['# GQA Native 60.63% 与官方 62.0%：协议差异审计','',
 '**审计状态：COMPLETE；差距归因：LIKELY / 尚未完全确定。** 最有实证支持的是 Vicuna System Prompt 与 square-pad 预处理的联合协议偏差。数据映射、计分实现、漏题及32-token截断未发现问题；不能把全部1.37 pp归因于Prompt。','',
 f"历史12,578条Native预测经独立逐题逻辑、官方转换脚本及官方GQA evaluator复算：**7,626 / 12,578 = {cpu['accuracy_percent']:.7f}%**。相对官方四舍五入62.0%的差为{62-cpu['accuracy_percent']:.7f} pp。历史预测、Gold和冻结Scorer均未修改。",'',
 '[官方Model Zoo](https://github.com/haotian-liu/LLaVA/blob/'+model_source['llava_current_commit']+'/docs/MODEL_ZOO.md)明确列出1.5-7B为62.0%，13B为63.3%。[2023原始GQA脚本](https://github.com/haotian-liu/LLaVA/blob/'+model_source['llava_release_commit']+'/scripts/v1_5/eval/gqa.sh)默认13B，因此只取其数据与评测流程，未误用13B权重或成绩。官方发布包只含13B预测，未用它冒充7B对照。','',
 '下表的“官方公开流程”来自原始发布脚本、7B模型配置与对话/预处理源码。62.0%的7B成绩本身已核实；产生该成绩的完整运行命令、依赖锁定与逐题预测未公开在所查发布包中，因此不能把公开通用脚本当作该次7B运行的逐项日志。HF_conversion_reference.py也仅作转换机制参考，不代表已证实本模型转换时使用了该版本脚本。','',
 '| 项目 | 官方公开流程 / 来源 | 历史ATE Native | 结论 |','|---|---|---|---|']
for row in protocol_rows:
 lines.append('| '+row['item']+' | '+row['official'].replace('|','/')+' | '+row['ATE'].replace('|','/')+' | '+row['status']+' |')
example=cpu['prompt_examples'][0]
lines+=['', '**Prompt的确切差别（VERIFIED）**：历史HF Prompt从 `USER: <image>` 开始；官方vicuna_v1在其前面加入标准System段，后面的图像换行、原问题、单词/短语回答指令和 ` ASSISTANT:` 均相同。示例qid201030233从616个展开位置变为646，增加30个文本token。完整字符串与Token IDs见cpu_audit_summary.json。原生-200图像标记与HF32000表示实现不同，二者均注入576个图像patch，不是缺图或多图。官方slow与HF fast对12578个官方Prompt的Token IDs完全一致。',
 '[官方对话模板](https://github.com/haotian-liu/LLaVA/blob/'+model_source['llava_release_commit']+'/llava/conversation.py)与[预处理实现](https://github.com/haotian-liu/LLaVA/blob/'+model_source['llava_release_commit']+'/llava/mm_utils.py)已固定到原始release commit；实际官方prepared GQA题干从eval.zip只按HTTP Range读取所需文本成员，不下载整包或图片。12578个qid、题干/回答后缀及图片ID逐项完全一致。','',
 '**数据、图片与Scorer（VERIFIED，保留来源限制）**：raw官方testdev_balanced题库SHA256为 `'+cpu['raw_official_annotation_sha256']+'`；当前/历史Question、Gold、Image ID、split均相同。398张图片无缺失，尺寸已检查；当前处理后FP16像素哈希全部匹配历史Native。旧官方ZIP的398个size/CRC检查、80张Stanford直接byte匹配记录与本轮迁移整体ZIP校验共同支持图片版本可信。本轮没有重新对398张做全量SHA或重新下载，未把80张的直接验证写成398张。',
 '实际执行[官方GQA转换](https://github.com/haotian-liu/LLaVA/blob/'+model_source['llava_release_commit']+'/scripts/convert_gqa_for_eval.py)和[LLaVA指定GQA evaluator](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187)：源码SHA与历史冻结evaluator相同，逐题及分母均一致。没有缺预测、重复qid、空输出、32-token截断或保存得分错误；历史最长输出仅6token（含EOS），增加cap不会改变这些已EOS结束的答案。','',
 '**冻结200题的GPU控制实验**：按structural类型比例分配（compare9/logical29/query108/verify36/choose18），仅按预先固定seed与qid哈希排序选题，未查看正确性来挑题。3个Native Smoke输入tensor、token和文本均匹配历史；同一模型一次加载，随后完成A/B/C各200题，共600次控制生成。没有额外完整基线、C3或参数搜索。','',
 '| 条件 | 正确数/200 | Accuracy | 相对同题历史Native | 改变答案 | 改善 / 恶化 |','|---|---:|---:|---:|---:|---:|',
 f'| 历史Native（复用） | {baseline}/200 | {100*baseline/N:.1f}% | — | — | — |']
for row in summary:lines.append(f"| {row['condition']} | {row['correct']}/200 | {row['accuracy_percent']:.1f}% | {row['paired_delta_pp']:+.1f} pp | {row['answers_changed']} | {row['improved']} / {row['worsened']} |")
lines+=['', 'A=仅换官方Vicuna-v1 Prompt；B=仅改官方square-pad；C=两者同时对齐。其余保持同一HF权重、FP16/Eager、greedy、EOS2和cap32；600条输出均有限、非空且未截断。此表是固定小样本诊断，不能作为正式GQA成绩，也不能直接外推完整1.37 pp。','',
 f'图像cluster数={len(images)}；4000次cluster bootstrap的配对差值探索区间：'+ '；'.join(f"{x['condition']} [{x['image_cluster_bootstrap95_low_pp']:+.2f}, {x['image_cluster_bootstrap95_high_pp']:+.2f}] pp" for x in summary)+'。小样本区间用于显示不确定性，不作为因果分解或恢复62.0的证明。',
 '389/398张原图的crop/pad像素不同，补方形后的当前CLIP核心输出与官方CLIP预处理逐像素一致。A单独-0.5 pp、B单独0 pp，C联合+3.0 pp，表明不能把效应简单相加或认定单独System Prompt必定提高成绩。','',
 '**逐题复核**：全部200题的三条件原输出/解析/分数见paired_predictions.csv；57条改变答案的条件记录见changed_answer_cases.csv。对四例原图做了人工复核（不改Gold）：','',
 '| qid | Gold | 历史 → A / B / C | 复核结论 |','|---|---|---|---|',
 '| 20886973 | blue | Gray → Gray / Blue / Blue | 右下角蓝鼠标处于中心裁剪边缘；B/C改善支持边缘图像证据变化 |',
 '| 201896132 | blue | White → White / Blue / Blue | 底部蛋糕有浅色主体和蓝装饰，目标/颜色有歧义；保留官方Gold，不宣传为确定视觉纠错 |',
 '| 20482509 | racket | Racket → Tennis racket / Racket / Tennis racket | 网球拍可见；更具体的tennis racket仍按官方exact-match计0，属于表述损失，不是Scorer错误 |',
 '| 201676091 | bottom | Bottom → Bottom / Top / Top | 图中多个手柄，指代有歧义；B/C按官方Gold恶化，pad并非保证提高每题 |','',
 '**模型与残余差异（UNRESOLVED）**：HF模型与原生模型的7B架构、SentencePiece及CLIP336/视觉层-2设定对应。原生词表32000、HF扩为32064；历史没有输出新增token。没有下载另一套大型权重，也没有独立逐tensor确认原生权重或原生Backend的数值等价；官方2023代码依赖与本次Transformers4.49/Eager不同。3个Native Smoke一致不能排除其余题的跨GPU浮点影响。因此本轮确认“协议不完全可比”，联合小样本正向证据使其成为最可信的部分解释，但完整差距的占比仍未确定。','',
 '**最少下一步**：'+conclusion['minimum_next_validation'],
 '', f"本轮环境：{g['gpu_before_load']}；代码 `{g['code_commit']}`；HF revision `{g['model_revision']}`；Torch2.6.0+cu124 / Transformers4.49.0。模型加载1次，allocated峰值{g['peak_allocated_mib']/1024:.3f}GiB。全部GPU工作已结束：`{gpu_after}`。原数据/模型保留本地盘，C3 Physical数值门禁未改动。本轮没有完整Benchmark重跑或任何V²Drop/其他研究任务。",
 '', '结果保存在独立目录featurize_4090_gqa_native_protocol_audit；GitHub只发布报告、协议清单、必要源码、固定样本/统计及小型逐题CSV。12,578条完整CPU审计表、600条完整原始GPU预测和完整样本绑定留在本地与Windows独立备份；同步提交见github_sync_receipt.json。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
names=['report_zh.md','audit_conclusion.json','cpu_audit_summary.json','gpu_progress.json','gpu_summary.csv','gpu_by_type.csv','paired_predictions.csv','changed_answer_cases.csv','protocol_differences.json','protocol_comparison.csv',
 'sample_freeze_summary.json','fixed_sample_ids.csv','native_by_type.csv','image_protocol_audit.csv','image_provenance_summary.json','official_prepared_input_audit.json','official_eval_zip_members.json','manual_case_review.json','official_eval.log',
 'protocol_helpers.py','cpu_audit.py','gpu_controls.py','supplement_cpu.py','finalize_audit.py','fetch_sources.py','inspect_official_zip.py','read_official_zip.py','gpu_controls.log','historical_image_provenance.json','native_per_question_audit.csv','gpu_predictions.jsonl','gpu_sample_plan.json']
names+=['official_sources/'+p.name for p in (O/'official_sources').iterdir() if p.is_file()]
manifest={n:{'bytes':(O/n).stat().st_size,'sha256':sha(O/n)} for n in names}
save('artifact_manifest.json',{'status':'COMPLETE','files':manifest,'large_files_local_only':['native_per_question_audit.csv','gpu_predictions.jsonl','gpu_sample_plan.json'],'no_model_or_image_bytes':True})
print('FINAL_AUDIT_COMPLETE',json.dumps({'correct_historical':cpu['correct'],'GPU_summary':summary,'sample_images':len(images),'files':len(manifest),'GPU_after_exit':gpu_after},ensure_ascii=False),flush=True)
