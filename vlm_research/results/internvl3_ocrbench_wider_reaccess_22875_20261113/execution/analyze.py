"""CPU-only independent scoring, paired image-cluster CI and compact publication."""
import sys,os,json,csv,gzip,shutil,hashlib,collections,subprocess
from pathlib import Path
OLD=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-wider-reaccess-22875-20261113')
sys.path.insert(0,str(OLD));from run_common import atomic,sha,score,RAW
import numpy as np
from transformers import AutoTokenizer
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((O/'selected_rows.json').read_text());byid={r['question_id']:r for r in rows};ids=P['ids']
assert json.loads((O/'progress.json').read_text())['status']=='COMPLETE_PENDING_CPU_AUDIT'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Wait for model process exit'
pub=O/'publication/vlm_research/results/internvl3_ocrbench_wider_reaccess_22875_20261113';pub.mkdir(parents=True,exist_ok=True)
data=json.loads((O/'baselines.json').read_text());tok=AutoTokenizer.from_pretrained('/home/featurize/ate-local-models/InternVL3-8B',trust_remote_code=True,local_files_only=True,use_fast=False)
decode_checks=0
for method in P['configs']:
 files=sorted((O/'predictions'/method).glob('*.json'));assert len(files)==100
 data[method]=[]
 for f in files:
  x=json.loads(f.read_text());q=x['question_id'];r=byid[q];g=x['generation'];b=x['input_binding_details']
  assert x['protocol_sha256']==pb and g['cutoff']==28 and g['reaccess_layers']==P['configs'][method] and g['independent_cache']
  assert [a['layer'] for a in g['kv_layout'] if a['visual_kv']]==P['configs'][method]
  text=tok.decode(g['token_ids'],skip_special_tokens=True).split('<|im_end|>')[0].strip();assert text==g['text'],('TOKEN_DECODE_MISMATCH',method,q)
  assert len(g['token_ids'])==g['generated_tokens']<=128 and g['eos']==(g['token_ids'][-1]==151645)
  assert g['truncated']==(len(g['token_ids'])==128 and not g['eos']);assert g['empty']==(not text)
  assert r['image_sha256']==x['image_sha256']==b['image_sha256'] and r['prompt_sha256']==x['prompt_sha256']==b['prompt_sha256']
  sc=score(q,text);assert sc['official_score']==x['official_score']==x['independent_score']
  data[method].append(dict(model='InternVL3-8B',method=method,question_id=q,category=r['category'],dataset_name=r['dataset_name'],image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding_sha256=x['input_binding'],gold=r['gold'] if isinstance(r['gold'],list) else [r['gold']],text=text,token_ids=g['token_ids'],official_score=sc['official_score'],independent_score=sc['independent_score'],eos=g['eos'],truncated=g['truncated'],empty=g['empty'],origin='NEW_FIXED_OCRBENCH100',source_record=str(f)))
  decode_checks+=1
for method,rs in data.items():
 assert len(rs)==100 and {r['question_id'] for r in rs}==set(ids)
 rs.sort(key=lambda x:int(x['question_id']))
 for x in rs:
  q=x['question_id'];r=byid[q];assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  assert x['gold']==(r['gold'] if isinstance(r['gold'],list) else [r['gold']])
  s=score(q,x['text']);assert s['official_score']==x['official_score']==x['independent_score']
 (pub/'predictions').mkdir(exist_ok=True)
 with gzip.GzipFile(str(pub/'predictions'/f'{method}.jsonl.gz'),'wb',mtime=0) as f:
  for x in rs:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
table=[];category=[]
for method,rs in data.items():
 table.append(dict(method=method,n=100,correct=sum(x['official_score'] for x in rs),accuracy_percent=sum(x['official_score'] for x in rs),empty=sum(x['empty'] for x in rs),truncated=sum(x['truncated'] for x in rs),new_generations=100 if method in P['configs'] else 0))
 for cat in sorted(P['allocation']):
  cr=[x for x in rs if x['category']==cat];category.append(dict(method=method,category=cat,n=len(cr),correct=sum(x['official_score'] for x in cr),accuracy_percent=100*sum(x['official_score'] for x in cr)/len(cr)))
def csvout(name,rs):
 with (pub/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
csvout('scores.csv',table);csvout('category_scores.csv',category)
imap={method:{x['question_id']:x for x in rs} for method,rs in data.items()}
clusters=sorted({r['image_sha256'] for r in rows});cluster_ids={s:[r['question_id'] for r in rows if r['image_sha256']==s] for s in clusters}
rng=np.random.default_rng(20261113);boots=rng.integers(0,len(clusters),size=(10000,len(clusters)));sizes=np.array([len(cluster_ids[s]) for s in clusters])
paired=[];cases=[]
for method in P['configs']:
 for ref in ['Late9','Native']:
  diffs={q:imap[method][q]['official_score']-imap[ref][q]['official_score'] for q in ids}
  sums=np.array([sum(diffs[q] for q in cluster_ids[s]) for s in clusters]);vals=100*sums[boots].sum(1)/sizes[boots].sum(1)
  ci=np.quantile(vals,[.025,.975]);p=dict(method=method,reference=ref,n=100,delta_pp=sum(diffs.values()),wrong_to_right=sum(v==1 for v in diffs.values()),right_to_wrong=sum(v==-1 for v in diffs.values()),changed_text=sum(imap[method][q]['text']!=imap[ref][q]['text'] for q in ids),image_clusters=len(clusters),bootstrap_replicates=10000,seed=20261113,ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),multiplicity='unadjusted exploratory')
  paired.append(p)
  for q in ids:
   if diffs[q]:
    r=byid[q];cases.append(dict(method=method,reference=ref,question_id=q,category=r['category'],question=r['question'],gold=json.dumps(r['gold'],ensure_ascii=False),reference_output=imap[ref][q]['text'],new_output=imap[method][q]['text'],score_change=diffs[q]))
csvout('paired_statistics.csv',paired);csvout('changed_score_cases.csv',cases)
inputrows=[{k:r[k] for k in ['question_id','question','image_id','image_sha256','prompt','prompt_sha256','input_ids','roles','category','dataset_name','max_new_tokens']} for r in rows]
with gzip.GzipFile(str(pub/'inputs_light.jsonl.gz'),'wb',mtime=0) as f:
 for x in inputrows:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
for n in ['protocol.json','cpu_audit.json','smoke.json','environment.json','expected_bindings.json','gold.json','progress.json','history_all24_source_manifest.json']:
 shutil.copy2(O/n,pub/n)
for f in (O/'numerical').glob('*.json'):
 (pub/'numerical').mkdir(exist_ok=True);shutil.copy2(f,pub/'numerical'/f.name)
for n in ['prepare.py','run.py','analyze.py']:
 (pub/'execution').mkdir(exist_ok=True);shutil.copy2(O/n,pub/'execution'/n)
history=json.loads((O/'github_history_inventory.json').read_text());atomic(pub/'github_history_inventory.json',history)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
summary=dict(status='COMPLETE_EXPLORATORY_QUALITY',scores=table,paired=paired,new_quality_generations=300,reused_predictions=300,dense_reference_requests=3,smoke_quality_predictions_included=12,independent_scoring_checks=600,cpu_token_decode_checks=decode_checks,unique_images=len(clusters),historical_full_OCRBench1000=dict(Native=751,Physical_D=32,Late9=563),GPU_after=gpu,official_protocol_alignment=False,scope='ATE frozen 1-tile protocol; not official dynamic24, not a full OCRBench score or efficiency study')
atomic(pub/'summary.json',summary)
smoke=json.loads((O/'smoke.json').read_text());num={m:json.loads((O/'numerical'/f'{m}.json').read_text()) for m in P['configs']}
report=['# InternVL3-8B 扩大视觉回看范围：OCRBench100 固定小样本',
 '', '状态：**COMPLETE_EXPLORATORY_QUALITY**。只运行预先固定的三组各100题；没有重跑Native、Physical-D、Late9，也没有扩大到1000题或测量效率。',
 '', '## 同一批100题结果', '', '| 方法 | 回看层（0-based） | 得分 /100 | 相对Late9（pp） | 空输出 | 截断 |', '|---|---|---:|---:|---:|---:|']
windows={'Native':'原生完整','Physical_D':'无回看','Late9':'L19–L27','Late16_L12_27':'L12–L27','Late20_L8_27':'L8–L27','All24_ExceptL2_5':'L0–L1及L6–L27'}
base=next(x['correct'] for x in table if x['method']=='Late9')
for x in table:report.append(f"| {x['method']} | {windows[x['method']]} | {x['correct']} | {x['correct']-base:+d} | {x['empty']} | {x['truncated']} |")
report+=['', '这里的分数是100题探索性结果，不能乘十当作OCRBench1000正式分数。既有全量结果为Native751、Physical-D32、Late9563；本轮未重复生成它们。',
 '', '## 配对比较', '', '| 新方法 | 对照 | 差值pp | 错→对 | 对→错 | 按图片SHA簇Bootstrap 95% CI（pp） |', '|---|---|---:|---:|---:|---|']
for x in paired:report.append(f"| {x['method']} | {x['reference']} | {x['delta_pp']:+d} | {x['wrong_to_right']} | {x['right_to_wrong']} | [{x['ci95_low_pp']:.2f}, {x['ci95_high_pp']:.2f}] |")
report+=['',f"固定seed=20261113，10000次图片簇配对Bootstrap，共{len(clusters)}个图像SHA簇。区间未经多重比较校正；样本小，同批三组用于探索，不用于证明非劣效或挑出配置后宣称独立验证。逐题改善与恶化均保存在changed_score_cases.csv。",'', '## GitHub历史依据与设计',
 '', '- [All24历史TextVQA实验](https://github.com/reyuyu/ATE/commit/5219e469d8b65f90f84669b29126ad573b6bfae9)：相同TextVQA128上All24=54.765625%，Late9=50.703125%。这是Image-only TextVQA，不是OCRBench。',
 '- [终端窗口实验](https://github.com/reyuyu/ATE/commit/1407275cbdffca5d4a23cbf8b8d233002e2840dd)：保留到L27的晚层信息有帮助；单纯扩展较早中层并不保证恢复。',
 '- 本轮预先冻结Late16、Late20、All24三组；共同终点L27、cutoff28、R读图边界20与答案Attention跳算L2–L5保持不变，只改变P−1及Decode允许读取原始视觉KV的层集合。',
 '- GPU前检查了GitHub历史分支、服务器实验目录与Windows已有备份目录。找到的宽窗口结果是TextVQA，未发现这三组OCRBench预测；三组基线各1000条则已存在并直接抽取。',
 '', '## 输入与评分',
 '', '按官方10个类别各取10%，由SHA256(seed|题号|图片SHA)排序确定：六类文字识别各5题，Scene/Doc/Key Information各20题，数学10题。保留原始顺序；选题不依赖Gold或预测。完整100题ID、Prompt、Token IDs、图片SHA和选取规则见protocol.json与inputs_light.jsonl.gz。',
 '', '100张输入绑定逐项核对冻结图片、Prompt、Gold、input_ids、BF16 pixel_values；Gold独立保存。初次CPU准备遇到单答案scalar与历史规范化singleton-list的序列化表示差异后停止，改为比较相同答案列表，未改变任何答案或评分规则，未因此重复GPU生成。',
 '', '使用原OCRBench v1官方评分循环并独立实现答案包含匹配：普通数据转小写、去首尾空白并换行转空格；HME100k区分大小写且移除空格/换行。600条预测全部重新核算且两种评分实现一致；新增300条Token IDs在CPU解码与保存原文一致。',
 '', '## Physical与数值检查',
 '', '每组4题结构Smoke（0000、0560、0743、0926），共12次生成计入各组100题。检查了真实视觉Prefill、允许层读图概率>0、禁止层和L20之后R读图概率=0、L2–L5答案Attention/QKV物理跳算、Norm/MLP正常、RoPE原始位置、独立KV、持久视觉KV仅在K集合保留，以及L27后/Final Norm前物理移除256行Hidden。全部通过；OOM/NaN/异常为0。',
 '', '每组首题0000仅生成2Token（包含EOS），另做同前缀Dense参考，共3次：这2步全部logits及共享Hidden逐元素一致，最大差值/相对RMS均0。未降低旧阈值；**该短参考仅覆盖这一题，不能消除历史严格门禁失败，也不能证明全局数值等价**。本轮继续明确标注历史EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER。',
 '', '直接导入历史冻结终端Physical执行器，核验全部59个来源文件SHA256，未修改核心。该实现有物理持久KV裁剪，但使用瞬时零scratch保持原Attention归约宽度；cutoff28意味着所有28层Prefill仍处理视觉Hidden。因此本轮不能据此声称延迟、FLOPs或KV收益，未运行任何效率测试。',
 '', '## 环境与限制',
 '', f"真实GPU：{gpu}。模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645。模型仅加载一次。",
 '', '继承ATE_SVH_1TILE_SCORER_ALIGNED协议：RGB单448×448图块/256视觉Token，原Chat Template及短答后缀不变；未混入官方动态max_num24图块协议。该预处理仍可能限制OCR质量，不能把恢复不足全部归因于回看层。',
 '', '执行结束后GPU无遗留计算进程；不关闭服务器。轻量报告、配置、逐题结果、统计、来源哈希与执行器外围代码在独立Git分支；原始记录另行归档并做Windows/持续盘SHA256校验。备份与Push是否成功以delivery_receipt.json为准。']
(pub/'report_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
manifest=[dict(path=str(p.relative_to(pub)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(pub.rglob('*')) if p.is_file()]
atomic(pub/'integrity_manifest.json',manifest)
atomic(O/'analysis_done.json',dict(status='PASS',publication=str(pub),new_quality=300,reused=300,GPU_after=gpu))
print(json.dumps(summary,ensure_ascii=False,indent=2))
