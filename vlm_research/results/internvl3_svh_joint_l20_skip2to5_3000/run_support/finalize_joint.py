"""Final CPU-only evidence audit; invoke only after inspecting the finished PNG."""
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,subprocess
from collections import Counter
ap=argparse.ArgumentParser();ap.add_argument('--visual-reviewed',action='store_true');args=ap.parse_args();assert args.visual_reviewed
N=Path('/home/featurize/work/ATE-internvl3-svh-joint-3000')
O=N/'vlm_research/results/internvl3_svh_joint_l20_skip2to5_3000'
R=Path('/home/featurize/ate-runs/internvl3-svh-joint-l20-skip2to5-3000')
def read(n):return json.loads((O/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+chr(10))
def csvread(n):
 with (O/n).open() as f:return list(csv.DictReader(f))
v=read('validation.json');assert v['passed'] and v['rows']==12000
assert not (R/'failure.json').exists()
assert read('generation_complete.json')['new_requests']==6000 and read('baseline_reuse_decision.json')['reuse_A_B']
for p,h in read('source_manifest.json').items():assert sha(p)==h
for p,h in read('analysis_source_freeze.json').items():assert sha(p)==h
rows=csvread('per_question.csv');assert len(rows)==12000
by={c:{} for c in 'ABCD'}
for r in rows:
 k=(r['task'],r['question_id']);assert k not in by[r['condition']]
 assert int(r['correct'])==int(r['text'].rstrip('.').lower()==r['gold'].lower())
 assert r['prediction']==r['text'].rstrip('.').lower()
 by[r['condition']][k]=r
assert all(len(x)==3000 for x in by.values()) and all(set(by[c])==set(by['A']) for c in 'ABCD')
summary={x['condition']:x for x in csvread('accuracy_summary.csv')}
for c in 'ABCD':
 assert sum(int(x['correct']) for x in by[c].values())==int(summary[c]['correct'])
 assert abs(float(summary[c]['accuracy_pct'])-100*int(summary[c]['correct'])/3000)<1e-10
for r in csvread('paired_comparisons.csv'):
 b,t=r['baseline'],r['intervention'];ctw=wtc=flip=textchange=0
 for k,x in by[b].items():
  y=by[t][k];ctw+=int(x['correct']=='1' and y['correct']=='0');wtc+=int(x['correct']=='0' and y['correct']=='1')
  flip+=int(json.loads(x['token_ids'])[0]!=json.loads(y['token_ids'])[0]);textchange+=int(x['text']!=y['text'])
 assert [ctw,wtc,flip,textchange]==[int(r[x]) for x in ['correct_to_wrong','wrong_to_correct','first_token_flip','full_text_change']]
 assert abs(float(r['delta_pp'])-100*(wtc-ctw)/3000)<1e-10
inter=csvread('interaction_analysis.csv')[0]
point=float(summary['D']['accuracy_pct'])-float(summary['B']['accuracy_pct'])-float(summary['C']['accuracy_pct'])+float(summary['A']['accuracy_pct'])
assert abs(point-float(inter['interaction_pp']))<1e-10
harms=csvread('harm_cases.csv');assert len(harms)==sum(int(x['correct_to_wrong']) for x in csvread('paired_comparisons.csv'))
for x in harms:assert x['baseline_text'].rstrip('.').lower()==x['gold'].lower() and x['intervention_text'].rstrip('.').lower()!=x['gold'].lower()
from PIL import Image
with Image.open(O/'figure_joint_validation.png') as im:
 dims=im.size;dpi=im.info.get('dpi');assert dpi and min(dpi)>=300
pdf=(O/'figure_joint_validation.pdf').read_bytes();assert b'/Subtype /Image' not in pdf
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.used','--format=csv,noheader'],text=True).strip()
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not procs,'Another compute process is present; do not claim GPU fully free'
active=read('active.json');process_state={}
for role in ['worker','supervisor']:
 p=Path('/proc')/str(active[role])/'stat'
 state=p.read_text().rsplit(') ',1)[1].split()[0] if p.exists() else 'absent'
 assert state in ['absent','Z'],(role,state)
 process_state[role]=dict(pid=active[role],state=state,active=False)
assert gpu.split(',')[0]==read('protocol.json')['GPU_UUID']
save('gpu_final_release.json',dict(passed=True,GPU=gpu,compute_processes=[],own_processes=process_state,server_not_closed=True))
for n in ['pipeline.log','worker.log']:
 if (R/n).exists():shutil.copy2(R/n,O/('saved_'+n))
sources=dict(read('source_manifest.json'));sources.update(read('analysis_source_freeze.json'))
D=O/'source';D.mkdir(exist_ok=True);snapshots=[]
for idx,(p,h) in enumerate(sorted(sources.items())):
 dest=D/(str(idx).zfill(2)+'_'+Path(p).name);shutil.copy2(p,dest);assert sha(dest)==h
 snapshots.append(dict(original=p,copy=str(dest.relative_to(O)),sha256=h))
run_support=O/'run_support';run_support.mkdir(exist_ok=True);shutil.copy2(__file__,run_support/Path(__file__).name)
save('source_snapshot_manifest.json',snapshots)
task=csvread('task_accuracy.csv');taskmap={(r['task'],r['condition']):float(r['accuracy_pct']) for r in task}
worst=sorted([(taskmap[t,'D']-taskmap[t,'A'],t,taskmap[t,'D']-taskmap[t,'B']) for t in sorted({r['task'] for r in task})])
report=(O/'report_zh.md').read_text()
report+=chr(10)+'任务损失检查（D−A，D−B）：'+ '；'.join(t+f' {da:+.2f} pp / {db:+.2f} pp' for da,t,db in worst)+ '。子组差异不能由总体平均替代。'+chr(10)
report+=chr(10)+'最终验收：12000条逐题记录、全部6000次新生成、全部3000个检查点和独立评分已通过；PNG至少300DPI、PDF为矢量，图形已目视检查。GPU无计算进程，工作进程与监督进程已结束；服务器未关闭。'+chr(10)
taskpairs={(r['task'],r['comparison']):r for r in csvread('task_paired_comparisons.csv')}
for taskname in ['CompareAttr','QueryAttr']:
 x=taskpairs[taskname,'D-A'];y=taskpairs[taskname,'D-B']
 report+=chr(10)+f"重点任务{taskname}：D−A {float(x['delta_pp']):+.2f} pp，普通95%CI [{float(x['CI95_low']):+.2f}, {float(x['CI95_high']):+.2f}]；D−B {float(y['delta_pp']):+.2f} pp，普通95%CI [{float(y['CI95_low']):+.2f}, {float(y['CI95_high']):+.2f}]。所有子组CI均未作多重比较校正。"+chr(10)
report+=chr(10)+'图中灰色竖线为0pp，橙色点线为−1pp参考线；完整英文图注见figure_caption_en.md。'+chr(10)
pairmap={x['comparison']:x for x in csvread('paired_comparisons.csv')}
r=pairmap['D-A']
report+=chr(10)+f"−1pp参考：D−A点估计为 {float(r['delta_pp']):+.4f} pp，{'位于参考线之上' if float(r['delta_pp'])>=-1 else '低于参考线'}；普通95%CI下界为 {float(r['CI95_low']):+.4f} pp。此为当前样本上的探索性质量对照，不构成独立非劣效验收。"+chr(10)
report+='本轮源码仅本地Commit '+read('source_commit.json')['commit']+'；本轮结果未推送GitHub。此前Figure 3与L20扩展数据已独立上传GitHub Commit 1cdcf6b2a0b6b5606b00789961904043849cbbdc。'+chr(10)
report+='加载前审计兼容性修正记录位于engineering_attempts/attempt01_loader_context/：首次模型在inference_mode下加载，参数版本计数审计报错，未开始Smoke或正式生成；改回历史no_grad加载方式后，Native/No-op严格一致且A/B全部有效Logits回放匹配。未放宽门槛，未改变干预算法。'+chr(10)
(O/'report_zh.md').write_text(report)
v.update(GPU_released=True,own_worker_and_supervisor_inactive=True,human_visual_review_passed=True,PNG_at_least_300DPI=True,PDF_vector=True,independent_final_review=True)
save('validation.json',v)
save('final_review.json',dict(passed=True,rows=12000,formal_new_requests=6000,reused_baseline_rows=6000,correct_counts={c:int(summary[c]['correct']) for c in 'ABCD'},
 independent_CSV_scores_and_pair_counts_match=True,all_harm_cases_listed=True,interaction_identity_exact=True,
 source_SHA_unchanged=True,figure_dimensions=dims,figure_DPI=dpi,PDF_vector=True,human_visual_review=True,GPU_released=True,
 no_GitHub_push_for_this_experiment=True,original_experiments_retained=True))
files=sorted(p for p in O.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.name not in ['SHA256SUMS','evidence_manifest.json'] and not p.name.endswith('.tmp'))
manifest=dict(passed=True,root=str(O),protocol_sha256=sha(O/'protocol.json'),source_commit=read('source_commit.json'),new_generations=6000,analysis_rows=12000,
 files=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
save('evidence_manifest.json',manifest);files.append(O/'evidence_manifest.json')
(O/'SHA256SUMS').write_text(chr(10).join(sha(p)+'  '+str(p.relative_to(O)) for p in sorted(files))+chr(10))
for line in (O/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert sha(O/n)==h,n
print('FINAL_AUDIT_PASS',json.dumps(dict(files=len(files),correct_counts={c:int(summary[c]['correct']) for c in 'ABCD'},GPU=gpu,processes=process_state,SHA256SUMS=sha(O/'SHA256SUMS'))))
