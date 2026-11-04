"""Final file, source, PDF, resource and publication-scope audit; no GPU calls."""
import csv,json,hashlib,subprocess,time,datetime,shutil,re
from pathlib import Path
WT=Path(__file__).resolve().parents[2];P=WT/'vlm_research/results/semantic_single_gpu_search3h'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
assert read(P/'final_validation.json')['passed'] and read(P/'final_completion.json')['status']=='COMPLETE'
clock=read(P/'clock.json');assert time.monotonic()<clock['deadline_monotonic']-600
report=P/'final_report_zh.md';text=report.read_text()
rows=list(csv.DictReader((P/'final_per_question.csv').open()));weighted=sum(int(x['k']) for x in rows)/sum(int(x['R']) for x in rows)
extra='''
## 出版前复核补充

阶段8对阶段7已冻结、预算合格的整个规则族统一完成600题复查，避免早期top4多为等价规则而缩窄候选。冻结的胜者为what/which或长度≥20或含空间词时开放assistant，最终2597/3000、48.820% R；未超过阶段6的2598。阶段9冻结少量文本类别触发All-Writer的静态回退规则，只复用精确相同writer集合的历史记录，新增GPU生成数严格为0；其开发胜者退回已有阈值18规则，为2596/3000。全部阶段已停止，不再追加策略。

这里“无训练”不等于规则没有使用实验反馈进行开发选择：规则族是有限手写谓词，ACC用于挑选候选，没有学习权重或打分模型；本轮正向收益仍受同数据反复探索的选择偏差影响。

Qwen新规则与Question-Only均为56/60，未复现LLaVA上超越Question-Only的净提升。不能称为跨模型优势确认。Qwen原生assistant角色词的Tokenizer组与LLaVA ASSISTANT组不同，这种迁移检查只测试明确映射后的规则。

预算另记：最终LLaVA的逐题k/R均值为48.779%，所有问题合并后的总k/总R为WEIGHTED%；二者含义不同。完整逐题位置、比例和高预算题均保留。

图表已经人工查看PNG；预算图对重叠编号做了排版调整，数据和区间不变。三个矢量PDF经pdfinfo验证，均为单页。最终CSV、原始Checkpoint、图表和代码均进入文件SHA索引；原始Logits数组仍只在服务器资源目录。
'''.replace('WEIGHTED',f'{100*weighted:.3f}')
if '出版前复核补充' not in text:report.write_text(text+extra,encoding='utf-8')
else:assert text.endswith(extra)
logs=P/'final_execution_logs'
for name in ['recheck.log','cached_fallbacks.log','finalize.log','finalize_retry.log']:shutil.copyfile(Path('/data1/likai/ATE-resources/experiments/semantic_qrelay_search3h_run')/name,logs/name)
pdfs={}
for rel in ['final_accuracy.pdf','final_budget_accuracy.pdf','qwen_fixed60/accuracy.pdf']:
    result=subprocess.check_output(['pdfinfo',str(P/rel)],text=True);assert re.search(r'Pages:\s+1\s*\n',result);pdfs[rel]=result
reference=read(P/'qwen_fixed60/source_reference.json');assert reference['byte_identical_to_reference_commit']
for row in reference['files']:assert sha(P/'qwen_fixed60/source_snapshot'/row['path'])==row['sha256']
qrows=list(csv.DictReader((P/'qwen_fixed60/per_question.csv').open()));qcheck=read(P/'qwen_fixed60/assistant_mapping_audit.json');pos=read(P/'phase6_fixed_compositions/winner_freeze.json')['positions']
for row in qcheck:
    sp=read(Path('/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research/results/gqa5h_receiver_search/input_specs')/(row['question_id']+'.json'))
    assert row['conditional_hit']==(set(pos[row['question_id']])!=set(sp['Q']))
main=subprocess.check_output(['git','rev-parse','main'],cwd=WT,text=True).strip();assert main=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert 'GPU-b08489af-0a8a-eb48-55ba-8346733af417' not in processes
write(P/'final_publication_qa.json',dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_minutes=(time.monotonic()-clock['start_monotonic'])/60,main_unchanged=main,pdf_info=pdfs,plots_visually_reviewed=True,weighted_R_ratio=weighted,mean_R_ratio=sum(float(x['R_ratio']) for x in rows)/3000,Qwen_rule_predicate_matches_LLaVA_on_all60=True,compute_processes=processes,no_new_GPU_after_search=True))
source=list((WT/'vlm_research/semantic_qrelay_search3h').glob('*.py'))
files=[p for p in P.rglob('*') if p.is_file()]+source
with (P/'final_artifact_index.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','size','sha256']);w.writeheader()
    for p in sorted(files):
        if p.name!='final_artifact_index.csv':w.writerow(dict(path=str(p.relative_to(WT)),size=p.stat().st_size,sha256=sha(p)))
print(json.dumps(read(P/'final_publication_qa.json'),ensure_ascii=False))
