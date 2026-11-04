"""Append observed timing, exact phase controls and unfiltered case tables."""
import pathlib,sys,json,csv,collections
root=pathlib.Path('/data/lk/ATE-worktrees/textvqa-partial-visual-layer-recovery/vlm_research');sys.path.insert(0,str(root))
from textvqa_layer_recovery.common import *
from receiver_cross_dataset.scoring import normalize
out=OUT;records=list(csv.DictReader((out/'per_question.csv').open()));by=collections.defaultdict(dict)
for r in records:by[r['question_id']][r['condition']]=r
order=[r['question_id'] for r in read(out/'sample_manifest.json')['rows']]
run=read(out/'gpu_run_complete.json');formal=read(out/'generation_completion.json');lines=['','## 实测运行与复用边界','',f"单GPU阶段（含加载、8题Smoke和正式矩阵）共{run['total_GPU_phase_seconds']/60:.2f}分钟；正式遍历阶段{formal['formal_seconds']/60:.2f}分钟。峰值已分配显存{run['peak_allocated_bytes']/1024**3:.2f}GiB、峰值已保留显存{run['peak_reserved_bytes']/1024**3:.2f}GiB，包含本实现审计和Cache开销，不能作为优化收益。",'', 'B的300条全部来自冻结Commit逐字节核验的历史检查点。A/C/D/E新检查点1200条，其中32条来自本轮Smoke并在全部验证通过后复用，正式阶段新增1168条；额外Smoke对照仅用于正确性，不纳入1500个评分标签重复计数。','', '首Token阶段不变性作为独立交付审计：全部300题D首Token原始FP16 Logits SHA必须与B相同，E必须与C相同。该检查与原始Token/输入/模型一致性一起验证阶段权限，结果见independent_statistics_audit.json。','', '## 真实改善、损害与残留差距案例','', '以下仅按冻结Manifest顺序展示各类前3题，不按故事效果挑选。全部负面和异常案例均保留在CSV；软评分升降不能一概写成二元答对/答错。']
def esc(t):return str(t).replace('|','/').replace('\n',' ')
for name,test in [('E相对B改善',lambda v:float(v['E']['score'])>float(v['B']['score'])+1e-12),('E相对B损害',lambda v:float(v['E']['score'])<float(v['B']['score'])-1e-12),('E仍低于Native Full',lambda v:float(v['E']['score'])<float(v['A']['score'])-1e-12)]:
 qs=[q for q in order if test(by[q])];lines+=['',f'### {name}：共{len(qs)}题','', '| ID / 问题 | 标注前三项（原始顺序） | A答案 / 分数 | B答案 / 分数 | E答案 / 分数 |','|---|---|---|---|---|']
 for q in qs[:3]:
  v=by[q];answers=json.loads(v['A']['answers']);cells=[q+' / '+v['A']['question'],' / '.join(answers[:3])]+[v[c]['answer']+' / '+v[c]['score'] for c in ['A','B','E']];lines.append('| '+' | '.join(esc(t) for t in cells)+' |')
residual=[]
for q in order:
 v=by[q]
 if float(v['E']['score'])<float(v['A']['score'])-1e-12:residual.append(dict(question_id=q,question=v['A']['question'],answers=v['A']['answers'],A_answer=v['A']['answer'],A_score=v['A']['score'],B_answer=v['B']['answer'],B_score=v['B']['score'],C_answer=v['C']['answer'],C_score=v['C']['score'],D_answer=v['D']['answer'],D_score=v['D']['score'],E_answer=v['E']['answer'],E_score=v['E']['score'],E_truncated=v['E']['truncated'],E_repeated=v['E']['repeated_trigram']))
fields=['question_id','question','answers','A_answer','A_score','B_answer','B_score','C_answer','C_score','D_answer','D_score','E_answer','E_score','E_truncated','E_repeated']
csvwrite(out/'residual_errors_vs_native.csv',residual,fields)
lines += ['', '恢复两层后仍低于Native Full的题全部另存residual_errors_vs_native.csv。E只在两个固定层开放直接视觉访问，其余30层仍关闭；因此差距与其余层直接读取的整体作用相容，但不能由本轮定位到某一个未恢复层。Native Full本身未达到满分，说明当前无OCR输入、图像分辨率和模型能力仍有限；具体错误来自文字读取、语言生成还是标注分歧，需要另行验证，本轮不据此改层或改Prompt。','', '实际Attention Mass只描述注意力分配，不等于因果重要性，不能据此在本批数据中重新选层。','', '图片字节和模型大文件未纳入Git；服务器原路径及SHA在sample_manifest.json和model_manifest.json。CPU/GPU日志、原子检查点和结果SHA随交付保留。']
p=out/'report_zh.md';text=p.read_text();text=text.replace('图表PNG/矢量PDF','图表PNG/PDF（坐标与文字为矢量，Mass热图嵌入栅格）');p.write_text(text+'\n'.join(lines)+'\n')
print('REPORT_DETAILS_COMPLETE',len(residual))
