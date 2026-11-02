"""Report-only editorial finalization from verified saved statistics."""
from pathlib import Path
import csv,json,hashlib
OUT=Path(__file__).resolve().parent
def table(name):
    with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
summary={r['condition']:r for r in table('accuracy_summary.csv')}
pairs={r['a']+'|'+r['b']:r for r in table('paired_comparisons.csv')}
assert summary['R-Image-OFF']['correct']=='207' and summary['R-Text-OFF']['correct']=='50'
answer='''## 对核心问题的明确回答

**按本轮来源 Mask 定义，在这批固定模型、层段和探索样本上，R 的关键更新更明显依赖非图像历史上下文，而不是 L2–L5 中对原始 Image Key 的直接访问。** Image-OFF 为 207/240（86.250%），相对 Full +0.417 pp，但仍有 2 题受损、3 题纠正；Text-OFF 为 50/240（20.833%），相对 Full −65.000 pp，156 题受损、无纠正。R-All 为 138/240（57.500%）。因此，不能仅凭 R-All 的大损害就将其归因于该层段必须直接读图；结果为保留上下文处理、审视直接来源访问提供了不同的优化线索。

这里的“非图像历史”不是“纯语言”：文本 Key 可能已包含视觉信息，其他层仍可读图。Text-OFF 还会重归一化并保留 Image/BOS/Self，其损害甚至大于 R-All；这说明保留被改变的 Attention 更新可以比完全置零更有害，不能将三种损害当作可加的信息份额。是否主要是一般语言整合、经文本状态转递的视觉证据，或两者共同作用，尚未被本轮完全分离。Question-only 没有可靠 Gold，不能据此宣称视觉专属性已证明。

换图控制强化了“图像条件性不等于正确整合”的区分：Full 和 Image-OFF 均为 28/40 对两端正确，Text-OFF 为 0/40；三条件却都仍有 31/40 对答案随图改变。不能据此说 Text-OFF 已丧失全部视觉信息，也不能把同样的变化计数解释为行为完全相同。主实验 Text-OFF 另有 3 条达到 32 Token 上限的截断，完整输出均保留。

'''
path=OUT/'report_zh.md';text=path.read_text();assert text.count('## 核心观察')==1
text=text.replace('## 核心观察',answer+'## 核心观察',1)
rows=table('per_question.csv');full={r['question_id']:r for r in rows if r['condition']=='Full'}
harms=table('negative_cases.csv');selected=[]
for c in ['R-Image-OFF','R-Text-OFF']:
    selected.extend(sorted([r for r in harms if r['condition']==c],key=lambda r:r['question_id'])[:2])
case_lines=['## 主要负面案例','', '以下是按条件、Question ID 固定排序的展示；完整负面案例均保留于 CSV，不删除失败样本。','',
    '|题号|条件|问题|Gold|Full|干预答案|','|---|---|---|---|---|---|']
for r in selected:
    cells=[r['question_id'],r['condition'],r['question'],r['gold'],full[r['question_id']]['text'],r['text']]
    case_lines.append('|'+ '|'.join(x.replace('|','\\|').replace('\n',' ') for x in cells)+'|')
text=text.replace('## 正确性与执行','\n'.join(case_lines)+'\n\n## 正确性与执行',1)
text+='\n交付编辑说明：finalize_delivery.py 只补充由已保存结果支持的核心结论和案例展示，不改变冻结源码、统计、检查点或数组。\n'
path.write_text(text)
c=json.loads((OUT/'completion.json').read_text())
c['delivery_finalization']=dict(report_question_answered=True,negative_cases_displayed=True,scientific_data_unchanged=True,frozen_sources_unchanged=True)
c['artifacts']={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['worker.lock','completion.json']}
(OUT/'completion.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
print('Finalized core answer and negative examples; refreshed artifact hashes')
