"""Documentation interpretation only; verify all scientific artifacts unchanged."""
from pathlib import Path
import json,hashlib,subprocess

r=Path('/data/lk/ATE-worktrees/sparse-visual-writing-frontier-240')
p=r/'vlm_research/results/sparse_visual_writing_frontier_240'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
c=json.loads((p/'completion.json').read_text());protocol=json.loads((p/'protocol.json').read_text())
for name,h in c['artifacts'].items():assert sha(p/name)==h,name
before={name:h for name,h in c['artifacts'].items() if name.endswith('.csv') or name.startswith(('checkpoints/','smoke_checkpoints/'))}
for name,h in protocol['sources'].items():assert sha(r/'vlm_research'/name)==h,name
assert not subprocess.check_output(['git','diff',protocol['code_commit'],'--','vlm_research/sparse_visual_writing_frontier'],cwd=r,text=True)
report=p/'report_zh.md';text=report.read_text();assert '## 完成后的研究判读' not in text
extra='''## 完成后的研究判读

**在这六个新增组合中，8层和12层均出现接近All-Writer的点估计候选，但尚未证明它们满足非劣效界限。** 采用−1pp点估计标准，最少保留12层（Keep-0-11，201/240，83.750%），即关闭其余20层R→Image直接访问。采用−2pp标准，最少保留8层（Keep-4-11，198/240，82.500%），即关闭24层。结论只覆盖本轮测试集合，不是全部层子集的最优解或已验证的安全关闭上限。

Keep-0-11相对All为−0.417 pp，95%CI [−3.750, +3.333]；Keep-4-11为−1.667 pp，CI [−5.833, +2.500]。16层Keep-0-15也达到−1pp点估计标准：200/240（83.333%），差−0.833 pp，CI [−2.500, +0.833]。12层Keep-4-15只达到−2pp标准；另两组8层Keep-8-15和Keep-0-3+8-11均为196/240（81.667%），差−2.500 pp，未达到两项点估计界限。**所有新增条件的95%CI下界均低于−2pp，当前证据尚不能排除超出质量界限的损害。** 所有界限都相对答案阶段已屏蔽直接视觉访问的All-Writer 202/240，而不是原生Full 206/240。

在“保留层数—净Accuracy”两维点估计前沿上，8层Keep-4-11与12层Keep-0-11最值得保留。16层Keep-0-15被12层条件在这两维上点支配，但其逐题行为更接近All：仅3题原对变错、1题原错变对，4条完整Token序列改变；12层Keep-0-11则有10题损害、9题纠正、19条序列改变；8层Keep-4-11有15题损害、11题纠正、26条序列改变。因此，“几乎相同的净Accuracy”不代表保留了同一批正确答案，点Pareto前沿也不概括全部质量稳定性。16层条件仍可作为后续独立验证时的稳定性候选，不能从本批少数变化直接外推鲁棒性。

同预算比较目前只能给出方向性线索。12层Keep-0-11比Keep-4-15高0.833 pp，CI [−2.500, +4.167]；8层Keep-4-11分别比Keep-8-15和Keep-0-3+8-11高0.833 pp，CI [−2.083, +3.750]、[−2.500, +4.167]。另两组8层点估计相同，但互有8题损害和8题纠正。四组直接比较的区间全部跨0，尚不能确定某个邻近窗口组合总体更优，更不能据此证明某种特定协同或补偿机制。

任务结果显示净质量仍有结构性代价。Keep-4-11在ChooseAttr、ChooseCat、LogicalObj的净Accuracy与All相同，但ChooseRel、CompareAttr分别−2.5pp，QueryAttr为−5.0pp。Keep-0-11在ChooseAttr为+5.0pp、ChooseRel为+2.5pp，同时CompareAttr和QueryAttr各−5.0pp。每类仅40题，均属描述性观察，不能将总体−0.417pp理解成各任务均无损。

反例包括：`02813015`问男人在哪一侧，Gold与All为Right，8层Keep-4-11及12层Keep-0-11均答Left，而16层Keep-0-15仍答Right；`0213048`问未站立的是斑马还是长颈鹿，Gold与All为Zebras，12层条件答Giraffes，而16层仍答Zebras。这些例子只说明本批个体损害，全部错误和纠正均已保留，没有据此重选样本或追加搜索。

机制层面，本轮把上一轮的单窗口不足推进到多窗口功能稀疏候选：L8–L11与邻近窗口组合，能在关闭大量R→Image直接连接后维持较接近All的点估计表现。不过，保留8或12个“写入层”不等于模型只在这些层处理视觉：Image Query和全部KV在32层仍正常计算，写入非图像状态的视觉信息也可跨层继续传播。这里的候选是功能性来源访问限制，不是已实现的低计算推理。高质量候选留待独立数据验证；本轮没有执行新的搜索、联合消融、训练或加速实现。

'''
text=text.replace('## 协议与复用\n',extra+'## 协议与复用\n',1)
report.write_text(text,encoding='utf-8')
assert all(sha(p/name)==h for name,h in before.items())
audit=dict(passed=True,changed_files=['report_zh.md'],scientific_csv_and_checkpoints_unchanged=True,frozen_execution_sources_unchanged=True,verified_scientific_files=len(before),new_model_generations=0)
(p/'delivery_editorial_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
c['delivery_editorial_audit']='delivery_editorial_audit.json'
c['artifacts']={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['completion.json','worker.lock']}
(p/'completion.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('EDITORIAL PASS: only Chinese interpretation changed; frozen sources and scientific CSV/checkpoints unchanged')
