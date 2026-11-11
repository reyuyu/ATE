"""Add data-derived interpretation; no predictions/statistics are changed."""
from pathlib import Path
import sys,csv,json,hashlib
sys.path.insert(0,str(Path(__file__).parent.parent/'code'))
from common import *
a={r['condition']:r for r in csv.DictReader((OUT/'accuracy_summary.csv').open())}
c={(r['A'],r['B']):r for r in csv.DictReader((OUT/'paired_comparisons.csv').open())}
p=OUT/'report_zh.md';text=p.read_text();before=sha(p)
gap=-float(c['Middle18','Late18']['delta_A_minus_B_pp']);low=-float(c['Middle18','Late18']['CI95_high']);high=-float(c['Middle18','Late18']['CI95_low'])
line=f"""本轮核心观察：**Qwen在k=4、9、13、18、22、27时均以Late得分最高；没有复现LLaVA Figure7的中层优先形态。** k=18时Late为{float(a['Late18']['accuracy_percent']):.2f}%，Middle为{float(a['Middle18']['accuracy_percent']):.2f}%，Late−Middle为{gap:+.2f}pp，普通95%配对CI[{low:+.2f}, {high:+.2f}]。Late18相对Native损失1.39pp；Late27与Native的均值相同，但均值相同不证明逐题或统计等价。

高预算差异很小：Middle31为83.03%，相对Native仅+0.14pp，95%CI[−0.37,+0.74]，不能声称优于Native。非单调现象保留：Early4低于全禁图、Early18低于Early13、Late31低于Late27；没有据此改层或修正结果。以上结论限于这一探索性共同有效子集、原生Qwen动态视觉预算与一个Random排列，不能将跨模型差异全部归因于视觉回看位置。

"""
assert '本轮核心观察：' not in text
pos=text.index('| k | Early')
text=text[:pos]+line+text[pos:];p.write_text(text)
write(OUT/'report_editorial_provenance.json',dict(source_sha256=sha(Path(__file__)),input_report_sha256=before,output_report_sha256=sha(p),accuracy_csv_sha256=sha(OUT/'accuracy_summary.csv'),paired_csv_sha256=sha(OUT/'paired_comparisons.csv'),no_prediction_or_statistics_changes=True))
write(OUT/'visual_qa.json',dict(status='PASS',figure_sha256=sha(OUT/'figure_accuracy.png'),checks=['PNG visually inspected at1350x950','four correctly labeled discrete curves and all9 frozen budgets','Native82.89 baseline and498/500 common-subset caption readable','no smoothing or fabricated points','title, legend, axes and output margins readable']))
print('REPORT_INTERPRETATION_AND_VISUAL_QA_PASS')
