"""Narrative attribution from the fixed accuracy comparisons; no new experiment."""
import csv
from .common import *
def load(name):
 with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
 s={r['condition']:r for r in load('accuracy_summary.csv')};p={(r['candidate'],r['reference']):r for r in load('paired_comparisons.csv')}
 names=dict(A='P−1视觉回看限制',B='Decode视觉回看限制',D='L1–L6早层Attention输出置零',E='L22–L35深层视觉写入限制')
 top=max(['A','B','D','E'],key=lambda c:int(s[c]['correct']))
 lines=['## 损失归因的直接回答','',f'在四项单通路恢复中，**{names[top]}**对应的恢复点估计最大：条件{top}为{s[top]["correct"]}/1000（{float(s[top]["accuracy_percent"]):.3f}%）。这说明在其余F限制仍存在的背景下，该通路是本组结果中最值得优先恢复的候选；未对所有单通路差值做统一显著性排名，不能据点估计断言其唯一重要。','']
 for c in ['A','B','D','E']:
  r=p[c,'F'];lines.append(f'- {names[c]}：恢复后Δ={float(r["delta_pp"]):+.3f}pp，95%CI [{float(r["ci_low_pp"]):+.3f}, {float(r["ci_high_pp"]):+.3f}]，F对→错{r["reference_correct_to_wrong"]}题，F错→对{r["reference_wrong_to_correct"]}题。')
 lines+=['']
 for a,b,desc in [('C','A','已有P−1回看时，额外恢复Decode'),('C','B','已有Decode回看时，额外恢复P−1')]:
  r=p[a,b];lines.append(f'{desc}：Δ={float(r["delta_pp"]):+.3f}pp，95%CI [{float(r["ci_low_pp"]):+.3f}, {float(r["ci_high_pp"]):+.3f}]。')
 if float(p['A','F']['ci_low_pp'])>0 and float(p['E','F']['ci_low_pp'])>0:
  lines+=['',f'**机制上的重点是两条视觉通路的补偿关系。** 单独恢复P−1直接回看恢复{float(p["A","F"]["delta_pp"]):.1f}pp，保持P−1禁图但单独恢复深层R写入也恢复{float(p["E","F"]["delta_pp"]):.1f}pp。两者各自都有效，支持F同时限制直接读取与文本中继写入时损害较大的解释。不能把全部损失唯一归给P−1，也不能断言深层R写入没有作用；尚未检验A+E联合或正式替代性交互。']
 lines+=['','B不改变第一个答案Token，质量变化仅能来自后续生成；A/C第一个Token也严格相同。上述证据帮助区分Prefill首答案形成与真实Decode后续生成。五项恢复均在固定F背景下测量，收益可能相互补偿或非加性，不能把它们相加作为原10.6pp损失的独立分解。']
 cn=p['C','Native'];ca=p['C','A']
 lines+=['',f'C相对Native的Δ={float(cn["delta_pp"]):+.3f}pp，95%CI [{float(cn["ci_low_pp"]):+.3f}, {float(cn["ci_high_pp"]):+.3f}]；仍有Native对→错{cn["reference_correct_to_wrong"]}题、错→对{cn["reference_wrong_to_correct"]}题。净正确数接近不等于逐题行为一致或已证明无损。C相对A仅{float(ca["delta_pp"]):+.3f}pp，区间[{float(ca["ci_low_pp"]):+.3f}, {float(ca["ci_high_pp"]):+.3f}]同时容纳下降与提高，尚不能支持Decode额外恢复有明确增益。','','**对应物理方案只作提案**：以最佳C为功能参考，P−1及Decode在L0/L7–L35访问真实视觉KV，L1–L6保留真实Attention跳算。为生成深层所需视觉KV，第一版正确性原型应保留Prefill视觉Hidden到深层，取消直接沿用L22视觉退出；随后先做Mask、mRoPE、DeepStack、Cache、数值及完整答案质量验证。若希望保持Decode禁图结构，已测A也是质量候选，可在Prefill结束后释放直接视觉KV；其物理正确性、质量和速度均未在本轮验证。','','本轮以完整答案Accuracy判断恢复效果，所有损害与恢复案例保留。未依据中途结果增加条件，未实施任何物理恢复、E2E、FLOPs或其他性能测试。']
 text='\n'.join(lines)+'\n';(OUT/'attribution_zh.md').write_text(text)
 report=OUT/'report_zh.md';marker='\n## 损失归因的直接回答\n';body=report.read_text().split(marker)[0];report.write_text(body+'\n'+text)
 print('TOP_SINGLE_PATHWAY',top,names[top])
if __name__=='__main__':main()
