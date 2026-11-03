"""Reconcile published counts and make the observed conclusions explicit."""
import csv,json,subprocess
from .failure_probe import P,ROOT,NAMES,PAIRS,read,sha256,atomic_json,stamp

def main():
    rows=list(csv.DictReader((P/'per_question.csv').open()));pairs=list(csv.DictReader((P/'paired_comparisons.csv').open()));summary=list(csv.DictReader((P/'accuracy_summary.csv').open()))
    val=read(P/'validation.json');done=read(P/'completion.json');assert val['passed'] and done['status']=='FAILURE_PROBE_COMPLETE'
    assert len(rows)==138 and sum(r['stratum']!='C' for r in rows)==114
    scores={r['question_id']:json.loads(r['correct_per_condition']) for r in rows};ab=[r for r in rows if r['stratum']!='C'];control=[r for r in rows if r['stratum']=='C']
    for r in summary:
        ids=rows if r['group']=='ALL_SELECTED' else ab if r['group']=='AB' else [v for v in rows if v['stratum']==r['group']]
        assert int(r['correct'])==sum(scores[v['question_id']][r['condition']] for v in ids) and len(ids)==int(r['n'])
    for r in pairs:
        ids=ab if r['group']=='AB' else [v for v in rows if v['stratum']==r['group']]
        harm=sum(scores[v['question_id']][r['right']] and not scores[v['question_id']][r['left']] for v in ids)
        fix=sum(not scores[v['question_id']][r['right']] and scores[v['question_id']][r['left']] for v in ids)
        assert harm==int(r['right_correct_left_wrong']) and fix==int(r['right_wrong_left_correct']) and abs(float(r['delta_pp'])-100*(fix-harm)/len(ids))<1e-10
    rep=list(csv.DictReader((P/'representative_cases.csv').open()))
    for r in rep:
        pool=[v for v in rows if v[r['category']]=='True'];assert r['question_id']==min(pool,key=lambda x:(x['selection_sha256'],x['question_id']))['question_id']
    old=(P/'report_zh.md').read_text(encoding='utf-8');text=old
    line4=next(l for l in text.splitlines() if l.startswith('4. 同预算模板'))
    text=text.replace(line4,'4. 同预算模板Control-1和Control-2各修复1/114题（0.88%）。NL比单Token模板净多20题（+17.54pp；调整99%CI [7.89, 27.19]）；SPACE净多12题（+10.53pp；[2.63, 19.30]）；联合比双Token模板净多25题（+21.93pp；[12.28, 32.46]）。三项在预设五比较的Bonferroni调整区间下均高于0，支持本批定向失败样本中边界位置优于这两个固定模板位置，不能推广为优于所有模板选择。')
    line6=next(l for l in text.splitlines() if l.startswith('6. 边界Token'))
    text=text.replace(line6,'6. 本批支持继续研究Boundary Receiver：边界补位在相同新增Token预算下显著超过两个冻结模板对照，说明不仅是增加接收Token数量。但效应依赖位置和样本：NL高于SPACE、联合高于NL都只是点估计趋势，其调整99%区间均跨0；联合仍破坏1道NL成功案例，负面控制也出现损害。恢复主要来自A组（NL 17/56、联合21/56），B组仅4/58和5/58。该结果提供限定样本中的干预效应，不证明高Attention Mass等于因果重要性，更不证明唯一视觉存储位置或3000题总体提升。下一步若研究边界接收者，应另行授权、在未按S失败选样的数据上验证净收益；本轮未追加任何实验。')
    text=text.replace('## 五项预设配对比较','负面控制中零事件的经验Bootstrap区间可能退化为[0,0]；它只反映24道观测样本的重采样分布，不代表总体损害率为零或已证明安全。\n\n## 五项预设配对比较')
    text=text.replace('原始首Token/全部有效Logits、逐层审计数组保存在服务器资源目录；','可复现命令：先运行failure_probe freeze，提交冻结源码与协议，再运行failure_probe driver，之后运行failure_analyze和failure_delivery；均以本包为Python module执行。CPU结果审计可按checkpoint_index重算，GPU阶段不得自动重新启动。\n\n原始首Token/全部有效Logits、逐层审计数组保存在服务器资源目录；')
    if text!=old:(P/'report_zh.md').write_text(text,encoding='utf-8')
    (P/'.gitignore').write_text('*.lock\n',encoding='utf-8')
    audit=dict(passed=True,independent_wide_table_count_reconciliation=True,all_paired_discordances_reconciled=True,all_representatives_follow_frozen_SHA_order=True,analysis_source_sha256=sha256(Path(__file__).with_name('failure_analyze.py')),delivery_source_sha256=sha256(__file__),report_sha256=sha256(P/'report_zh.md'),raw_arrays_in_git=False,utc=stamp())
    assert audit['analysis_source_sha256']==val['analysis_source_sha256']
    atomic_json(P/'delivery_audit.json',audit)
    print(json.dumps(audit),flush=True)

from pathlib import Path
if __name__=='__main__':main()
