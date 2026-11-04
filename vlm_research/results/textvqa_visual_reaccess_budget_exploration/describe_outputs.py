"""Additional descriptive audits only; no conditions, generations or tests added."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys,json,csv,collections,math
root=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget/vlm_research');sys.path.insert(0,str(root))
from textvqa_reaccess.common import *
from transformers import AutoTokenizer
from receiver_cross_dataset.scoring import normalize
import numpy as np
def main():
 rows=list(csv.DictReader((OUT/'per_question.csv').open()));by=collections.defaultdict(dict);order=[r['question_id'] for r in read(OUT/'sample_manifest.json')['rows']]
 for r in rows:by[r['question_id']][r['condition']]=r
 for q,v in by.items():
  for c,r in v.items():
   for base in ['A','B']:
    for field in ['answer','token_ids','first_token','eos','truncated','repeated_trigram']:r[field+'_changed_vs_'+base]=r[field]!=v[base][field]
    r['length_delta_vs_'+base]=int(r['length'])-int(v[base]['length'])
 csvwrite(OUT/'per_question.csv',rows)
 # Literal first model token and first prefix that decodes to visible content.
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True);tokenrows=[];content={}
 for r in rows:
  ids=json.loads(r['token_ids']);at=next((j for j in range(len(ids)) if tok.decode(ids[:j+1],skip_special_tokens=True).strip()),None)
  first=tok.decode(ids[:1],skip_special_tokens=True);item=dict(question_id=r['question_id'],condition=r['condition'],first_token_id=ids[0],first_token_decoded_repr=repr(first),first_token_whitespace_or_special=not bool(first.strip()),first_content_prefix_end_index=at,first_content_subword_id=ids[at] if at is not None else None,first_content_prefix=tok.decode(ids[:at+1],skip_special_tokens=True) if at is not None else '',first_content_is_not_necessarily_a_whole_word=True)
  tokenrows.append(item);content[(r['question_id'],r['condition'])]=item
 csvwrite(OUT/'first_token_content_audit.csv',tokenrows)
 summaries=list(csv.DictReader((OUT/'accuracy_summary.csv').open()));toksummary=[]
 for s in summaries:
  c=s['condition'];qs=order if s['scope']=='300' else [q for q in order if by[q]['B']['stage2_subset']=='True'];aggregate=c.startswith('RandomMean')
  if aggregate:
   s.update(Pminus1_layers=c.split('-')[1],Decode_layers=c.split('-')[1],total_decode_queries='',pooled_actual_edge_ratio='',budget_source='Three seeds; no single trajectory')
  else:
   rr=[by[q][c] for q in qs];s.update(Pminus1_layers=DESIGN['conditions'][c]['Pminus1'].__len__(),Decode_layers=DESIGN['conditions'][c]['Decode'].__len__(),total_decode_queries=sum(int(r['length'])-1 for r in rr),pooled_actual_edge_ratio=sum(float(r['allowed_edge_ratio'])*int(r['length']) for r in rr)/sum(int(r['length']) for r in rr),budget_source='Actual condition trajectories')
   for base in ['A','B']:
    for field in ['answer','token_ids','first_token','eos','truncated','repeated_trigram']:s[field+'_changed_vs_'+base]=sum(r[field]!=by[q][base][field] for q,r in zip(qs,rr))
    s['length_changed_vs_'+base]=sum(r['length']!=by[q][base]['length'] for q,r in zip(qs,rr))
   toksummary.append(dict(scope=s['scope'],condition=c,n=len(qs),first_token_whitespace_or_special=sum(content[(q,c)]['first_token_whitespace_or_special'] for q in qs),literal_first_token_flip_vs_B=sum(by[q][c]['first_token']!=by[q]['B']['first_token'] for q in qs),first_content_subword_flip_vs_B=sum(content[(q,c)]['first_content_subword_id']!=content[(q,'B')]['first_content_subword_id'] for q in qs)))
 # Union field list preserves blanks for aggregate rows.
 fields=list(dict.fromkeys(k for r in summaries for k in r));csvwrite(OUT/'accuracy_summary.csv',summaries,fields)
 csvwrite(OUT/'phase_budget_summary.csv',[r for r in summaries if r['scope']=='120' and not r['condition'].startswith('Random')],fields)
 csvwrite(OUT/'random_control_summary.csv',[r for r in summaries if r['scope']=='120' and (r['condition'].startswith('Random') or r['condition'] in ['Both-8','Both-16'])],fields)
 csvwrite(OUT/'first_token_content_summary.csv',toksummary)
 # Every seed-mean negative includes the three actual answers rather than a fabricated averaged answer.
 negative=list(csv.DictReader((OUT/'harm_cases.csv').open()));positive=list(csv.DictReader((OUT/'recovery_cases.csv').open()));mean_cases=[]
 for k in [8,16]:
  for q in order:
   if by[q]['B']['stage2_subset']!='True':continue
   left=by[q]['Both-'+str(k)];rr=[by[q][f'Random-{k}-s{j+1}'] for j in range(3)];avg=np.mean([float(r['score']) for r in rr]);delta=float(left['score'])-avg
   if abs(delta)<=1e-12:continue
   case=dict(family='stage2',scope=120,question_id=q,image_sha256=left['image_sha256'],image_path=left['image_path'],question=left['question'],answers=left['answers'],left=left['condition'],right='RandomMean-'+str(k),left_answer=left['answer'],right_answer=json.dumps([r['answer'] for r in rr],ensure_ascii=False),left_score=left['score'],right_score=avg,delta_score=delta,left_token_ids=left['token_ids'],right_token_ids=json.dumps([json.loads(r['token_ids']) for r in rr]),kind='harm' if delta<0 else 'recovery')
   mean_cases.append(case);(negative if delta<0 else positive).append(case)
 csvwrite(OUT/'harm_cases.csv',negative);csvwrite(OUT/'recovery_cases.csv',positive);csvwrite(OUT/'random_mean_cases.csv',mean_cases,fields=list(negative[0]))
 # More stringent all-observed-sparse diagnostics, retaining partial scores.
 unrescued=[]
 for q in order:
  v=by[q];sparse=[c for c in v if c not in ['A','B','F','G']]
  if float(v['A']['score'])>=1-1e-12 and all(float(v[c]['score'])<1-1e-12 for c in sparse):unrescued.append(dict(question_id=q,question=v['A']['question'],answers=v['A']['answers'],native_answer=v['A']['answer'],tested_sparse_conditions=json.dumps(sparse),n_tested=len(sparse),all_sparse_scores_zero=all(float(v[c]['score'])==0 for c in sparse),sparse_answers=json.dumps({c:v[c]['answer'] for c in sparse},ensure_ascii=False),sparse_scores=json.dumps({c:float(v[c]['score']) for c in sparse})))
 csvwrite(OUT/'all_observed_sparse_failures.csv',unrescued,fields=['question_id','question','answers','native_answer','tested_sparse_conditions','n_tested','all_sparse_scores_zero','sparse_answers','sparse_scores'])
 targets=list(csv.DictReader((OUT/'descriptive_targets.csv').open()));curve=list(csv.DictReader((OUT/'budget_curve.csv').open()))
 for r in targets:
  test=(lambda x:float(x['delta_vs_A_pp'])>=-2-1e-12) if r['target']=='Full_minus_2pp' else (lambda x:float(x['Full_gap_recovery_ratio'])>=float(r['target'].split('_')[-1]));valid=[int(x['k']) for x in curve if int(x['k'])<32 and test(x)];r['minimum_observed_sparse_k']=min(valid) if valid else None;r['sparse_point_target_achieved']=bool(valid)
 csvwrite(OUT/'descriptive_targets.csv',targets)
 p=OUT/'report_zh.md';text=p.read_text();text+='\n## 补充审计：实际连接计数与首Token的含义\n\n'
 text+='accuracy_summary.csv、phase_budget_summary.csv同时给出每阶段层数、实际Decode查询总数、逐题平均连接比例与按实际查询加权的合并比例。统计样本量仍为300或120；随机均值不被当作一条真实生成轨迹。\n\n'
 text+='首Token指模型Tokenizer的第一个生成位置，可能只是空格、换行或特殊Token；不能直接等同于首个词。first_token_content_audit.csv以“最短生成前缀解码后含非空白字符”的确定规则记录第一个内容子词，保留原始Token和位置，不改任何Prompt或生成。该子词也不必是完整单词。\n\n'
 for s in toksummary:
  if s['scope']=='300' and s['condition'] in ['A','B','F','G']:text+=f"{s['condition']}：首Token为空白/特殊位置{s['first_token_whitespace_or_special']}/300；相对B的字面首Token变化{s['literal_first_token_flip_vs_B']}题，第一个内容子词变化{s['first_content_subword_flip_vs_B']}题。\n\n"
 text+='随机均值比较的逐题正负例已补齐在harm_cases.csv和recovery_cases.csv，右侧明确保存三个种子的真实答案与Token列表，不制造一条“平均回答”。其余物理条件的所有损害和异常未删除。\n\n'
 text+=f"Native Full得满分但所有该题已测试稀疏条件均未得满分，共{len(unrescued)}题，详见all_observed_sparse_failures.csv；不同题是否属于120子集影响已测试条件数，不能据此比较子集难度。部分得分与零分严格区分。\n\n"
 for t in targets:
  if t['sparse_point_target_achieved']:text+=f"{t['target']}：稀疏点最小观察到k={t['minimum_observed_sparse_k']}。\n\n"
  else:text+=f"{t['target']}：所有已测试k<32的稀疏点均未达到，仅全开放Native k=32达到。\n\n"
 p.write_text(text.rstrip()+'\n')
 print('DESCRIPTIVE_ENRICHMENT_PASS',len(tokenrows),len(mean_cases),len(unrescued),flush=True)
if __name__=='__main__':main()
