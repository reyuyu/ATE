"""All prespecified comparisons and failures, with actual query-weighted edges."""
from .common import *
from receiver_cross_dataset.analyze import paired
from receiver_cross_dataset.scoring import normalize
import numpy as np,collections,math,re,textwrap
def load():
 for j in [0,1]:assert read(OUT/'workers'/str(j)/'generation_validation.json')['passed']
 assert sources()==read(OUT/'source_manifest.json')['files']
 manifest=read(OUT/'sample_manifest.json')['rows'];history=historical_rows();rows=[];by={};edge=[];agg=collections.defaultdict(list);index=[];raw=collections.defaultdict(list)
 for r in manifest:
  q=r['question_id'];s=read(OUT/'input_specs'/(q+'.json'));by[q]={}
  available=list('ABCDE')+STAGE1+(STAGE2 if r['stage2_subset'] else [])
  for c in available:
   p=ROOT/history[q][c]['checkpoint'] if c in 'ABCDE' else OUT/'checkpoints'/q/(c+'.json');v=read(p);g=v['generation'];assert abs(score(g['text'],r['answers'])-v['score'])<1e-12
   design=DESIGN['conditions'][c];layersp=len(design['Pminus1']);layersd=len(design['Decode']);decode=g['length']-1;I=len(s['I']);head=32;ep=layersp*I;ed=layersd*I*decode;den=32*I*(1+decode)
   row=dict(question_id=q,image_id=r['image_id'],image_sha256=r['image_sha256'],image_path=r['image_path'],question=r['question'],answers=json.dumps(r['answers'],ensure_ascii=False),condition=c,score=v['score'],answer=g['text'],token_ids=json.dumps(g['token_ids']),first_token=v['first_token'],first_logits_sha256=v['first_logits_sha256'],length=g['length'],non_EOS_length=g['length']-int(g['eos']),eos=g['eos'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],empty=g['empty'],R=len(s['R']),R_allowed_ratio=1.,Pminus1_layers=layersp,Decode_layers=layersd,Decode_queries=decode,image_keys=I,allowed_edge_ratio=(ep+ed)/den,worker=r['worker'],stage2_subset=r['stage2_subset'],checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=sha(p),origin='audited frozen history' if c in 'ABCDE' else v['origin'])
   rows.append(row);by[q][c]=row
   edge.append(dict(question_id=q,condition=c,Pminus1_layers=layersp,Decode_layers=layersd,Pminus1_queries=1,Decode_queries=decode,original_image_keys=I,heads=head,Pminus1_allowed_edges_per_head=ep,Decode_allowed_edges_per_head=ed,total_allowed_edges_all_heads=(ep+ed)*head,total_possible_edges_all_heads=den*head,Pminus1_allowed_ratio=layersp/32,Decode_allowed_ratio=layersd/32,total_actual_allowed_ratio=(ep+ed)/den,restricted_layer_mask_fraction_Pminus1=1-layersp/32,restricted_layer_mask_fraction_Decode=1-layersd/32,R_allowed_ratio=1.,generation_length=g['length']))
   if c not in 'ABCDE':index.append(dict(question_id=q,condition=c,path=str(p.relative_to(OUT)),sha256=sha(p),worker=r['worker']))
   if c=='B':
    for phase,num in [('Pminus1',1),('Decode',decode)]:
     for l in range(32):agg[(c,phase,l)].append((0.,num,'Historical all forbidden probabilities exactly0; mass inferred0'))
   else:
    for m in v['attention_mass']:
     agg[(c,m['phase'],m['layer'])].append((m['image_mass_mean'],1,'Fresh this run' if c not in 'ABCDE' else 'Audited historical actual attention'))
     raw[c].append(dict(question_id=q,condition=c,**{k:value for k,value in m.items() if k!='per_head_mass'}))
  assert by[q]['B']['first_logits_sha256']==by[q]['G']['first_logits_sha256'] and by[q]['A']['first_logits_sha256']==by[q]['F']['first_logits_sha256']
  if r['stage2_subset']:
   for k in [4,8,16]:assert by[q]['B']['first_logits_sha256']==by[q]['Donly-'+str(k)]['first_logits_sha256'] and by[q]['Ponly-'+str(k)]['first_logits_sha256']==by[q]['Both-'+str(k)]['first_logits_sha256']
 assert len(rows)==5040 and len(index)==3540 and len({(r['question_id'],r['condition']) for r in rows})==5040
 for r in rows:r['delta_score_vs_B']=r['score']-by[r['question_id']]['B']['score'];r['delta_score_vs_A']=r['score']-by[r['question_id']]['A']['score']
 csvwrite(OUT/'per_question.csv',rows);csvwrite(OUT/'visual_edge_budget.csv',edge);csvwrite(OUT/'checkpoint_index.csv',index)
 for c,rr in raw.items():csvwrite(OUT/'attention_mass'/(c+'.csv'),rr)
 layer=[]
 for (c,phase,l),values in agg.items():
  n=sum(x[1] for x in values);layer.append(dict(condition=c,phase=phase,layer=l,query_count=n,image_mass_head_query_mean=sum(x[0]*x[1] for x in values)/n if n else None,source=values[0][2]))
 csvwrite(OUT/'attention_layer_summary.csv',layer)
 return rows,by,manifest
def actual_changes(aa,bb):
 if any(r.get('aggregate',False) for r in aa+bb):return dict(answer_changed=None,token_sequence_changed=None,first_token_flip=None,length_changed=None,mean_length_delta=None,aggregate_note='Seed-averaged score only; no single generated answer')
 out=dict(answer_changed=sum(x['answer']!=y['answer'] for x,y in zip(aa,bb)),token_sequence_changed=sum(x['token_ids']!=y['token_ids'] for x,y in zip(aa,bb)),first_token_flip=sum(x['first_token']!=y['first_token'] for x,y in zip(aa,bb)),length_changed=sum(x['length']!=y['length'] for x,y in zip(aa,bb)),mean_length_delta=float(np.mean([x['length']-y['length'] for x,y in zip(aa,bb)])),aggregate_note='')
 for flag in ['eos','truncated','repeated_trigram','empty']:out[flag+'_gained']=sum(x[flag] and not y[flag] for x,y in zip(aa,bb));out[flag+'_lost']=sum(y[flag] and not x[flag] for x,y in zip(aa,bb))
 return out
def augmented(by):
 result={q:dict(v) for q,v in by.items()}
 for q,v in result.items():
  if 'Random-8-s1' not in v:continue
  for k in [8,16]:
   rr=[v[f'Random-{k}-s{j+1}'] for j in range(3)];v['RandomMean-'+str(k)]=dict(score=float(np.mean([r['score'] for r in rr])),aggregate=True,image_sha256=v['B']['image_sha256'])
 return result
def statistics(rows,by):
 augmented_by=augmented(by);summary=[];comparisons=[];cases=[];casefields=['family','scope','question_id','image_sha256','image_path','question','answers','left','right','left_answer','right_answer','left_score','right_score','delta_score','left_token_ids','right_token_ids','kind']
 for family,n,subset in [('stage1',300,list(by)),('stage2',120,[q for q,v in by.items() if v['B']['stage2_subset']])]:
  allconds=list('ABCDE')+STAGE1+(STAGE2+['RandomMean-8','RandomMean-16'] if n==120 else [])
  A=100*np.mean([by[q]['A']['score'] for q in subset]);B=100*np.mean([by[q]['B']['score'] for q in subset]);images=[by[q]['B']['image_sha256'] for q in subset]
  for c in allconds:
   rr=[augmented_by[q][c] for q in subset];sc=[r['score'] for r in rr];ci=paired(sc,np.zeros(n),images);vals=np.asarray(sc);base=np.array([by[q]['B']['score'] for q in subset]);full=np.array([by[q]['A']['score'] for q in subset]);acc=100*float(vals.mean());isagg=c.startswith('RandomMean')
   summary.append(dict(family=family,scope=n,condition=c,n=n,accuracy_pct=acc,CI95_low_pct=ci['CI95_low_pp'],CI95_high_pct=ci['CI95_high_pp'],delta_vs_B_pp=acc-B,delta_vs_A_pp=acc-A,Full_gap_recovery_ratio=(acc-B)/(A-B) if A!=B else None,score_up_vs_B=int((vals>base+1e-12).sum()),score_down_vs_B=int((vals<base-1e-12).sum()),score_equal_vs_B=int((abs(vals-base)<=1e-12).sum()),score_up_vs_A=int((vals>full+1e-12).sum()),score_down_vs_A=int((vals<full-1e-12).sum()),score_equal_vs_A=int((abs(vals-full)<=1e-12).sum()),eos=None if isagg else sum(r['eos'] for r in rr),truncated=None if isagg else sum(r['truncated'] for r in rr),repeated=None if isagg else sum(r['repeated_trigram'] for r in rr),empty=None if isagg else sum(r['empty'] for r in rr),mean_tokens=None if isagg else float(np.mean([r['length'] for r in rr])),mean_actual_edge_ratio=None if isagg else float(np.mean([r['allowed_edge_ratio'] for r in rr])),**actual_changes(rr,[by[q]['B'] for q in subset])))
  familyrows=[]
  for a,b in read(OUT/'protocol.json')['comparisons'][family]:
   aa=[augmented_by[q][a] for q in subset];bb=[augmented_by[q][b] for q in subset];r=paired([x['score'] for x in aa],[x['score'] for x in bb],images);r.update(family=family,scope=n,left=a,right=b,**actual_changes(aa,bb));familyrows.append(r)
   if a.startswith('RandomMean') or b.startswith('RandomMean'):continue
   for q,x,y in zip(subset,aa,bb):
    delta=x['score']-y['score']
    if abs(delta)>1e-12:cases.append(dict(family=family,scope=n,question_id=q,image_sha256=x['image_sha256'],image_path=x['image_path'],question=x['question'],answers=x['answers'],left=a,right=b,left_answer=x['answer'],right_answer=y['answer'],left_score=x['score'],right_score=y['score'],delta_score=delta,left_token_ids=x['token_ids'],right_token_ids=y['token_ids'],kind='harm' if delta<0 else 'recovery'))
  previous=0
  for j,r in enumerate(sorted(familyrows,key=lambda x:x['cluster_signflip_p'])):previous=max(previous,min(1.,(len(familyrows)-j)*r['cluster_signflip_p']));r['Holm_p_within_family']=previous;r['family_size']=len(familyrows)
  comparisons.extend(familyrows)
 csvwrite(OUT/'accuracy_summary.csv',summary);csvwrite(OUT/'paired_comparisons.csv',comparisons);csvwrite(OUT/'harm_cases.csv',[r for r in cases if r['kind']=='harm'],casefields);csvwrite(OUT/'recovery_cases.csv',[r for r in cases if r['kind']=='recovery'],casefields)
 csvwrite(OUT/'abnormal_outputs.csv',[r for r in rows if not r['eos'] or r['empty'] or r['truncated'] or r['repeated_trigram']],fields=list(rows[0]))
 phase=[r for r in summary if r['scope']==120 and r['condition'] not in STAGE2[6:] and not r['condition'].startswith('Random')];csvwrite(OUT/'phase_budget_summary.csv',phase)
 random=[r for r in summary if r['scope']==120 and (r['condition'].startswith('Random') or r['condition'] in ['Both-8','Both-16'])];csvwrite(OUT/'random_control_summary.csv',random)
 curve=[]
 for k in KS:
  c=DESIGN['curve_aliases'][str(k)];r=next(x for x in summary if x['scope']==300 and x['condition']==c);curve.append(dict(k=k,**r))
 csvwrite(OUT/'budget_curve.csv',curve)
 targets=[]
 for threshold in [.5,.75,.9]:
  achieved=[r for r in curve if r['Full_gap_recovery_ratio']>=threshold];targets.append(dict(target='gap_recovery_'+str(threshold),minimum_observed_k=min(r['k'] for r in achieved) if achieved else None,point_estimate_only=True))
 achieved=[r for r in curve if r['delta_vs_A_pp']>=-2-1e-12];targets.append(dict(target='Full_minus_2pp',minimum_observed_k=min(r['k'] for r in achieved) if achieved else None,point_estimate_only=True));csvwrite(OUT/'descriptive_targets.csv',targets)
 return summary,comparisons,curve,targets
def diagnostic(rows,by,manifest):
 records=[];features={r['question_id']:r for r in csv.DictReader((OUT/'descriptive_features.csv').open())};groups=collections.defaultdict(list);alias=DESIGN['curve_aliases']
 for r in rows:
  f=features[r['question_id']]
  for axis in ['first_word','reference_length','answer_form']:groups[(axis,f[axis],r['condition'])].append(r)
 table=[]
 for (axis,value,c),rr in groups.items():table.append(dict(axis=axis,value=value,condition=c,n=len(rr),accuracy_pct=100*np.mean([r['score'] for r in rr]),delta_vs_B_pp=100*np.mean([r['delta_score_vs_B'] for r in rr]),delta_vs_A_pp=100*np.mean([r['delta_score_vs_A'] for r in rr])))
 csvwrite(OUT/'descriptive_groups.csv',table)
 for r in manifest:
  q=r['question_id'];v=by[q]
  def save(kind,left,right,note=''):
   a,b=v[left],v[right];records.append(dict(category=kind,question_id=q,image_path=r['image_path'],image_sha256=r['image_sha256'],question=r['question'],answers=json.dumps(r['answers']),left=left,right=right,left_answer=a['answer'],right_answer=b['answer'],left_score=a['score'],right_score=b['score'],left_tokens=a['token_ids'],right_tokens=b['token_ids'],first_token_same=a['first_token']==b['first_token'],note=note))
  for lo,hi in zip(KS,KS[1:]):
   if v[alias[str(hi)]]['score']<v[alias[str(lo)]]['score']-1e-12:save('higher_budget_harms',alias[str(hi)],alias[str(lo)],f'k {lo} -> {hi}')
  for c in ['Both-8','Both-16']:
   if v[c]['score']>v['B']['score']+1e-12:save('recovery_vs_B',c,'B')
   if v[c]['score']<v['B']['score']-1e-12:save('harm_vs_B',c,'B')
  if v['A']['score']>=1-1e-12 and all(v[alias[str(k)]]['score']<1-1e-12 for k in [2,4,8,12,16,24]):save('Full_credit_but_all_main_sparse_below_full_credit','A','Both-24')
  if v['G']['first_token']==v['B']['first_token'] and v['G']['token_ids']!=v['B']['token_ids']:save('decode_changes_later_tokens','G','B')
  if r['stage2_subset']:
   for k in [2,4,8,16,32]:
    p='C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k);d='D' if k==2 else 'G' if k==32 else 'Donly-'+str(k)
    if (v[p]['score']-v['B']['score'])*(v[d]['score']-v['B']['score'])< -1e-12:save('opposite_phase_score_directions',p,d)
  for c,a in v.items():
   if a['truncated'] or a['repeated_trigram'] or (a['eos'] and a['length']==1):save('output_anomaly',c,'A')
 fields=['category','question_id','image_path','image_sha256','question','answers','left','right','left_answer','right_answer','left_score','right_score','left_tokens','right_tokens','first_token_same','note'];csvwrite(OUT/'diagnostic_cases.csv',records,fields)
 return records
def recovery_intervals(by,curve):
 qs=list(by);images=sorted({by[q]['B']['image_sha256'] for q in qs});cluster={s:[] for s in images}
 for q in qs:cluster[by[q]['B']['image_sha256']].append(q)
 values=np.array([[sum(by[q][r['condition']]['score']-by[q]['B']['score'] for q in cluster[s]) for r in curve] for s in images]);den=np.array([sum(by[q]['A']['score']-by[q]['B']['score'] for q in cluster[s]) for s in images]);rng=np.random.default_rng(SEED);results=[];zero=0
 for start in range(0,20000,200):
  weights=rng.multinomial(len(images),np.ones(len(images))/len(images),size=200);d=weights@den;ok=abs(d)>1e-12;zero+=int((~ok).sum());results.append((weights@values)[ok]/d[ok,None])
 samples=np.concatenate(results)
 for j,r in enumerate(curve):r['recovery_CI95_low'],r['recovery_CI95_high']=map(float,np.percentile(samples[:,j],[2.5,97.5]));r['zero_gap_bootstrap_draws']=zero
 csvwrite(OUT/'budget_curve.csv',curve)
def plots(summary,comparisons,curve,diagnostics):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 def error(ax,x,r,color,label=None,marker='o'):
  ax.errorbar(x,r['accuracy_pct'],yerr=[[r['accuracy_pct']-r['CI95_low_pct']],[r['CI95_high_pct']-r['accuracy_pct']]],fmt=marker,color=color,label=label,capsize=3)
 fig,ax=plt.subplots(figsize=(8,4.5))
 for r in curve:error(ax,r['k'],r,'#0072B2')
 ax.set_xticks(KS);ax.set_xlabel('Allowed layers per phase (P-1 and Decode)');ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_title('A. Same TextVQA300: measured visual re-access budgets\nk=0: B; k=2: E; k=32: native A; pointwise 95% CI');save(fig,'figure_A_accuracy_layer_budget')
 fig,ax=plt.subplots(figsize=(8,4.5))
 for r in curve:ax.errorbar(r['k'],100*r['Full_gap_recovery_ratio'],yerr=[[100*(r['Full_gap_recovery_ratio']-r['recovery_CI95_low'])],[100*(r['recovery_CI95_high']-r['Full_gap_recovery_ratio'])]],fmt='o',color='#009E73',capsize=3)
 for p in [50,75,90]:ax.axhline(p,color='gray',ls='--',alpha=.45,lw=1);ax.text(32.5,p,str(p)+'%',va='center',fontsize=8)
 ax.set_xticks(KS);ax.set_xlabel('Allowed layers per phase');ax.set_ylabel('Fraction of native Full gap recovered (%)');ax.set_title('B. Descriptive Full gap recovery; TextVQA300\nPaired ratio bootstrap 95% CI; not noninferiority');save(fig,'figure_B_gap_recovery')
 scope={r['condition']:r for r in summary if r['scope']==120};fig,axes=plt.subplots(1,2,figsize=(12,4.5));phase_data=[]
 for label,color,marker in [('P-1 only','#E69F00','^'),('Decode only','#CC79A7','s'),('Both','#0072B2','o')]:
  for j,k in enumerate([0,2,4,8,16,32]):
   c='B' if k==0 else ('C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k)) if label=='P-1 only' else ('D' if k==2 else 'G' if k==32 else 'Donly-'+str(k)) if label=='Decode only' else DESIGN['curve_aliases'][str(k)]
   r=scope[c];error(axes[0],k,r,color,label if j==0 else None,marker);phase_data.append(dict(phase=label,k=k,**r))
 for k in [2,4,8,16,32]:
  a='C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k);b='D' if k==2 else 'G' if k==32 else 'Donly-'+str(k);r=next(r for r in comparisons if r['scope']==120 and r['left']==a and r['right']==b);axes[1].errorbar(k,r['delta_pp'],yerr=[[r['delta_pp']-r['CI95_low_pp']],[r['CI95_high_pp']-r['delta_pp']]],fmt='o',capsize=3,color='#444444')
 axes[0].set_xlabel('Allowed layers');axes[0].set_ylabel('Official consensus accuracy (%)');axes[0].legend(fontsize=8);axes[0].set_xticks([0,2,4,8,16,32]);axes[1].axhline(0,color='gray',lw=1);axes[1].set_xlabel('Same allowed layer count');axes[1].set_ylabel('P-1-only minus Decode-only (pp)');axes[1].set_xticks([2,4,8,16,32]);fig.suptitle('C. Frozen120 exploratory subset; pointwise paired 95% CI',y=1.03);save(fig,'figure_C_phase_contribution');csvwrite(OUT/'figure_C_plot_data.csv',phase_data)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,k in zip(axes,[8,16]):
  names=['Both-'+str(k)]+[f'Random-{k}-s{j+1}' for j in range(3)]+['RandomMean-'+str(k)]
  for j,c in enumerate(names):error(ax,j,scope[c],'#0072B2' if j==0 else '#999999' if j<4 else '#D55E00');ax.text(j,scope[c]['CI95_high_pct']+1,f"{scope[c]['accuracy_pct']:.1f}",ha='center',fontsize=8)
  ax.set_xticks(range(5),['Nested','Seed1','Seed2','Seed3','Seed mean']);ax.set_ylim(0,max(scope[c]['CI95_high_pct'] for c in names)+7);ax.set_title(str(k)+' layers / phase');ax.set_ylabel('Official consensus accuracy (%)')
 fig.suptitle('D. Matched layer budgets on frozen120; seed mean averaged within question',y=1.03);save(fig,'figure_D_random_layers')
 first=[r for r in summary if r['scope']==300];edgeplot=[]
 for r in first:
  if r['condition'] not in 'ABCDE'+''.join(STAGE1):continue
  edgeplot.append(r)
 fig,ax=plt.subplots(figsize=(9,5))
 for j,r in enumerate(first):
  ax.scatter(r['mean_actual_edge_ratio']*100,r['accuracy_pct'],color='#0072B2' if r['condition'].startswith('Both') else '#D55E00',s=45);offset=(5,6) if j%2 else (5,-12);ax.annotate(r['condition'],(100*r['mean_actual_edge_ratio'],r['accuracy_pct']),xytext=offset,textcoords='offset points',fontsize=8)
 ax.set_xlabel('Mean actual allowed answer-query to Image edges (%)');ax.set_ylabel('Official consensus accuracy (%)');ax.set_title('E. Actual query-weighted edge permissions; TextVQA300\nCondition-dependent lengths; dense execution retained');save(fig,'figure_E_actual_edges');csvwrite(OUT/'figure_E_plot_data.csv',first)
 categories=['recovery_vs_B','harm_vs_B','higher_budget_harms','Full_credit_but_all_main_sparse_below_full_credit','decode_changes_later_tokens','output_anomaly'];titles=['Recovery vs B','Harm vs B','More layers can harm','Sparse misses native Full','Decode changes continuation','Output anomaly'];chosen=[]
 fig,axes=plt.subplots(2,3,figsize=(15,12))
 for ax,category,title in zip(axes.flat,categories,titles):
  case=next((r for r in diagnostics if r['category']==category),None);ax.axis('off');ax.set_title(title,fontsize=11)
  if case is None:ax.text(.05,.5,'No case observed under frozen rule',transform=ax.transAxes);continue
  chosen.append(case)
  with Image.open(case['image_path']) as im:ax.imshow(im.convert('RGB'),extent=[0,1,.42,1],aspect='auto')
  ax.set_xlim(0,1);ax.set_ylim(0,1);gold=collections.Counter(normalize(x) for x in json.loads(case['answers'])).most_common(1)[0][0];text='ID '+case['question_id']+'\n'+textwrap.fill(case['question'],48)+'\nReference (modal): '+gold+'\n'
  for key in ['right','left']:
   answer=case[key+'_answer'];shown=answer if len(answer)<=110 else answer[:107]+'...';text+=case[key]+': '+shown+f"  [score {case[key+'_score']:.1f}]\n"
  text+=case['note'];ax.text(.01,.4,textwrap.fill(text,60,replace_whitespace=False),va='top',transform=ax.transAxes,fontsize=8)
 fig.suptitle('F. First frozen-manifest case per category; all failures retained in CSV',y=1.01);save(fig,'figure_F_cases');write(OUT/'figure_F_case_selection.json',dict(rule='First manifest-order case per frozen category; no GPU conditions changed',cases=chosen))
def report(summary,comparisons,curve,targets,diagnostics):
 first={r['condition']:r for r in summary if r['scope']==300};second={r['condition']:r for r in summary if r['scope']==120};text=['# TextVQA Visual Re-access Budget：探索性质量—回看预算','',f'冻结基线 `{BASE}`；同一批无OCR TextVQA300，LLaVA-1.5-7B固定revision，FP16/Eager、Greedy32、logits_to_keep=1，官方TextVQA/MMF十人留一共识软评分。原生R和其他非目标路径完全保持原生。本批历史已研究，不能称为独立确认集。','', '## 第一阶段：完整300题','', '| 条件 | P−1/Decode开放层数 | Accuracy % | 相对B pp | 相对A pp | 截断/重复 |','|---|---:|---:|---:|---:|---:|']
 for c in ['A','B','C','D','E','F','G']+['Both-'+str(k) for k in [4,8,12,16,24]]:
  r=first[c];d=DESIGN['conditions'][c];text.append(f"| {c} | {len(d['Pminus1'])}/{len(d['Decode'])} | {r['accuracy_pct']:.3f} | {r['delta_vs_B_pp']:+.3f} | {r['delta_vs_A_pp']:+.3f} | {r['truncated']}/{r['repeated']} |")
 text+=['',f"仅P−1全32层（F）：{first['F']['accuracy_pct']:.3f}%；仅Decode全32层（G）：{first['G']['accuracy_pct']:.3f}%。真正Native Full（A）为{first['A']['accuracy_pct']:.3f}%。A为无任何实验Mask的原生行为，CPU/GPU均验证32层全开放与A严格一致。",'', '## Both预算曲线与描述性目标','', '| k | Accuracy % | Full Gap恢复比例 % | 相对A pp |','|---:|---:|---:|---:|']
 for r in curve:text.append(f"| {r['k']} | {r['accuracy_pct']:.3f} | {100*r['Full_gap_recovery_ratio']:.2f} | {r['delta_vs_A_pp']:+.3f} |")
 text+=['', '层集合严格按最远距离添加、并列取较小编号，全部嵌套且GPU前冻结；实际集合见layer_sets.json。没有按得分改层，也没有平滑或插值。恢复比例为(条件−B)/(A−B)，不同于Accuracy；恢复比例区间在同一Image簇重采样中联合计算分子和分母，保留超出0–100%的可能取值。']
 for r in targets:text.append(f"{r['target']}：最小观察到k={r['minimum_observed_k']}；仅在本轮测试的离散预算中成立，点估计描述标准，不是非劣效检验。")
 text+=['', '## 第二阶段：冻结120题子集','', '子集仅用固定seed、Question ID和Image SHA抽取，两卡各60题。所有对照仍是历史300题的一部分。三个随机种子先逐题平均后再统计，不能当成360道独立题。','', '| k | P−1-only % | Decode-only % | Both % |','|---:|---:|---:|---:|']
 for k in [2,4,8,16,32]:
  p='C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k);d='D' if k==2 else 'G' if k==32 else 'Donly-'+str(k);b=DESIGN['curve_aliases'][str(k)];text.append(f"| {k} | {second[p]['accuracy_pct']:.3f} | {second[d]['accuracy_pct']:.3f} | {second[b]['accuracy_pct']:.3f} |")
 text+=['', '| k | Nested % | 随机seed1/2/3 % | 随机均值 % | Nested−随机均值 pp [95%CI] / Holm p |','|---:|---:|---:|---:|---|']
 for k in [8,16]:
  r=next(r for r in comparisons if r['scope']==120 and r['left']=='Both-'+str(k) and r['right']=='RandomMean-'+str(k));seed='/'.join(f"{second[f'Random-{k}-s{j+1}']['accuracy_pct']:.3f}" for j in range(3));text.append(f"| {k} | {second['Both-'+str(k)]['accuracy_pct']:.3f} | {seed} | {second['RandomMean-'+str(k)]['accuracy_pct']:.3f} | {r['delta_pp']:+.3f} [{r['CI95_low_pp']:+.3f}, {r['CI95_high_pp']:+.3f}] / {r['Holm_p_within_family']:.5f} |")
 text+=['', '这些随机对照只检验8/16层预算下固定嵌套集合的表现；没有随机两层对照，因此不能直接判断L14/L19是否优于随机两层，更不能由三个种子证明最优选择机制。','', '## 统计与阶段归因','', f"两阶段分别预声明{len(read(OUT/'protocol.json')['comparisons']['stage1'])}和{len(read(OUT/'protocol.json')['comparisons']['stage2'])}个比较，各自做Holm校正；完整比较而非仅显著项保存在paired_comparisons.csv。Image SHA整簇配对Bootstrap20,000次，seed={SEED}，普通逐点95%CI不是同时区间；双侧整簇符号翻转20,000次、plus-one p。没有定义或证明非劣效，也不能把新p值当作独立外部确认。",'', '同等层数Ponly−Donly的逐题配对差及区间见Figure C与完整CSV；不能仅根据点估计排名断言阶段更重要。所有Donly/F/G阶段边界通过首Token原始Logits SHA验证：G和所有Donly的首Token必须与B相同，F首Token必须与A相同，Ponly-k首Token必须与对应Both-k相同。只有后续Decode发生变化时，答案变化可在保持首Token不变下发生。该检查不等于确定某个视觉概念的唯一存储位置。','', '## 负面、非单调和异常记录','']
 counter=collections.Counter(r['category'] for r in diagnostics)
 for name,n in sorted(counter.items()):text.append(f'{name}：{n}条案例记录（同题可因多个预算出现多条，不作为独立样本数）。')
 text+=['', '全部预声明比较的Score损害保存在harm_cases.csv，改善保存在recovery_cases.csv；逐题所有回答、Token序列、标注与检查点SHA在per_question.csv。异常保存在abnormal_outputs.csv，主评分没有删除截断、重复、提前EOS或非EOS输出。Full满分而所有主稀疏点未得满分不等于所有点都零分，diagnostic_cases.csv明确保留软评分。','', '问题首词、参考答案长度、数字/字母/其他形式在GPU前冻结，只作词面代理，不能未经人工核实认定为OCR因果属性。描述性分组见descriptive_groups.csv；既不据此改预算，也不追加验证。','', '## Layer Budget、真实连接预算及资源','', 'Layer Budget分别给出P−1与Decode开放层数；视觉Edge Budget按真实查询数统计：P−1为1个查询，Decode为实际前向次数−1，含EOS预测。每阶段允许层数×原始Image Key数×该阶段查询数，再乘32个Head；总分母为32层×全部真实答案查询数×576原始Image Key×32个Head。visual_edge_budget.csv保留逐题计数与比例。R开放比例始终100%，不能混用R Receiver预算。','', '实际生成长度依赖条件，因此观察到的Query-Key预算比例也依赖各自轨迹；其不是预先固定的延迟或FLOPs预算。实际Image Mass逐层记录：新条件本轮测量，A/C/D/E审计复用历史实际Mass，B由历史禁止概率严格0推得Mass为0，明确标记来源。高Mass仅表示分配，不等于因果必要性。','', '每卡独立一份模型，按Image SHA分片，同题全部条件在同一Worker。两卡完成全部Smoke并对同题所有条件的完整有效Logits序列SHA逐位一致后才启动正式矩阵；所有32层Mask非目标条目不变、禁用实际概率严格0，保持独立真实KV、视觉特征、MLP/Residual/QKV及正常/异常Hook清理。阶段1两卡完成后进入冻结阶段2；原子检查点支持同源码/协议/输入哈希的精确恢复。','', '完整执行所有QKV、Attention、MLP、Residual和KV更新；本轮不声称物理FLOPs、KV、显存或时延收益，不进行训练、层/Token追加搜索或动态策略优化。GPU耗时和峰值显存见workers/*/gpu_run_complete.json，退出审计见gpu_release.json。原始模型和图片在服务器资源目录，Git仅保存路径、SHA、生成、统计和图表。','', '## 图表','']
 for name,title in [('A_accuracy_layer_budget','Accuracy与Layer预算'),('B_gap_recovery','Full Gap恢复比例'),('C_phase_contribution','阶段贡献与配对区间'),('D_random_layers','预算匹配随机层'),('E_actual_edges','真实Query-Key连接预算'),('F_cases','恢复与损害案例')]:text+=['!['+title+'](figure_'+name+'.png)','']
 text+=['A–E为矢量PDF，F保留原始照片栅格并以矢量绘制文字；全部PNG/PDF及绘图CSV随交付保存，analyze.py可重现。达到高质量的点只能作为未来新数据独立验证候选，本轮结束后停止。']
 (OUT/'report_zh.md').write_text('\n'.join(text).rstrip()+'\n',encoding='utf-8')
def main():
 rows,by,manifest=load();summary,comparisons,curve,targets=statistics(rows,by);recovery_intervals(by,curve);cases=diagnostic(rows,by,manifest);plots(summary,comparisons,curve,cases);report(summary,comparisons,curve,targets,cases)
 write(OUT/'analysis_validation.json',dict(passed=True,UTC=utc(),canonical_records=5040,new_checkpoints=3540,historical_records=1500,all_phase_first_logit_invariances_exact=True,ordinary_CI_and_Holm_separate=True,all_negative_and_abnormal_outputs_retained=True))
 print(json.dumps(dict(stage1=[r for r in summary if r['scope']==300],targets=targets,random=[r for r in comparisons if 'RandomMean' in r['right']]),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
