"""CPU-only post hoc OCR annotation/effect analysis of frozen TextVQA outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import pathlib,json,csv,hashlib,datetime,collections,sys,re,contextlib,subprocess,shutil,math
import numpy as np
from tokenizers import Tokenizer
O=pathlib.Path(__file__).resolve().parent;ROOT=O.parents[1];W=ROOT.parent
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration'
MODEL=pathlib.Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
SEED=20261105;REPS=20000
GROUPS=['OCR-Direct','OCR-Reasoning','OCR-Combined','Non-OCR Visual','Ambiguous']
CONDS=['A','B','C','D','E','F','G','Both-4','Both-8','Both-12','Both-16','Both-24']
EFFECTS=[('Full_reaccess_effect','A','B'),('Pminus1_effect','F','B'),('Decode_effect','G','B'),('Both16_effect','Both-16','B'),('Pminus1_minus_Decode','F','G')]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rr,fields=None):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rr[0]));w.writeheader();w.writerows(rr)
def loadcsv(p):return list(csv.DictReader(p.open()))
def quant(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return tuple(map(float,np.percentile(x,[2.5,97.5]))) if len(x) else (None,None)
def flag(x):return x in [True,'True','true','1',1]
def holm(rr,p='raw_p'):
 prev=0
 for j,r in enumerate(sorted(rr,key=lambda r:r[p])):
  prev=max(prev,min(1.,(len(rr)-j)*r[p]));r['Holm_p']=prev;r['family_size']=len(rr)
def modal(aa):
 counts=collections.Counter(aa);return max(aa,key=lambda a:counts[a])
def firstword(s):
 m=re.search(r'[A-Za-z]+',s);return m.group().lower() if m else ''
def divergence(a,b):
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return i+1,'token_mismatch'
 if len(a)!=len(b):return min(len(a),len(b))+1,'prefix_extension'
 return None,'unchanged'
def cluster_boot(ids,values,rng=None):
 rng=np.random.default_rng(SEED) if rng is None else rng
 images=sorted(set(ids));sums=np.array([np.asarray(values)[np.array(ids)==im].sum(axis=0) for im in images]);counts=np.array([ids.count(im) for im in images]);out=[]
 for start in range(0,REPS,500):
  ww=rng.multinomial(len(images),np.ones(len(images))/len(images),size=min(500,REPS-start));out.append((ww@sums)/(ww@counts)[:,None])
 return np.concatenate(out)
def main():
 frozen=read(O/'classification_freeze.json');assert sha(O/'ocr_labels.csv')==frozen['labels_sha256'] and sha(O/'classification_evidence.csv')==frozen['evidence_sha256'] and sha(O/'classification_protocol.md')==frozen['protocol_sha256']
 labels=loadcsv(O/'ocr_labels.csv');blind=read(O/'blind_inputs.json');assert len(labels)==len(blind)==300
 evidence={r['question_id']:r for r in loadcsv(O/'classification_evidence.csv')};ocr={r['question_id']:r for r in read(O/'aligned_ocr_annotations.json')};lb={r['question_id']:r for r in labels};qs=[r['question_id'] for r in blind]
 history=loadcsv(H/'per_question.csv');assert len(history)==5040
 sys.path.insert(0,str(ROOT));from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
 scorer=TextVQAAccuracyEvaluator();by=collections.defaultdict(dict);diff=[];check=[]
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
  for r in history:
   v=scorer.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=json.loads(r['answers']))]);diff.append(abs(v-float(r['score'])));assert diff[-1]<1e-12
   cp=ROOT/r['checkpoint']
   if not cp.exists():cp=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget/vlm_research')/r['checkpoint']
   assert sha(cp)==r['checkpoint_sha256'];g=read(cp);assert g['generation']['token_ids']==json.loads(r['token_ids'])
   r['score']=v;r['ids']=json.loads(r['token_ids']);by[r['question_id']][r['condition']]=r
   check.append(dict(question_id=r['question_id'],condition=r['condition'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],score=v,original_score=float(r['score'])))
 assert set(qs)==set(by) and all(len(by[q]) in [12,24] for q in qs)
 tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'));toksha=sha(MODEL/'tokenizer.json')
 original_content={(r['question_id'],r['condition']):r for r in loadcsv(H/'first_token_content_audit.csv')}
 tokeninfo={};tokens=[];features=[];divs=[]
 for b in blind:
  q=b['question_id'];a=by[q];cat=lb[q]['category'];gold=modal(json.loads(a['A']['answers']));refwords=gold.split();refids=tok.encode(gold,add_special_tokens=False).ids
  f=dict(question_id=q,image_sha256=b['image_sha256'],category=cat,question_words=len(b['question'].split()),question_length_group='short<=10' if len(b['question'].split())<=10 else 'long>10',modal_reference=gold,reference_words=len(refwords),reference_length_group='single_word' if len(refwords)<=1 else 'multi_word',reference_token_count=len(refids),reference_token_group='single_token' if len(refids)<=1 else 'multi_token',numeric_group='contains_digit' if re.search(r'[0-9]',gold) else 'no_digit',first_word_group='what' if b['question'].lower().lstrip().startswith('what ') else 'other',OCR_text_count=int(evidence[q]['OCR_legible_count']),OCR_density_group='<=20_legible' if int(evidence[q]['OCR_legible_count'])<=20 else '>20_legible' if ocr[q]['TextOCR_matched'] else 'unmatched')
  if not ocr[q]['TextOCR_matched']:f['OCR_density_group']='unmatched'
  features.append(f)
  for c,r in a.items():
   ids=r['ids'];prefixes=[tok.decode(ids[:j+1],skip_special_tokens=True) for j in range(len(ids))];idx=next((j for j,s in enumerate(prefixes) if s.strip()),None);content=ids[idx] if idx is not None else None;prefix=prefixes[idx] if idx is not None else ''
   old=original_content[(q,c)];assert (int(old['first_content_subword_id']) if old['first_content_subword_id'] else None)==content
   info=dict(question_id=q,condition=c,category=cat,first_token_id=ids[0],first_token_piece=tok.decode(ids[:1],skip_special_tokens=False),first_content_position_1based=idx+1 if idx is not None else None,first_content_subword_id=content,first_content_prefix=prefix,first_english_word_proxy=firstword(r['answer']),generated_tokens_including_EOS=len(ids),generated_tokens_excluding_EOS=len(ids)-int(ids[-1]==2),generated_whitespace_words=len(r['answer'].split()),first_complete_word_is_text_proxy=True)
   tokeninfo[(q,c)]=info;tokens.append(info)
  gb=a['G'];bb=a['B'];pos,kind=divergence(gb['ids'],bb['ids']);assert gb['ids'][0]==bb['ids'][0] and gb['first_logits_sha256']==bb['first_logits_sha256']
  def changedword(r):
   if pos is None:return ''
   if pos>len(r['ids']) or r['ids'][pos-1] in [1,2]:return ''
   piece=tok.decode(r['ids'][pos-1:pos],skip_special_tokens=True)
   if not re.search('[A-Za-z0-9]',piece):return ''
   text=tok.decode(r['ids'],skip_special_tokens=True);before=tok.decode(r['ids'][:pos-1],skip_special_tokens=True);offset=min(len(before),max(0,len(text)-1));spans=list(re.finditer(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*",text));m=next((m for m in spans if m.end()>offset),None);return m.group() if m else ''
  gw,bw=changedword(gb),changedword(bb)
  strings=[x['text'].lower() for x in ocr[q]['TextOCR_words'] if x['text']!='.'];norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower());sw={norm(s) for s in strings if norm(s)}
  gmatch=bool(norm(gw)) and norm(gw) in sw;bmatch=bool(norm(bw)) and norm(bw) in sw
  proxy='unchanged' if pos is None else 'divergent_word_matches_official_OCR' if gmatch or bmatch else 'unresolved_no_lexical_match' if ocr[q]['TextOCR_matched'] else 'unresolved_no_official_OCR'
  divs.append(dict(question_id=q,image_sha256=b['image_sha256'],category=cat,question=b['question'],modal_reference=gold,B_answer=bb['answer'],G_answer=gb['answer'],B_score=bb['score'],G_score=gb['score'],delta_score=gb['score']-bb['score'],B_tokens=json.dumps(bb['ids']),G_tokens=json.dumps(gb['ids']),first_logits_exact=gb['first_logits_sha256']==bb['first_logits_sha256'],first_token_same=gb['ids'][0]==bb['ids'][0],first_content_subword_same=tokeninfo[(q,'G')]['first_content_subword_id']==tokeninfo[(q,'B')]['first_content_subword_id'],first_english_word_same=tokeninfo[(q,'G')]['first_english_word_proxy']==tokeninfo[(q,'B')]['first_english_word_proxy'],first_divergence_position_1based=pos,divergence_kind=kind,B_token_at_divergence=bb['ids'][pos-1] if pos and pos<=len(bb['ids']) else None,G_token_at_divergence=gb['ids'][pos-1] if pos and pos<=len(gb['ids']) else None,B_divergent_word_proxy=bw,G_divergent_word_proxy=gw,divergence_at_EOS=bool(pos and ((pos<=len(bb['ids']) and bb['ids'][pos-1]==2) or (pos<=len(gb['ids']) and gb['ids'][pos-1]==2))),B_word_matches_OCR=bmatch,G_word_matches_OCR=gmatch,OCR_fragment_proxy=proxy,proxy_is_not_causal_grounding=True,B_length=len(bb['ids']),G_length=len(gb['ids']),reference_words=len(refwords),reference_token_count=len(refids),truncated_B=flag(bb['truncated']),truncated_G=flag(gb['truncated'])))
 csvwrite(O/'token_position_audit.csv',tokens);csvwrite(O/'decode_divergence.csv',divs);csvwrite(O/'descriptive_features.csv',features);csvwrite(O/'historical_checkpoint_audit.csv',check)
 groups={g:[q for q in qs if lb[q]['category']==g or (g=='OCR-Combined' and lb[q]['category'].startswith('OCR-'))] for g in GROUPS};groups['All300']=qs
 images=sorted({by[q]['B']['image_sha256'] for q in qs});rng=np.random.default_rng(SEED);weights=np.concatenate([rng.multinomial(len(images),np.ones(len(images))/len(images),size=500) for _ in range(REPS//500)])
 def bootgroup(qsub,conditions=CONDS):
  sums=np.zeros((len(images),len(conditions)));counts=np.zeros(len(images));imidx={im:j for j,im in enumerate(images)}
  for q in qsub:
   i=imidx[by[q]['B']['image_sha256']];counts[i]+=1;sums[i]+=[by[q][c]['score'] for c in conditions]
  den=weights@counts;valid=den>0;out=np.full((REPS,len(conditions)),np.nan);out[valid]=(weights[valid]@sums)/den[valid,None];return out,int((~valid).sum())
 boot={};summary=[];effects=[];all_effects={};descriptive=[]
 def flags_summary(rr):
  return {k:sum(flag(r[k]) for r in rr) for k in ['eos','truncated','repeated_trigram','empty']}
 for g,qsub in groups.items():
  bs,invalid=bootgroup(qsub);boot[g]=bs
  for j,c in enumerate(CONDS):
   vals=np.array([by[q][c]['score'] for q in qsub]);acc=100*vals.mean() if len(vals) else None;lo,hi=quant(100*bs[:,j]);bb=np.array([by[q]['B']['score'] for q in qsub]);aa=np.array([by[q]['A']['score'] for q in qsub]);rr=[by[q][c] for q in qsub];n=len(qsub)
   z=dict(group=g,condition=c,n_questions=n,n_images=len({by[q]['B']['image_sha256'] for q in qsub}),accuracy_pct=acc,CI95_low_pct=lo,CI95_high_pct=hi,delta_vs_B_pp=100*float((vals-bb).mean()) if n else None,delta_vs_A_pp=100*float((vals-aa).mean()) if n else None,score_up_vs_B=int((vals>bb+1e-12).sum()),score_down_vs_B=int((vals<bb-1e-12).sum()),answer_change_count=sum(by[q][c]['answer']!=by[q]['B']['answer'] for q in qsub),token_sequence_change_count=sum(by[q][c]['ids']!=by[q]['B']['ids'] for q in qsub),first_token_change_count=sum(by[q][c]['ids'][0]!=by[q]['B']['ids'][0] for q in qsub),first_content_subword_change_count=sum(tokeninfo[(q,c)]['first_content_subword_id']!=tokeninfo[(q,'B')]['first_content_subword_id'] for q in qsub),first_english_word_change_count=sum(tokeninfo[(q,c)]['first_english_word_proxy']!=tokeninfo[(q,'B')]['first_english_word_proxy'] for q in qsub),mean_generated_tokens=np.mean([len(r['ids']) for r in rr]).item() if n else None,bootstrap_empty_group_draws=invalid,**flags_summary(rr))
   for key in ['answer_change','token_sequence_change','first_token_change','first_content_subword_change','first_english_word_change']:z[key+'_ratio']=z[key+'_count']/n if n else None
   summary.append(z)
  for name,left,right in EFFECTS:
   d=np.array([by[q][left]['score']-by[q][right]['score'] for q in qsub]);samples=100*(bs[:,CONDS.index(left)]-bs[:,CONDS.index(right)]);ci=quant(samples);point=100*d.mean() if len(d) else None;ids=[by[q]['B']['image_sha256'] for q in qsub];cl=np.array([d[np.array(ids)==im].sum() for im in sorted(set(ids))]);prng=np.random.default_rng(SEED+11);ge=0
   if len(cl):
    for start in range(0,REPS,500):signs=prng.integers(0,2,size=(500,len(cl)))*2-1;ge+=int((abs(signs@cl)>=abs(cl.sum())-1e-12).sum())
   r=dict(group=g,effect=name,left=left,right=right,n_questions=len(qsub),n_images=len(cl),delta_pp=point,CI95_low_pp=ci[0],CI95_high_pp=ci[1],score_up=int((d>1e-12).sum()),score_down=int((d< -1e-12).sum()),raw_p=(ge+1)/(REPS+1) if len(cl) else None,CI_type='ordinary pointwise cluster paired bootstrap',p_method='cluster paired sign-flip; plus-one',small_group=len(qsub)<30)
   all_effects[(g,name)]=(r,samples)
   if g!='All300' and len(qsub):effects.append(r)
 holm(effects);csvwrite(O/'paired_ocr_effects.csv',effects)
 interactions=[]
 for g in ['OCR-Combined','OCR-Direct','OCR-Reasoning']:
  for name,left,right in EFFECTS:
   r,x=all_effects[(g,name)];n,y=all_effects[('Non-OCR Visual',name)];point=r['delta_pp']-n['delta_pp'];bs=x-y;lo,hi=quant(bs);valid=bs[np.isfinite(bs)];p=(int((abs(valid-point)>=abs(point)-1e-12).sum())+1)/(len(valid)+1)
   interactions.append(dict(OCR_group=g,reference_group='Non-OCR Visual',effect=name,n_OCR=r['n_questions'],n_NonOCR=n['n_questions'],OCR_effect_pp=r['delta_pp'],NonOCR_effect_pp=n['delta_pp'],interaction_pp=point,CI95_low_pp=lo,CI95_high_pp=hi,raw_p=p,p_method='centered whole-image-cluster bootstrap null; plus-one',valid_draws=len(valid),interpretation='observed-label effect heterogeneity; not OCR causal intervention'))
 holm(interactions);csvwrite(O/'ocr_vs_nonocr_interaction.csv',interactions);csvwrite(O/'ocr_group_accuracy.csv',summary)
 recovery=[]
 for g,qsub in groups.items():
  bs=boot[g];den=bs[:,CONDS.index('A')]-bs[:,CONDS.index('B')];zero=int((abs(den)<1e-12).sum());negative=int((den< -1e-12).sum());valid=np.isfinite(den)&(abs(den)>1e-12);pointden=np.mean([by[q]['A']['score']-by[q]['B']['score'] for q in qsub])
  for k,c in zip([0,2,4,8,12,16,24,32],['B','E','Both-4','Both-8','Both-12','Both-16','Both-24','A']):
   numerator=np.mean([by[q][c]['score']-by[q]['B']['score'] for q in qsub]);ratio=numerator/pointden if abs(pointden)>1e-12 else None;ratioCI=quant((bs[:,CONDS.index(c)]-bs[:,CONDS.index('B')])[valid]/den[valid]) if zero==0 and negative==0 else (None,None)
   recovery.append(dict(group=g,n=len(qsub),budget_layers=k,condition=c,Full_gap_recovery_ratio=ratio,CI95_low_ratio=ratioCI[0],CI95_high_ratio=ratioCI[1],zero_denominator_bootstrap_draws=zero,negative_denominator_draws=negative,ratio_interval_note='NA if any zero/negative sampled Full gap; ratio remains descriptive'))
 csvwrite(O/'group_budget_recovery.csv',recovery)
 # Retain every stage2 record separately, never silently pool the different denominator.
 subsummary=[]
 for g,qsub in groups.items():
  for c in sorted(set(r['condition'] for r in history)-set(CONDS)):
   use=[q for q in qsub if c in by[q]]
   if not use:continue
   vv=np.array([[by[q][c]['score'],by[q]['B']['score'],by[q]['A']['score']] for q in use]);bs=cluster_boot([by[q]['B']['image_sha256'] for q in use],vv);lo,hi=quant(100*bs[:,0]);subsummary.append(dict(scope='frozen120_only',group=g,condition=c,n=len(use),accuracy_pct=100*vv[:,0].mean(),CI95_low_pct=lo,CI95_high_pct=hi,delta_vs_B_pp=100*(vv[:,0]-vv[:,1]).mean(),delta_vs_A_pp=100*(vv[:,0]-vv[:,2]).mean()))
 csvwrite(O/'stage2_group_accuracy.csv',subsummary)
 # Prespecified formal proxies; report every stratum, including empty cells.
 feat={r['question_id']:r for r in features};strata=[]
 axes=dict(question_length_group=['short<=10','long>10'],reference_length_group=['single_word','multi_word'],reference_token_group=['single_token','multi_token'],numeric_group=['contains_digit','no_digit'],first_word_group=['what','other'],OCR_density_group=['<=20_legible','>20_legible','unmatched'])
 for axis,levels in axes.items():
  for g,qsub in groups.items():
   for level in levels:
    use=[q for q in qsub if feat[q][axis]==level];v=np.array([[by[q][l]['score']-by[q][r]['score'] for name,l,r in EFFECTS] for q in use]);bs=cluster_boot([by[q]['B']['image_sha256'] for q in use],v) if use else np.full((REPS,5),np.nan)
    for j,(name,left,right) in enumerate(EFFECTS):
     lo,hi=quant(100*bs[:,j]);strata.append(dict(axis=axis,value=level,group=g,effect=name,n_questions=len(use),n_images=len({by[q]['B']['image_sha256'] for q in use}),delta_pp=100*v[:,j].mean() if use else None,CI95_low_pp=lo,CI95_high_pp=hi,inference='descriptive unadjusted; no subgroup selection',A_accuracy_pct=100*np.mean([by[q]['A']['score'] for q in use]) if use else None,B_accuracy_pct=100*np.mean([by[q]['B']['score'] for q in use]) if use else None,F_accuracy_pct=100*np.mean([by[q]['F']['score'] for q in use]) if use else None,G_accuracy_pct=100*np.mean([by[q]['G']['score'] for q in use]) if use else None))
 csvwrite(O/'confound_stratified_effects.csv',strata);csvwrite(O/'answer_length_analysis.csv',[r for r in strata if r['axis'] in ['reference_length_group','reference_token_group']])
 # Explicit length interaction; descriptive only, no extra confirmatory family.
 lint=[]
 for g,qsub in groups.items():
  for axis,short,long in [('reference_length_group','single_word','multi_word'),('reference_token_group','single_token','multi_token')]:
   use1=[q for q in qsub if feat[q][axis]==short];use2=[q for q in qsub if feat[q][axis]==long]
   if not use1 or not use2:continue
   u,_=bootgroup(use1,['G','B']);v,_=bootgroup(use2,['G','B']);s=100*((v[:,0]-v[:,1])-(u[:,0]-u[:,1]));pt=100*(np.mean([by[q]['G']['score']-by[q]['B']['score'] for q in use2])-np.mean([by[q]['G']['score']-by[q]['B']['score'] for q in use1]));lo,hi=quant(s);lint.append(dict(group=g,axis=axis,n_short=len(use1),n_long=len(use2),Decode_gain_long_minus_short_pp=pt,CI95_low_pp=lo,CI95_high_pp=hi,inference='descriptive length association, not causal decoding benefit per token'))
 csvwrite(O/'decode_length_interaction.csv',lint)
 dsummary=[]
 for g,qsub in groups.items():
  dd=[r for r in divs if r['question_id'] in qsub];n=len(dd);v=np.array([[float(not r['first_token_same']),float(not r['first_content_subword_same']),float(not r['first_english_word_same']),float(r['divergence_kind']!='unchanged'),float(r['OCR_fragment_proxy']=='divergent_word_matches_official_OCR')] for r in dd]);bs=cluster_boot([r['image_sha256'] for r in dd],v)
  for j,key in enumerate(['first_token_changed','first_content_subword_changed','first_english_word_changed','any_token_divergence','divergent_word_OCR_lexical_match']):
   lo,hi=quant(100*bs[:,j]);dsummary.append(dict(group=g,metric=key,n=n,count=int(v[:,j].sum()),percentage=100*v[:,j].mean(),CI95_low_pct=lo,CI95_high_pct=hi))
 csvwrite(O/'decode_divergence_summary.csv',dsummary)
 negatives=[];allpairs=[(c,base) for c in CONDS for base in ['A','B'] if c!=base]+[('F','G'),('G','F')]
 curve=['B','E','Both-4','Both-8','Both-12','Both-16','Both-24','A']
 allpairs+=list(zip(curve[1:],curve[:-1]))
 for q in qs:
  for left,right in dict.fromkeys(allpairs):
   x,y=by[q][left],by[q][right]
   if x['score']>=y['score']-1e-12:continue
   pos,kind=divergence(x['ids'],y['ids']);negatives.append(dict(question_id=q,category=lb[q]['category'],question=lb[q]['question'],image_sha256=x['image_sha256'],left=left,right=right,left_answer=x['answer'],right_answer=y['answer'],left_score=x['score'],right_score=y['score'],delta_score=x['score']-y['score'],left_tokens=json.dumps(x['ids']),right_tokens=json.dumps(y['ids']),first_divergence_position_1based=pos,kind=kind,answers=x['answers']))
 csvwrite(O/'negative_cases.csv',negatives)
 case_rules=[('OCR_Direct_A_full_B_zero',lambda q:lb[q]['category']=='OCR-Direct' and by[q]['A']['score']==1 and by[q]['B']['score']==0),('OCR_Direct_B_full',lambda q:lb[q]['category']=='OCR-Direct' and by[q]['B']['score']==1),('OCR_Direct_F_recovers_G_not',lambda q:lb[q]['category']=='OCR-Direct' and by[q]['F']['score']>by[q]['B']['score'] and by[q]['G']['score']<=by[q]['B']['score']),('OCR_Direct_G_recovers_F_not',lambda q:lb[q]['category']=='OCR-Direct' and by[q]['G']['score']>by[q]['B']['score'] and by[q]['F']['score']<=by[q]['B']['score']),('NonOCR_reaccess_gain',lambda q:lb[q]['category']=='Non-OCR Visual' and by[q]['A']['score']>by[q]['B']['score']),('Higher_budget_harms',lambda q:any(by[q][h]['score']<by[q][l]['score']-1e-12 for l,h in zip(curve,curve[1:]))),('OCR_Reasoning_Both24_below_A',lambda q:lb[q]['category']=='OCR-Reasoning' and by[q]['Both-24']['score']<by[q]['A']['score'])]
 cases=[]
 from PIL import Image
 for name,predicate in case_rules:
  chosen=next((q for q in qs if predicate(q)),None)
  if chosen is None:cases.append(dict(category=name,question_id=None,status='No case satisfying frozen criterion'));continue
  b=next(x for x in blind if x['question_id']==chosen);q=chosen;answerrows=[]
  for c in ['A','B','F','G','E','Both-4','Both-8','Both-12','Both-16','Both-24']:
   rr=by[q][c];pos,kind=divergence(rr['ids'],by[q]['B']['ids']);answerrows.append(dict(condition=c,answer=rr['answer'],score=rr['score'],tokens=rr['ids'],first_divergence_vs_B_1based=pos,divergence_kind=kind))
  thumb=O/'case_images'/(q+'.jpg');thumb.parent.mkdir(exist_ok=True)
  with Image.open(b['image_path']) as im:im=im.convert('RGB');im.thumbnail((900,700));im.save(thumb,quality=90)
  cases.append(dict(category=name,question_id=q,status='first frozen blind-ID case',classification=lb[q]['category'],rationale=evidence[q]['rationale'],question=b['question'],image_path=b['image_path'],image_sha256=b['image_sha256'],thumbnail='case_images/'+q+'.jpg',answers=json.loads(by[q]['A']['answers']),predictions=answerrows))
 write(O/'case_selection.json',dict(rule='First numeric QuestionID under protocol criteria; labels frozen first; all eligible losses retained in negative_cases.csv',cases=cases))
 # Outcome-free labels remain byte-identical throughout every analysis stage.
 assert sha(O/'ocr_labels.csv')==frozen['labels_sha256'] and sha(O/'classification_evidence.csv')==frozen['evidence_sha256']
 write(O/'analysis_validation.json',dict(passed=True,UTC=utc(),base_commit='65c7a9c66cd4ea3747fc11e132c4d6a01b5cd5b4',all_records=5040,all_questions=300,all_original_checkpoints_verified=True,official_score_max_difference=max(diff),classification_unchanged=True,group_family_size=len(effects),interaction_family_size=len(interactions),bootstrap=REPS,seed=SEED,tokenizer_sha256=toksha,first_content_matches_historical_all_records=True,first_G_B_logits_and_token_exact=300,model_loaded=False,new_generation=False,GPU_used=False,CPU_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
 print(json.dumps(dict(status='CPU_ANALYSIS_PASS',group_A_B=[dict(group=r['group'],condition=r['condition'],n=r['n_questions'],ACC=r['accuracy_pct']) for r in summary if r['condition'] in ['A','B']],interaction=[r for r in interactions if r['effect']=='Full_reaccess_effect'],decode_length=[r for r in lint if r['group']=='OCR-Combined']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
