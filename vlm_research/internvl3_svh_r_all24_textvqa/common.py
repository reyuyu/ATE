"""TextVQA-only analysis and frozen selection utilities; unchanged V2 model engine."""
import os,json,hashlib,time,csv,re,sqlite3
from pathlib import Path
import numpy as np
RUN=Path(os.environ.get("ATE_DEEP_RUN","/home/featurize/ate-runs/internvl3-svh-r-all24-textvqa-3090"));OUT=RUN/"results"
BASE=Path("/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090")
GPU="GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27";SEED=20261112
SPACE={"R0":[],"R6":list(range(14,20)),"R14":list(range(6,20))}
CONFIGS={"All24_ExceptL2_5":dict(keep=[i for i in range(28) if i not in range(2,6)],cutoff=28)}
SPACE.update({"Late8_L20_27":list(range(20,28)),"Late9_L19_27":list(range(19,28)),"Late6_L21_26":list(range(21,27))})
SPACE.update({"Shift8_L18_25":list(range(18,26)),"Shift8_L17_24":list(range(17,25)),"Shift8_L16_23":list(range(16,24))})
SPACE["Deep7_L20_26"]=list(range(20,27))
SPACE.update({"Deep6_L20_25":list(range(20,26)),"Deep8_L19_26":list(range(19,27))})
CONTROLS={}
ALL_CONFIGS={**CONFIGS,**CONTROLS}
SPACE.update({k:v["keep"] for k,v in ALL_CONFIGS.items()})
ORDER=list(ALL_CONFIGS)
MODEL=Path("/home/featurize/ate-local-models/InternVL3-8B")
PYTHON="/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python"
PKG="vlm_research/internvl3_svh_r_all24_textvqa"
PARENT_COMMIT="1407275cbdffca5d4a23cbf8b8d233002e2840dd"
OLD_GPU="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e"
PARENT=Path("/home/featurize/work/ATE-internvl3-svh-r-terminal-windows-textvqa-3090/vlm_research/results/internvl3_svh_r_terminal_windows_textvqa_3090")
TYPES=["signs","products","titles","numbers","short_words","multiword","other"]
INSTRUCTION="Answer with the text visible in the image, using a word or short phrase. Do not explain."
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(tmp,p)
def csvout(name,rows,fields=None):
 rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(r.get(k,""),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k,"") for k in fields})
def rank(r,label):return digest([SEED,label,r["question_id"],r["question"],r["image_sha256"]])
def subtype(question):
 q=question.lower()
 if re.search(r"\b(number|price|cost|date|year|time|percent|percentage|dollars?|cents?|digit|speed|temperature)\b|how (?:many|much)",q):return "numbers"
 if re.search(r"\b(book|title|cover|novel|magazine|newspaper|headline|author)\b",q):return "titles"
 if re.search(r"\b(brand|product|label|bottle|package|pack|company|logo|manufacturer|cereal|soda|beer|wine)\b",q):return "products"
 if re.search(r"\b(sign|street|road|banner|billboard|poster|store|shop|restaurant|advertisement|building)\b",q):return "signs"
 if re.search(r"\b(word|letter|letters|abbreviation|initials)\b",q) and not re.search(r"\b(words|sentence|phrase|slogan|motto)\b",q):return "short_words"
 if re.search(r"\b(words|sentence|phrase|slogan|motto|text|written|writing|read|say|says)\b",q):return "multiword"
 return "other"
def anomaly(x):
 text=x["text"].lower().strip();words=text.split()
 answering=bool(re.match(r"^answering\b",text))
 meta=answering or bool(re.search(r"does not require|do(?:n't| not) need|the answer is not in (?:the )?(?:image|text)|cannot determine|can(?:not|'t) (?:read|see|view|identify)|unable to (?:read|determine|identify)|as an ai|not (?:visible|legible|readable) in (?:the )?(?:image|text)",text))
 repeat=bool(re.search(r"(.{3,}?)\1\1",text)) or (len(words)>=12 and len(set(zip(words,words[1:],words[2:])))/max(1,len(words)-2)<.35)
 return dict(meta=meta,answering_prefix=answering,repeat=repeat,empty=not text,truncated=x["truncated"])
def quality(x,r):
 from vlm_research.internvl3_svh_seven_benchmarks import scoring
 parsed,score=scoring.score("TextVQA",x["text"],r);other=scoring.independent_score("TextVQA",x["text"],r);assert abs(score-other)<1e-10
 return {**x,"question_id":r["question_id"],"analysis_type":r["analysis_type"],"image_sha256":r["image_sha256"],"gold":r["gold"],"score":score,"parsed_answer":parsed,"flags":anomaly(x)}
def summarize(rows,phase):
 out=[]
 for m in dict.fromkeys(x["method"] for x in rows):
  rr=[x for x in rows if x["method"]==m]
  out.append(dict(phase=phase,method=m,N=len(rr),soft_accuracy_pct=100*float(np.mean([x["score"] for x in rr])),soft_score_sum=sum(x["score"] for x in rr),reaccess_layers=SPACE.get(m,list(range(28))),meta_count=sum(x["flags"]["meta"] for x in rr),meta_pct=100*np.mean([x["flags"]["meta"] for x in rr]),answering_prefix_count=sum(x["flags"]["answering_prefix"] for x in rr),empty_count=sum(x["empty"] for x in rr),repeat_count=sum(x["flags"]["repeat"] for x in rr),truncated_count=sum(x["truncated"] for x in rr),mean_KV_live_MiB=float(np.mean([x["KV_live_MiB"] for x in rr])),mean_KV_capacity_MiB=float(np.mean([x["KV_capacity_MiB"] for x in rr]))))
 return out
def pair(rows,base,cand,phase):
 a={x["question_id"]:x for x in rows if x["method"]==base};b={x["question_id"]:x for x in rows if x["method"]==cand};ids=sorted(set(a)&set(b))
 if not ids:return None
 aa=[a[q] for q in ids];bb=[b[q] for q in ids];clusters={}
 for x,y in zip(aa,bb):
  z=clusters.setdefault(x["image_sha256"],[0.,0.]);z[0]+=1;z[1]+=y["score"]-x["score"]
 ar=np.array(list(clusters.values()));rng=np.random.default_rng(SEED);weights=rng.multinomial(len(ar),np.full(len(ar),1/len(ar)),size=5000);diff=100*(weights@ar[:,1])/(weights@ar[:,0]);lo,hi=np.quantile(diff,[.025,.975]);d=[y["score"]-x["score"] for x,y in zip(aa,bb)]
 return dict(phase=phase,base=base,candidate=cand,N=len(ids),image_clusters=len(ar),difference_pp=100*float(np.mean(d)),CI95_low_pp=float(lo),CI95_high_pp=float(hi),improved=sum(z>1e-10 for z in d),declined=sum(z< -1e-10 for z in d),unchanged=sum(abs(z)<=1e-10 for z in d),meta_before=sum(x["flags"]["meta"] for x in aa),meta_after=sum(x["flags"]["meta"] for x in bb),meta_resolved=sum(x["flags"]["meta"] and not y["flags"]["meta"] for x,y in zip(aa,bb)),meta_resolved_and_score_improved=sum(x["flags"]["meta"] and not y["flags"]["meta"] and y["score"]>x["score"]+1e-10 for x,y in zip(aa,bb)),meta_to_scored_answer=sum(x["flags"]["meta"] and not y["flags"]["meta"] and y["score"]>0 for x,y in zip(aa,bb)),first_token_changes=sum(x["token_ids"][0]!=y["token_ids"][0] for x,y in zip(aa,bb)),full_text_changes=sum(x["text"]!=y["text"] for x,y in zip(aa,bb)),zero_to_positive=sum(x["score"]<=1e-10 and y["score"]>1e-10 for x,y in zip(aa,bb)),positive_to_zero=sum(x["score"]>1e-10 and y["score"]<=1e-10 for x,y in zip(aa,bb)),not_full_to_full=sum(x["score"]<1-1e-10 and y["score"]>=1-1e-10 for x,y in zip(aa,bb)),full_to_not_full=sum(x["score"]>=1-1e-10 and y["score"]<1-1e-10 for x,y in zip(aa,bb)),meta_new=sum(not x["flags"]["meta"] and y["flags"]["meta"] for x,y in zip(aa,bb)))

