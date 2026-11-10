"""Frozen fast-search protocol utilities; inference engine is imported unchanged."""
import os,json,hashlib,time,csv,re,sqlite3
from pathlib import Path
import numpy as np
RUN=Path(os.environ.get("ATE_FAST_RUN","/home/featurize/ate-runs/internvl3-svh-r-fast-query-search-3090"))
OUT=RUN/"results"
OLD=Path("/home/featurize/ate-runs/internvl3-svh-r-middle-reaccess-search-3090")
BASE=Path("/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090")
GPU="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e"
SPACE={"R0":[],"R2":[16,19],"R4-M":[12,13,14,15],"R4-L":[16,17,18,19],"R6":list(range(14,20)),"R8":list(range(12,20)),"R10":list(range(10,20))}
ORDER=list(SPACE);SEED=20261110
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(tmp,p)
def csvout(name,rows,fields=None):
 p=OUT/name;rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def rank(r,label):return digest([SEED,label,r["question_id"]])
def subtype(r):
 q=r["question"].lower();t=r["types"]
 if re.search(r"^how many\b|\bnumber of\b",q):return "quantity"
 if re.search(r"\bcolou?r\b",q):return "color"
 if t["detailed"]=="positionQuery" or re.search(r"^where\b|^on which side\b",q):return "spatial"
 if t["semantic"]=="rel":return "relation"
 if t["semantic"] in ("cat","obj"):return "object"
 if t["semantic"]=="attr":return "attribute"
 return "other"
def anomaly(x):
 text=x["text"].lower();words=text.split()
 meta=bool(re.search(r"does not require|do(?:n't| not) need|as an ai|cannot (?:see|view|determine)|answering (?:this|the) question|reading text in the image",text))
 repeat=bool(re.search(r"(.{3,}?)\1\1",text)) or (len(words)>=12 and len(set(zip(words,words[1:],words[2:])))/max(1,len(words)-2)<.35)
 return {"meta":meta,"repeat":repeat,"empty":not text.strip(),"truncated":x["truncated"],"explanation":len(words)>12 or bool(re.search(r"\bbecause\b|\btherefore\b|the answer is|in the image",text))}
def quality(x,r):
 parsed=x["text"].rstrip(".").lower();score=float(parsed==r["gold"])
 return {**x,"question_id":r["question_id"],"category":r["category"],"subtype":subtype(r),"image_sha256":r["image_sha256"],"gold":r["gold"],"score":score,"parsed_answer":parsed,"flags":anomaly(x)}
def summarize(records,phase):
 out=[]
 for method in dict.fromkeys(x["method"] for x in records):
  rr=[x for x in records if x["method"]==method]
  for group in ["all","query","nonquery"]:
   a=[x for x in rr if group=="all" or (x["category"]=="query")== (group=="query")]
   if not a:continue
   out.append(dict(phase=phase,method=method,group=group,N=len(a),correct=sum(x["score"] for x in a),accuracy_pct=100*np.mean([x["score"] for x in a]),K=len(SPACE.get(method,range(28))),mean_KV_live_MiB=float(np.mean([x["KV_live_MiB"] for x in a])),mean_KV_capacity_MiB=float(np.mean([x["KV_capacity_MiB"] for x in a])),anomalies=sum(any(x["flags"].values()) for x in a),meta=sum(x["flags"]["meta"] for x in a),empty=sum(x["empty"] for x in a),truncated=sum(x["truncated"] for x in a)))
 return out
def pair(rows,base,cand,label,group="all"):
 aa={r["question_id"]:r for r in rows if r["method"]==base};bb={r["question_id"]:r for r in rows if r["method"]==cand}
 ids=[q for q in aa if q in bb and (group=="all" or (aa[q]["category"]=="query")== (group=="query"))]
 if not ids:return None
 a=[aa[q] for q in ids];b=[bb[q] for q in ids];clusters={}
 for x,y in zip(a,b):
  z=clusters.setdefault(x["image_sha256"],[0,0]);z[0]+=1;z[1]+=y["score"]-x["score"]
 arr=np.array(list(clusters.values()),float);rng=np.random.default_rng(SEED);weights=rng.multinomial(len(arr),np.full(len(arr),1/len(arr)),size=5000);diff=100*(weights@arr[:,1])/(weights@arr[:,0]);lo,hi=np.quantile(diff,[.025,.975])
 return dict(phase=label,base=base,candidate=cand,group=group,N=len(ids),image_clusters=len(arr),difference_pp=100*np.mean([y["score"]-x["score"] for x,y in zip(a,b)]),CI95_low_pp=float(lo),CI95_high_pp=float(hi),wrong_to_right=sum(x["score"]==0 and y["score"]==1 for x,y in zip(a,b)),right_to_wrong=sum(x["score"]==1 and y["score"]==0 for x,y in zip(a,b)),first_token_changes=sum(x["token_ids"][0]!=y["token_ids"][0] for x,y in zip(a,b)),full_text_changes=sum(x["text"]!=y["text"] for x,y in zip(a,b)))
def selections(rows,stage):
 sm=summarize(rows,stage);d={(x["method"],x["group"]):x for x in sm};base=d["R0","all"];bq=d["R0","query"]["accuracy_pct"];bn=d["R0","nonquery"]["accuracy_pct"];br=base["anomalies"]/base["N"]
 ranks=[]
 for m in dict.fromkeys(x["method"] for x in rows):
  if m=="Native":continue
  allr=d[m,"all"];q=d[m,"query"]["accuracy_pct"];n=d[m,"nonquery"]["accuracy_pct"]
  penalty=max(0,bn-n)+100*max(0,allr["anomalies"]/allr["N"]-br)+.5*max(0,allr["mean_KV_capacity_MiB"]-base["mean_KV_capacity_MiB"])
  ranks.append(dict(method=m,query_pct=q,nonquery_pct=n,penalized_gain=q-bq-penalty,mean_KV_capacity_MiB=allr["mean_KV_capacity_MiB"],K=len(SPACE[m])))
 if stage=="A":
  selected=sorted([x for x in ranks if x["method"]!="R0"],key=lambda x:(-x["penalized_gain"],-x["query_pct"],x["mean_KV_capacity_MiB"],x["K"],ORDER.index(x["method"])))[:3]
  return {"stage":"A","ranking":ranks,"selected":["R0"]+[x["method"] for x in selected],"rule":"Frozen penalized query gain; no outcome-based candidate expansion"}
 # best_quality is highest query accuracy; non-query loss and anomalies are explicit tradeoffs, not hidden.
 best=sorted(ranks,key=lambda x:(-x["query_pct"],x["mean_KV_capacity_MiB"],x["K"],ORDER.index(x["method"])))[0]
 near=[x for x in ranks if x["query_pct"]>=best["query_pct"]-2.5-1e-8 and x["nonquery_pct"]>=bn-2.5-1e-8]
 eff=min(near,key=lambda x:(x["K"],x["mean_KV_capacity_MiB"],-x["query_pct"],ORDER.index(x["method"]))) if near else next(x for x in ranks if x["method"]=="R0")
 return {"stage":"B","ranking":ranks,"best_quality":best["method"],"best_efficiency":eff["method"],"holdout_methods":list(dict.fromkeys(["R0",best["method"],eff["method"]]))}
