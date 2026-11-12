"""Pinned official answer normalization/scoring; independent audit implementations."""
from .common import *
import ast,re,string,copy,logging,importlib.util
from collections import Counter
def ast_scope(path,names,extra=None):
    nodes=ast.parse(Path(path).read_text()).body
    chosen=[n for n in nodes if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in names for x in n.targets))]
    scope={"re":re,"string":string,"cp":copy,"os":os,"logger":logging.getLogger(__name__)}
    if extra:scope.update(extra)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),"exec"),scope)
    return scope
def official_matcher():
    p=RUN/"references/VLMEvalKit/vlmeval/utils/matching_util.py"
    return ast_scope(p,{"can_infer","can_infer_text","can_infer_option","_VERBOSE_ANSWER_RE"})["can_infer"]
_MATCH=None;_TV=None;_MME=None
def parse(b,text,r):
    global _MATCH,_TV,_MME
    if b=="GQA":return text.rstrip(".").lower()
    if b=="POPE":
        words=text.split(".")[0].replace(",","").split(" ")
        return "no" if any(w in words for w in ("No","not","no")) else "yes"
    if b=="MME":
        if _MME is None:_MME=ast_scope(RUN/"references/InternVL/internvl_chat/eval/mme/calculation.py",{"calculate_metrics"})["calculate_metrics"]()
        return _MME.parse_pred_ans(text.lower())
    if b=="ScienceQA-IMG":return text.strip() if text.strip() in "ABCDE" and len(text.strip())==1 else ""
    if b=="MMBench":
        if _MATCH is None:_MATCH=official_matcher()
        return _MATCH(text,copy.deepcopy(r["choices"])) or "Z"
    if b=="TextVQA":
        if _TV is None:
            p=RUN/"references/InternVL/internvl_chat/eval/vqa/textvqa_eval.py"
            sp=importlib.util.spec_from_file_location("frozen_textvqa",p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);_TV=m.TextVQAAccuracyEvaluator()
        return _TV.answer_processor(text)
    return text
def score(b,text,r):
    parsed=parse(b,text,r);gold=r["gold"]
    if b=="TextVQA":return parsed,float(_TV._compute_answer_scores(gold).get(parsed,0))
    if b=="OCRBench":
        answers=gold if isinstance(gold,list) else [gold]
        if r["dataset_name"]=="HME100k":
            norm=lambda s:s.strip().replace("\n"," ").replace(" ","")
        else:norm=lambda s:s.lower().strip().replace("\n"," ")
        return parsed,float(any(norm(a) in norm(text) for a in answers))
    if b=="ScienceQA-IMG":gold="ABCDE"[gold]
    return parsed,float(parsed==str(gold))
def independent_score(b,text,r):
    """Separate scalar implementation checked against official branches for every saved prediction."""
    if b=="TextVQA":
        normalized=[_TV.answer_processor(x) for x in r["gold"]]
        p=_TV.answer_processor(text)
        return sum(min(sum(x==p for j,x in enumerate(normalized) if j!=i)/3,1) for i in range(10))/10
    if b=="OCRBench":
        answers=r["gold"] if isinstance(r["gold"],list) else [r["gold"]]
        def norm(s):
            s=s.strip().replace("\n"," ")
            return s.replace(" ","") if r["dataset_name"]=="HME100k" else s.lower()
        return float(any(norm(text).find(norm(x))>=0 for x in answers))
    if b=="GQA":return float(text.rstrip(".").lower()==r["gold"])
    if b=="ScienceQA-IMG":return float(text.strip()=="ABCDE"[r["gold"]])
    if b=="POPE":
        first=text.partition(".")[0].replace(",","");p="no" if set(first.split(" ")).intersection(["No","not","no"]) else "yes"
        return float(p==r["gold"])
    if b=="MME":
        p=text.lower()
        if p not in ("yes","no"):
            p="yes" if "yes" in p[:4] else ("no" if "no" in p[:4] else "other")
        return float(p==r["gold"])
    # Official exact_matching is reused for extraction; independent circular reduction is audited separately.
    return float(parse(b,text,r)==r["gold"])
def official_ocr_batch(rows):
    """Execute the original MultimodalOCR scoring loop, without importing its model loader."""
    p=RUN/"references/OCRBench_official_LLaVA1_5.py";tree=ast.parse(p.read_text());node=None
    for n in ast.walk(tree):
        if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="i" and any(isinstance(x,ast.Name) and x.id=="dataset_name" for x in ast.walk(n)):
            if "data_type" in ast.unparse(n) and "data[i]['result']" in ast.unparse(n):node=n;break
    assert node is not None,"Official OCR scoring loop not found"
    data=[dict(type=r["category"],dataset_name=r["dataset_name"],answers=r["gold"],predict=r["text"]) for r in rows]
    scope={"data":data};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec"),scope)
    return [x.get("result",0) for x in data]
def anomalies(b,res,r):
    text=res["text"];words=re.findall(r"\w+",text)
    # Declared output-form checks; no gold/accuracy is consulted.
    repeated=bool(re.search(r"(.{3,}?)\1\1",text,re.S)) or (len(words)>=12 and len(set(zip(words,words[1:],words[2:])))<len(words)-5)
    explanation=False
    if b in ("POPE","MME"):explanation=not bool(re.fullmatch(r"\s*(yes|no)[.!]?\s*",text,re.I))
    elif b in ("ScienceQA-IMG","MMBench"):explanation=not bool(re.fullmatch(r"\s*[A-E][.!]?\s*",text))
    elif b=="GQA":explanation=len(words)>12 or bool(re.search(r"\b(because|therefore|the answer is|in the image)\b",text,re.I))
    elif b=="TextVQA":explanation=len(words)>24 or bool(re.search(r"\b(the image shows|the answer is|because)\b",text,re.I))
    else:explanation=len(words)>80 and r["category"]!="Handwritten Mathematical Expression Recognition"
    parsed=parse(b,text,r)
    fail=(b in ("POPE","MME") and not re.fullmatch(r"(yes|no)[.!]?",text.strip(),re.I)) or (b in ("ScienceQA-IMG","MMBench") and parsed not in ("ABCDE"[:len(r["choices"])] if isinstance(r.get("choices"),list) else r.get("choices",{})))
    flags=[]
    if repeated:flags.append("REPETITION")
    if explanation:flags.append("UNEXPECTED_ANSWER_FORM")
    if b!="OCRBench" and (not res["eos"] or res["truncated"]):flags.append("ABNORMAL_STOP_OR_TRUNCATION")
    if res["empty"]:flags.append("EMPTY_OUTPUT")
    return {"parsed_answer":parsed,"parse_failure":bool(fail),"explanation":explanation,"repetition":repeated,"flags":flags}
