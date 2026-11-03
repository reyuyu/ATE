"""CPU-only, text-only POS/dependency extraction. No labels or model outputs."""
import os,json,hashlib,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import spacy,en_core_web_sm
spacy.require_cpu()
root=Path('/data/lk/ATE-worktrees/noun-article-visual-writers-pilot/vlm_research')
out=root/'results/noun_article_visual_writers_pilot';out.mkdir(exist_ok=True)
nlp=spacy.load('en_core_web_sm',disable=['ner'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
rules=dict(parser='spacy==3.8.7',pipeline='en_core_web_sm==3.8.0',noun='Question NOUN or PROPN; include every compound noun token independently',article='literal a/an/the with dep=det and head POS NOUN/PROPN; direct dependency permits intervening adjectives; no outcome-dependent POS correction',mapping='Intersect parser character spans with original native tokenizer offsets; close to original whole-word groups; audit closures',selection='SHA256(noun-article-v1|task|question_id|image_sha256), 20 eligible per task, no repeated image; exclude all Semantic240 and historical12 harm images',placebo='two seeds 20261103/20261107; SHA-priority whole lexical Question groups outside base and containing no literal a/an/the; exact subset-sum equal to D minus base, otherwise unavailable; never split words',swap='historical12 only; add D minus S, remove exact whole-group subset of S ordered by SHA independently of answers; random swaps share same removal and exact added budget; unavailable if no exact subset',bootstrap='20000 image-cluster paired resamples, seed 20261104; random seeds average per question; exploratory 95% intervals; no noninferiority claims')
dump(out/'selection_rules.json',rules)
samples=json.loads((root/'results/gqa3000_token_visual_writing/sample_manifest.json').read_text())['samples']
parsed={}
for s,doc in zip(samples,nlp.pipe((s['question'] for s in samples),batch_size=64)):
    tokens=[dict(i=t.i,text=t.text,start=t.idx,end=t.idx+len(t.text),pos=t.pos_,tag=t.tag_,dep=t.dep_,head=t.head.i,head_text=t.head.text,head_pos=t.head.pos_,noun=t.pos_ in ('NOUN','PROPN'),article=t.lower_ in ('a','an','the') and t.dep_=='det' and t.head.pos_ in ('NOUN','PROPN')) for t in doc]
    parsed[s['question_id']]=dict(question=s['question'],tokens=tokens)
dump(out/'pos_parse_all3000.json',parsed)
modelpath=Path(en_core_web_sm.__file__).parent
dump(out/'parser_environment.json',dict(spacy=spacy.__version__,model=en_core_web_sm.__version__,python=os.sys.version,executable=os.sys.executable,require_cpu=True,pipeline=nlp.pipe_names,model_files={str(p.relative_to(modelpath)):sha(p) for p in sorted(modelpath.rglob('*')) if p.is_file() and '__pycache__' not in str(p)},pip_freeze=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),official_source='https://spacy.io/models/en#en_core_web_sm',user_authorized_isolated_install=True))
assert len(parsed)==3000 and spacy.__version__=='3.8.7' and en_core_web_sm.__version__=='3.8.0'
print('PARSED',len(parsed),flush=True)
for qid in ['10763612','001020372']:
    print(qid,parsed[qid],flush=True)
