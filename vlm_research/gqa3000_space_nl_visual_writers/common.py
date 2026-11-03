"""Input-only SPACE/NL design. No GPU/model launch is implemented here."""
import csv,json,hashlib,subprocess
from pathlib import Path
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,read,sha256,digest
from semantic_visual_writers.groups import intact,flatten
BASE='11fe70e82d6b608da21d9bd375ce49abf390d4d5'
SEM='01c82b1d39360fc73bca3d10d12886672ffee370'
ATTENTION='2e1e02bc7c9b523c7878701cf0f563be2e0cf570'
HIST=ROOT/'results/gqa3000_token_visual_writing'
PILOT=ROOT/'results/noun_article_visual_writers_pilot'
OUT=ROOT/'results/gqa3000_space_nl_visual_writers'
CONDITIONS=['All-Writer','Semantic-k','S+SPACE','S+NL','S+SPACE+NL','Noun+Article','Noun+Article+SPACE+NL','S+Template-Control-2','N+D+Template-Control-2']
COMPARISONS=[['S+SPACE+NL','Semantic-k'],['Noun+Article+SPACE+NL','Noun+Article'],['S+SPACE+NL','S+Template-Control-2'],['Noun+Article+SPACE+NL','N+D+Template-Control-2'],['S+SPACE','S+NL']]
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),'Do not overwrite frozen output: '+str(p)
    p.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def csvwrite(name,rows):
    rows=list(rows);assert rows;p=OUT/name;assert not p.exists()
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
class GitHistory:
    """Batch exact committed-byte audit without thousands of shell processes."""
    def __init__(self):self.p=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT.parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE);self.files={}
    def bytes(self,path,commit):
        path=Path(path);rel=str(path.relative_to(ROOT.parent));query=(commit+':'+rel).encode();self.p.stdin.write(query+b'\n');self.p.stdin.flush();header=self.p.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',(query,header)
        n=int(header[2]);blob=self.p.stdout.read(n);assert len(blob)==n and self.p.stdout.read(1)==b'\n';assert path.read_bytes()==blob,'History differs: '+rel
        h=hashlib.sha256(blob).hexdigest();self.files[commit+':'+rel]=h;return blob
    def load(self,path,commit):return json.loads(self.bytes(path,commit))
    def close(self):self.p.stdin.close();assert self.p.wait()==0
def control_two(groups,question_positions,boundary,instruction_span,offsets):
    # Earliest two complete one-subword lexical instruction groups, by native position.
    choices=[g for g in groups if not g['question'] and g['words'] and len(g['positions'])==1 and not set(g['positions'])&(set(question_positions)|set(boundary)) and all(offsets[p][1]>instruction_span[0] and offsets[p][0]<instruction_span[1] for p in g['positions'])]
    choices=sorted(choices,key=lambda g:g['positions'][0]);return flatten(choices[:2]) if len(choices)>=2 else None
