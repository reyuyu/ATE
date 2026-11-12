"""Frozen official TextVQA evaluator plus independent normalization and soft-score arithmetic."""
import importlib.util,json,re
from pathlib import Path
def load(path):
 spec=importlib.util.spec_from_file_location('frozen_TextVQA_eval',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class Scorer:
 def __init__(self,rows,official_path):
  m=load(official_path);self.official=m.TextVQAAccuracyEvaluator();cls=m.EvalAIAnswerProcessor
  self.numbers=dict(cls.NUMBER_MAP);self.contractions=dict(cls.CONTRACTIONS);self.articles=set(cls.ARTICLES);self.punctuation=list(cls.PUNCTUATIONS)
  self.period=cls.PERIOD_STRIP.pattern;self.comma=cls.COMMA_STRIP.pattern;self.golds={};self.official_golds={};self.normalization_checks=0
  for r in rows:
   q=r['question_id'];assert len(r['gold'])==10
   self.golds[q]=[self.checked_norm(x) for x in r['gold']];self.official_golds[q]=self.official._compute_answer_scores(r['gold'])
 def independent_norm(self,text):
  s=text.lower().replace(',','').replace('?','').replace("'s"," 's").strip();s=s.replace('\n',' ').replace('\t',' ').strip();out=s
  for p in self.punctuation:
   replacement='' if p+' ' in s or ' '+p in s or re.search(self.comma,s) is not None else ' '
   out=out.replace(p,replacement)
  # The frozen official third positional argument is count=re.UNICODE(32), not flags. Preserve it.
  out=re.sub(self.period,'',out,count=int(re.UNICODE))
  words=[self.numbers.get(w,w) for w in out.lower().split()];return ' '.join(self.contractions.get(w,w) for w in words if w not in self.articles)
 def checked_norm(self,text):
  a=self.official.answer_processor(text);b=self.independent_norm(text);assert a==b,('NORMALIZATION_MISMATCH',text,a,b);self.normalization_checks+=1;return a
 def score(self,qid,text):
  p=self.checked_norm(text);official=float(self.official_golds[qid].get(p,0));matches=sum(a==p for a in self.golds[qid])
  independent=(0,.3,.6,.9,1,1,1,1,1,1,1)[matches];assert abs(official-independent)<1e-12,(qid,p,official,independent)
  return dict(parsed_answer=p,official_score=official,independent_score=independent)
