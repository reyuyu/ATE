"""Author rule scorers, isolated CPU adapters; no paid judge or random fallback."""
from pathlib import Path
from collections import Counter,defaultdict
from functools import lru_cache
import ast,copy,importlib.util,logging,os,re,string
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
S=O/'official_sources'
@lru_cache(None)
def text_evaluator():
 path=S/'facebookresearch__mmf/mmf/utils/m4c_evaluators.py'
 spec=importlib.util.spec_from_file_location('frozen_textvqa_author',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.TextVQAAccuracyEvaluator()
@lru_cache(None)
def option_matcher():
 path=S/'open-compass__VLMEvalKit/vlmeval/utils/matching_util.py';tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['can_infer_option','can_infer_text','can_infer']]
 assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VERBOSE_ANSWER_RE' for t in n.targets))
 context=dict(cp=copy,os=os,re=re,string=string,logger=logging.getLogger('official_matching'))
 exec(compile(ast.Module(body=[assign]+nodes,type_ignores=[]),str(path),'exec'),context)
 return context['can_infer']
@lru_cache(None)
def mme_parser():
 path=S/'MME_calculation.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='calculate_metrics')
 method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='parse_pred_ans')
 reduced=ast.ClassDef(name=cls.name,bases=[],keywords=[],body=[method],decorator_list=[]);ast.fix_missing_locations(reduced)
 ctx={};exec(compile(ast.Module(body=[reduced],type_ignores=[]),str(path),'exec'),ctx)
 return ctx['calculate_metrics']().parse_pred_ans
def score(s,text):
 b=s['benchmark'];gold=s['answer'];failed=False
 if b=='GQA':parsed=text.rstrip('.').lower();value=float(parsed==gold)
 elif b=='TextVQA':
  e=text_evaluator();parsed=e.answer_processor(text);value=e.eval_pred_list([dict(pred_answer=text,gt_answers=gold)])
 elif b=='POPE':
  words=text.split('.')[0].replace(',','').split(' ');parsed='no' if any(w in words for w in ['No','not','no']) else 'yes';value=float(parsed==gold)
 elif b=='MME':parsed=mme_parser()(text.lower());failed=parsed=='other';value=float(parsed==gold)
 elif b in ['ScienceQA-IMG','MMBench']:
  choices=s['choices'];choices=choices if isinstance(choices,dict) else {string.ascii_uppercase[i]:v for i,v in enumerate(choices)}
  parsed=option_matcher()(text,dict(choices));failed=not parsed or parsed=='Z';value=float(parsed==gold)
 elif b=='OCRBench-v1':
  def normalize(x):return x.strip().replace('\n',' ').replace(' ','') if s['dataset_name']=='HME100k' else x.lower().strip().replace('\n',' ')
  parsed=normalize(text);answers=gold if isinstance(gold,list) else [gold];value=float(any(normalize(a) in parsed for a in answers))
 else:raise ValueError(b)
 return dict(score=value,parsed=parsed,parse_failed=failed)
