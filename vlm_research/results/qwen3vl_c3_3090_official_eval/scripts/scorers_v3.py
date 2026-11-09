"""Offline canonical ATE extraction; never alters generation or raw records.

ScienceQA's original ATE extraction is copied verbatim as an AST branch from
e281754; the generic VLMEvalKit parser remains available as a named alternate.
Official ScienceQA matches integer option indices, without a canonical parser
for arbitrary free-form responses. No random fallback or judge is used here.
"""
from pathlib import Path
from functools import lru_cache
import ast,hashlib,re,string
from scorers_v2 import score as generic_score
REFERENCE=Path(__file__).parents[1]/'scoring_reference/score.py'
REFERENCE_SHA256='50bc02bb5376af0f2aec9d32e0ce507e050b54f5b78137e1bd1ac8c8549a47f7'

@lru_cache(None)
def frozen_scienceqa():
 data=REFERENCE.read_bytes();assert hashlib.sha256(data).hexdigest()==REFERENCE_SHA256
 function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='score')
 branch=next(n for n in function.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and any(isinstance(c,ast.Constant) and c.value=='ScienceQA-IMG' for c in n.test.comparators))
 function.body=branch.body;ast.fix_missing_locations(function)
 ctx={'re':re};exec(compile(ast.Module(body=[function],type_ignores=[]),str(REFERENCE),'exec'),ctx)
 return ctx['score']

def score(s,text):
 if s['benchmark']!='ScienceQA-IMG':return generic_score(s,text)
 choices=s['choices'];choices=choices if isinstance(choices,dict) else {string.ascii_uppercase[i]:v for i,v in enumerate(choices)}
 assert isinstance(s['answer'],int) and 0<=s['answer']<len(choices)
 sample=dict(s,options=choices);result=frozen_scienceqa()(sample,text)
 result.update(parsed_index=result['parsed'],Gold_index=s['answer'],Gold_letter=chr(65+s['answer']),score_protocol='Original frozen ATE extraction -> official integer option match')
 return result
