"""Offline-only correction of the new ScienceQA adapter's index/letter bridge.

Original annotation, question, prompt, model generation and raw records stay intact.
The initial GPU-record score is provisional for ScienceQA and is never used as a
formal score. Official ScienceQA compares integer option indices.
"""
from scorers import score as initial_score,option_matcher
import string
def score(s,text):
 if s['benchmark']!='ScienceQA-IMG':return initial_score(s,text)
 choices=s['choices'];choices=choices if isinstance(choices,dict) else {string.ascii_uppercase[i]:v for i,v in enumerate(choices)}
 parsed=option_matcher()(text,dict(choices));failed=not parsed or parsed=='Z'
 gold=s['answer'];assert isinstance(gold,int) and 0<=gold<len(choices)
 index=ord(parsed)-65 if parsed and parsed in choices else -1
 return dict(score=float(index==gold),parsed=parsed,parsed_index=index,Gold_index=gold,Gold_letter=chr(65+gold),parse_failed=failed)
