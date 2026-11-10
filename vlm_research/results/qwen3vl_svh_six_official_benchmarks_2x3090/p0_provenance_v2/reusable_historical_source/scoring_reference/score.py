"""Pinned author lexical scoring, no LLM judge or answer-ranking generation."""
from functools import lru_cache
import ast
import copy
import importlib.util
import logging
import os
import re
import string
import subprocess
import base64

from .common import *


@lru_cache(None)
def text_evaluator():
    p = AUTHOR / 'llava/eval/m4c_evaluator.py'
    spec = importlib.util.spec_from_file_location('author_textvqa_evaluator',p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


@lru_cache(None)
def matcher():
    # Exact official VLMEvalKit functions already frozen in historical ATE.
    raw = subprocess.check_output(['git','show',
        'e2022e7e195fcd121ac2275c4c42edfa30642e17:vlm_research/results/seven_benchmark_native_physical_quality/official/matching_util_api.json'],cwd=ROOT)
    entry = json.loads(raw)
    content = base64.b64decode(entry['content'])
    assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == entry['sha'] == '84ee053d7d267ffe7ad397d3a99c2f5e80716c56'
    module = ast.parse(content)
    names = {'can_infer_option','can_infer_text','can_infer'}
    nodes = [n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names
        or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VERBOSE_ANSWER_RE' for t in n.targets)]
    scope = dict(cp=copy,os=os,re=re,string=string,logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'official/vlmeval_matching_util.py'),'exec'),scope)
    return scope['can_infer']


def score(sample,text):
    bench = sample['benchmark']
    if bench == 'TextVQA':
        e = text_evaluator()
        scores = e._compute_answer_scores(sample['answer'])
        prediction = e.answer_processor(text)
        return dict(score=float(scores.get(prediction,0.0)),parsed=prediction)
    if bench == 'MME':
        p = text.lower()
        parsed = p if p in ('yes','no') else 'yes' if 'yes' in p[:4] else 'no' if 'no' in p[:4] else 'other'
        return dict(score=float(parsed==sample['answer']),parsed=parsed,parse_failed=parsed=='other')
    if bench == 'POPE':
        # Exact eval_pope lexical rule; unknown/empty output defaults Yes.
        words = text.split('.')[0].replace(',','').split(' ')
        parsed = 'no' if any(w in words for w in ('No','not','no')) else 'yes'
        return dict(score=float(parsed==sample['answer']),parsed=parsed,
                    lexical_empty_defaults_yes=not bool(text))
    if bench == 'ScienceQA-IMG':
        options = 'ABCDE'
        if text in options and len(text)==1:
            answer = text
        elif len(text)>=3 and text[0] in options and text[1:3]=='. ':
            answer = text[0]
        else:
            matches = re.findall(r'The answer is ([A-Z]).',text)
            answer = matches[0] if len(matches)==1 else 'FAILED'
        parsed = options.index(answer) if answer in sample['options'] else -1
        return dict(score=float(parsed==sample['answer']),parsed=parsed,parse_failed=parsed<0)
    parsed = matcher()(text,sample['options']) or 'Z'
    return dict(score=float(parsed==sample['answer']),parsed=parsed,parse_failed=parsed=='Z',
                score_protocol='OFFLINE_REPRODUCTION/VLMEvalKit exact matching')
