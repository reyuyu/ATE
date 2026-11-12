"""Small, explicitly non-author-official paired evaluation; no full-run entrypoint."""
import ast
import copy
import importlib.util
import json
import logging
import os
import re
import string
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch

from .freeze import AUDIT, OUT, TASKS, digest, prepare, sha, table, write
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_svh_numerical_repair.engine import generate

AUTHOR = AUDIT.parent / 'qwen3vl_c3_3090_official_eval/official_sources'
AUTHOR_EXISTING = __import__('pathlib').Path('/data/lk/ATE-worktrees/qwen3vl-svh-six-official-benchmarks-2x3090/vlm_research/results/qwen3vl_c3_3090_official_eval/official_sources')
SOURCES = AUDIT / 'p0_provenance_v2/sources'


def scorer_sources():
    return dict(scienceqa=str(SOURCES / 'EvolvingLMMs-Lab__lmms-eval/lmms_eval/tasks/scienceqa/utils.py'),
                textvqa=str(AUTHOR_EXISTING / 'facebookresearch__mmf/mmf/utils/m4c_evaluators.py'),
                pope=str(AUTHOR_EXISTING / 'RUCAIBox__POPE/evaluate.py'),
                mme=str(AUTHOR_EXISTING / 'MME_calculation.py'),
                mmbench=str(SOURCES / 'open-compass__VLMEvalKit/vlmeval/utils/matching_util.py'))


@lru_cache(None)
def adapters():
    paths = scorer_sources()
    tree = ast.parse(Path(paths['scienceqa']).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('sqa_doc_to_target','sqa_process_results')]
    sqa = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), paths['scienceqa'], 'exec'), sqa)
    spec = importlib.util.spec_from_file_location('pinned_textvqa_evaluator', paths['textvqa'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(Path(paths['mme']).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'calculate_metrics')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_pred_ans')
    reduced = ast.ClassDef(name=cls.name, bases=[], keywords=[], body=[method], decorator_list=[])
    ast.fix_missing_locations(reduced)
    mme = {}
    exec(compile(ast.Module(body=[reduced], type_ignores=[]), paths['mme'], 'exec'), mme)
    tree = ast.parse(Path(paths['mmbench']).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('can_infer_option','can_infer_text','can_infer')]
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == '_VERBOSE_ANSWER_RE' for t in n.targets))
    mm = dict(cp=copy, os=os, re=re, string=string, logger=logging.getLogger('pinned_exact_matching'))
    exec(compile(ast.Module(body=[assignment]+nodes,type_ignores=[]),paths['mmbench'],'exec'), mm)
    return sqa['sqa_process_results'], module.TextVQAAccuracyEvaluator(), mme['calculate_metrics']().parse_pred_ans, mm['can_infer']


def score(row, text):
    sqa, tv, mme, mm = adapters()
    task, gold = row['benchmark'], row['gold']
    failed = False
    if task == 'GQA':
        parsed = text.strip().lower()
        value = float(parsed == gold.strip().lower())
    elif task == 'ScienceQA-IMG':
        value = sqa(dict(choices=list(row['options'].values()), answer=gold), [text])['exact_match']
        pred = text.strip()
        valid = pred.upper() in row['options'] or (len(pred) >= 2 and pred[0] in row['options'] and pred[1] == '.')
        parsed = pred[0].upper() if valid else None
        failed = not valid
    elif task == 'TextVQA':
        parsed = tv.answer_processor(text)
        value = tv.eval_pred_list([dict(pred_answer=text,gt_answers=gold)])
    elif task == 'POPE':
        words = text.split('.')[0].replace(',','').split(' ')
        parsed = 'no' if any(w in words for w in ['No','not','no']) else 'yes'
        value = float(parsed == gold)
    elif task == 'MME':
        parsed = mme(text.lower())
        failed = parsed == 'other'
        value = float(parsed == gold)
    elif task == 'MMBench':
        parsed = mm(text, dict(row['options']))
        failed = not parsed or parsed == 'Z'
        value = float(parsed == gold)
    else:
        raise ValueError(task)
    return dict(score=value, parsed=parsed, parse_failed=failed)


def evaluate(model, processor):
    gate = json.loads((OUT/'numerical_completion.json').read_text())
    assert gate['status'] == 'PASS' and gate['pretest_allowed']
    samples = json.loads((OUT/'sample_manifest.json').read_text())
    bindings = json.loads((OUT/'input_bindings.json').read_text())
    hooks = hook_counts(model)
    rows = []
    for sample in samples:
        inputs, layout = prepare(processor, sample)
        assert layout == bindings[sample['key']]
        for name, condition in [('Native','N'),('SVH','C')]:
            path = OUT/'pretest'/sample['key'].replace(':','_')/(name+'.json')
            assert not path.exists(), 'Do not overwrite existing prediction'
            write(OUT/'current.json',dict(stage='PRETEST48',key=sample['key'],condition=name))
            result, vectors = generate(model,processor,inputs,layout,condition,repaired=name=='SVH',
                                       audit=False,capture=True,max_tokens=sample['max_new_tokens'])
            assert hook_counts(model) == hooks
            assert {k:tensor_digest(v) for k,v in inputs.items()} == layout['input_tensor_hashes']
            scored = score(sample,result['text'])
            record = dict(key=sample['key'],benchmark=sample['benchmark'],question_id=sample['question_id'],
                          condition=name,image_sha256=sample['image_sha256'],gold=sample['gold'],
                          text=result['text'],token_ids=result['token_ids'],generated_tokens=result['generated_tokens'],
                          eos=result['eos'],truncated=result['truncated'],empty=result['empty'],
                          input_binding_sha256=digest(layout),peak_allocated=result['peak_allocated'],
                          peak_reserved=result['peak_reserved'],strategy=sample.get('strategy'),
                          circular_group=sample.get('circular_group'),pair_id=sample.get('pair_id'),**scored)
            write(path,record)
            rows.append(record)
            print('PRETEST',len(rows),'/',len(samples)*2,sample['key'],name,scored['score'],flush=True)
    table(OUT/'per_question.csv',rows)
    summaries=[]
    for task in TASKS:
        pool=[r for r in rows if r['benchmark']==task]
        by_key=defaultdict(dict)
        for r in pool:
            by_key[r['key']][r['condition']]=r
        for name in ['Native','SVH']:
            selected=[r for r in pool if r['condition']==name]
            assert len(selected)==48
            summaries.append(dict(benchmark=task,condition=name,n=48,accuracy=100*sum(r['score'] for r in selected)/48,
                harms=sum(p['SVH']['score']<p['Native']['score'] for p in by_key.values()) if name=='SVH' else 0,
                rescues=sum(p['SVH']['score']>p['Native']['score'] for p in by_key.values()) if name=='SVH' else 0,
                empty=sum(r['empty'] for r in selected),truncated=sum(r['truncated'] for r in selected),
                parse_failed=sum(r['parse_failed'] for r in selected),max_generated_tokens=max(r['generated_tokens'] for r in selected),
                peak_allocated=max(r['peak_allocated'] for r in selected),peak_reserved=max(r['peak_reserved'] for r in selected)))
    table(OUT/'accuracy_summary.csv',summaries)
    write(OUT/'pretest_completion.json',dict(status='COMPLETE',records=len(rows),expected=576,full_evaluation_started=False))
