"""Generated-answer metrics never accept candidate scores as official predictions."""
import math
import re

from .dataset_registry import sha256
from ._vendor import m4c_evaluator

VQA_SHA256 = '8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774'


def pope_answer(text):
    # Exact AoiDragon/POPE evaluate.py / LLaVA eval_pope.py lexical rule.
    first_sentence = text.split('.')[0].replace(',', '')
    return 'no' if any(w in first_sentence.split(' ') for w in ('No', 'not', 'no')) else 'yes'


def pope_metrics(predictions, labels):
    if not predictions or len(predictions) != len(labels) or any(x not in ('yes', 'no') for x in labels):
        raise ValueError('Require nonempty, fully aligned valid gold labels')
    parsed = [pope_answer(p) for p in predictions]
    tp = sum(p == g == 'yes' for p, g in zip(parsed, labels))
    tn = sum(p == g == 'no' for p, g in zip(parsed, labels))
    fp = sum(p == 'yes' and g == 'no' for p, g in zip(parsed, labels))
    fn = sum(p == 'no' and g == 'yes' for p, g in zip(parsed, labels))
    # Upstream divides by zero on degenerate subsets. Explicit zero extension,
    # with denominator flags, preserves ordinary official values without crashes.
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {'official_generation_accuracy': (tp+tn)/len(labels), 'official_generation_precision': precision,
            'official_generation_recall': recall,
            'official_generation_f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'official_generation_yes_ratio': parsed.count('yes')/len(parsed),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'undefined_denominators': {'precision': tp+fp == 0, 'recall': tp+fn == 0}}


def scienceqa_index(text, choice_count):
    options = 'ABCDE'
    if text in options and len(text) == 1:
        answer = text
    elif len(text) >= 3 and text[0] in options and text[1:3] == '. ':
        answer = text[0]
    else:
        found = re.findall(r'The answer is ([A-Z]).', text)
        answer = found[0] if len(found) == 1 else 'FAILED'
    return options.index(answer) if answer in options[:choice_count] and len(answer) == 1 else -1


def textvqa_score(text, answers):
    if len(answers) != 10:
        raise ValueError('TextVQA requires all 10 original human answers')
    if sha256(m4c_evaluator.__file__) != VQA_SHA256:
        raise ValueError('Pinned TextVQA scorer source changed')
    evaluator = m4c_evaluator.TextVQAAccuracyEvaluator()
    scores = evaluator._compute_answer_scores(answers)
    return scores.get(evaluator.answer_processor(text), 0.0)


def diagnostic_candidate_metrics(sample, scores):
    candidates = sample['candidates']
    if sample['answer_type'] != 'closed' or candidates is None:
        raise ValueError('Open tasks do not have a fabricated Margin')
    if len(scores) != len(candidates) or any(not math.isfinite(x) or x > 1e-10 for x in scores):
        raise ValueError('Require all finite full-vocabulary sequence LogP')
    gold = candidates.index(sample['answer'])
    wrong = max(v for i, v in enumerate(scores) if i != gold)
    margin = scores[gold]-wrong
    return {'diagnostic_candidate_margin': margin, 'diagnostic_candidate_correct': margin > 0,
            'diagnostic_gold_sequence_logp': scores[gold],
            'diagnostic_all_candidate_sequence_logp': dict(zip(candidates, scores)),
            'diagnostic_margin_definition': 'gold minus strongest wrong; sums over all answer tokens; no EOS'}


def generated_score(sample, text):
    if not isinstance(text, str):
        raise TypeError('Official metrics require actual decoded generation, not candidate LogP')
    if sample['dataset'] == 'pope':
        return {'official_generation_correct': pope_answer(text) == sample['answer']}
    if sample['dataset'] == 'scienceqa_img':
        parsed = scienceqa_index(text, len(sample['candidates']))
        return {'official_generation_correct': parsed == sample['metadata']['gold_answer_index'], 'parsed_index': parsed}
    if sample['dataset'] == 'textvqa':
        return {'official_generation_vqa_soft_accuracy': textvqa_score(text, sample['answer'])}
    if sample['dataset'].startswith('gqa_'):
        # Mechanism subset only. Official GQA full evaluator is separately blocked.
        return {'mechanism_generation_literal_exact_match': text.strip().lower() == sample['answer'].strip().lower()}
    raise ValueError('Official GQA scorer not installed/verified')
