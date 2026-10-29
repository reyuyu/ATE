"""Verified published primary metrics; caption scoring stays corpus-aware."""
import ast
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from benchmarks.dataset_registry import sha256
from benchmarks.scoring import pope_answer, scienceqa_index, textvqa_score
from .inventory import OLD, OUTPUT, RESOURCES

GQA_EVALUATOR_SHA = '4433a8959d82c14b37ffc0c17c6f8d3e34ef692515b79939eb66220a4b74d130'
GQA_CONVERTER_SHA = '016cea68c91cbc141c276e9e7568420e2f5db5aba90c4f594701663c72929932'
DEPS = RESOURCES/'metric_deps'


def gqa_prediction(text):
    # Exact pinned LLaVA official-GQA conversion, after native decoder output.strip().
    return text.rstrip('.').lower()


def short_score(sample, text):
    unit = sample['dataset']
    if unit == 'gqa_balanced':
        return float(gqa_prediction(text) == sample['answer'])
    if unit == 'scienceqa_img':
        return float(scienceqa_index(text, len(sample['candidates'])) == sample['metadata']['gold_answer_index'])
    if unit == 'pope_adversarial':
        return float(pope_answer(text) == sample['answer'])
    if unit == 'textvqa':
        return float(textvqa_score(text, sample['answer']))
    if unit in ('coco_caption', 'textcaps'):
        return None
    raise ValueError('Unsupported frozen dataset')


def caption_scores(references, predictions):
    if set(references) != set(predictions) or not references:
        raise ValueError('Caption refs/predictions must fully align')
    if str(DEPS) not in sys.path:
        sys.path.insert(0, str(DEPS))
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize({k: [dict(caption=t) for t in v] for k, v in references.items()})
    res = tokenizer.tokenize({k: [dict(caption=t)] for k, t in predictions.items()})
    cider, cider_rows = Cider().compute_score(gts, res)
    spice, spice_rows = Spice().compute_score(gts, res)
    # Cider retains insertion order; Spice explicitly sorts keys.
    by_cider = dict(zip(gts, map(float, cider_rows)))
    by_spice = {k: float(v['All']['f']) for k, v in zip(sorted(gts), spice_rows)}
    if not np.isfinite(list(by_cider.values())+list(by_spice.values())).all():
        raise ValueError('Nonfinite official caption score')
    return dict(CIDEr=float(cider), SPICE=float(spice)), {k: dict(CIDEr=by_cider[k], SPICE=by_spice[k]) for k in gts}


def validate():
    started = time.perf_counter()
    audit = dict(passed=False, checks={}, source_sha256={}, scope='Official primary accuracy/soft scoring and caption CIDEr/SPICE; not full GQA auxiliary suite')
    try:
        official = RESOURCES/'annotations/gqa_eval.py'
        converter = OLD/'sources/llava/scripts/convert_gqa_for_eval.py'
        if sha256(official) != GQA_EVALUATOR_SHA or sha256(converter) != GQA_CONVERTER_SHA:
            raise ValueError('GQA official evaluator/converter hash mismatch')
        tree = ast.parse(official.read_text())
        expressions = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'correct' for t in n.targets)]
        expected = ast.dump(ast.parse('predicted == gold', mode='eval').body)
        if not any(ast.dump(n) == expected for n in expressions):
            raise ValueError('Official GQA accuracy expression differs')
        if "text = res['text'].rstrip('.').lower()" not in converter.read_text():
            raise ValueError('GQA official submission conversion differs')
        check = compile(ast.Expression(next(n for n in expressions if ast.dump(n) == expected)), '<official-GQA-equality>', 'eval')
        for text, gold in [('Dog.', 'dog'), ('red', 'blue'), ('two cats', 'two cats'), ('yes.', 'yes')]:
            predicted = gqa_prediction(text)
            reference = eval(check, {'__builtins__': {}}, dict(predicted=predicted, gold=gold))
            if short_score(dict(dataset='gqa_balanced', answer=gold), text) != float(reference):
                raise AssertionError('GQA primary scoring differs')
        audit['checks']['official_gqa_primary_and_llava_conversion'] = True
        audit['checks']['pope_parser'] = pope_answer('No, there is not.') == 'no' and pope_answer('Yes.') == 'yes'
        audit['checks']['scienceqa_parser'] = scienceqa_index('B', 3) == 1
        audit['checks']['textvqa_ten_vote_fixture'] = textvqa_score('cat', ['cat']*10) == 1.
        refs = {0: ['A red bus drives on a road.'], 1: ['A cat sits on a mat.']}
        summary, rows = caption_scores(refs, {k: v[0] for k, v in refs.items()})
        if summary['CIDEr'] <= 0 or not np.isclose(summary['SPICE'], 1., atol=1e-6):
            raise AssertionError('Official caption identical-reference fixture failed')
        audit['checks']['official_caption_identical_reference'] = dict(passed=True, metrics=summary, per_image=rows)
        code = sorted((DEPS/'pycocoevalcap').rglob('*.py'))
        code += sorted((DEPS/'pycocoevalcap').rglob('*.jar'))
        code += [official, converter]
        audit['source_sha256'] = {str(p): sha256(p) for p in code}
        audit['passed'] = all(bool(v) for v in audit['checks'].values())
    except Exception as exc:
        audit.update(error=repr(exc), traceback=traceback.format_exc())
    audit['seconds'] = time.perf_counter()-started
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/f'scorer_validation_attempt_{time.time_ns()}.json'
    path.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    (OUTPUT/'scorer_validation.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(audit, indent=2), flush=True)
    if not audit['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    validate()
