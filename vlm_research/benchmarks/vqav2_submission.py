"""Offline-only VQAv2 export; never submits to a server or generates predictions."""
import argparse
import json
from pathlib import Path
import zipfile

from ._vendor.m4c_evaluator import EvalAIAnswerProcessor
from .dataset_registry import sha256


def question_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError('Question ID must be an integer or decimal string')
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError('Question ID must be decimal')
    result = int(value)
    if result < 0:
        raise ValueError('Negative Question ID')
    return result


def convert(records, dev_ids, full_ids=None, synthetic=False):
    dev = [question_id(q) for q in dev_ids]
    if len(set(dev)) != len(dev):
        raise ValueError('Duplicate manifest Question ID')
    allowed = set(dev)
    predictions = {}
    processor = EvalAIAnswerProcessor()
    for record in records:
        qid = question_id(record['question_id'])
        if qid not in allowed:
            raise ValueError('Prediction is outside the inference Test-dev split')
        if qid in predictions:
            raise ValueError('Duplicate prediction Question ID')
        if not synthetic and record.get('synthetic_for_structure_only'):
            raise ValueError('Synthetic fixture cannot be exported as genuine predictions')
        if synthetic and record.get('synthetic_for_structure_only') is not True:
            raise ValueError('Structure fixtures must be explicitly marked synthetic')
        if ('text' in record) == ('answer' in record):
            raise ValueError('Use exactly one prediction field: text or answer')
        answer = record.get('text', record.get('answer'))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Prediction must be a nonempty string')
        predictions[qid] = processor(answer)
    if not synthetic and set(predictions) != allowed:
        raise ValueError(f'Incomplete inference: {len(predictions)} predictions for {len(dev)} questions')
    target = dev if full_ids is None else [question_id(q) for q in full_ids]
    if len(set(target)) != len(target) or not allowed <= set(target):
        raise ValueError('Invalid official submission Question ID universe')
    output = [dict(question_id=q, answer=predictions.get(q, '')) for q in target]
    audit = dict(inference_manifest_questions=len(dev), actual_prediction_records=len(predictions),
                 official_submission_records=len(output),
                 padding_records=sum(q not in predictions for q in target),
                 empty_normalized_real_predictions=sum(a == '' for a in predictions.values()),
                 synthetic_for_structure_only=synthetic, upload_allowed=not synthetic,
                 scoring_performed=False, submitted=False,
                 normalization='Pinned LLaVA EvalAIAnswerProcessor; padding is not inference')
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015'))
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('testdev-only', 'llava-full-test2015'), default='llava-full-test2015')
    parser.add_argument('--synthetic-structure-only', action='store_true')
    args = parser.parse_args()
    audit = json.loads((args.dataset / 'dataset_audit.json').read_text())
    if audit['status'] != 'COMPLETE' or audit['actual_questions'] != 107394 or audit['gold_available'] is not False:
        raise ValueError('Require validated hidden-answer Test-dev dataset')
    inputs = args.dataset / 'inputs.jsonl'
    if sha256(inputs) != audit['inputs_sha256']:
        raise ValueError('Inference manifest changed')
    with inputs.open() as f:
        dev = [json.loads(line)['question_id'] for line in f]
    with args.predictions.open() as f:
        records = [json.loads(line) for line in f]
    full = None
    if args.mode == 'llava-full-test2015':
        source = json.loads((args.dataset / 'source_manifest.json').read_text())['question_source']
        archive = Path(source['local_path'])
        if sha256(archive) != source['sha256']:
            raise ValueError('Official source archive changed')
        with zipfile.ZipFile(archive) as z:
            full = [q['question_id'] for q in json.loads(z.read('v2_OpenEnded_mscoco_test2015_questions.json'))['questions']]
        if len(full) != 447793:
            raise ValueError('Wrong official full Test2015 ID count')
    if args.synthetic_structure_only and not args.output.name.endswith('.structure-test.json'):
        raise ValueError('Synthetic export filename must end with .structure-test.json')
    output, evidence = convert(records, dev, full, args.synthetic_structure_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False)
        f.write('\n')
    with args.output.with_suffix(args.output.suffix + '.audit.json').open('x') as f:
        json.dump(dict(evidence, output_sha256=sha256(args.output)), f, indent=2)
        f.write('\n')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
