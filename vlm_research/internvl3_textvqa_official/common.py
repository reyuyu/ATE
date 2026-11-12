"""Input/provenance helpers. Upstream functions are executed from pinned source ASTs."""
import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/internvl3_textvqa_official_alignment'
UPSTREAM = OUT / 'release_sources'
MODEL = Path('/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9')
HISTORY = Path('/data/lk/ATE-worktrees/internvl3-svhr-late9-sevenbench-2x3090')
COMMIT = '1cc6d16d34b8f164523388edb0a9642ef8ab7314'
DATASET_MD5 = 'b233b31f551bbf4056f2f955da3a92cd'
KEEP = list(range(19, 28))
GENERATION = dict(do_sample=False, max_new_tokens=4096, top_p=0.95, temperature=0.7)
INDEPENDENT_SCORER = HISTORY/'vlm_research/results/internvl3_svh_r_terminal_windows_textvqa_3090/references/textvqa_eval.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n')
    os.replace(tmp, path)


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def execute_functions(path, scope, names=None, class_name=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body if class_name is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    functions = [n for n in body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    if names is not None:
        assert {n.name for n in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def istype(value, typ):
    if isinstance(value, typ):
        return True
    if not isinstance(value, str):
        return False
    try:
        return isinstance(ast.literal_eval(value), typ)
    except (ValueError, SyntaxError):
        return False


def upstream():
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    scope = dict(torch=torch, T=T, Image=Image, InterpolationMode=InterpolationMode,
        IMAGENET_MEAN=(0.485, 0.456, 0.406), IMAGENET_STD=(0.229, 0.224, 0.225),
        np=np, os=os, osp=os.path, io=io, base64=base64, re=re, Optional=Optional,
        listinstr=lambda words, text: any(word in text for word in words), istype=istype,
        DATASET_TYPE=lambda _: 'VQA', DATASET_MODALITY=lambda _: 'IMAGE')
    execute_functions(UPSTREAM/'vlmeval/vlm/internvl/utils.py', scope,
        ['build_transform', 'find_closest_aspect_ratio', 'dynamic_preprocess', 'load_image', 'reorganize_prompt'])
    execute_functions(UPSTREAM/'vlmeval/vlm/internvl/internvl_chat.py', scope,
        ['build_prompt', 'set_max_num'], 'InternVLChat')
    execute_functions(UPSTREAM/'vlmeval/smp/vlm.py', scope,
        ['decode_base64_to_image', 'decode_base64_to_image_file'])
    execute_functions(UPSTREAM/'vlmeval/smp/misc.py', scope,
        ['process_punctuation', 'istype', 'listinstr'])
    execute_functions(UPSTREAM/'vlmeval/dataset/utils/vqa_eval.py', scope)
    return scope


def official_request(scope, question, image_path):
    stub = SimpleNamespace(use_mpo_prompt=False, use_cot=False,
        use_custom_prompt=lambda _: True, dump_image=lambda *_: [str(image_path)])
    scope['set_max_num'](stub, 'TextVQA_VAL')
    assert stub.max_num == 6
    messages = scope['build_prompt'](stub, {'question': question}, 'TextVQA_VAL')
    return scope['reorganize_prompt'](messages, 1, dataset='TextVQA_VAL')


def conversation():
    spec = importlib.util.spec_from_file_location('ate_official_internvl_conversation', MODEL/'conversation.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = json.loads((MODEL/'config.json').read_text())
    return module.get_conv_template(config['template'])


def render(tok, request, tiles):
    conv = conversation()
    conv.append_message(conv.roles[0], request)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt().replace('<image>', '<img>' + '<IMG_CONTEXT>' * (256 * tiles) + '</img>', 1)
    ids = tok(prompt, return_tensors='pt')['input_ids']
    image_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    image = [i for i, token in enumerate(ids[0].tolist()) if token == image_id]
    assert len(image) == tiles * 256 and image == list(range(image[0], image[-1]+1))
    roles = dict(n=ids.shape[1], image=image, R=list(range(image[-1]+1, ids.shape[1]-1)),
                 A_prefill=[ids.shape[1]-1], image_token_id=image_id)
    return prompt, ids, roles


def manifest():
    with gzip.open(OUT/'sample_manifest.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def score(scope, prediction, gold):
    assert len(gold) == 10
    value = scope['process_line']({'prediction': prediction, 'answer': repr(gold)}, method='vqa_score')
    actual = sum(value['match']) / 10
    normalized = value['pred']
    count = value['gt'].count(normalized)
    independent = sum(min(1.0, (count-int(answer == normalized))/3) for answer in value['gt']) / 10
    assert abs(actual-independent) < 1e-12
    return actual, normalized


def independent_scorer():
    tree = ast.parse(INDEPENDENT_SCORER.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
               n.name in ['EvalAIAnswerProcessor', 'TextVQAAccuracyEvaluator']]
    assert len(classes) == 2
    scope = dict(re=re, tqdm=lambda values, **_: values)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(INDEPENDENT_SCORER), 'exec'), scope)
    return scope['TextVQAAccuracyEvaluator']()
