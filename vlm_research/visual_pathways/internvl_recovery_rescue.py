"""The already-requested optional three-condition rescue, with pilot-frozen layers."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import gc
import time
import traceback

import torch

from .common import ROOT, IMODEL, EXPERIMENT_END, read, write, sha, digest, preflight, command
from .engine import InternPrefill, metrics
from .relay import Controller
from .scan import load_intern
from .internvl_recovery import RECOVERY, verify
from ..internvl3_smoke.engine import hooks_count, tensor_hash, official_preprocessor
from ..internvl3_smoke.run import ChatCapture

OUT = RECOVERY/'internvl_rescue'


class IController(Controller):
    def __init__(self, runner, image_off=(), noop=False, capture=False):
        self.model = runner.model.language_model
        self.inputs = runner.inputs
        self.layers = runner.layers
        self.n = runner.layout['input_length']
        self.image = runner.layout['positions']['image']
        self.config = dict(relay_off=True, skip=False, image_off=image_off, noop=noop,
                           tolerance=2**-8+2e-6)
        self.capture = capture
        self.calls, self.audits, self.state_hashes = [], {}, {}


@torch.inference_mode()
def generate(runner, tokenizer, image_off=(), native=False, noop=False, capture=False):
    from transformers.cache_utils import DynamicCache
    model = runner.model
    n = runner.layout['input_length']
    eos = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    cache = DynamicCache()
    baseline = hooks_count(model)
    input_hashes = {k:tensor_hash(v) for k,v in runner.inputs.items()}
    with ExitStack() as stack:
        control = None if native else stack.enter_context(IController(runner,image_off,noop,capture))
        result = model.language_model.generate(**runner.inputs,past_key_values=cache,
            use_cache=True,do_sample=False,num_beams=1,max_new_tokens=32,
            eos_token_id=eos,pad_token_id=eos,logits_to_keep=1,
            return_dict_in_generate=True,output_scores=True,output_attentions=False,output_hidden_states=False)
        ids = result.sequences[0].tolist()
        assert ids and len(ids)==len(result.scores) and result.past_key_values is cache
        assert all(torch.isfinite(score).all() for score in result.scores)
        length = n+len(ids)-1
        assert all(cache.get_seq_length(i)==length for i in range(28))
        if control:
            assert control.calls==[n]+[1]*(len(ids)-1) and len(control.audits)==28
        text = tokenizer.decode(ids,skip_special_tokens=True).split(model.conv_template.sep.strip())[0].strip()
        first = result.scores[0][0].detach().cpu().clone()
        value = dict(text=text,prediction=text.rstrip('.').lower(),token_ids=ids,
            eos=ids[-1]==eos,truncated=len(ids)==32 and ids[-1]!=eos,empty=not bool(text),
            generated_tokens=len(ids),prefill_mask_audits=control.audits if control else {},
            decode_hooks_removed=True,fresh_cache=True,final_cache_length=length)
        states = control.state_hashes if control else {}
    assert hooks_count(model)==baseline
    assert input_hashes=={k:tensor_hash(v) for k,v in runner.inputs.items()}
    runner.verify()
    return value,first,states


def freeze():
    verify()
    assert not (OUT/'protocol.json').exists()
    pilot = read(RECOVERY/'pilot_sensitive_layers.json')
    assert pilot['questions']==240 and pilot['frozen_before_3000_extension']
    keep = pilot['keep_layers']
    assert len(set(keep))==2
    samples = [s for s in read(RECOVERY/'sample_manifest.json')['samples'] if s['pilot']]
    write(OUT/'sample_manifest.json',dict(samples=samples))
    names = ['internvl_recovery_rescue.py','test_internvl_recovery_rescue.py','relay.py','engine.py']
    write(OUT/'protocol.json',dict(source_sha256={f'vlm_research/visual_pathways/{name}':sha(ROOT/'vlm_research/visual_pathways'/name) for name in names},
        parent_protocol_sha256=sha(RECOVERY/'protocol.json'),
        pilot_selection_sha256=sha(RECOVERY/'pilot_sensitive_layers.json'),
        manifest_sha256=sha(OUT/'sample_manifest.json'),keep_layers=keep,
        conditions={'FullImageAccess':[],'AllImageOFF':list(range(28)),
                    'KeepSparse':[i for i in range(28) if i not in keep]},
        relay='All nonvisual original Prefill Query strictly before P-1 -> Image OFF, all28layers',
        direct_access='Only original P-1 -> Image masked at specified layers, softmax-before -inf',
        decode='Entirely native; independent fresh DynamicCache per condition',
        native_full='One native complete generation per question, verify exact first logits against scan',
        model_revision=IMODEL.name,dtype='bfloat16',attention='eager',max_num=1,image_tokens=256,
        max_new_tokens=32,greedy=True,original_experiment_end=EXPERIMENT_END,
        exploratory=True,no_new_layer_search=True,
        bootstrap=read(RECOVERY/'protocol.json')['bootstrap']))
    print('Frozen original Task3 InternVL rescue, keep:',keep,flush=True)


@torch.inference_mode()
def smoke(model, tokenizer, fast, loader, sample, conditions):
    runner = InternPrefill(model,tokenizer,fast,sample,loader)
    native,full,_ = generate(runner,tokenizer,native=True)
    with ChatCapture(model) as chat:
        text = model.chat(tokenizer,runner.pixels,runner.request,
            dict(do_sample=False,num_beams=1,max_new_tokens=32))
    assert chat.generated_ids==native['token_ids'] and text.strip()==native['text']
    assert torch.equal(chat.first_logits.float(),full.float())
    noop,noop_logits,reference = generate(runner,tokenizer,range(28),noop=True,capture=True)
    assert noop['token_ids']==native['token_ids'] and torch.equal(full,noop_logits)
    prior = None
    checks = []
    for name,layers in conditions.items():
        value,_,states = generate(runner,tokenizer,layers,capture=True)
        assert all(audit['masked_probability_max']==0 for audit in value['prefill_mask_audits'].values())
        assert all(states[i]['image']==reference[i]['image'] for i in range(28))
        if prior is not None:
            assert all(states[i]['prefix']==prior[i]['prefix'] for i in range(28))
        prior = states
        checks.append(dict(condition=name,masked_probability_max=0,image_states_native=True,
            non_p1_prefix_identical_across_direct_access_conditions=True,decode_native=True))
    return dict(passed=True,question_id=sample['question_id'],official_chat_tokens_exact=True,
        official_logits_max_abs=0,noop_logits_max_abs=0,checks=checks)


@torch.no_grad()
def worker(gpu):
    verify()
    protocol = read(OUT/'protocol.json')
    assert all(sha(ROOT/path)==expected for path,expected in protocol['source_sha256'].items())
    assert sha(OUT/'sample_manifest.json')==protocol['manifest_sha256']
    assert sha(RECOVERY/'pilot_sensitive_layers.json')==protocol['pilot_selection_sha256']
    folder = OUT/f'gpu{gpu}'
    preflight(gpu,folder)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,tokenizer,fast = load_intern()
    loader,_ = official_preprocessor(IMODEL)
    samples = read(OUT/'sample_manifest.json')['samples']
    validations = [smoke(model,tokenizer,fast,loader,s,protocol['conditions']) for s in (samples[0],samples[-1])]
    write(folder/'validation.json',dict(passed=True,checks=validations,code_commit=command('git','rev-parse','HEAD')))
    print('INTERNVL RESCUE SMOKE PASS',flush=True)
    weights = [(p._version,p.data_ptr()) for p in model.parameters()]
    baseline = hooks_count(model)
    start = time.perf_counter()
    done = 0
    for sample in samples:
        if datetime.now().astimezone()>=datetime.fromisoformat(EXPERIMENT_END):
            break
        key = f'{sample["task"]}__{sample["question_id"]}'
        path = OUT/'checkpoints'/f'{key}.json'
        if path.exists() and read(path).get('complete'):
            record = read(path)
            assert record['sample_sha256']==digest(sample) and record['protocol_sha256']==sha(OUT/'protocol.json')
            done += 1
            continue
        with torch.inference_mode():
            runner = InternPrefill(model,tokenizer,fast,sample,loader)
            native,full,_ = generate(runner,tokenizer,native=True)
            old = read(RECOVERY/'internvl/checkpoints'/f'{key}.json')
            assert old['complete'] and old['sample_sha256']==digest(sample)
            assert old['input_binding']==digest(runner.layout)
            assert old['full_logits_sha256']==tensor_hash(full.to(torch.bfloat16)), 'Native generation differs from native scan'
            conditions = {}
            for name,layers in protocol['conditions'].items():
                value,changed,_ = generate(runner,tokenizer,layers)
                value.update(metrics(full,changed),correct=int(value['prediction']==sample['answer']),physical_gpu=gpu)
                conditions[name]=value
            write(path,dict(complete=True,task=sample['task'],question_id=sample['question_id'],
                image_id=sample['image_id'],image_sha256=sample['image_sha256'],
                sample_sha256=digest(sample),protocol_sha256=sha(OUT/'protocol.json'),
                input_binding=digest(runner.layout),full_logits_sha256=old['full_logits_sha256'],
                native_full_accuracy=int(native['prediction']==sample['answer']),
                native_full_generation=native,conditions=conditions))
        done += 1
        write(folder/'progress.json',dict(done=done,total=240,seconds=time.perf_counter()-start))
        if done%20==0:
            print('RESCUE',done,240,round(time.perf_counter()-start,1),flush=True)
    assert weights==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==baseline
    write(folder/'complete.json',dict(passed=True,done=done,total=240,seconds=time.perf_counter()-start,
        weights_unchanged=True,hooks_clean=True,no_nan_inf_oom=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','worker'])
    parser.add_argument('--gpu',type=int,choices=[0,2,3],default=3)
    args = parser.parse_args()
    if args.action=='freeze':
        freeze()
    else:
        try:
            worker(args.gpu)
        except BaseException:
            write(OUT/f'gpu{args.gpu}'/'error.json',dict(traceback=traceback.format_exc()))
            raise
