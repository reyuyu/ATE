"""Native-first gated TextVQA run. No polling, no training, no speed benchmark."""
import argparse
from contextlib import ExitStack
import gc
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback

from .common import *


def resource_audit():
    gpus = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader'],text=True)
    active = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    choices = []
    for line in gpus.splitlines():
        idx,name,uuid,used,total,util = [x.strip() for x in line.split(',')]
        if '3090' in name and uuid not in active and int(used) < 256 and int(util) == 0:
            choices.append(uuid)
    result = dict(status='IDLE_GPU_AVAILABLE' if choices else 'BLOCKED_GPU_BUSY',
                  gpus=gpus, processes=processes, idle_uuids=choices, current_pid=os.getpid(),
                  timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), other_processes_untouched=True)
    write(OUT/'gpu_preflight.json', result)
    return result


def freeze():
    import torch
    from transformers import AutoTokenizer
    from .dynamic import engines
    assert not torch.cuda.is_initialized()
    assert not (OUT/'protocol.json').exists(), 'Never replace frozen protocol'
    validation = json.loads((OUT/'dataset_validation.json').read_text())
    assert validation['questions'] == 5000
    source = json.loads((OUT/'release_source_manifest.json').read_text())
    assert source['commit'] == COMMIT
    base, late, edits = engines()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    conv = conversation()
    write(OUT/'dynamic_length_source_audit.json', edits)
    paths = list((ROOT/'vlm_research/internvl3_textvqa_official').glob('*.py'))
    paths += [UPSTREAM/e['path'] for e in source['entries'] if e.get('commit') == COMMIT]
    for package in ['internvl3_svh_physical_d_pilot','internvl3_visual_writing_figure3_3000',
                    'internvl3_svh_r_terminal_windows_textvqa']:
        paths += list((HISTORY/'vlm_research'/package).glob('*.py'))
    assets = [p for p in MODEL.iterdir() if p.is_file() and p.suffix in ['.py','.json','.txt']]
    assets += [INDEPENDENT_SCORER]
    write(OUT/'protocol.json',dict(experiment='INTERNVL3_TEXTVQA_VAL_OFFICIAL_HF_ALIGNMENT',
        status='FROZEN_NOT_GPU_VALIDATED', framework_commit=COMMIT, dataset='TextVQA_VAL', questions=5000,
        model='OpenGVLab/InternVL3-8B', revision=MODEL.name, dtype='bfloat16', attention='eager',
        runtime='Official VLMEvalKit release-time HF branch; not the later LMDeploy default',
        official_leaderboard_runtime_exact_commit='UNPUBLISHED_UNRESOLVED', official_reference_score_percent=80.2,
        native_diagnostic_gate=dict(max_absolute_difference_pp=1.0,
            meaning='Predeclared reproduction check only; no parameter tuning against score.'),
        generation=GENERATION, min_tiles=1,max_grid_tiles=6,use_thumbnail=True,max_tiles_including_thumbnail=7,
        eos_token=conv.sep.strip(),eos_token_id=tokenizer.convert_tokens_to_ids(conv.sep.strip()),
        pad_token_id=tokenizer.pad_token_id,greedy_seed=20261113,
        image_size=448,vision_tokens_per_tile=256,external_OCR=False,
        prompt_suffix='Answer the question using a single word or phrase.',chat_template='internvl2_5',
        scorer='Pinned VLMEvalKit process_line(vqa_score); independent archived MMF/LLaVA evaluator. Normalization differences reported, not silently substituted.',
        late9=dict(reaccess_layers=KEEP, stages=['P-1','all_real_decode'],R_writing_layers=list(range(20)),
            answer_attention_skip=list(range(2,6)),hidden_exit_after_layer=27,persistent_image_KV_layers=KEEP),
        numerical_gate=dict(native_noop='torch.equal all effective logits and token IDs',
            dense_physical_max_abs=0.02, dense_physical_tokens='identical',
            historical_numerical_waiver='NOT_INHERITED; fail stops SVH'),
        manifest_sha256=sha(OUT/'sample_manifest.jsonl.gz'),
        source_sha256={str(p):sha(p) for p in paths+assets},
        weights_stat={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in MODEL.glob('*.safetensors')},
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','Pillow','numpy']},
        smoke_ids=validation['smoke_ids'], bootstrap_seed=20261113,bootstrap_replicates=20000,
        no_GPU_work_started=True))


def check():
    p = json.loads((OUT/'protocol.json').read_text())
    assert p['manifest_sha256'] == sha(OUT/'sample_manifest.jsonl.gz')
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, ('SOURCE_CHANGED',path)
    for path, expected in p['weights_stat'].items():
        stat = Path(path).stat()
        assert [stat.st_size,stat.st_mtime_ns] == expected, ('WEIGHT_CHANGED',path)
    return p


class ChatCapture:
    def __init__(self, model, logits=False):
        self.model, self.logits, self.scores, self.ids = model, logits, [], None

    def __enter__(self):
        self.stack = ExitStack()
        old = self.model.generate
        def capture_input(*args,**kw):
            self.input_ids = kw['input_ids'].detach().cpu().clone()
            return old(*args,**kw)
        self.model.generate = capture_input
        self.stack.callback(setattr,self.model,'generate',old)
        lm = self.model.language_model
        generate = lm.generate
        def capture_output(*args,**kw):
            result = generate(*args,**kw)
            self.ids = result[0].detach().cpu().tolist()
            return result
        lm.generate = capture_output
        self.stack.callback(setattr,lm,'generate',generate)
        self.finite = None
        def forward(_module,_args,out):
            import torch
            finite = torch.isfinite(out.logits).all()
            self.finite = finite if self.finite is None else self.finite & finite
            if self.logits:
                self.scores.append(out.logits[0,-1].detach().cpu().clone())
        self.stack.callback(lm.register_forward_hook(forward).remove)
        return self

    def __exit__(self,*args):
        return self.stack.__exit__(*args)


def inputs(row,tok,scope):
    import torch
    assert sha(row['image_path']) == row['image_sha256']
    pixels = scope['load_image'](row['image_path'],max_num=6).to(torch.bfloat16)
    assert tensor_sha(pixels) == row['pixels_bf16_sha256']
    prompt,ids,roles = render(tok,row['request'],row['tiles'])
    assert prompt == row['prompt'] and ids[0].tolist() == row['input_ids']
    assert tensor_sha(ids) == row['input_ids_sha256'] and roles == row['roles']
    return dict(ids=ids.to('cuda:0'),pixels=pixels.to('cuda:0'),roles=roles,
                request=row['request'],prompt=prompt)


def native(model,tok,ready,row,logits=False):
    with ChatCapture(model,logits) as capture:
        text = model.chat(tok,ready['pixels'],row['request'],dict(GENERATION),num_patches_list=[row['tiles']])
    assert capture.input_ids[0].tolist() == row['input_ids'] and bool(capture.finite)
    eos = tok.convert_tokens_to_ids(model.conv_template.sep.strip())
    assert capture.ids is not None
    result = dict(text=text,token_ids=capture.ids,generated_tokens=len(capture.ids),
        eos=bool(capture.ids and capture.ids[-1] == eos),empty=not text.strip(),
        truncated=len(capture.ids)==GENERATION['max_new_tokens'] and capture.ids[-1]!=eos)
    return result,capture.scores


def smoke_native(model,tok,scope,rows):
    import torch
    selected = json.loads((OUT/'protocol.json').read_text())['smoke_ids']
    evidence = []
    for row in [r for r in rows if r['question_id'] in selected]:
        ready = inputs(row,tok,scope)
        a,x = native(model,tok,ready,row,True)
        before = sum(len(m._forward_pre_hooks) for m in model.modules())
        with ExitStack() as stack:
            for layer in model.language_model.model.layers:
                stack.callback(layer.register_forward_pre_hook(lambda _module, _args: None).remove)
            b,y = native(model,tok,ready,row,True)
        assert before == sum(len(m._forward_pre_hooks) for m in model.modules())
        assert a['token_ids'] == b['token_ids'] and len(x)==len(y)
        assert all(torch.equal(v,w) for v,w in zip(x,y)), 'NATIVE_NOOP_NUMERICAL_FAIL'
        evidence.append(dict(question_id=row['question_id'],tiles=row['tiles'],input_tokens=row['input_tokens'],
            token_ids=a['token_ids'],native_noop_effective_logits_exact=True,
            input_binding_sha256=row['input_binding_sha256']))
        write(OUT/'native_smoke.json',dict(status='PARTIAL',checks=evidence))
    write(OUT/'native_smoke.json',dict(status='PASS',checks=evidence))


def smoke_late(model,tok,scope,rows):
    import torch
    from .dynamic import engines
    _,late,_ = engines()
    selected = json.loads((OUT/'protocol.json').read_text())['smoke_ids']
    evidence = []
    for row in [r for r in rows if r['question_id'] in selected]:
        ready = inputs(row,tok,scope)
        before = late.frozen.hooks_count(model)
        dense,ds,dc = late.generate(model,tok,ready,'Late9-Dense',KEEP,cutoff=28,
            max_tokens=GENERATION['max_new_tokens'],physical=False,audit=True)
        physical,ps,pc = late.generate(model,tok,ready,'Late9',KEEP,cutoff=28,
            max_tokens=GENERATION['max_new_tokens'],physical=True,audit=True)
        errors = [float((a.float()-b.float()).abs().max()) for a,b in zip(ds,ps)]
        record = dict(question_id=row['question_id'],tiles=row['tiles'],dense_tokens=dense['token_ids'],
            physical_tokens=physical['token_ids'],per_step_max_abs=errors,
            physical_audit=physical['audit'],kv_layout=physical['kv_layout'])
        evidence.append(record)
        write(OUT/'late9_smoke.json',dict(status='PARTIAL',checks=evidence))
        assert before == late.frozen.hooks_count(model)
        assert dense['token_ids'] == physical['token_ids'] and len(ds)==len(ps), 'DENSE_PHYSICAL_TOKEN_FAIL'
        assert errors and max(errors) <= 0.02, 'DENSE_PHYSICAL_NUMERICAL_FAIL'
        assert all(r['visual_kv'] == (row['image_tokens'] if r['layer'] in KEEP else 0) for r in physical['kv_layout'])
        del ds,ps,dc,pc,ready
        gc.collect()
    write(OUT/'late9_smoke.json',dict(status='PASS',checks=evidence))


def worker(uuid):
    import torch
    from transformers import AutoModel,AutoTokenizer
    protocol = check()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == uuid
    preflight = resource_audit()
    assert uuid in preflight['idle_uuids'], 'GPU_BECAME_BUSY'
    torch.set_num_threads(1)
    torch.manual_seed(protocol['greedy_seed'])
    model = tok = None
    current_request = None
    db = sqlite3.connect(OUT/'server_only/predictions.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS records(method TEXT,qid TEXT,sha256 TEXT,payload TEXT,PRIMARY KEY(method,qid))')
    ph = sha(OUT/'protocol.json')
    previous = db.execute('SELECT value FROM meta WHERE key="protocol_sha256"').fetchone()
    if previous:
        assert previous[0] == ph
    else:
        db.execute('INSERT INTO meta VALUES (?,?)',('protocol_sha256',ph));db.commit()
    try:
        assert torch.__version__ == '2.6.0+cu124'
        tok = AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
        model = AutoModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16,trust_remote_code=True,
            low_cpu_mem_usage=True,use_flash_attn=False,device_map={'':'cuda:0'},local_files_only=True).eval()
        model.img_context_token_id = tok.convert_tokens_to_ids('<IMG_CONTEXT>')
        assert model.language_model.config._attn_implementation == 'eager'
        actual = 'GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
        assert actual == uuid
        scope,rows = upstream(),manifest()
        write(OUT/'gpu_environment.json',dict(uuid=uuid,model_revision=MODEL.name,
            vision_attention='eager',language_attention='eager',dtype='bfloat16',protocol_sha256=ph))
        with torch.inference_mode():
            smoke_native(model,tok,scope,rows)
            for method in ['Native','Late9']:
                if method == 'Late9':
                    from .analyze import analyze
                    gate = analyze(require_native=True)
                    assert gate['native_protocol_gate'] == 'PASS', 'NATIVE_REPRODUCTION_GATE_FAILED'
                    smoke_late(model,tok,scope,rows)
                for number,row in enumerate(rows):
                    current_request = dict(method=method,question_id=row['question_id'],
                        input_binding_sha256=row['input_binding_sha256'],image_tokens=row['image_tokens'])
                    old = db.execute('SELECT sha256,payload FROM records WHERE method=? AND qid=?',
                                     (method,row['question_id'])).fetchone()
                    if old:
                        assert hashlib.sha256(old[1].encode()).hexdigest() == old[0]
                        assert json.loads(old[1])['input_binding_sha256'] == row['input_binding_sha256']
                        continue
                    ready = inputs(row,tok,scope)
                    if method == 'Native':
                        result,_ = native(model,tok,ready,row)
                    else:
                        from .dynamic import engines
                        _,late,_ = engines()
                        result,_,control = late.generate(model,tok,ready,'Late9',KEEP,cutoff=28,
                            max_tokens=GENERATION['max_new_tokens'],physical=True,audit=False)
                        result.pop('audit',None)
                        del control
                    value,normalized = score(scope,result['text'],row['gold'])
                    record = dict(result,method=method,question_id=row['question_id'],gold=row['gold'],score=value,
                        normalized_prediction=normalized,input_binding_sha256=row['input_binding_sha256'],
                        prompt_sha256=row['prompt_sha256'],input_ids_sha256=row['input_ids_sha256'],
                        pixels_bf16_sha256=row['pixels_bf16_sha256'],image_sha256=row['image_sha256'],
                        image_tokens=row['image_tokens'],tiles=row['tiles'],gpu_uuid=uuid,protocol_sha256=ph)
                    raw = json.dumps(record,sort_keys=True,ensure_ascii=True)
                    db.execute('INSERT INTO records VALUES (?,?,?,?)',(method,row['question_id'],hashlib.sha256(raw.encode()).hexdigest(),raw))
                    db.commit()
                    del ready,result
                    if (number+1) % 50 == 0:
                        write(OUT/'progress.json',dict(method=method,visited=number+1,total=5000))
                        print('PROGRESS',method,number+1,5000,flush=True)
            from .analyze import analyze
            analyze(require_native=True)
    except BaseException:
        write(OUT/'gpu_failure.json',dict(status='FAIL_OR_BLOCKED',traceback=traceback.format_exc(),uuid=uuid,
            current_request=current_request,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))
        raise
    finally:
        db.close()
        del model,tok
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        write(OUT/'gpu_release_audit.json',dict(status='WORKER_EXITING',pid=os.getpid(),uuid=uuid,
            other_processes_untouched=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','launch','worker','preflight'])
    parser.add_argument('--uuid')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'preflight':
        print(json.dumps(resource_audit(),indent=2))
    elif args.action == 'worker':
        worker(args.uuid)
    else:
        check()
        audit = resource_audit()
        if not audit['idle_uuids']:
            print('BLOCKED_GPU_BUSY: no model loaded; no polling or external process interruption.')
            return 2
        uuid = audit['idle_uuids'][0]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (OUT/'worker.log').open('a') as stream:
            result = subprocess.run([sys.executable,'-u','-m','vlm_research.internvl3_textvqa_official.run',
                'worker','--uuid',uuid],env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(OUT/'worker_exit.json',dict(exit_code=result.returncode,uuid=uuid))
        return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
