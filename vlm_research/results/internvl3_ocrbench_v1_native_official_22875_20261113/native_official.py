#!/usr/bin/env python3
"""Isolated OCRBench Native harness. Executes pinned VLMEvalKit AST unchanged.

No model, tokenizer, vision, attention, chat or generation implementation is patched.
Transparent observation wrappers only validate and record native inputs/outputs.
"""
import argparse, ast, collections, copy, csv, gc, hashlib, importlib.metadata, json
import os, platform, subprocess, sys, time, traceback, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / 'audit'
OUT = ROOT / 'results'
MODEL = Path('/home/featurize/ate-local-models/InternVL3-8B')
DATA = Path('/home/featurize/ate-local-data/bundles/OCRBench-v1')
REVISION = '853e3a797a661694b1b8ece0cb72dc2b23e3dac9'
AUDIT_COMMIT = '93ff60f839644fabf81f202d699bbc0638484f42'
KIT_COMMIT = '3c192143007e4eb3f5c1a36558a4f8c1451980cb'
SMOKE_IDS = ['0098', '0759', '0053', '0743']

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def json_bytes(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def atomic(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(json.dumps(x, ensure_ascii=False, indent=2).encode() + b'\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
def jsonl(p, rows):
    with Path(p).open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
def tensor_sha(t):
    return sha_bytes(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
def dataset_type(name):
    assert name == 'OCRBench'; return 'VQA'
def dataset_modality(name):
    assert name == 'OCRBench'; return 'IMAGE'
def listinstr(ss, s): return any(x in s for x in ss)

def load_official():
    global torch
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    kit = AUDIT/'official/VLMEvalKit_baseline_pinned/vlmeval/vlm/internvl'
    env = dict(torch=torch, T=T, Image=Image, InterpolationMode=InterpolationMode,
               IMAGENET_MEAN=(.485, .456, .406), IMAGENET_STD=(.229, .224, .225),
               listinstr=listinstr, DATASET_TYPE=dataset_type, DATASET_MODALITY=dataset_modality, os=os)
    records = []
    tree = ast.parse((kit/'utils.py').read_text())
    names = ['build_transform', 'find_closest_aspect_ratio', 'dynamic_preprocess', 'load_image', 'reorganize_prompt']
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(funcs) == len(names)
    for n in funcs: records.append(dict(file='utils.py', name=n.name, ast_sha256=sha_bytes(ast.dump(n).encode())))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(kit/'utils.py'), 'exec'), env)
    tree = ast.parse((kit/'internvl_chat.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InternVLChat')
    names = ['use_custom_prompt', 'build_prompt', 'set_max_num', 'generate_v2']
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(funcs) == len(names)
    for n in funcs: records.append(dict(file='internvl_chat.py', name=n.name, ast_sha256=sha_bytes(ast.dump(n).encode())))
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(kit/'internvl_chat.py'), 'exec'), env)
    wrapper = type('PinnedOCRBenchWrapper', (), {n: env[n] for n in names})()
    wrapper._use_custom_prompt = True
    wrapper.use_mpo_prompt = False; wrapper.use_cot = False; wrapper.use_postprocess = False
    wrapper.use_lmdeploy = False; wrapper.best_of_n = 1; wrapper.system_prompt = None
    wrapper.dump_image = lambda line, dataset: [line['image_path']]
    wrapper.kwargs = dict(do_sample=False, max_new_tokens=4096, top_p=None)
    wrapper.set_max_num('OCRBench')
    assert wrapper.max_num == 24 and wrapper.total_max_num == 64
    return wrapper, env, records

def official_score(raw, text):
    p = AUDIT/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py'
    tree = ast.parse(p.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and
                "data[i]['result']" in ast.unparse(n) and 'data_type' in ast.unparse(n) and 'dataset_name' in ast.unparse(n))
    data = [dict(copy.deepcopy(raw), predict=text)]
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(p), 'exec'), {'data': data})
    assert 'result' in data[0]
    return int(data[0]['result'])

def independent_score(raw, text):
    answers = raw['answers'] if isinstance(raw['answers'], list) else [raw['answers']]
    def norm(s):
        s = s.strip().replace('\n', ' ')
        return s.replace(' ', '') if raw['dataset_name'] == 'HME100k' else s.lower()
    return int(any(norm(str(a)) in norm(text) for a in answers))

def get_rows():
    raw = json.loads((DATA/'OCRBench.json').read_text())
    assert len(raw) == 1000
    with (AUDIT/'question_image_sha256.csv').open() as f: maps = list(csv.DictReader(f))
    assert len(maps) == 1000
    return raw, maps

def make_query(tokenizer, convmod, config, prompt, tiles):
    conv = convmod.get_conv_template(config['template'])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    query = conv.get_prompt().replace('<image>', '<img>' + '<IMG_CONTEXT>'*(tiles*256) + '</img>', 1)
    ids = tokenizer(query, return_tensors='pt')
    return query, ids, conv

def prepare():
    import importlib.util
    from PIL import Image
    from transformers import AutoTokenizer
    wrapper, env, records = load_official()
    torch.set_num_threads(4); torch.set_num_interop_threads(4)
    assert os.getenv('USE_COT') != '1'
    receipt = json.loads((ROOT/'control/model_integrity.json').read_text())
    assert receipt['status'] == 'VERIFIED' and receipt['revision'] == REVISION
    raw, maps = get_rows()
    assert sha_file(DATA/'OCRBench.json') == 'df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864'
    config = json.loads((MODEL/'config.json').read_text())
    spec = importlib.util.spec_from_file_location('pinned_conversation', MODEL/'conversation.py')
    convmod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = convmod; spec.loader.exec_module(convmod)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    ratios = sorted(set((i,j) for n in range(1,25) for i in range(1,n+1) for j in range(1,n+1) if i*j<=24),key=lambda x:x[0]*x[1])
    plans = []; gold = []; counts = collections.Counter(); checked = {}
    for i, (row, m) in enumerate(zip(raw, maps)):
        qid = f'{i:04d}'; assert m['question_id'] == qid and m['image_id'] == row['image_path']
        image = DATA/'OCRBench_Images'/row['image_path']
        if str(image) not in checked:
            checked[str(image)] = sha_file(image)
        assert checked[str(image)] == m['sha256']
        with Image.open(image) as im:
            width,height=im.size
        grid = env['find_closest_aspect_ratio'](width/height,ratios,width,height,448)
        tiles = grid[0]*grid[1]; tiles += int(tiles>1); counts[tiles] += 1
        message = wrapper.build_prompt(dict(question=row['question'], image_path=str(image)), 'OCRBench')
        prompt = env['reorganize_prompt'](message,1,'OCRBench')
        assert prompt == '<image>\n'+row['question']+'\nAnswer the question using a single word or phrase.'
        query, ids, conv = make_query(tokenizer,convmod,config,prompt,tiles)
        assert int((ids['input_ids']==tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')).sum()) == tiles*256
        plans.append(dict(question_id=qid, official_id=row['id'], image_id=row['image_path'], image_sha256=m['sha256'],
                          image_size=[width,height], grid=list(grid), tiles=tiles, visual_tokens=tiles*256,
                          question=row['question'], prompt=prompt, prompt_sha256=sha_bytes(prompt.encode()),
                          chat_query_sha256=sha_bytes(query.encode()), input_ids_sha256=tensor_sha(ids['input_ids']),
                          attention_mask_sha256=tensor_sha(ids['attention_mask']), input_tokens=ids['input_ids'].numel(),
                          eos_token_id=tokenizer.convert_tokens_to_ids(conv.sep.strip())))
        gold.append(dict(question_id=qid, official_id=row['id'], answers=row['answers'], category=row['type'], dataset_name=row['dataset_name']))
    assert len(checked)==930 and max(counts)==25
    for qid in SMOKE_IDS:
        row=plans[int(qid)]
        px=env['load_image'](DATA/'OCRBench_Images'/row['image_id'],max_num=24,upscale=False)
        assert px.shape[0]==row['tiles']; row['cpu_smoke_pixel_values_bf16_sha256']=tensor_sha(px.to(torch.bfloat16))
    protocol = dict(model='OpenGVLab/InternVL3-8B', revision=REVISION, audit_commit=AUDIT_COMMIT,
                    vlmevalkit_commit=KIT_COMMIT, method='Native', dtype='bfloat16', attention='eager',
                    model_all_on_gpu=True, quantization=False, cpu_offload=False, batch_size=1,
                    model_max_dynamic_patch_config=config['max_dynamic_patch'], effective_max_num=24,
                    image_size=448, use_thumbnail=True, max_total_tiles=25, visual_tokens_per_tile=256,
                    rgb=True, upscale=False, imagenet_mean=[.485,.456,.406], imagenet_std=[.229,.224,.225],
                    use_custom_prompt=True,use_cot=False,use_mpo_prompt=False,best_of_n=1,version='V2.0',
                    kwargs_at_wrapper_init=wrapper.kwargs, kwargs_at_chat=dict(do_sample=False,max_new_tokens=4096,temperature=.6,top_p=.95),
                    template=config['template'], system_prompt=conv.system_message, eos_token_id=plans[0]['eos_token_id'],
                    annotation_sha256=sha_file(DATA/'OCRBench.json'), sample_order='original OCRBench.json, 0000..0999',
                    samples=1000, unique_original_images=930, smoke_ids=SMOKE_IDS,
                    smoke_reuse='Successful smoke predictions reused for identical formal inputs; final rows in original order',
                    ast_extraction=records, tile_histogram=dict(sorted(counts.items())),
                    dependency_shim='Only OCRBench DATASET_TYPE=VQA, modality=IMAGE, dump_image returns byte-identical local original; unmodified official AST',
                    model_source_sha256={p.name:sha_file(p) for p in MODEL.glob('*.py')}, harness_sha256=sha_file(__file__))
    assert not (OUT/'protocol.json').exists()
    jsonl(OUT/'inputs.jsonl',plans); jsonl(OUT/'gold.jsonl',gold)
    protocol['inputs_manifest_sha256']=sha_file(OUT/'inputs.jsonl');protocol['gold_manifest_sha256']=sha_file(OUT/'gold.jsonl')
    atomic(OUT/'protocol.json',protocol)
    atomic(OUT/'cpu_input_checks.json',dict(status='PASS',questions=1000,unique_images=930,all_original_image_sha256=True,
        all_prompts_tokenized=True,all_image_token_counts=True,smoke_cpu_tiles={q:plans[int(q)]['tiles'] for q in SMOKE_IDS},
        environment={p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','timm','Pillow','einops','tokenizers']}))
    print('CPU_INPUTS_PASS',json.dumps(dict(tile_histogram=protocol['tile_histogram'],smoke={q:plans[int(q)]['tiles'] for q in SMOKE_IDS})),flush=True)

def run():
    from transformers import AutoModel, AutoTokenizer
    wrapper, env, _ = load_official()
    protocol=json.loads((OUT/'protocol.json').read_text()); binding=sha_file(OUT/'protocol.json')
    assert protocol['harness_sha256']==sha_file(__file__)
    assert sha_file(OUT/'inputs.jsonl')==protocol['inputs_manifest_sha256']
    assert sha_file(OUT/'gold.jsonl')==protocol['gold_manifest_sha256']
    plans=[json.loads(s) for s in (OUT/'inputs.jsonl').read_text().splitlines()]
    raw,maps=get_rows(); state={}; observed={}; handles=[]; model=None; started=time.time()
    def progress(status,**kw): atomic(OUT/'progress.json',dict(status=status,protocol_sha256=binding,complete=len(state),total=1000,elapsed_s=time.time()-started,**kw))
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>=23*2**30
        free,total=torch.cuda.mem_get_info(); assert free>22*2**30,(free,total)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.free','--format=csv,noheader'],text=True).strip()
        progress('LOADING_MODEL',gpu=gpu)
        tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
        model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,use_flash_attn=False,attn_implementation='eager').eval().cuda()
        assert all(p.device.type=='cuda' for p in model.parameters())
        assert {p.dtype for p in model.parameters() if p.is_floating_point()}=={torch.bfloat16}
        assert model.language_model.config._attn_implementation=='eager'
        assert model.num_image_token==256 and model.config.template==protocol['template']
        assert model.system_message==protocol['system_prompt']
        wrapper.model=model;wrapper.tokenizer=tokenizer;wrapper.device=model.device
        atomic(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,gpu=gpu,torch=torch.__version__,
            transformers=importlib.metadata.version('transformers'),cuda_runtime=torch.version.cuda,
            frozen_dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            language_attention=model.language_model.config._attn_implementation,vision_flash=model.vision_model.config.use_flash_attn,
            all_weights_bfloat16=True,all_weights_gpu=True,num_image_token=model.num_image_token,
            system_message=model.system_message,chat_template=model.template,
            language_generation_config=model.language_model.generation_config.to_dict(),
            torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads()))
        orig_gen=model.generate;orig_feature=model.extract_feature
        def feature(*args,**kwargs):
            x=orig_feature(*args,**kwargs)
            assert list(x.shape[:2])==[observed['tiles'],256]
            assert bool(torch.isfinite(x).all()),'Nonfinite visual features'
            observed['visual_feature_shape']=list(x.shape);observed['feature_finite']=True
            return x
        def generate(*args,**kwargs):
            ids=kwargs['input_ids'];mask=kwargs['attention_mask'];px=kwargs['pixel_values']
            row=plans[int(observed['question_id'])]
            observed.update(tiles=px.shape[0],pixel_values_bf16_sha256=tensor_sha(px),
                input_ids_sha256=tensor_sha(ids),attention_mask_sha256=tensor_sha(mask),
                image_context_tokens=int((ids==tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')).sum()),
                actual_generation_kwargs={k:v for k,v in kwargs.items() if k not in ['pixel_values','input_ids','attention_mask']})
            assert px.shape[0]==row['tiles'] and observed['image_context_tokens']==row['tiles']*256
            assert observed['input_ids_sha256']==row['input_ids_sha256']
            assert observed['attention_mask_sha256']==row['attention_mask_sha256']
            if row.get('cpu_smoke_pixel_values_bf16_sha256'):
                assert observed['pixel_values_bf16_sha256']==row['cpu_smoke_pixel_values_bf16_sha256']
            assert observed['actual_generation_kwargs']==dict(do_sample=False,max_new_tokens=4096,temperature=.6,top_p=.95,eos_token_id=row['eos_token_id'])
            out=orig_gen(*args,**kwargs)
            observed['generated_token_ids']=out[0].detach().cpu().tolist()
            observed['raw_decode_with_special_tokens']=tokenizer.decode(out[0],skip_special_tokens=False)
            return out
        def lm_hook(module,args,kwargs,output):
            assert bool(torch.isfinite(output.logits).all()),'Nonfinite LM logits'
            observed['lm_forward_calls']=observed.get('lm_forward_calls',0)+1
            if observed['lm_forward_calls']==1:
                observed['prefill_logits_shape']=list(output.logits.shape)
                embeds=kwargs.get('inputs_embeds');assert embeds is not None
                assert embeds.shape[1]==plans[int(observed['question_id'])]['input_tokens']
                observed['prefill_input_embeds_shape']=list(embeds.shape)
            observed['all_logits_finite']=True
        model.generate=generate;model.extract_feature=feature
        handles.append(model.language_model.register_forward_hook(lm_hook,with_kwargs=True))
        stored=OUT/'per_question';stored.mkdir(exist_ok=True)
        for p in stored.glob('*.json'):
            r=json.loads(p.read_text());assert r['protocol_sha256']==binding and r['status']=='OK'
            assert r['input']==plans[int(r['question_id'])];state[r['question_id']]=r
        def infer(qid,phase):
            if qid in state:return state[qid]
            row=plans[int(qid)];observed.clear();observed['question_id']=qid
            progress(phase,current_question_id=qid,current_tiles=row['tiles'])
            message=wrapper.build_prompt(dict(question=row['question'],image_path=str(DATA/'OCRBench_Images'/row['image_id'])),'OCRBench')
            torch.cuda.reset_peak_memory_stats();t=time.time()
            with torch.inference_mode():text=wrapper.generate_v2(message,'OCRBench')
            torch.cuda.synchronize()
            tokens=observed['generated_token_ids'];assert tokens and observed.get('feature_finite') and observed.get('all_logits_finite')
            score=official_score(raw[int(qid)],text); independent=independent_score(raw[int(qid)],text);assert score==independent
            r=dict(question_id=qid,status='OK',protocol_sha256=binding,input=row,text=text,
                official_score=score,independent_score=independent,empty_output=not text.strip(),
                generated_tokens=len(tokens),stopped_on_eos=tokens[-1]==row['eos_token_id'],
                truncated=len(tokens)>=4096 and tokens[-1]!=row['eos_token_id'],observation=copy.deepcopy(observed),
                elapsed_s=time.time()-t,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,phase=phase)
            atomic(stored/(qid+'.json'),r);state[qid]=r
            print('PREDICTION',qid,'tiles',row['tiles'],'tokens',len(tokens),'score',score,'seconds',round(r['elapsed_s'],2),flush=True)
            return r
        for qid in SMOKE_IDS:infer(qid,'SMOKE')
        atomic(OUT/'smoke.json',dict(status='PASS',ids=SMOKE_IDS,protocol_sha256=binding,
            cases=[state[q] for q in SMOKE_IDS],checks=['native model.chat','all tile visual features','tiles*256 image token mapping',
            'native ChatML input_ids and attention_mask equal CPU','same CPU/GPU BF16 pixels','finite vision features and LM logits',
            'unchanged official generate_v2 kwargs','raw decode saved','official and independent scoring equal']))
        progress('SMOKE_PASS')
        for row in plans:infer(row['question_id'],'FULL_NATIVE')
        ordered=[state[r['question_id']] for r in plans];assert len(ordered)==1000
        jsonl(OUT/'predictions.jsonl',ordered)
        categories=collections.defaultdict(lambda:dict(n=0,score=0))
        for r,x in zip(ordered,raw):categories[x['type']]['n']+=1;categories[x['type']]['score']+=r['official_score']
        score=sum(r['official_score'] for r in ordered)
        atomic(OUT/'summary.json',dict(status='COMPLETE',N=1000,unique_images=930,official_score=score,independent_score=sum(r['independent_score'] for r in ordered),
            paper_reference=880,difference_from_paper=score-880,categories=dict(categories),empty_outputs=sum(r['empty_output'] for r in ordered),
            truncated=sum(r['truncated'] for r in ordered),non_eos=sum(not r['stopped_on_eos'] for r in ordered),
            peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in ordered),peak_reserved_MiB=max(r['peak_reserved_MiB'] for r in ordered),
            protocol_sha256=binding,elapsed_s=time.time()-started,model_loads_this_process=1))
        progress('COMPLETE',official_score=score)
    except BaseException as e:
        atomic(OUT/'failure.json',dict(status='BLOCKED',type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),
            current_observation=observed,completed_ids=sorted(state),elapsed_s=time.time()-started,
            cuda_allocated_MiB=torch.cuda.memory_allocated()/2**20 if torch.cuda.is_initialized() else None,
            cuda_reserved_MiB=torch.cuda.memory_reserved()/2**20 if torch.cuda.is_initialized() else None))
        progress('BLOCKED',error_type=type(e).__name__,error=str(e));traceback.print_exc();raise
    finally:
        for h in handles:h.remove()
        wrapper.model=None
        if model is not None:
            model.generate=None;model.extract_feature=None
        model=None;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        print('EXPERIMENT_FINISHED_GPU_RELEASE_ON_PROCESS_EXIT',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.mode=='prepare' else run()
