"""One frozen 60-question, four-condition transfer check; no training or search."""
import sys,os,json,hashlib,time,subprocess,gc,traceback,unittest,threading,signal,csv
from pathlib import Path
WT=Path('/data1/likai/ATE-worktrees/semantic-single-gpu-search3h')
REF=Path('/data/lk/ATE-worktrees/qwen3vl-gqa3000-visual-writing-necessity')
P=WT/'vlm_research/results/semantic_single_gpu_search3h/qwen_fixed60'
ROOT=P.parent;PRIVATE=Path('/data1/likai/ATE-resources/experiments/semantic_qwen_fixed60')
sys.path.insert(0,str(REF))
import torch,numpy as np
from transformers import AutoProcessor,AutoConfig,Qwen3VLForConditionalGeneration
from vlm_research.qwen3vl_early_windows.experiment import MODEL,REVISION,prepare,tensor_digest,hook_counts
from vlm_research.qwen3vl_semantic_visual_writers.run import make_spec,cache_fingerprint
from vlm_research.qwen3vl_semantic_visual_writers.engine import generate,WriterMask
from vlm_research.qwen3vl_query_role.engine import generate as native,feature_hashes
from vlm_research.qwen3vl_query_role.run import environment,check_models
from vlm_research.qwen3vl_semantic_visual_writers import tests
from vlm_research.qwen3vl_query_role import tests as role_tests
from vlm_research.semantic_visual_writers.groups import intact
from vlm_research.qwen3vl_semantic_visual_writers.common import hardware

GPU='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
CONDITIONS=['All-Writer','Question-Only','Semantic-k','Fixed-Q-Conditional-Assistant']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def gold(s):return read(Path('/data1/likai/ATE-resources/experiments/gqa5h_receiver_search/gold')/(s['question_id']+'.json'))['answer']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
def table(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def guard():assert time.monotonic()<read(ROOT/'clock.json')['gpu_stop_monotonic']-120
def preflight():
    state=hardware();fields=[s.strip() for s in next(s for s in state['table'].splitlines() if GPU in s).split(',')]
    assert GPU not in state['compute'] and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0
    assert os.environ['CUDA_VISIBLE_DEVICES']==GPU
    return state
def inputs(processor,s,sp):
    x,l=prepare(processor,s);assert l==sp['layout'];return {k:v.to('cuda:0') for k,v in x.items()}

def main():
    P.mkdir();PRIVATE.mkdir(exist_ok=True);guard()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),unittest.defaultTestLoader.loadTestsFromModule(role_tests)])
    test=unittest.TextTestRunner(verbosity=1).run(suite);assert test.wasSuccessful() and not torch.cuda.is_initialized()
    write(P/'cpu_tests.json',dict(passed=True,tests=test.testsRun))
    splits=read(ROOT/'split_manifest.json');ids=splits['dev_stages']['60'];assert len(ids)==60
    samples=[read(Path('/data1/likai/ATE-resources/experiments/gqa5h_receiver_search/samples')/(q+'.json')) for q in ids]
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    specs={};groupaudit=[]
    for s in samples:
        q=s['question_id'];_,layout=prepare(proc,s);sp=make_spec(proc,config,s['question'],layout);groups=sp['selection']['groups'];Q=layout['question_positions']
        assistants=[g for g in groups if any(w.lower()=='assistant' for w in g['words']) and min(g['positions'])>max(Q)]
        assert len(assistants)==1 and set(assistants[0]['positions'])<=set(sp['roles']['R'])
        ll=read(Path('/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research/results/gqa5h_receiver_search/input_specs')/(q+'.json'))
        words=s['question'].lower().split();hit=words[0] in ['what','which'] or len(ll['Q'])>=15
        sp['selection']['writers']['Question-Only']=Q
        sp['selection']['writers']['Fixed-Q-Conditional-Assistant']=sorted(set(Q)| (set(assistants[0]['positions']) if hit else set()))
        for c in CONDITIONS:assert intact(sp['selection']['writers'][c],groups)
        specs[q]=sp;groupaudit.append(dict(question_id=q,assistant_positions=assistants[0]['positions'],assistant_token_ids=[layout['input_ids'][j] for j in assistants[0]['positions']],assistant_decoded=assistants[0]['decoded'],conditional_hit=hit,llava_Q_tokens=len(ll['Q']),qwen_Q_tokens=len(Q)))
    assert sorted(__import__('collections').Counter(s['task'] for s in samples).values())==[10]*6
    write(P/'sample_manifest.json',dict(samples=samples,source_split_sha256=sha(ROOT/'split_manifest.json'),selection='Exact previously frozen DEV60; six tasks x ten; no new selection'))
    write(P/'token_manifest.json',specs);write(P/'assistant_mapping_audit.json',groupaudit)
    manifest=read(REF/'vlm_research/results/qwen3vl_semantic_visual_writers_240/model_manifest.json');check_models(manifest);write(P/'model_manifest.json',manifest)
    env=environment();assert env['versions']['transformers']=='4.57.6';write(P/'environment.json',env)
    src=[p for folder in ['qwen3vl_semantic_visual_writers','qwen3vl_query_role','qwen3vl_early_windows','qwen3vl_smoke','semantic_visual_writers'] for p in (REF/'vlm_research'/folder).glob('*.py')]
    write(P/'protocol.json',dict(no_training=True,no_search=True,model='Qwen3-VL-8B-Instruct',revision=REVISION,dtype='bfloat16',layers=36,attention='eager',conditions=CONDITIONS,n=60,greedy=True,max_new_tokens=32,logits_to_keep=1,source_sha256={str(p):sha(p) for p in src},driver_sha256=sha(__file__),sample_sha256=sha(P/'sample_manifest.json'),token_sha256=sha(P/'token_manifest.json'),gpu=GPU,original_deadline=read(ROOT/'clock.json'),rule='All Question writers; add actual native assistant role word if first word what/which OR ORIGINAL LLAVA Question token count >=15. Native Qwen token positions; no literal ASSISTANT inserted.',cross_model_limit='Native Qwen prompt/tokenizer/BF16/36 layers differ from LLaVA; descriptive fixed-sample transfer check. No fitting. R includes vision_end. All-Writer differs from Native Full. Mean budget is measured, not assumed <50%.'))
    write(P/'resource_preflight.json',preflight());model=None;rows=[]
    def watchdog():
        while time.monotonic()<read(ROOT/'clock.json')['gpu_stop_monotonic']-60:time.sleep(1)
        os.kill(os.getpid(),signal.SIGINT)
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        guard();model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
        assert torch.cuda.device_count()==1 and len(model.model.language_model.layers)==36 and {v.dtype for v in model.parameters()}=={torch.bfloat16}
        checks=[]
        smoke=sorted(samples,key=lambda s:(specs[s['question_id']]['layout']['input_length'],s['question_id']))
        for s in [smoke[0],smoke[len(smoke)//2],smoke[-1]]:
            guard();q=s['question_id'];sp=specs[q];x=inputs(proc,s,sp);before={k:tensor_digest(v) for k,v in x.items()};versions=[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
            g,ss,obs,parent=native(model,proc,x,sp['roles'],native=True,audit=True,trace=True);fh=feature_hashes(obs);ph=cache_fingerprint(parent)
            for c in CONDITIONS:
                gg,sc,oo,own=generate(model,proc,x,sp,c,noop=True,audit=True,trace=True)
                assert gg['token_ids']==g['token_ids'] and len(ss)==len(sc) and all(torch.equal(a,b) for a,b in zip(ss,sc)) and feature_hashes(oo)==fh
                assert own is not parent and all(a.keys.data_ptr()!=b.keys.data_ptr() for a,b in zip(parent.layers,own.layers)) and cache_fingerprint(parent)==ph
                del own,oo,sc
                gg,sc,oo,own=generate(model,proc,x,sp,c,audit=True,trace=True)
                assert feature_hashes(oo)==fh and oo.mlp_calls==[len(gg['token_ids'])]*36 and cache_fingerprint(parent)==ph
                assert all(e['blocked_probability_max']==0 and e['non_target_mask_exact'] for e in gg['source_events'])
                assert own is not parent and all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(parent.layers,own.layers))
                del own,oo,sc
            assert before=={k:tensor_digest(v) for k,v in x.items()} and versions==[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
            count=hook_counts(model)
            try:
                with WriterMask(model,sp,[]):raise RuntimeError('cleanup-test')
            except RuntimeError as e:assert str(e)=='cleanup-test'
            assert hook_counts(model)==count
            checks.append(dict(question_id=q,native_noop_all_logits_torch_equal=True,mask_zero=True,MLP36=True,independent_cache=True,inputs_and_weights_unchanged=True,hook_cleanup=True))
            del x,parent,obs,ss;gc.collect();torch.cuda.empty_cache();print('QWEN_SMOKE_PASS',q,flush=True)
        write(P/'gpu_smoke_validation.json',dict(passed=True,cases=checks))
        for j,s in enumerate(samples):
            guard();q=s['question_id'];sp=specs[q];x=inputs(proc,s,sp)
            for c in CONDITIONS:
                g,sc,_,_=generate(model,proc,x,sp,c);assert all(e['blocked_probability_max']==0 for e in g['source_events'])
                w=sp['selection']['writers'][c];raw=PRIVATE/q/(c+'.npz');raw.parent.mkdir(parents=True,exist_ok=True)
                with raw.open('xb') as f:np.savez_compressed(f,first_logits=sc[0].numpy())
                row=dict(question_id=q,task=s['task'],image_sha256=s['image_sha256'],condition=c,question=s['question'],gold=gold(s),answer=g['text'],correct=int(g['prediction']==gold(s).lower()),writers=json.dumps(w),k=len(w),R=len(sp['roles']['R']),R_ratio=len(w)/len(sp['roles']['R']),token_ids=json.dumps(g['token_ids']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],first_token=g['first_token_id'])
                write(P/'checkpoints'/q/(c+'.json'),dict(**row,generation=g,input_hashes=sp['layout']['input_tensor_hashes'],raw_path=str(raw),raw_sha256=sha(raw)))
                rows.append(row);del sc
            del x;gc.collect();torch.cuda.empty_cache();print('QWEN_PROGRESS',j+1,60,flush=True)
        table(P/'per_question.csv',rows);summary=[]
        for c in CONDITIONS:
            rr=[r for r in rows if r['condition']==c];summary.append(dict(condition=c,n=len(rr),correct=sum(r['correct'] for r in rr),ACC=100*np.mean([r['correct'] for r in rr]),mean_R_ratio=np.mean([r['R_ratio'] for r in rr])))
        table(P/'accuracy_summary.csv',summary);write(P/'completion.json',dict(status='COMPLETE',n=60,records=len(rows),no_training=True,summary=summary));print('QWEN_COMPLETE',json.dumps(summary),flush=True)
    except BaseException:
        write(P/'STOP.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache();write(P/'gpu_release.json',dict(resources=hardware(),note='Snapshot before interpreter exit; final independent snapshot confirms release.'))

if __name__=='__main__':main()
