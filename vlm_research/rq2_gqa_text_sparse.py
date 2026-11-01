"""Minimal scheduling/aggregation only; call the unchanged historical generators."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

from benchmarks.dataset_registry import digest, sha256, freeze_json
from benchmarks.smoke import hook_footprint
from knockout_probe import tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from paper_rq1_early_joint_attention.statistics import bootstrap
from rq2_visual_full import run as full
from rq2_decode_visual import run as sparse
from rq2_visual_handoff import run as text
from rq2_visual_sufficiency.engine import generate as native_generate
from vlm_common import load_model, gpu_memory

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/rq2_gqa_text_vs_sparse_3000'
BASE='9edcbceb437b0eb30c712001c299a2a9108dc6dd'
GPUS=[(0,text.ALLOWED[0]),(2,text.ALLOWED[2]),(3,text.ALLOWED[3])]
KEYS=['Full','Only_Text','Sparse']
read=text.read


def source_hashes():
    return dict(text.sources(), **{'rq2_gqa_text_sparse.py':sha256(__file__)})


def score(g,s):
    assert g['score']==int(gqa_prediction(g['text'])==s['answer'])
    assert 1<=len(g['token_ids'])<=32 and g['eos']==(g['token_ids'][-1]==2)
    assert g['truncated']==(len(g['token_ids'])==32 and not g['eos'])
    return g


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    fp=read(full.OUT/'protocol.json');tp=text.verify();sp=sparse.verify()
    for p in [fp,tp,sp]:
        assert p['model_revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
        assert p['model_files']==fp['model_files'] and p['transformers_source']==fp['transformers_source']
    assert fp['generation']==sp['generation']==dict(do_sample=False,num_beams=1,max_new_tokens=32,eos_token_id=2,use_cache=True)
    assert tp['generation']['gqa_max_new_tokens']==32 and tp['generation']['use_cache']
    samples=read(full.OUT/'sample_manifest.json')['samples']
    assert len(samples)==len({s['question_id'] for s in samples})==3000
    assert Counter(s['task'] for s in samples)==Counter({t:500 for t in full.TASKS})
    tm=read(text.OUT/'sample_manifest.json');sm=read(sparse.OUT/'sample_manifest.json')
    reused={s['question_id']:s for s in tm['samples']}
    assert reused=={s['question_id']:s for s in sm['samples']} and len(reused)==240
    bindings={};correct=0
    for s in samples:
        q=s['question_id']; assert sha256(s['image_path'])==s['image_sha256']
        h=full.history()[q];p=ROOT/h['full_path'];r=read(p)
        assert sha256(p)==h['full_sha256'] and r['sample_sha256']==h['sample_sha256']==digest(s)
        assert h['prompt_sha256']==digest(s['input_mapping']) and digest(r['generation'])==r['result_sha256']
        g=score(r['generation'],s); assert g['edits']==0;correct+=g['score']
        binding=dict(Full=dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),generation=g))
        if q in reused:
            assert reused[q]==s
            rank=tm['assignments']['main'][q]
            p=text.OUT/f"shard{rank}/main/{s['task']}__{q}/B.json"
            r=text.valid_checkpoint(p,digest(tp),s,'B');g=score(r['result']['generation'],s)
            assert g['condition']=='B' and not g['noop'] and g['mask_layers']==list(range(32))
            assert g['own_cache'] and g['own_prefix'] and g['p_minus_one']==len(s['input_mapping']['prompt_ids'])-1
            assert g['image_indices']==[i for i,t in enumerate(s['input_mapping']['prompt_ids']) if t==32000]
            assert all(z['image_probability_max']==0 for z in r['result']['attention_rows'] if z['query_group']!='Relay')
            assert all(not z['blocked'] for z in r['result']['attention_rows'] if z['query_group']=='Relay')
            binding['Only_Text']=dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),generation=g,
                raw=str((text.OUT/r['raw_file']).relative_to(ROOT)),raw_sha256=r['raw_sha256'])
            r=sparse.checkpoint(s,digest(sp));assert r is not None
            p=sparse.OUT/f"checkpoints/{s['task']}__{q}.json";g=score(r['generation'],s)
            assert g['masked_layers']==sparse.BLOCKED and g['own_cache'] and g['own_prefix']
            assert g['original_p_minus_one']==len(s['input_mapping']['prompt_ids'])-1
            assert g['image_key_indices']==binding['Only_Text']['generation']['image_indices']
            for step in g['decode_mask_audit']:
                assert [z['layer'] for z in step['probabilities']]==list(range(32))
                assert all(z['image_probability_max']==0 for z in step['probabilities'] if z['blocked'])
            binding['Sparse']=dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),generation=g,
                raw=str((sparse.OUT/r['logits_file']).relative_to(ROOT)),raw_sha256=r['logits_sha256'])
        bindings[q]=dict(sample_sha256=digest(s),conditions=binding)
    assert correct==2627
    groups=defaultdict(list)
    for s in samples:groups[s['image_sha256']].append(s)
    assignments={};loads=[0]*3
    for image,ss in sorted(groups.items(),key=lambda kv:(-len(kv[1]),digest(['rq2-text-sparse',kv[0]]))):
        rank=min(range(3),key=lambda k:(loads[k],k))
        for s in ss:assignments[s['question_id']]=rank
        loads[rank]+=sum(2 for s in ss if s['question_id'] not in reused)
    freeze_json(OUT/'sample_manifest.json',dict(samples=samples,assignments=assignments,
        original_manifest_sha256=sha256(full.OUT/'sample_manifest.json'),new_generations_per_shard=loads,
        smoke_ids=sorted(reused)[:3]))
    freeze_json(OUT/'history_bindings.json',bindings)
    stream=io.StringIO();suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_rq2_decode_visual','test_rq2_visual_handoff'])
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    freeze_json(OUT/'cpu_validation.json',dict(passed=tests.wasSuccessful(),tests=tests.testsRun,output=stream.getvalue()))
    assert tests.wasSuccessful(),stream.getvalue()
    for p,b in fp['model_files'].items():assert sha256(p)==b['sha256']
    freeze_json(OUT/'protocol.json',dict(base=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=source_hashes(),
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json','history_bindings.json','cpu_validation.json']},
        historical_protocols={str(p.relative_to(ROOT)):sha256(p) for p in [full.OUT/'protocol.json',text.OUT/'protocol.json',sparse.OUT/'protocol.json']},
        model_revision=fp['model_revision'],model_files=fp['model_files'],transformers_source=fp['transformers_source'],
        generation=fp['generation'],dtype='FP16',attention='eager',transformers='4.49.0',
        conditions=dict(Full='Native historical Full3000',Only_Text='Native relay; original P-1 and all Decode Image OFF all32',
            Sparse='Native relay; original P-1 and all Decode Image OFF except L14/L19'),
        full_reused=3000,only_text_reused=240,sparse_reused=240,new_generations=5520,
        statistics=dict(comparison='Only_Text minus Sparse, accuracy percentage points',seed=20261102,
            repetitions=100000,unit='whole image SHA256 cluster paired bootstrap, retain question weighting',
            interval='ordinary two-sided percentile 95%; no NI/equivalence claim'),
        worktree_created_epoch=(ROOT.parent/'.git').stat().st_mtime,
        stop='Only original3000, two existing strategies. No plots, new ablations, training or algorithm development.'))
    print('Frozen: Full2627/3000; reuse240+240; new5520; shard loads',loads,'CPU tests',tests.testsRun,flush=True)


def verify():
    p=read(OUT/'protocol.json');assert p['source']==source_hashes()
    for n,h in p['bindings'].items():assert sha256(OUT/n)==h
    for n,h in dict(p['historical_protocols'],**p['transformers_source']).items():
        assert sha256(ROOT/n if not Path(n).is_absolute() else n)==h
    for n,b in p['model_files'].items():
        st=Path(n).stat();assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    return p


def execute(model,inputs,bos,key,**kwargs):
    if key=='Only_Text':return text.generate(model,inputs,bos,'B',**kwargs)
    assert key=='Sparse'
    return sparse.generate(model,inputs,bos,**kwargs)


def smoke(model,processor,s,h,rank):
    data=text.prompt(model,processor,s);bos=processor.tokenizer.bos_token_id
    foot=hook_footprint(model);before=tensor_hashes(data);versions={n:p._version for n,p in model.named_parameters()}
    g,f,trace,debug=native_generate(model,data,bos,[],collect=True,audit=True)
    assert g['token_ids']==h['Full']['generation']['token_ids'] and torch.equal(f,full.native_logits(s,True))
    for key in KEYS[1:]:
        noop,nf,nt,nd=execute(model,data,bos,key,noop=True,collect=True,audit=True)
        assert torch.equal(debug['prefill_logits'],nd['prefill_logits'])
        assert noop['token_ids']==g['token_ids'] and len(nt)==len(trace) and all(torch.equal(a,b) for a,b in zip(nt,trace))
        actual,first,_,d=execute(model,data,bos,key,audit=True)
        assert actual['token_ids']==h[key]['generation']['token_ids']
        with np.load(ROOT/h[key]['raw']) as raw:assert np.array_equal(first.numpy(),raw['first_logits'])
        assert torch.equal(debug['prefill_logits'][:,:-1],d['prefill_logits'][:,:-1])
        if key=='Only_Text':
            assert all(r['image_probability_max']==0 for r in d['attention_rows'] if r['query_group']!='Relay')
        else:
            assert all(r['Image_max']==0 if r['layer'] in sparse.BLOCKED else r['Image']>0 for r in d['observer'].rows)
    text.guard(model,data,foot,versions,before)
    atomic_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,protocol_sha256=sha256(OUT/'protocol.json'),
        question_id=s['question_id'],native_history_exact=True,noop_all_logits_exact=True,
        historical_both_strategy_tokens_first_logits_exact=True,non_p1_prefill_logits_exact=True,
        masked_probabilities_zero=True,hooks_clean=True))


def worker(rank):
    p=verify();binding=digest(p);folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding;print('Already complete');return
    gpu,uuid=GPUS[rank];assert os.environ.get('CUDA_VISIBLE_DEVICES') in [str(gpu),uuid] and gpu!=1
    table,apps=text.snapshot();fields=next(r for r in table.splitlines() if r.startswith(str(gpu)+', ')).split(', ')
    assert fields[1]==uuid and int(fields[4])>=18000 and int(fields[5])==0
    assert not any(r.startswith(uuid) for r in apps.splitlines())
    load_start=time.time();model,processor,env=load_model(full.MODEL_PATH)
    assert env['transformers']=='4.49.0' and env['model_revision']==p['model_revision']
    assert env['parameter_dtypes']==['torch.float16'] and not model.training
    manifest=read(OUT/'sample_manifest.json');hist=read(OUT/'history_bindings.json')
    byq={s['question_id']:s for s in manifest['samples']};q=manifest['smoke_ids'][rank]
    smoke(model,processor,byq[q],hist[q]['conditions'],rank)
    deadline=time.time()+300
    while not all((OUT/f'shard{k}/smoke.json').exists() for k in range(3)):
        assert time.time()<deadline,'GPU smoke barrier timeout; inspect other worker logs'
        assert not list(OUT.glob('failure_*.json')),'Another worker failed; stop'
        time.sleep(1)
    assert all(read(OUT/f'shard{k}/smoke.json')['passed'] for k in range(3))
    start=time.time();done=resumed=0;foot=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    for s in manifest['samples']:
        q=s['question_id']
        if manifest['assignments'][q]!=rank or len(hist[q]['conditions'])==3:continue
        data=None
        for key in KEYS[1:]:
            path=folder/f'checkpoints/{q}/{key}.json'
            if path.exists():
                r=read(path);assert r['binding']==binding and r['sample_sha256']==digest(s) and r['condition']==key
                assert digest(r['generation'])==r['generation_sha256'];resumed+=1;done+=1;continue
            if data is None:data=text.prompt(model,processor,s);before=tensor_hashes(data)
            g,first,_,_=execute(model,data,processor.tokenizer.bos_token_id,key)
            result={k:g[k] for k in ['token_ids','eos','truncated','decode_steps','own_cache','own_prefix']}
            result['text']=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            result.update(prediction=gqa_prediction(result['text']),score=int(gqa_prediction(result['text'])==s['answer']))
            score(result,s);text.guard(model,data,foot,versions,before)
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),condition=key,
                generation=result,generation_sha256=digest(result),first_logits_sha256=tensor_hash(first),
                physical_gpu=gpu,hooks_clean=True,inputs_unchanged=True,parameters_unchanged=True))
            done+=1
        if done%100==0:print(f'GPU{gpu} {done}/{manifest["new_generations_per_shard"][rank]}; {time.time()-start:.1f}s',flush=True)
    assert done==manifest['new_generations_per_shard'][rank]
    verify();atomic_json(folder/'completion.json',dict(passed=True,binding=binding,physical_gpu=gpu,pid=os.getpid(),
        generations=done,resumed=resumed,started_epoch=start,finished_epoch=time.time(),
        formal_seconds=time.time()-start,worker_seconds=time.time()-load_start,environment=env,memory=gpu_memory(),
        hooks_clean=hook_footprint(model)==foot))
    print('GPU',gpu,'complete, exiting',flush=True)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate():
    p=verify();binding=digest(p);manifest=read(OUT/'sample_manifest.json');hist=read(OUT/'history_bindings.json')
    completions=[read(OUT/f'shard{k}/completion.json') for k in range(3)]
    assert all(c['passed'] and c['binding']==binding for c in completions)
    assert sum(c['generations'] for c in completions)==5520
    records={};rows=[];reused=Counter()
    for s in manifest['samples']:
        q=s['question_id'];results={};row=dict(task=s['task'],question_id=q,image_id=s['image_id'],image_sha256=s['image_sha256'],gold=s['answer'])
        for key in KEYS:
            if key in hist[q]['conditions']:
                old=hist[q]['conditions'][key];assert sha256(ROOT/old['path'])==old['sha256']
                g=old['generation'];source='historical';reused[key]+=1
            else:
                path=OUT/f"shard{manifest['assignments'][q]}/checkpoints/{q}/{key}.json";r=read(path)
                assert r['binding']==binding and r['sample_sha256']==digest(s) and r['condition']==key
                assert digest(r['generation'])==r['generation_sha256'] and r['hooks_clean'] and r['inputs_unchanged'] and r['parameters_unchanged']
                g=r['generation'];source='new'
            score(g,s);results[key]=g
            row.update({key+'_score':g['score'],key+'_text':g['text'],key+'_token_ids':str(g['token_ids']),key+'_source':source})
        records[q]=results;rows.append(row)
    assert dict(reused)==dict(Full=3000,Only_Text=240,Sparse=240)
    assert sum(records[q]['Full']['score'] for q in records)==2627
    summary=[];tasks=[];transitions=[]
    for task in ['all']+list(full.TASKS):
        ss=[s for s in manifest['samples'] if task=='all' or s['task']==task]
        for key in KEYS:
            gs=[records[s['question_id']][key] for s in ss]
            row=dict(task=task,condition=key,n=len(gs),correct=sum(g['score'] for g in gs),
                accuracy_percent=100*np.mean([g['score'] for g in gs]),eos=sum(g['eos'] for g in gs),
                truncated=sum(g['truncated'] for g in gs),empty=sum(not bool(g['text']) for g in gs))
            (summary if task=='all' else tasks).append(row)
        a=np.array([records[s['question_id']]['Only_Text']['score'] for s in ss]);b=np.array([records[s['question_id']]['Sparse']['score'] for s in ss])
        transitions.append(dict(task=task,n=len(ss),only_text_correct_sparse_wrong=int(((a==1)&(b==0)).sum()),
            sparse_correct_only_text_wrong=int(((a==0)&(b==1)).sum()),both_correct=int(((a==1)&(b==1)).sum()),both_wrong=int(((a==0)&(b==0)).sum())))
    differences=np.array([[100*(records[s['question_id']]['Only_Text']['score']-records[s['question_id']]['Sparse']['score'])] for s in manifest['samples']])
    stats=bootstrap(differences,[s['image_sha256'] for s in manifest['samples']],[1],
        repetitions=p['statistics']['repetitions'],seed=p['statistics']['seed'])[0]
    comparison=dict(comparison='Only_Text minus Sparse',n=3000,delta_accuracy_pp=stats['mean'],ci95_low_pp=stats['ci95_low'],ci95_high_pp=stats['ci95_high'],image_clusters=stats['clusters'],bootstrap_repetitions=stats['repetitions'])
    write_csv('accuracy_summary.csv',summary);write_csv('task_accuracy.csv',tasks);write_csv('paired_comparison.csv',[comparison]);write_csv('mutual_transitions.csv',transitions);write_csv('per_question.csv',rows)
    table,apps=text.snapshot();pids={str(c['pid']) for c in completions}
    assert not any(any(pid==field.strip() for field in line.split(',')) for pid in pids for line in apps.splitlines())
    atomic_json(OUT/'gpu_release.json',dict(released=True,gpus=[0,2,3],gpu1_used=False,snapshot=table,compute_apps=apps))
    lines=['# GQA3000：Only Text 与 Sparse Visual + Text','', '|条件|正确数/3000|Accuracy|','|---|---:|---:|']
    for r in summary:lines.append(f"|{r['condition']}|{r['correct']}/3000|{r['accuracy_percent']:.4f}%|")
    lines+=['',f"Only Text − Sparse = **{stats['mean']:+.4f} pp**；图片SHA256整簇配对95%CI **[{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}] pp**。固定种子20261102、100,000次Bootstrap，保留题目加权，共{stats['clusters']}个图片簇。未作非劣效或等价认定。",'',
        '|任务|题数|Full|Only Text|Sparse|','|---|---:|---:|---:|---:|']
    for task in full.TASKS:
        values=[next(r['accuracy_percent'] for r in tasks if r['task']==task and r['condition']==k) for k in KEYS]
        lines.append(f'|{task}|500|'+ '|'.join(f'{v:.2f}%' for v in values)+'|')
    t=transitions[0];lines+=['',f"Only Text正确、Sparse错误：**{t['only_text_correct_sparse_wrong']}题**；Sparse正确、Only Text错误：**{t['sparse_correct_only_text_wrong']}题**；两者均正确{t['both_correct']}题，均错误{t['both_wrong']}题。",'',
        '复用原六类各500题、LLaVA-1.5-7B固定revision、FP16/eager/Transformers4.49.0、原始Prompt、Greedy、独立KV Cache、EOS及最多32 Token，并沿用原GQA完整答案归一化评分。',
        'Full全部3000题复用RQ2.3绑定的历史结果；Only Text复用RQ2.6的240题；Sparse只复用RQ2.4的P−1+Decode 240题，未复用RQ2.3的P−1-only Sparse。Question ID、完整样本/Prompt绑定、图片SHA256、模型和Mask定义均逐题核验。',
        '直接调用未修改的RQ2.6 Only Text生成器和RQ2.4 Sparse生成器。Mask为Softmax前来源−inf；非P−1 Prefill读图正常，MLP/残差/位置及全部视觉Token保留。没有新算法、图表或额外条件。',
        '三卡均通过简短No-op及Mask Smoke：历史Full/两策略Token与首步Logits精确一致，No-op全部有效Prefill/Decode Logits精确一致，非P−1 Prefill输出精确一致。仅使用GPU0/2/3；GPU1未使用。',
        f"正式新增5520次生成。三卡正式墙钟约{(max(c['finished_epoch'] for c in completions)-min(c['started_epoch'] for c in completions))/60:.2f}分钟；从Worktree创建至汇总约{(time.time()-p['worktree_created_epoch'])/60:.2f}分钟，不含最初阅读和之后推送。",'',
        '本轮比较为已有研究样本上的策略复核，不作为全新图像隔离确认集。来源Mask会重分配概率，不能直接解释为独立语义贡献；没有实际剪枝或加速结论。本轮到此停止。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'validation.json',dict(passed=True,total_questions=3000,reused=dict(reused),new_generations=5520,
        cpu_tests=read(OUT/'cpu_validation.json')['tests'],all_shards_complete=True,no_duplicate_questions=True,
        source_unchanged=True,gpus_released=True,code_commit=p['code_commit'],paired=comparison,
        elapsed_since_worktree_seconds=time.time()-p['worktree_created_epoch']))
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name not in ['worker.lock','artifact_hashes.json']]
    atomic_json(OUT/'artifact_hashes.json',{str(f.relative_to(OUT)):sha256(f) for f in sorted(files)})
    print(summary,comparison,transitions[0],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','worker','aggregate']);parser.add_argument('--rank',type=int,default=0);args=parser.parse_args()
    try:
        if args.stage=='worker':worker(args.rank)
        else:globals()[args.stage]()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/f'failure_{args.stage}_{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
