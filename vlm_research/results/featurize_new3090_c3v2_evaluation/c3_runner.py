"""One-condition single-GPU adapter; imported frozen calculation paths unchanged."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
    CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
import gc,gzip,hashlib,json,platform,shutil,sqlite3,subprocess,sys,time,traceback
from pathlib import Path

REPO=Path('/home/featurize/work/ATE')
DATA=Path('/home/featurize/data/ate-recovery')
REC=REPO/'vlm_research/results/featurize_new3090_recovery'
OUT=REPO/'vlm_research/results/featurize_new3090_c3v2_evaluation'
AUDIT=REPO/'vlm_research/results/featurize_new3090_scoring_audit/audit_summary.json'
HIST=REC/'history/cvpr2026-information-horizon-alignment/vlm_research/results/cvpr2026_information_horizon_alignment'
MODEL=DATA/'models/llava-1.5-7b-hf'
REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
COMMIT='e281754ef17b3accc09d6938b5b9f020048b2aec'
COUNTS={'MME':2374,'ScienceQA-IMG':2017,'POPE':8910,'TextVQA':5000,'MMBench':4377}
ORDER=list(COUNTS)
VIEWS={'MME':'MME-paper','ScienceQA-IMG':'ScienceQA-IMG-paper','POPE':'POPE-paper',
    'TextVQA':'TextVQA-OCR','MMBench':'MMBench-paper'}
MME_CHANGED='landmark/03d5e3bfc958be38.jpg'
MME_NEW_SHA='ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a'
sys.path[:0]=[str(REPO),str(REC/'scripts')]
import torch,transformers
from PIL import Image
import ate_data
from vlm_research.information_horizon_alignment.common import digest,tensor_sha,hook_counts
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.information_horizon_alignment import score as scorer
from vlm_research.information_horizon_alignment import analyze
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.fastp2_bypass_writer.run import operators
from vlm_research.physical_visual_handoff.engine import equivalence


def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,cwd=REPO,text=True,timeout=45).strip()
def atomic(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def loadjson(p):return json.loads(Path(p).read_text())


def sources_check(protocol):
    for p,h in protocol['sources'].items():assert sha(REPO/p)==h,p
    assert command(['git','rev-parse','HEAD'])==COMMIT
    assert not command(['git','status','--porcelain','--untracked-files=no'])


def manifest():
    protocol=loadjson(HIST/'protocol_ate_paper.json')
    assert protocol['model_revision']==REV and protocol['backend']=='eager'
    assert transformers.__version__==protocol['transformers']=='4.49.0'
    sources_check(protocol)
    assert sha(HIST/'sample_manifest.jsonl.gz')==protocol['artifacts']['sample_manifest.jsonl.gz']
    with gzip.open(HIST/'sample_manifest.jsonl.gz','rt') as f: historical=[json.loads(l) for l in f]
    assert Counter(s['benchmark'] for s in historical)==COUNTS
    bykey={s['key']:s for s in historical};mapped=[];bindings={}
    for bench in ORDER:
        view=VIEWS[bench];rows=list(ate_data.iter_inputs(view));gold=ate_data.load_answers(view)
        assert len(rows)==COUNTS[bench] and {r['question_id'] for r in rows}==set(gold)
        bindings[bench]=dict(view=view,rows=len(rows),inputs_sha256=sha(ate_data.dataset_dir(view)/'inputs.jsonl'),
            answers_sha256=sha(ate_data.dataset_dir(view)/'answers.json'))
        if bench=='POPE':assert Counter(r['strategy'] for r in rows)=={'random':2910,'popular':3000,'adversarial':3000}
        for row in rows:
            key=f'{bench}:{row["question_id"]}';h=bykey[key]
            assert row['question']==h['question'] and row['image_id']==h['image_id']
            assert row['prompt']==h['prompt'] and row['user_text']==h['user_text']
            assert gold[row['question_id']]==h['answer'] and Path(row['image_path']).is_file()
            assert h['max_new_tokens']==protocol['max_new_tokens'][bench]
            if bench in ['MMBench','ScienceQA-IMG']:
                choices=row['choices'];choices=dict(zip('ABCDE',choices)) if isinstance(choices,list) else choices
                assert choices==h['options']
            if bench=='TextVQA':assert h['paper_ocr'] and row['paper_ocr'] and 'Reference OCR token:' in h['prompt']
            if bench=='POPE':assert h['strategy']==row['strategy'] and h['question_id'] in row['official_question_ids']
            s=dict(h,image_path=row['image_path'],answer=gold[row['question_id']],
                original_sample_sha256=digest(h),normalized_question_id=row['question_id'],input_view=view)
            if bench=='MME' and row['image_id']==MME_CHANGED:
                s.update(historical_image_sha256=h['image_sha256'],image_sha256=MME_NEW_SHA,
                    known_recovered_image_version_difference=True)
            mapped.append(s)
    mapped.sort(key=lambda s:(ORDER.index(s['benchmark']),s['key']))
    assert len(mapped)==22678 and len({s['key'] for s in mapped})==22678
    pairs=defaultdict(list);groups=defaultdict(list)
    for s in mapped:
        if s['benchmark']=='MME':pairs[s['pair_id']].append(s)
        if s['benchmark']=='MMBench':groups[s['circular_group']].append(s)
    assert len(pairs)==1187 and all(len(g)==2 for g in pairs.values())
    historical_groups=defaultdict(set)
    for h in historical:
        if h['benchmark']=='MMBench':historical_groups[h['circular_group']].add(h['key'])
    assert set(groups)==set(historical_groups)
    assert all({s['key'] for s in g}==historical_groups[k] and
        len({s['options'][s['answer']] for s in g})==1 for k,g in groups.items())
    bindings['MMBench']['circular_groups']=len(groups)
    bindings['MMBench']['group_size_counts']=dict(Counter(len(g) for g in groups.values()))
    bindings['MMBench']['group_scope']='Every rotation present in original frozen Manifest; fixed D options can yield 3 rotations among 4 choices'
    scorer.AUTHOR=DATA/'sources/Information-Horizon';scorer.text_evaluator();scorer.matcher()
    acceptance=loadjson(REC/'model_acceptance.json')
    assert acceptance['success'] and acceptance['revision']==REV
    for name,e in acceptance['files'].items():assert (MODEL/name).stat().st_size==e['bytes']
    return mapped,protocol,bindings


def setup(rows,protocol,bindings):
    OUT.mkdir(exist_ok=True)
    mapped_digest=digest(rows)
    config=dict(code_commit=COMMIT,model='llava-hf/llava-1.5-7b-hf',model_revision=REV,
        model_path=str(MODEL),condition='Physical-C3',algorithm='Original Physical C3-V2',
        protocol_sha256=sha(HIST/'protocol_ate_paper.json'),source_sha256=protocol['sources'],
        original_manifest_sha256=sha(HIST/'sample_manifest.jsonl.gz'),mapped_manifest_digest=mapped_digest,
        inputs=bindings,expected_counts=COUNTS,expected_total=22678,
        max_new_tokens=protocol['max_new_tokens'],dtype='float16',attention='eager',batch_size=1,
        greedy=True,num_beams=1,temperature=0,eos='model generation_config; no additional stop strings',
        semantics=dict(R_image_layers=list(range(20)),physical_visual_exit_before=20,
            answer_image_layers=[],physical_answer_attention_skip=[2,3,4,5],
            norm_mlp_residual='unchanged frozen decoder layers',rope='original unrenumbered positions',
            cache='physical compact SelectiveKV; no image persistence or concatenation'),
        MME_difference=dict(image=MME_CHANGED,old_sha256='61d010ea9fabdc0bf930362d3cc1e4dc554789036f219feba927551effecd510',
            recovered_sha256=MME_NEW_SHA,affected_questions=2,data_not_replaced=True),
        formal_condition_only=True,no_gqa=True,no_native_generation=True,no_reaccess_generation=True)
    binding=digest(config)
    if (OUT/'run_metadata.json').exists():
        metadata=loadjson(OUT/'run_metadata.json');assert metadata['binding']==binding
    else:
        report=(HIST/'report_zh.md').read_text()
        assert '0/68,034' in report
        dbs=[str(p) for p in (REPO/'vlm_research/results').rglob('*.sqlite') if OUT not in p.parents]
        assert not dbs,'Unexpected existing database requires explicit comparability review before generation'
        completion=json.loads(command(['git','show','HEAD:vlm_research/results/cvpr2026_information_horizon_alignment/completion.json']))
        assert completion['stage_B_completed']==0
        metadata=dict(status='NOT_VALIDATED',created_at=now(),binding=binding,config=config,
            runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
                transformers=transformers.__version__),runner_path=str(Path(__file__).resolve()),runner_sha256=sha(__file__),
            existing_result_check=dict(local_result_sqlite=dbs,historical_alignment_completion=completion,
                historical_stage_B_completed=0,compatible_complete_C3_found=False,
                legacy_results_not_reused='Different frozen prompt / OCR view / generation cap or implementation; GQA outside scope'),
            native_comparison=dict(status='NOT_COMPARABLE_FULL_CONTROL',
                reason='Only same-protocol Native 15-question Smoke; no complete HF Paper Native cohort. Historical Image-only/Author-Native are different protocols/backends.'),
            model_loads_this_run=0,physical_smoke_status='NOT_VALIDATED')
        with gzip.GzipFile(filename=str(OUT/'mapped_manifest.jsonl.gz'),mode='wb',mtime=0) as f:
            for s in rows:f.write((json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        atomic(OUT/'run_metadata.json',metadata)
    return metadata


@contextmanager
def execution_trace(adapter,layout):
    result=dict(vision_calls=0,projector_calls=0,image_embedding_exact=False,
        layer_events=[],norm_calls=[0]*32,post_norm_calls=[0]*32,mlp_calls=[0]*32,
        skip_residual_exact_checks=0)
    handles=[];projected=[];layer_input={}
    mask=adapter.inputs['input_ids'][0]==32000;n=layout.length
    def vision(m,args,out):
        result['vision_calls']+=1;assert out.last_hidden_state.device.type=='cuda'
        result['vision_shape']=list(out.last_hidden_state.shape)
    def project(m,args,out):
        result['projector_calls']+=1;assert list(out.shape)==[1,576,4096];projected.append(out.detach())
    def language(m,args,kw):
        if projected:
            assert torch.equal(kw['inputs_embeds'][0,mask],projected.pop().reshape(576,4096))
            result['image_embedding_exact']=True
    try:
        handles.append(adapter.model.vision_tower.register_forward_hook(vision))
        handles.append(adapter.model.multi_modal_projector.register_forward_hook(project))
        handles.append(adapter.model.language_model.register_forward_pre_hook(language,with_kwargs=True))
        for i,layer in enumerate(adapter.layers):
            def before(m,args,kw,i=i):
                step=sum(e['layer']==i for e in result['layer_events'])
                positions=list(range(n)) if step==0 and i<20 else list(layout.text) if step==0 else [n+step-1]
                assert args[0].shape[1]==len(positions)
                assert kw['cache_position'].tolist()==positions and kw['position_ids'][0].tolist()==positions
                if i in [2,3,4,5]:layer_input[i]=args[0].detach().clone()
                result['layer_events'].append(dict(layer=i,step=step,rows=len(positions),
                    position_ids_sha256=digest(positions),first_position=positions[0],last_position=positions[-1]))
            handles.append(layer.register_forward_pre_hook(before,with_kwargs=True))
            def norm(m,args,out,i=i):result['norm_calls'][i]+=1
            def postnorm(m,args,i=i):
                result['post_norm_calls'][i]+=1
                if i in [2,3,4,5]:
                    assert torch.equal(args[0][:,-1],layer_input.pop(i)[:,-1])
                    result['skip_residual_exact_checks']+=1
            def mlp(m,args,out,i=i):result['mlp_calls'][i]+=1
            handles.append(layer.input_layernorm.register_forward_hook(norm))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(postnorm))
            handles.append(layer.mlp.register_forward_hook(mlp))
        yield result
    finally:
        for h in handles:h.remove()
        projected.clear();layer_input.clear()


def prepare(adapter,s):
    return frozen.prepare(adapter,dict(prompt=s['prompt'],user_text=s['user_text'],image_path=s['image_path']))


def smoke(adapter,rows):
    selected=[next(s for s in rows if s['benchmark']==b) for b in ORDER]
    details=[]
    for s in selected:
        assert sha(s['image_path'])==s['image_sha256']
        layout,binding=prepare(adapter,s)
        assert layout.cutoff==20 and len(layout.image)==576 and layout.length-1 not in layout.relay
        input_hash=digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})
        with operators(adapter) as (calls,mlp,shapes),execution_trace(adapter,layout) as trace:
            actual,pv=frozen.hf_generate(adapter,layout,'Physical-C3',3,capture=True,audit=True)
        assert len(actual['token_ids'])>=2,'Smoke needs real decode'
        steps=len(actual['token_ids']);n=layout.length
        assert trace['vision_calls']==trace['projector_calls']==1 and trace['image_embedding_exact']
        assert trace['norm_calls']==trace['post_norm_calls']==trace['mlp_calls']==[steps]*32
        assert trace['skip_residual_exact_checks']==4*steps
        assert all(mlp[i]==steps for i in range(32))
        for i in [2,3,4,5]:
            assert all(calls[i,op]==1 for op in ['q_proj','k_proj','v_proj','o_proj'])
            projection={r['op']:r['shape'][1] for r in shapes if r['layer']==i}
            assert projection=={'q_proj':n-1,'k_proj':n,'v_proj':n,'o_proj':n-1}
        assert all(e['blocked_probability_max']==0 for e in actual['attention_audit'])
        assert all(e['answer_image_mass']==0 and not e['image_keys_permitted'] for e in actual['visual_audit'])
        assert all(not e['image_keys_present'] for e in actual['visual_audit'] if not e['prefill'] or e['layer']>=20)
        assert any(e['relay_image_mass']>0 for e in actual['visual_audit'] if e['prefill'] and e['layer']<20)
        kv=actual['kv_audit'];assert kv['retained_image_layers']==[] and kv['historical_cache_copies']==kv['append_concatenations']==0
        for r in kv['layers']:
            if r['layer'] in [2,3,4,5]:assert r['used_tokens']==0 and r['storage_bytes']==0
            else:assert r['original_positions']==list(layout.text)+list(range(n,n+steps-1)) and r['used_tokens']==n-576+steps-1
        # Same C3 semantics as a short numerical reference; never a formal dense run.
        dense,dv=frozen.hf_generate(adapter,layout,'Physical-C3',3,capture=True,audit=True,dense=True)
        assert actual['token_ids']==dense['token_ids']
        gate=equivalence('llava',dv,pv,dense.pop('_hidden'),actual.pop('_hidden'))
        assert gate['passed'],gate
        ar=actual.pop('_rotary');dr=dense.pop('_rotary')
        assert len(ar)==len(dr)==steps
        assert all(a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a) for a,b in zip(ar,dr))
        assert digest({k:tensor_sha(v) for k,v in adapter.inputs.items()})==input_hash
        scoring=scorer.score(s,actual['text'])
        for r in [actual,dense]:
            r.pop('attention_audit',None);r.pop('visual_audit',None)
        details.append(dict(benchmark=s['benchmark'],key=s['key'],status='PASS',actual=actual,scoring=scoring,
            numerical_equivalence=gate,rope_exact=True,trace=trace,projection_shapes=shapes,
            visual_hidden_removed_before_L20=True,answer_attention_physically_skipped=True,
            compact_kv_original_positions_exact=True,blocked_image_probability_zero=True,
            reference='Only three-token same-algorithm dense C3 numerical reference, no formal/reference variant cohort'))
        atomic(OUT/'physical_smoke.json',dict(status='RUNNING',at=now(),cases=details))
        print('PHYSICAL_SMOKE_PASS',s['benchmark'],s['key'],gate,flush=True)
        del actual,dense,pv,dv,ar,dr;adapter.inputs={};gc.collect();torch.cuda.empty_cache()
    final=dict(status='PASS',at=now(),cases=details,no_native_or_reaccess_generation=True)
    atomic(OUT/'physical_smoke.json',final)
    return final


def audit_gate(wait=False):
    start=time.monotonic()
    while not AUDIT.exists():
        if not wait:return None
        assert time.monotonic()-start<900,'CPU audit not available within 15 minutes'
        assert not (OUT/'STOP_REQUESTED').exists()
        time.sleep(2)
    audit=loadjson(AUDIT)
    assert not audit.get('input_or_generation_protocol_issue',True),'CPU audit found protocol issue'
    assert not audit.get('scoring_issue_affects_c3',True),'CPU audit found C3 scorer issue'
    assert audit.get('records_audited')==15 and audit.get('overall_status')=='PASS',audit
    return audit


def open_db(binding):
    db=sqlite3.connect(OUT/'predictions.sqlite',timeout=60)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY,benchmark TEXT,payload TEXT,sha256 TEXT)')
    old=db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    if old:assert old[0]==binding
    else:
        with db:db.execute('INSERT INTO metadata VALUES(?,?)',('binding',binding))
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    return db


def progress(db,metadata,started,last=None):
    done=dict(db.execute('SELECT benchmark,COUNT(*) FROM results GROUP BY benchmark'))
    value=dict(status='PARTIAL',at=now(),pid=os.getpid(),condition='Physical-C3',
        completed=sum(done.values()),expected=22678,benchmarks={b:dict(completed=done.get(b,0),expected=n) for b,n in COUNTS.items()},
        last_key=last,elapsed_seconds=time.monotonic()-started,binding=metadata['binding'],resumable=True)
    atomic(OUT/'progress.json',value)
    return value


def summarize(db,rows):
    lookup={s['key']:s for s in rows};scores=defaultdict(list);gens={};seen=set()
    with gzip.GzipFile(filename=str(OUT/'predictions.jsonl.gz'),mode='wb',mtime=0) as export:
        for key,bench,payload,checksum in db.execute('SELECT key,benchmark,payload,sha256 FROM results ORDER BY key'):
            r=json.loads(payload);assert digest(r)==checksum and r['sample_sha256']==digest(lookup[key])
            assert scorer.score(lookup[key],r['generation']['text'])==r['scoring']
            scores[bench].append((lookup[key],r['scoring']['score']));gens[key]=r['generation'];seen.add(key)
            export.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    assert seen==set(lookup)
    summary={}
    for bench in ORDER:
        entries=scores[bench];samples=[s for s,v in entries];values=[v for s,v in entries]
        entry=dict(status='COMPLETE',questions=len(entries),expected=COUNTS[bench],condition='Physical-C3')
        assert len(entries)==COUNTS[bench]
        if bench=='MME':
            totals,tasks=analyze.mme_score(samples,values)
            entry.update(metric='MME Score',scores=totals,tasks=tasks,MME_image_version_difference=True)
        elif bench=='MMBench':
            units,circular=analyze.circular_rows(samples,values)
            entry.update(metric='Circular Accuracy',value=100*float(circular.mean()),unit='%',
                circular_groups=len(units),row_accuracy_secondary=100*sum(values)/len(values),
                scorer='frozen official VLMEvalKit matcher + all-rotations-correct',server_judge=False)
        else:
            entry.update(metric='VQA Soft Accuracy' if bench=='TextVQA' else 'Accuracy',value=100*sum(values)/len(values),unit='%')
            if bench=='POPE':
                from vlm_research.benchmarks.scoring import pope_metrics
                entry['strategies']={strategy:pope_metrics([gens[s['key']]['text'] for s in samples if s['strategy']==strategy],
                    [s['answer'] for s in samples if s['strategy']==strategy]) for strategy in ['random','popular','adversarial']}
        summary[bench]=entry
    atomic(OUT/'score_summary.json',dict(status='COMPLETE',at=now(),benchmarks=summary,
        records=22678,all_saved_scores_independently_recomputed=True,native_paired_delta=None))
    return summary


def report(metadata):
    p=loadjson(OUT/'progress.json') if (OUT/'progress.json').exists() else dict(completed=0,benchmarks={})
    summaries=loadjson(OUT/'score_summary.json')['benchmarks'] if (OUT/'score_summary.json').exists() else {}
    lines=['# RTX 3090 原始 Physical C3-V2 正式评测','',f"**状态：{metadata['status']}**；正式完成 {p['completed']}/22,678。未完成数据不作为正式成绩。",'',
        f"代码：`{COMMIT}`；模型：`llava-hf/llava-1.5-7b-hf@{REV}`；FP16 / Eager / Transformers 4.49.0。",
        f"实际环境：{metadata['runtime']}。模型加载 {metadata.get('model_loads_this_run',0)} 次；单一 GPU 推理进程。",
        f"物理正确性 Smoke：{metadata.get('physical_smoke_status','NOT_VALIDATED')}。具体门禁、算子形状、位置与数值误差见 physical_smoke.json。",'',
        '| Benchmark | 完成/应完成 | 正式分数 |','|---|---:|---|']
    for b,n in COUNTS.items():
        completed=p.get('benchmarks',{}).get(b,{}).get('completed',0);v=summaries.get(b)
        text='NOT_VALIDATED / 不发布部分前缀均分'
        if v:text=json.dumps(v['scores'],ensure_ascii=False) if b=='MME' else f"{v['value']:.6f}% ({v['metric']})"
        lines.append(f'| {b} | {completed}/{n} | {text} |')
    lines+=['','协议：五类输入逐字段绑定冻结 Paper Manifest 和独立 Gold；TextVQA 为 Paper-OCR，POPE 为 8,910 行（Random 2,910 / Popular 3,000 / Adversarial 3,000）。各题生成上限沿用 128 / 1024；无新增 stop strings。',
        '原始 C3：R 仅 L0–L19 可读图像；L20 前物理移除视觉 Hidden；P−1 与 Decode 无原始 Image Key；L2–L5 物理跳过答案 Attention；原 Norm/MLP/Residual 保留；使用未重新编号的 RoPE 与紧凑物理 KV。',
        'Smoke 仅在五类各一题上检查实际 Physical C3，并对照同算法三 Token dense 数值参考；dense 不是正式评测实现，未重跑 HF Native 或视觉回看变体。门限沿用 atol=0.10、rtol=0.003、hidden RMSE≤0.01、cosine≤1e-4。',
        'MME 保留 landmark/03d5e3bfc958be38.jpg 单图 SHA 差异及两题标记，未替换或删除图片；因此与旧全量 MME 不具备全部图片逐字节一致的条件。',
        'Native 对照：无同模型、同完整 Paper 输入与后端的已完成 HF Native 对照；本轮只报告 C3 单条件指标，不计算配对差值。此前 15 条 Native Smoke 不能作为完整 Benchmark 基线。',
        '检查点：predictions.sqlite 单题事务、FULL 同步与记录 SHA；progress.json 记录真实进度。恢复时先核对运行配置绑定和已完成记录，再跳过已完成键。最终 predictions.jsonl.gz 保存可追溯原始预测。',
        'CPU Native 评分审计独立写入 ../featurize_new3090_scoring_audit/，仅 CPU，不重新生成 Native。',
        f"CPU 审计门禁：{metadata.get('cpu_audit_gate','等待/未确认')}。",
        f"GPU/峰值与释放：{metadata.get('gpu',{})}；{metadata.get('release',{})}。",
        '未重新下载模型/数据、未全图 SHA 或重复 452 项 CPU 回归、未运行 GQA、其他层组合、性能 Profiling 或 Push。']
    if metadata.get('failure'):lines+=['',f"停止原因：{metadata['failure']['phase']} — {metadata['failure']['error']}"]
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    rows,protocol,bindings=manifest();metadata=setup(rows,protocol,bindings)
    if args.preflight:
        print('PREFLIGHT_PASS',len(rows),metadata['binding'],flush=True);report(metadata);return
    assert not (OUT/'STOP_REQUESTED').exists()
    adapter=None;db=None;phase='GPU_LOAD';started=time.monotonic()
    try:
        assert not command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
        assert torch.cuda.mem_get_info()[0]>20*2**30
        metadata['gpu']=dict(before=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free','--format=csv,noheader']))
        loader.MODELS=dict(loader.MODELS,llava=MODEL);torch.cuda.reset_peak_memory_stats()
        metadata['model_loads_this_run']+=1;atomic(OUT/'run_metadata.json',metadata)
        print('LOADING_MODEL_ONCE',flush=True);adapter=frozen.load('B')
        assert {str(p.device) for p in adapter.model.parameters()}=={'cuda:0'} and {p.dtype for p in adapter.model.parameters()}=={torch.float16}
        assert adapter.model.config.text_config._attn_implementation==adapter.model.config.vision_config._attn_implementation=='eager'
        phase='PHYSICAL_SMOKE';metadata['status']='NOT_VALIDATED';report(metadata)
        smoke(adapter,rows);metadata['physical_smoke_status']='PASS';atomic(OUT/'run_metadata.json',metadata)
        phase='CPU_AUDIT_GATE';print('WAITING_CPU_AUDIT_GATE',flush=True)
        audit=audit_gate(wait=True);metadata['cpu_audit_gate']=dict(status='PASS',path=str(AUDIT),sha256=sha(AUDIT))
        sources_check(protocol);phase='FORMAL_C3_ONLY';metadata['status']='PARTIAL';metadata['formal_started_at']=now()
        atomic(OUT/'run_metadata.json',metadata);db=open_db(metadata['binding'])
        lookup={s['key']:s for s in rows};existing=set()
        for key,payload,check in db.execute('SELECT key,payload,sha256 FROM results'):
            r=json.loads(payload);assert key in lookup and digest(r)==check and r['sample_sha256']==digest(lookup[key])
            assert r['binding']==metadata['binding'] and scorer.score(lookup[key],r['generation']['text'])==r['scoring'];existing.add(key)
        p=progress(db,metadata,started);report(metadata)
        print('FORMAL_START',len(existing),'of',len(rows),flush=True)
        for s in rows:
            if s['key'] in existing:continue
            assert not (OUT/'STOP_REQUESTED').exists(),'Explicit task stop'
            if len(existing)%25==0:audit_gate()
            if len(existing)%500==0:
                sources_check(protocol);assert shutil.disk_usage(OUT).free>2*2**30 and shutil.disk_usage(DATA).free>20*2**30
            if s.get('known_recovered_image_version_difference'):assert sha(s['image_path'])==MME_NEW_SHA
            layout,binding=prepare(adapter,s)
            generation,_=frozen.hf_generate(adapter,layout,'Physical-C3',s['max_new_tokens'])
            scoring=scorer.score(s,generation['text'])
            record=dict(key=s['key'],benchmark=s['benchmark'],condition='Physical-C3',question_id=s['question_id'],
                normalized_question_id=s['normalized_question_id'],author_question_id=s['author_question_id'],
                image_id=s['image_id'],image_sha256=s['image_sha256'],input_view=s['input_view'],
                sample_sha256=digest(s),original_sample_sha256=s['original_sample_sha256'],binding=metadata['binding'],
                prompt_sha256=digest(binding['prompt']),input_ids_sha256=digest(binding['input_ids']),
                pixel_values_sha256=binding['pixel_values_sha256'],expanded_length=binding['expanded_length'],
                question=s['question'],gold=s['answer'],generation=generation,scoring=scoring,
                decoded_unstripped=adapter.tokenizer.decode(generation['token_ids'],skip_special_tokens=True),at=now())
            with db:db.execute('INSERT INTO results VALUES(?,?,?,?)',(s['key'],s['benchmark'],json.dumps(record,ensure_ascii=False,separators=(',',':')),digest(record)))
            existing.add(s['key']);adapter.inputs={}
            if len(existing)%100==0 or len(existing)==len(rows):
                p=progress(db,metadata,started,s['key']);report(metadata)
                print('PROGRESS',p['completed'],'/',p['expected'],s['benchmark'],s['key'],flush=True)
        phase='FINAL_SCORING';summarize(db,rows)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');p=progress(db,metadata,started)
        metadata['status']='COMPLETE';metadata['completed_at']=now();metadata['formal_records']=len(existing)
        print('EVALUATION_COMPLETE',len(existing),flush=True)
    except BaseException as e:
        metadata['status']='BLOCKED' if phase in ['GPU_LOAD','PHYSICAL_SMOKE','CPU_AUDIT_GATE'] else 'PARTIAL'
        if phase=='PHYSICAL_SMOKE':metadata['physical_smoke_status']='FAILED'
        metadata['failure']=dict(phase=phase,error=repr(e),traceback=traceback.format_exc()[-12000:],at=now())
        atomic(OUT/'failure.json',metadata['failure'])
        if db:progress(db,metadata,started)
        print('STOPPED',phase,traceback.format_exc(),flush=True)
    finally:
        if db:db.close()
        if torch.cuda.is_initialized():
            metadata.setdefault('gpu',{}).update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            adapter=None;gc.collect();torch.cuda.empty_cache()
            metadata['release']=dict(allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,process_exit_releases_context=True)
        metadata['last_updated_at']=now();atomic(OUT/'run_metadata.json',metadata);report(metadata)
        print('FINAL_STATUS',metadata['status'],flush=True)


if __name__=='__main__':main()
