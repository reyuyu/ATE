"""Final integrity and provenance checks, with stopped attempts kept separate."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from .common import ROOT,OUT,QH,read,write,sha,digest,verify,command,START,DEADLINE


def run():
    verify()
    samples=read(OUT/'sample_manifest.json')['samples']
    coverage=read(OUT/'qwen/coverage.json')
    assert coverage['questions']==3000 and coverage['status']=='complete'
    assert coverage['conditions']==108000 and coverage['historical_questions']==240 and coverage['new_questions']==2760
    assert coverage['tasks']=={t:500 for t in sorted({s['task'] for s in samples})}
    seen=set();devices=Counter();image_tokens=[];native_targets=0
    for s in samples:
        key=f'{s["task"]}__{s["question_id"]}'
        r=read(OUT/'qwen/checkpoints'/f'{key}.json')
        assert r['complete'] and r['sample_sha256']==digest(s)
        assert r['protocol_sha256']==sha(OUT/'protocol.json') and r['image_sha256']==s['image_sha256']
        assert len(r['rows'])==36
        tokens={x['full_first_token'] for x in r['rows']}
        logps={x['full_top1_logp'] for x in r['rows']}
        assert len(tokens)==len(logps)==1
        native_targets+=1
        for row in r['rows']:
            item=(s['task'],s['question_id'],row['layer'])
            assert item not in seen;seen.add(item)
            assert row['physical_gpu'] in (0,2,3)
            devices[row['physical_gpu']]+=1
            assert row['absolute_logp_change']==abs(row['signed_logp_loss'])
            assert row.get('masked_probability_max',row.get('image_probability_max'))==0
        image_tokens.append(r['input_layout']['image_tokens'])
        if s['pilot']:
            assert r['historical_protocol_sha256']==sha(QH/'protocol.json')
            assert sha(r['reused_from'])==r['historical_record_sha256']
    validations=[]
    for folder in sorted((OUT/'qwen').glob('all_gpu*_shard*')):
        v=read(folder/'validation.json');c=read(folder/'complete.json')
        assert v['passed'] and c['passed'] and c['status']=='complete' and c['complete_count']==1500
        assert v['protocol_sha256']==sha(OUT/'protocol.json')
        assert c['weights_unchanged'] and c['hooks_clean'] and c['no_nan_inf_oom']
        validations.append(dict(folder=str(folder.relative_to(OUT)),**v))
    assert len(validations)==2 and len({v['code_commit'] for v in validations})==1
    generation_checks=[]
    for name,expected in [('llava_relay',4),('llava_rescue',3),('qwen_rescue',3)]:
        folder=OUT/name;p=read(folder/'protocol.json')
        assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
        assert sha(folder/'sample_manifest.json')==p['manifest_sha256']
        ss=read(folder/'sample_manifest.json')['samples']
        assert len(ss)==240 and len({s['image_sha256'] for s in ss})==235
        count=0;token_checks=0;empty=trunc=0
        for s in ss:
            r=read(folder/'checkpoints'/f'{s["task"]}__{s["question_id"]}.json')
            assert r['complete'] and r['sample_sha256']==digest(s) and r['protocol_sha256']==sha(folder/'protocol.json')
            assert len(r['conditions'])==expected
            for value in r['conditions'].values():
                assert value['correct']==int(value['prediction']==s['answer'])
                assert value['fresh_cache'] and value['decode_hooks_removed']
                assert value['token_ids'] and value['eos'] and value['generated_tokens']==len(value['token_ids'])
                assert value['physical_gpu']==3
                assert all(a['masked_probability_max']==0 for a in value['prefill_mask_audits'].values())
                count+=1;empty+=value['empty'];trunc+=value['truncated']
            if name=='qwen_rescue':
                old=read(ROOT/'vlm_research/results/qwen3vl_keep_l26_generation_240'/f'shard_{s["shard"]}'/'checkpoints'/f'{s["task"]}__{s["question_id"]}.json')
                assert old['input_binding']==s['input_binding'] and old['task']==s['task'] and old['question_id']==s['question_id']
                assert old['image_sha256']==s['image_sha256']
                full=old['conditions']['Full']
                assert full['correct']==int(full['prediction']==s['answer'])==r['native_full_accuracy']
                token_checks+=1
        assert empty==trunc==0
        generation_checks.append(dict(experiment=name,questions=240,condition_rows=count,empty=empty,truncated=trunc,
            qwen_historical_full_score_bindings_checked=token_checks))
    assert not read(OUT/'llava_relay/extension_decision.json')['extend_to_other_models']
    assert not (OUT/'internvl/checkpoints').exists(), 'InternVL formal data must not exist after stopped smoke'
    assert read(OUT/'qwen_model_files_validation.json')['passed'] and read(OUT/'llava_model_files_validation.json')['passed']
    now=datetime.now().astimezone()
    result=dict(passed=True,overall_status='partial',reason='InternVL exact native-logit smoke failed; stopped with zero formal samples',
        qwen_questions=3000,qwen_rows=len(seen),same_full_targets_checked=native_targets,
        qwen_new_rows=99360,qwen_historical_rows=8640,qwen_image_token_range=[min(image_tokens),max(image_tokens)],
        qwen_row_devices=dict(devices),qwen_source_freeze_code_commit=validations[0]['code_commit'],
        generation_checks=generation_checks,bootstrap_unit='Image SHA256 whole clusters, question-weighted',
        no_unrequested_interaction_extension=True,time=now.isoformat(),elapsed_seconds=(now-datetime.fromisoformat(START)).total_seconds())
    write(OUT/'integrity.json',result)
    write(OUT/'attempts_audit.json',dict(failed_model=dict(model='InternVL',phase='native reference smoke',
        formal_samples=0,action='Stopped without numerical tolerance changes or retry',
        logs=[str(p.relative_to(OUT)) for p in sorted(OUT.glob('internvl_pilot_gpu*.log'))]),
        repaired_startup_issues=[dict(experiment='Qwen Task1',failure='Inference tensor weights lack version counter',
            action='Load under no_grad, run forwards under inference_mode; no intervention/tolerance change',
            protocol_before='protocol_initial.json',protocol_after='protocol.json',
            logs=['qwen_all_gpu0.log','qwen_all_gpu2.log']),
            dict(experiment='Qwen rescue',failure='CPU smoke inputs with CUDA model, before numerical smoke',
            action='Move inputs to CUDA, no intervention/tolerance change',protocol_before='qwen_rescue/protocol_attempt0.json',
            protocol_after='qwen_rescue/protocol.json',logs=['qwen_rescue_gpu3.log'])],
        valid_formal_runs='Only passed current validation/complete files counted; attempt0 failures retained separately'))
    print(result,flush=True)


def delivery():
    assert read(OUT/'integrity.json')['passed']
    now=datetime.now().astimezone()
    inventory=command('nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader')
    apps=command('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader')
    lines=inventory.splitlines()
    own=[line.split(',')[1].strip() for line in lines if int(line.split(',')[0]) in (0,2,3)]
    assert not any(uuid in apps for uuid in own), 'A permitted GPU still has a compute process; do not claim release'
    write(OUT/'delivery_audit.json',dict(time=now.isoformat(),start=START,deadline=DEADLINE,
        elapsed_seconds=(now-datetime.fromisoformat(START)).total_seconds(),within_budget=now<datetime.fromisoformat(DEADLINE),
        gpu_ids_used=[0,2,3],gpu1_used=False,gpu_released=True,gpu_inventory=inventory,compute_apps=apps,
        branch=command('git','branch','--show-current'),pre_results_commit=command('git','rev-parse','HEAD'),
        main_modified_by_this_experiment=False,scope='No training, acceleration algorithm, extra layer combination or dataset'))
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='result_hashes.json' and not p.name.endswith('.tmp')}
    write(OUT/'result_hashes.json',files)
    print('GPU0/2/3 released; hashed',len(files),'artifacts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['integrity','delivery']);args=parser.parse_args()
    run() if args.action=='integrity' else delivery()
