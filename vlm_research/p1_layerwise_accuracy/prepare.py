"""Audit committed historical records before allowing any GPU reuse."""
from collections import Counter,defaultdict
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .common import *


def legacy_digest(value):
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def git_verified(path,commit,relative):
    blob = subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest()==sha(path), str(path)
    return dict(path=str(path),commit=commit,git_path=relative,sha256=sha(path))


def canonical(g,layer,answer):
    ids = g['token_ids']
    text = g['text']
    prediction = text.rstrip('.').lower()
    assert prediction==g['prediction']
    score = int(prediction==answer.lower())
    assert score==g.get('correct',g.get('score'))
    return dict(token_ids=ids,text=text,prediction=prediction,correct=score,condition_layer=layer,
        eos=g['eos'],truncated=g['truncated'],empty=not bool(text),first_token_id=ids[0],
        repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2)),
        historical=True)


def balanced(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[s['image_sha256']].append(key(s))
    shards = [[],[]]
    for image,keys in sorted(groups.items(),key=lambda pair:(-len(pair[1]),digest([SEED,pair[0]]))):
        i = min(range(2),key=lambda j:(len(shards[j]),j))
        shards[i].extend(sorted(keys))
    return [sorted(s) for s in shards]


def summarize_first(kind):
    path = ORIGINAL/'first_token_per_question.csv' if kind=='llava' else SCAN/'qwen_per_question_layer.csv'
    commit = '221cdde56dde53908aaa953fe26672fd33334f84' if kind=='llava' else '96a66a30bc81db9ffe5c3ed77716673a81aaaa01'
    relative = 'vlm_research/results/'+('paper_rq1_single_layer_gqa6/' if kind=='llava' else 'three_vlm_layer_sensitivity_3000/')+path.name
    evidence = git_verified(path,commit,relative)
    groups = defaultdict(list)
    with path.open() as stream:
        for r in csv.DictReader(stream):
            layer = int(r['layer'])
            loss = -float(r['delta_logp']) if kind=='llava' else float(r['signed_loss'])
            flip = r['top1_flip'] in ('True','true','1')
            for task in (r['task'],'overall'):
                groups[(task,layer)].append((loss,abs(loss),int(flip)))
    rows = [dict(task=t,layer=i,questions=len(v),signed_loss=float(np.mean(v,axis=0)[0]),
        absolute_change=float(np.mean(v,axis=0)[1]),top1_flip_rate=float(np.mean(v,axis=0)[2]))
        for (t,i),v in sorted(groups.items())]
    assert len(rows)==7*MODELS[kind]['layers']
    assert all(r['questions']==(3000 if r['task']=='overall' else 500) for r in rows)
    return rows,evidence


def prepare():
    samples = read(ORIGINAL/'sample_manifest.json')['samples']
    assert len(samples)==3000 and len({s['image_sha256'] for s in samples})==2409
    assert sorted(Counter(s['task'] for s in samples).values())==[500]*6
    assert len({key(s) for s in samples})==3000
    initial = git_verified(ORIGINAL/'sample_manifest.json','221cdde56dde53908aaa953fe26672fd33334f84',
        'vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json')
    for s in samples:
        assert sha(s['image_path'])==s['image_sha256']
    shards = balanced(samples)
    by = {key(s):s for s in samples}
    cluster_sets = [{by[k]['image_sha256'] for k in part} for part in shards]
    assert cluster_sets[0].isdisjoint(cluster_sets[1])
    assert sorted(shards[0]+shards[1])==sorted(by)
    qinputs = read(SCAN/'qwen_input_manifest.json')
    qspec = {key(s):s for s in qinputs['samples']}
    assert set(qspec)==set(by)
    evidence_qinputs = git_verified(SCAN/'qwen_input_manifest.json','96a66a30bc81db9ffe5c3ed77716673a81aaaa01',
        'vlm_research/results/three_vlm_layer_sensitivity_3000/qwen_input_manifest.json')
    old_completion = read(ORIGINAL/'completion.json')
    old_index = old_completion['files_sha256']
    git_verified(ORIGINAL/'completion.json','221cdde56dde53908aaa953fe26672fd33334f84',
        'vlm_research/results/paper_rq1_single_layer_gqa6/completion.json')
    figtree = subprocess.check_output(['git','ls-tree','-r','c25001e135408ad72d9f797da03265464b01f7b1',
        'vlm_research/results/figure3_decode_window_comparison'],cwd=ROOT,text=True)
    figblobs = {line.split('\t')[1]:line.split()[2] for line in figtree.splitlines()}
    qindex_evidence = git_verified(SPARSE/'result_hashes.json','3132d923459a2ab80f38fb5e4108d6c27d3b9016',
        'vlm_research/results/qwen_internvl_sparse_visual_3000/result_hashes.json')
    qindex = read(SPARSE/'result_hashes.json')['files']
    figure_full = {}
    for f in FIG3.glob('shard*/checkpoints/*/Full.json'):
        data = f.read_bytes()
        relative = 'vlm_research/results/figure3_decode_window_comparison/'+str(f.relative_to(FIG3))
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==figblobs[relative]
        r = read(f)
        s = by[key(r)]
        assert r['sample_sha256']==legacy_digest(s) and legacy_digest(r['generation'])==r['result_sha256']
        assert r['condition']=='Full' and r['generation']['edits']==0
        figure_full[key(r)] = (r,str(f),sha(f))
    assert len(figure_full)==3000
    legacy, rawmeta, source_records = {},{},[]
    for f in ORIGINAL.glob('first_token/shard*/checkpoints/*.json'):
        assert sha(f)==old_index[str(f.relative_to(ORIGINAL))]
        r = read(f)
        assert r['sample_sha256']==legacy_digest(by[key(r)])
        rawmeta[key(r)] = dict(path=r['result']['logits_path'],sha256=r['result']['logits_sha256'],
            first_logp=r['result']['metrics'][0]['full_token_logp'],first_token=r['result']['full_first_token_id'])
    assert len(rawmeta)==3000
    for f in ORIGINAL.glob('accuracy/shard*/checkpoints/*.json'):
        assert sha(f)==old_index[str(f.relative_to(ORIGINAL))]
        r = read(f); s = by[key(r)]
        assert s['accuracy_subset'] and r['sample_sha256']==legacy_digest(s)
        gs = r['result']['generations']
        assert r['result']['all_33_initial_logits_exact']
        assert [g['layer'] for g in gs]==[None]+list(range(32))
        assert gs[0]['token_ids']==figure_full[key(r)][0]['generation']['token_ids']
        for g in gs[1:]:
            assert len(g['intervention_audit'])==1
            a = g['intervention_audit'][0]
            assert a['time']==0 and a['layer']==g['layer'] and a['target_zero'] and a['other_queries_exact']
        legacy[key(r)] = {conditions('llava')[i]:canonical(g,g['layer'],s['answer']) for i,g in enumerate(gs)}
        source_records.append(dict(path=str(f),sha256=sha(f),conditions=33))
    assert len(legacy)==480
    for kind in ('llava','qwen'):
        folder = OUT/kind
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ORIGINAL/'sample_manifest.json',folder/'sample_manifest.json')
        baselines,records = {},[]
        for s in samples:
            identifier = key(s)
            if kind=='llava':
                r,path,h = figure_full[identifier]
                g = canonical(r['generation'],None,s['answer'])
                g['first_logp'] = rawmeta[identifier]['first_logp']
                assert g['token_ids'][0]==rawmeta[identifier]['first_token']
                records.append(dict(key=identifier,path=path,sha256=h))
            else:
                f = SPARSE/'qwen/checkpoints'/(identifier+'.json')
                assert sha(f)==qindex[str(f.relative_to(SPARSE))]
                r = read(f)
                assert r['image_sha256']==s['image_sha256'] and r['answer']==s['answer'] and r['question']==s['question']
                assert r['input_binding']==digest(qspec[identifier]['layout'])
                assert r['complete'] and r['conditions']['Full']['mask_audits']=={}
                g = canonical(r['conditions']['Full'],None,s['answer'])
                g['first_logp'] = r['conditions']['Full']['full_top1_logp']
                g['all_step_logits_sha256'] = r['conditions']['Full']['all_step_logits_sha256']
                records.append(dict(key=identifier,path=str(f),sha256=sha(f)))
            baselines[identifier] = g
        # Independent tokenizer decode checks do not initialize CUDA.
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(model_path(kind)/'tokenizer.json'))
        for identifier,g in baselines.items():
            assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
        if kind=='llava':
            for gs in legacy.values():
                for g in gs.values():
                    assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
        seal(folder/'baselines.json',baselines)
        seal(folder/'legacy_generation.json',legacy if kind=='llava' else {})
        seal(folder/'first_token_raw_bindings.json',rawmeta if kind=='llava' else {})
        seal(folder/'input_bindings.json',{key(s):s['input_mapping'] for s in samples} if kind=='llava' else qspec)
        seal(folder/'shards.json',dict(keys=shards,image_exclusive=True))
        first,evidence = summarize_first(kind)
        seal(folder/'historical_first_token_summary.json',first)
        seal(folder/'historical_reuse_audit.json',dict(passed=True,model=kind,
            native_full_count=3000,native_full_correct=sum(g['correct'] for g in baselines.values()),
            legacy_drop_count=15360 if kind=='llava' else 0,legacy_full_count=480 if kind=='llava' else 0,
            manifest=initial,first_token=evidence,input_source=initial if kind=='llava' else evidence_qinputs,
            baseline_source_commit='c25001e135408ad72d9f797da03265464b01f7b1' if kind=='llava' else qindex_evidence['commit'],
            baseline_records=records,legacy_records=source_records if kind=='llava' else [],
            original_native_not_All_Writer=True,
            legacy_reuse_gate='Must pass matched-shape native/legacy GPU Smoke and per-question full-logit exact replay'))
    print('AUDITED: native Full 3000 each; LLaVA legacy Drop 15360; 3000 inputs and 2409 clusters',flush=True)


if __name__=='__main__':
    prepare()
