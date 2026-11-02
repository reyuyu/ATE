"""Read-only binding of audited endpoints; do not substitute P-1-only or relay results."""
import numpy as np
import torch

from .common import OTHER, LLAVA, MODELS, compact_digest, read, sha, digest


def normalized(g, answer):
    text = g['text'].strip(); pred = text.rstrip('.').lower()
    score = g.get('correct', g.get('score'))
    assert score == int(pred == answer)
    assert 1 <= len(g['token_ids']) <= 32
    return dict(text=text, prediction=pred, token_ids=g['token_ids'], correct=score,
        eos=g['eos'], truncated=g['truncated'], empty=not bool(text), generated_tokens=len(g['token_ids']),
        **{k:g[k] for k in ['full_first_token', 'changed_first_token', 'full_top1_logp',
            'off_full_top1_logp', 'signed_logp_loss', 'absolute_logp_change', 'top1_flip'] if k in g})


def token_metrics(first, full):
    target = full['full_first_token']; value = float(first.float().log_softmax(-1)[target])
    loss = full['full_top1_logp']-value
    return dict(full_first_token=target, changed_first_token=int(first.argmax()),
        full_top1_logp=full['full_top1_logp'], off_full_top1_logp=value,
        signed_logp_loss=loss, absolute_logp_change=abs(loss), top1_flip=int(int(first.argmax())!=target))


def bind(samples):
    result = {}
    root = OTHER/'results/qwen_internvl_sparse_visual_3000'
    protocol = read(root/'protocol.json')
    assert protocol['generation'] if 'generation' in protocol else protocol['max_new_tokens']==32
    assert protocol['greedy'] and protocol['other_prefill'].startswith('Native')
    assert protocol['conditions']['qwen']['Only Text'] == list(range(36))
    assert protocol['conditions']['internvl']['Only Text'] == list(range(28))
    assert protocol['qwen_deepstack']=='Native' and protocol['internvl']['logits_to_keep']==1
    for model in ['qwen', 'internvl']:
        assert protocol['model_revisions'][model] == MODELS[model]['revision']
        records = {}
        for s in samples:
            key = s['task']+'__'+s['question_id']; path=root/model/'checkpoints'/(key+'.json'); old=read(path)
            base = {k:s[k] for k in ['task','question_id','image_id','image_path','image_sha256','question','answer']}
            assert old['sample_sha256']==digest(base) and old['protocol_sha256']==sha(root/'protocol.json')
            assert old['complete'] and all(old[k]==v for k,v in base.items())
            full=old['conditions']['Full']; off=old['conditions']['Only Text']
            assert full['fresh_cache'] and off['fresh_cache']
            assert sorted(map(int, off['mask_audits']))==list(range(MODELS[model]['layers']))
            assert all(a['prefill_calls']==1 and a['decode_calls']==len(off['token_ids'])-1 and
                a['masked_probability_max']==0 for a in off['mask_audits'].values())
            assert full['all_step_logits_sha256'] and off['all_step_logits_sha256']
            records[s['question_id']]=dict(input_binding=old['input_binding'], sample_sha256=compact_digest(s),
                protocol_path=str(root/'protocol.json'), protocol_sha256=sha(root/'protocol.json'),
                source_path=str(path), source_sha256=sha(path), full_logits_sha256=old['full_logits_sha256'],
                full_step_logits_sha256=full['all_step_logits_sha256'],
                only_step_logits_sha256=off['all_step_logits_sha256'],
                Full=normalized(full,s['answer']), Only_Text=normalized(off,s['answer']))
        result[model]=records
    root=LLAVA/'results/rq2_gqa_text_vs_sparse_3000'
    p=read(root/'protocol.json');manifest=read(root/'sample_manifest.json')
    assert p['model_revision']==MODELS['llava']['revision']
    assert p['generation']==dict(do_sample=False,num_beams=1,max_new_tokens=32,eos_token_id=2,use_cache=True)
    assert manifest['samples']==samples
    bound=read(root/'history_bindings.json')
    raw=read(LLAVA/'results/rq2_visual_sufficiency_full3000/history_bindings.json')['records']
    records={}
    for s in samples:
        q=s['question_id']; b=bound[q]; first=raw[q]
        assert b['sample_sha256']==first['sample_sha256']==compact_digest(s)
        assert first['prompt_sha256']==compact_digest(s['input_mapping'])
        full=b['conditions']['Full']; fp=LLAVA/full['path']; fr=read(fp)
        assert sha(fp)==full['sha256']==first['full_sha256']
        assert fr['sample_sha256']==compact_digest(s) and fr['generation']==full['generation']
        assert compact_digest(fr['generation'])==fr['result_sha256']
        assert fr['generation']['edits']==0
        assert sha(first['raw_logits_path'])==first['raw_logits_sha256']
        with np.load(first['raw_logits_path']) as z: logits=torch.from_numpy(z['full_logits'].copy())
        assert __import__('hashlib').sha256(logits.numpy().tobytes()).hexdigest()==first['first_logits_sha256']
        g=normalized(full['generation'],s['answer'])
        g.update(full_first_token=int(logits.argmax()), full_top1_logp=float(logits.float().log_softmax(-1).max()))
        g.update(token_metrics(logits,g))
        if 'Only_Text' in b['conditions']:
            old=b['conditions']['Only_Text']; op=LLAVA/old['path']; r=read(op)
            assert sha(op)==old['sha256'] and r['sample_sha256']==compact_digest(s)
            off=old['generation']; assert off==r['result']['generation'] and off['condition']=='B'
            assert off['mask_layers']==list(range(32)) and off['own_cache'] and off['own_prefix']
            assert off['image_indices']==s['input_mapping']['visual_positions']
            assert all(x['image_probability_max']==0 for x in r['result']['attention_rows'] if x['query_group']!='Relay')
            assert all(not x['blocked'] for x in r['result']['attention_rows'] if x['query_group']=='Relay')
            assert sha(LLAVA/old['raw'])==old['raw_sha256']
            with np.load(LLAVA/old['raw']) as z: of=torch.from_numpy(z['first_logits'].copy())
            og=normalized(off,s['answer']);og.update(token_metrics(of,g))
            oh=__import__('hashlib').sha256(of.numpy().tobytes()).hexdigest()
        else:
            op=root/f"shard{manifest['assignments'][q]}/checkpoints/{q}/Only_Text.json";r=read(op)
            assert r['binding']==compact_digest(p) and r['sample_sha256']==compact_digest(s)
            assert r['condition']=='Only_Text' and compact_digest(r['generation'])==r['generation_sha256']
            assert r['hooks_clean'] and r['inputs_unchanged'] and r['parameters_unchanged']
            og=normalized(r['generation'],s['answer']);oh=r['first_logits_sha256']
        records[q]=dict(sample_sha256=compact_digest(s), input_binding=compact_digest(s['input_mapping']),
            Full=g, Only_Text=og, full_logits_sha256=first['first_logits_sha256'], only_logits_sha256=oh,
            full_path=str(fp), full_path_sha256=sha(fp), source_path=str(op), source_sha256=sha(op),
            protocol_path=str(root/'protocol.json'), protocol_sha256=sha(root/'protocol.json'),
            full_raw_path=first['raw_logits_path'], full_raw_sha256=first['raw_logits_sha256'])
    result['llava']=records
    return result


def model_files():
    from ..qwen3vl_early_windows.experiment import MODEL
    from ..visual_pathways.common import IMODEL
    files={}
    for modelpath,manifest in [(MODEL,OTHER/'results/qwen3vl_image_source_240/model_manifest.json'),
        (IMODEL,OTHER/'results/visual_pathways_seven_hour/internvl_model_manifest.json')]:
        for e in read(manifest)['files']: files[str(modelpath/e['path'])]=e['sha256']
    p=read(LLAVA/'results/rq2_gqa_text_vs_sparse_3000/protocol.json')
    files.update({name:entry['sha256'] for name,entry in p['model_files'].items()})
    return files
