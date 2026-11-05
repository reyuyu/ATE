import argparse
import datetime
import gc
import os
import subprocess
import sys
import time
import traceback

import torch

from .common import *
from .probe import generate, describe_generation, forced_logp, VisionMappingAudit
from ..reverse_writer.engine import prepare, generate as frozen_generate
from ..accuracy_autoresearch.engine import generate as research_generate
from ..accuracy_autoresearch.common import config
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence
from ..fastp2_bypass_writer.common import hardware


def now():
    return datetime.datetime.now().astimezone().isoformat()


def run():
    frozen = read(OUT/'freeze.json')
    assert frozen['sources'] == sources()
    assert sha(OUT/'sample_manifest.json') == frozen['sample_manifest_sha256']
    assert sha(OUT/'protocol.json') == frozen['protocol_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU['uuid']
    state = hardware()
    assert GPU['uuid'] not in state['processes'], 'Authorized GPU0 no longer idle; do not load'
    atomic(OUT/'gpu_preflight.json', dict(at=now(), pid=os.getpid(), **state))
    assert all(sha(p)==h for p,h in frozen['weights'].items())
    samples = read(OUT/'sample_manifest.json'); protocol=read(OUT/'protocol.json')
    atomic(OUT/'execution_commit.json', dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), sources=sources(), at=now()))
    a=None; started=time.time(); smokes=[]
    try:
        a=Adapter('llava'); atomic(OUT/'environment.json', a.environment)
        for key,s in samples.items():
            qid=str(s['question_id']); layout,w,spec=prepare(a,s,'C3')
            with torch.inference_mode(), VisionMappingAudit(a):
                native=research_generate(a,layout,config('native',head=1),w,2,True)
                noop=research_generate(a,layout,config('native',head=1),w,2,True,reference=True)
            assert native[0]['token_ids']==noop[0]['token_ids']
            assert all(torch.equal(x,y) for x,y in zip(native[1],noop[1]))
            del native,noop
            checks=[]
            for label in ('C3','E1'):
                layout,w,spec=prepare(a,s,label)
                plain=frozen_generate(a,layout,w,32,True,False)
                observed,rows,audit=generate(a,layout,w,True)
                old=load_record(HISTORY/'checkpoints'/key/(label+'.json'))
                assert spec == old['input_binding'], 'Historical image, Prompt and writer bindings differ'
                desc=describe_generation(observed)
                assert desc['token_ids']==old['generation']['token_ids']==plain[0]['token_ids']
                assert desc['first_logits_sha256']==old['generation']['first_logits_sha256']
                assert all(torch.equal(x,y) for x,y in zip(plain[1],observed[1]))
                assert desc['text'].lower()==('yes' if label=='C3' else 'no')
                ref=frozen_generate(a,layout,w,2,True,True,True)
                actual=frozen_generate(a,layout,w,2,True,True)
                eq=equivalence('llava',ref[1],actual[1],ref[3],actual[3])
                assert eq['passed'] and ref[0]['token_ids']==actual[0]['token_ids']
                assert all(e['blocked_probability_max']==0 for e in actual[0]['audit'])
                p=RESOURCES/qid/(label+'_R_image_attention.npz'); p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,weights=rows,query_positions=np.array(layout.relay),image_positions=np.array(layout.image))
                atomic(OUT/'attention'/qid/(label+'.json'),dict(path=str(p),sha256=sha(p),shape=list(rows.shape),dtype=str(rows.dtype),audit=audit))
                checks.append(dict(condition=label,history_logits_exact=True,history_tokens_exact=True,
                    observer_all_logits_exact=True,observer_all_tokens_exact=True,native_noop_exact=True,
                    dense_reference=eq,attention_shape=list(rows.shape),blocked_actual_probability_zero=True))
                del plain,observed,ref,actual,rows
            smokes.append(dict(sample=key,checks=checks)); a.verify()
            atomic(OUT/'smoke_progress.json',dict(at=now(),samples=smokes))
        atomic(OUT/'validation.json',dict(status='PASS',at=now(),sources=sources(),samples=smokes,
            all_three_reproduced=True,full_generation_and_cache_isolation=True,hook_cleanup=True,
            precision_and_tolerances_unchanged=True))
        for key,s in samples.items():
            qid=str(s['question_id']); forms=protocol['scoring']['variants'][key]
            for name,restored in conditions(10):
                path=checkpoint(qid,name)
                if path.exists():
                    old=load_record(path)
                    assert old['sources']==sources() and old['sample_sha256']==digest(s) and old['restored']==restored
                    continue
                layout,_,spec=prepare(a,s,'E1'); w=writers(s['input_mapping'],restored)
                result,_,_=generate(a,layout,w)
                g=describe_generation(result); first=result[1][0]
                variants={form:dict(token_ids=ids,logp=forced_logp(a,layout,w,ids,first)) for form,ids in forms.items()}
                margin=variants[' Yes']['logp']-variants[' No']['logp']
                if name in ('C3','E1'):
                    old=load_record(HISTORY/'checkpoints'/key/(name+'.json'))
                    assert g['first_logits_sha256']==old['generation']['first_logits_sha256']
                    assert g['token_ids']==old['generation']['token_ids']
                save_record(path,dict(sample=key,question_id=qid,sample_sha256=digest(s),sources=sources(),
                    condition=name,restored=restored,writer_positions=w,generation=g,variants=variants,
                    yes_no_margin=margin,at=now()))
                print(qid,name,restored,g['text'],'margin',round(margin,6),flush=True)
                del result,first; a.verify()
        assert all(checkpoint(str(s['question_id']),c).exists() for s in samples.values() for c,_ in conditions(10))
        atomic(OUT/'gpu_completion.json',dict(status='COMPLETE',at=now(),elapsed_seconds=time.time()-started,
            records=len(samples)*len(conditions(10)),sources=sources(),no_runtime_failures=True))
    except BaseException:
        atomic(OUT/'failure.json',dict(at=now(),traceback=traceback.format_exc())); raise
    finally:
        del a; gc.collect(); torch.cuda.empty_cache()
        atomic(OUT/'gpu_release.json',dict(at=now(),pid=os.getpid(),allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),process_exit_releases_remaining=True))


if __name__=='__main__':
    run()
