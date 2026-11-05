"""Freeze the unchanged historical cohort before running either GPU worker."""
import collections
import subprocess
from PIL import Image

from .common import *
from ..fastp2_bypass_writer.common import hardware


def main():
    assert not (OUT/'execution_freeze.json').exists(), 'Already frozen; do not reselect'
    samples={k:s for k,s in read(HISTORY/'sample_manifest.json').items() if s['dataset']=='textvqa'}
    assert len(samples)==500 and all(len(s['answer'])==10 for s in samples.values())
    groups=collections.defaultdict(list)
    for k,s in samples.items(): groups[s['image_sha256']].append(k)
    assert len(groups)==309
    images={s['image_path']:s['image_sha256'] for s in samples.values()}
    for path,h in images.items():
        assert sha(path)==h,path
        with Image.open(path) as im: im.verify()
    unique=[min(keys,key=lambda k:digest([SEED,'representative',k])) for keys in groups.values()]
    ordered=sorted(unique,key=lambda k:(len(samples[k]['input_mapping']['prompt_ids']),k))
    perf=[]
    for quartile in range(4):
        pool=ordered[len(ordered)*quartile//4:len(ordered)*(quartile+1)//4]
        perf+=sorted(pool,key=lambda k:digest([SEED,'performance48',k]))[:12]
    assert len(perf)==len({samples[k]['image_sha256'] for k in perf})==48
    smoke=[ordered[len(ordered)*q//4] for q in (0,2,3)]
    shard={}; loads=[0,0]; counts=[0,0]
    for h,keys in sorted(groups.items(),key=lambda kv:(-len(kv[1])-64*sum(k in perf for k in kv[1]),digest([SEED,'shard',kv[0]]))):
        rank=min(range(2),key=lambda i:(loads[i],i))
        shard[h]=rank; loads[rank]+=len(keys)+64*sum(k in perf for k in keys); counts[rank]+=len(keys)
    weights=read(HISTORY/'execution_freeze.json')['weights']
    assert all(sha(p)==h for p,h in weights.items()),'Frozen model weights changed'
    atomic(OUT/'sample_manifest.json',samples)
    atomic(OUT/'partition.json',dict(image_shard=shard,question_counts=counts,weighted_loads=loads,
        performance48=perf,smoke=smoke,performance_counts=[sum(shard[samples[k]['image_sha256']]==r for k in perf) for r in range(2)]))
    atomic(OUT/'protocol.json',dict(created_at=now(),seed=SEED,model='LLaVA-1.5-7B',
        revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',dtype='FP16',attention='eager',
        transformers='4.49.0',logits_to_keep=1,max_new_tokens=32,greedy=True,
        labels=LABELS,answer_image_layers=KEEP,visual_hidden_exit=CUTOFF,
        receiver_image_layers=list(range(20)),answer_attention_bypass=[2,3,4,5],
        native_exception='Native is completely unmodified; native Image KV exists in all32 layers.',
        writer_definition='Frozen R: post-image text positions before original P-1. L20 receiver image access remains blocked even for Middle10.',
        physical='Image KV retained only in9/10 selected layers; absent in bypass layers; other persistent KV is text-only. In-place append, no historical copies.',
        sample_count=500,image_clusters=309,history_manifest=str(HISTORY/'sample_manifest.json'),
        history_manifest_sha256=sha(HISTORY/'sample_manifest.json'),all_four_regenerated=True,
        bootstrap=dict(replicates=20000,seed=SEED,cluster='image_sha256',weighting='equal questions',
            comparisons=['C3+Middle9 - C3-V2','C3+Middle10 - C3-V2','C3+Middle9 - Native','C3+Middle10 - Native']),
        exploratory=True,selection='Same frozen500 questions, no output filtering. Performance48:12/image-distinct prompt-length quartile, stable input hash.',
        performance=dict(samples=48,warmups=3,repetitions=5,conditions_interleaved=True,
            modes=['natural_eos','fixed16'],vision_recomputed=True,cold_independent_KV=True,
            prefill='CUDA events: Vision plus Language first forward',ttft='Wall request start to first greedy token ready',
            e2e='Wall start before control/cache setup through last generated token and final synchronize; excludes scoring, token string decode, JSON/KV auditing.',
            fixed16='Exactly16 predicted tokens with own cache/prefix; ignores EOS only in length-control timing.',
            dense_reference_not_timed=True),
        tolerances=TOLERANCES['llava'],correctness='Native/Noop and unchanged C3 strict torch.equal; shape-changing dense reference uses unchanged historical FP16 gates AND exact greedy tokens.',
        devices=DEVICES,no_hardware_speed_claim_for_masks=True))
    atomic(OUT/'data_audit.json',dict(at=now(),samples=500,clusters=309,images_verified=len(images),
        all_image_sha256_verified=True,official_ten_answers=True,model_weights_rehashed=True,
        image_paths_sha256=digest(images),historical_results_not_used_for_selection=True))
    atomic(OUT/'execution_freeze.json',dict(at=now(),sources=sources(),weights=weights,
        **{n+'_sha256':sha(OUT/(n+'.json')) for n in ('sample_manifest','partition','protocol')},
        cpu_tests_sha256=sha(OUT/'cpu_tests.log'),base_commit=BASE_COMMIT,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    atomic(OUT/'initial_resources.json',dict(at=now(),**hardware()))
    print('FROZEN',len(samples),'questions,',len(groups),'clusters; shards',counts,'performance',48,flush=True)


if __name__=='__main__': main()
