from pathlib import Path
import csv,json,hashlib,subprocess
import numpy as np

r=Path('/data/lk/ATE-worktrees/layer-semantic-joint-writing-240')
p=r/'vlm_research/results/layer_semantic_joint_writing_240'
def read(n):return json.loads((p/n).read_text())
def table(n):
    with (p/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
c=read('completion.json');pr=read('protocol.json');assert c['status']=='COMPLETE' and c['new_main_generations']==720 and c['reused_main_records']==1200
for n,h in c['artifacts'].items():assert sha(p/n)==h,n
for n,h in pr['sources'].items():
    assert sha(r/'vlm_research'/n)==h
    assert hashlib.sha256(subprocess.check_output(['git','show',pr['code_commit']+':vlm_research/'+n],cwd=r)).hexdigest()==h,n
spec=read('token_group_manifest.json');manifest=read('sample_manifest.json')['samples'];rows=table('per_question.csv');sums=table('accuracy_summary.csv')
assert len(rows)==1920 and len(manifest)==240
conds=[x['condition'] for x in sums];images=sorted(s['image_sha256'] for s in manifest);assert len(set(images))==240
samples={s['image_sha256']:s for s in manifest};lookup={(v['image_sha256'],v['condition']):v for v in rows}
scores=np.array([[int(lookup[(im,co)]['prediction']==samples[im]['answer']) for co in conds] for im in images],float)
rng=np.random.default_rng(20261103);draws=np.concatenate([scores[rng.integers(0,240,size=(500,240))].mean(1) for _ in range(40)])
with np.load(p/'bootstrap_replications.npz') as z:assert np.array_equal(draws,z['accuracy'])
for x in rows:
    assert int(x['correct'])==int(x['prediction']==x['gold'])
    s=samples[x['image_sha256']];sp=spec['main'][s['task']+'__'+s['question_id']]
    selected=json.loads(x['writer_positions']);R=sp['relay_positions'];L=json.loads(x['writer_layers'])
    assert selected==(sp['selection']['writers']['Semantic-k'] if x['condition'] in ['Semantic-k','Joint-16','Joint-12','Joint-8'] else R)
    assert int(x['allowed_R_image_connections_per_head'])==len(selected)*len(L)*576
    assert int(x['all_R_image_connections_per_head'])==len(R)*32*576
    assert float(x['connection_budget_ratio'])==len(selected)*len(L)/(len(R)*32)
for x in table('paired_comparisons.csv'):
    a,b=conds.index(x['a']),conds.index(x['b']);lo,hi=np.percentile(100*(draws[:,a]-draws[:,b]),[2.5,97.5])
    for k,v in [('delta_pp',100*(scores[:,a]-scores[:,b]).mean()),('ci95_low_pp',lo),('ci95_high_pp',hi)]:assert abs(float(x[k])-v)<1e-12,(x['comparison'],k)
    assert int(x['b_correct_a_wrong'])==((scores[:,b]==1)&(scores[:,a]==0)).sum()
    assert int(x['b_wrong_a_correct'])==((scores[:,b]==0)&(scores[:,a]==1)).sum()
for x in table('interaction_summary.csv'):
    j,l,se,a=[conds.index(co) for co in [x['joint'],x['layer'],'Semantic-k','All-Writer']]
    individual=scores[:,j]-scores[:,l]-scores[:,se]+scores[:,a];bootstrap=100*(draws[:,j]-draws[:,l]-draws[:,se]+draws[:,a]);lo,hi=np.percentile(bootstrap,[2.5,97.5])
    for k,v in [('interaction_pp',100*individual.mean()),('ci95_low_pp',lo),('ci95_high_pp',hi)]:assert abs(float(x[k])-v)<1e-12,(x['joint'],k)
    unique=(scores[:,a]==1)&(scores[:,l]==1)&(scores[:,se]==1)&(scores[:,j]==0)
    assert int(x['joint_unique_harm'])==unique.sum()
    assert int(x['both_singles_correct_joint_wrong'])==((scores[:,l]==1)&(scores[:,se]==1)&(scores[:,j]==0)).sum()
ref=int(scores[:,conds.index('All-Writer')].sum())
for x in sums:
    j=conds.index(x['condition']);correct=int(scores[:,j].sum());assert int(x['correct'])==correct
    delta=100*(correct-ref)/240
    assert (x['point_minus1_candidate']=='True')==(delta>=-1.) and (x['point_minus2_candidate']=='True')==(delta>=-2.)
    rr=[v for v in rows if v['condition']==x['condition']]
    assert abs(float(x['connection_budget_pooled_pct'])-100*sum(int(v['allowed_R_image_connections_per_head']) for v in rr)/sum(int(v['all_R_image_connections_per_head']) for v in rr))<1e-12
negative=table('negative_cases.csv');expected=set()
for d in pr['comparisons']:
    for im in images:
        if lookup[(im,d['b'])]['correct']=='1' and lookup[(im,d['a'])]['correct']=='0':expected.add((im,d['a'],d['b']))
assert {(x['image_sha256'],x['joint'],x['reference']) for x in negative}==expected
jr=table('per_question_joint.csv');assert len(jr)==720
assert {(x['image_sha256'],x['joint']) for x in table('unique_joint_harm_cases.csv')}=={(x['image_sha256'],x['joint']) for x in jr if x['joint_unique_harm']=='True'}
assert {(x['image_sha256'],x['condition']) for x in table('incorrect_answers.csv')}=={(x['image_sha256'],x['condition']) for x in rows if x['correct']=='0'}
va=read('validation.json');assert len(va['checkpoint_hashes'])==720 and len(va['reused_checkpoint_hashes'])==1200 and len(va['raw_hashes'])==1920
for n,h in {**va['checkpoint_hashes'],**va['reused_checkpoint_hashes']}.items():assert sha(p/n)==h
for n,h in va['raw_hashes'].items():assert sha(Path(n))==h
assert sha(p/'token_group_manifest.json')==sha(Path('/data/lk/ATE-worktrees/semantic-visual-writers-240/vlm_research/results/semantic_visual_writers_240/token_group_manifest.json'))
print('INDEPENDENT PASS: full answers, nine paired comparisons, three four-way interactions, frozen Token sets, actual edge budgets, point screens, all negative cases, raw/checkpoint/Git hashes')
print('RESULTS',[(co,int(scores[:,j].sum())) for j,co in enumerate(conds)])
