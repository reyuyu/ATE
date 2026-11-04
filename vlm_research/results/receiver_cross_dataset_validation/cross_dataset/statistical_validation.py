import os,sys,pathlib
os.environ['CUDA_VISIBLE_DEVICES']=''
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation');sys.path.insert(0,str(W/'vlm_research'))
from receiver_cross_dataset.common import *
from receiver_cross_dataset.analyze import paired
# Unequal questions per image: pooled question target is 50pp, image-equal would be 37.5pp.
r=paired([1.,.5,0.],[0.,0.,0.],['A','A','B'])
assert r['delta_pp']==50 and r['n_image_clusters']==2 and r['CI95_low_pp']==0 and r['CI95_high_pp']==75
assert r['score_up']==2 and r['score_equal']==1 and r['score_down']==0
# Signed soft-score differences must retain negative values.
n=paired([0.,0.],[.3,.6],['A','B']);assert abs(n['delta_pp']+45)<1e-12 and n['score_down']==2
assert n['CI95_low_pp']==-60 and n['CI95_high_pp']==-30
assert paired([1.,.5,0.],[0.,0.,0.],['A','A','B'])==r
z=paired([.3,.9],[.3,.9],['A','B']);assert z['delta_pp']==0 and z['cluster_signflip_p']==1 and z['CI95_low_pp']==z['CI95_high_pp']==0
write(OUT/'statistical_validation.json',dict(passed=True,UTC=utc(),unequal_cluster_size_question_weighting=r,signed_soft_score=n,zero_difference=z,deterministic_seed=True,bootstrap_replicates=20000))
print('STATISTICAL_TESTS_PASS',flush=True)
