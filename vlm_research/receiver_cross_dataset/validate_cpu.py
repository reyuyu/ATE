import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,pathlib,unittest,io
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation');sys.path.insert(0,str(W/'vlm_research'))
from receiver_cross_dataset.common import *
import receiver_cross_dataset.tests as new
import gqa3000_token_visual_writing.tests as old
import semantic_visual_writers.tests as masks
old.OUT=pathlib.Path('/data1/likai/ATE-worktrees/semantic-single-gpu-search3h/vlm_research/results/gqa3000_token_visual_writing')
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [new,old,masks]])
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
write(OUT/'CPU_test_suite.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),output=stream.getvalue(),historical_data_root=str(old.OUT),initial_invocation_errors='Two historical integrity tests initially lacked old results in sparse worktree; full unmodified tests rerun using original frozen history directory. No test or tolerance removed.'))
print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not result.skipped
from receiver_cross_dataset.prepare import cpu_inputs,AutoProcessor
proc=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True);proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
r=read('/data1/likai/ATE-resources/experiments/gqa5h_receiver_search/samples/001020372.json');s=cpu_inputs(proc,r);print('ASSISTANT_IDS',s['ASSISTANT_ids'],'INPUT_KEYS',list(s),flush=True)
assert s['prompt_ids']==r['input_mapping']['prompt_ids']
write(OUT/'historical_native_input_regression.json',dict(passed=True,question_id=r['question_id'],ASSISTANT_ids=s['ASSISTANT_ids'],exact_historical_prompt_ids=True,input_hashes=s['input_hashes']))
