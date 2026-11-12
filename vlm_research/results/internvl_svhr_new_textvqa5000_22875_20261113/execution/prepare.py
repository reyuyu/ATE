import sys,json,gzip,csv,hashlib,struct,zipfile,shutil
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-textvqa5000-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113');C=Path('/home/featurize/ate-experiments/internvl-svh-r-new-ocrbench1000-22875-20261113')
sys.path.insert(0,str(L));from run_common import atomic,sha,gpu_idle
sys.path.insert(0,str(O));from scoring import Scorer
assert not (O/'protocol.json').exists();print('GPU',gpu_idle())
receipt=json.loads((O/'reference_receipt.json').read_text());assert sha(O/'references.zip')==receipt['zip_sha256']
R=O/'references';R.mkdir(exist_ok=True)
with zipfile.ZipFile(O/'references.zip') as z:
 assert all(len(Path(n).parts)==1 for n in z.namelist());z.extractall(R)
for x in receipt['files']:assert sha(R/x['path'])==x['sha256']
rows=[json.loads(s) for s in gzip.open(R/'TextVQA.jsonl.gz','rt')];assert len(rows)==5000 and sha(R/'TextVQA.jsonl.gz')=='975b095b2048f6ef4f79e102e6fde07ae33946fa5eadc04436cb181d5cd804ff'
D=Path('/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr');bundle=json.loads((D/'bundle_manifest.json').read_text());meta=json.loads((D/'image_sha256.json').read_text())
assert sha(D/'image_sha256.json')==bundle['metadata_files']['image_sha256.json']
ann=D/'source_annotations/textvqa/TextVQA_0.5.1_val.json';assert sha(ann)==bundle['metadata_files']['source_annotations/textvqa/TextVQA_0.5.1_val.json']
raw={str(r['question_id']):r for r in json.loads(ann.read_text())['data']};assert set(raw)=={r['question_id'] for r in rows}
scorer=Scorer(rows,R/'official_textvqa_eval.py');baselines={m:[] for m in ['native','physical_d']};expected={};byid={r['question_id']:r for r in rows}
for x in csv.DictReader(gzip.open(R/'baselines_native_physical.csv.gz','rt')):
 m=x['method'];assert m in baselines;q=x['question_id'];r=byid[q];b=json.loads(x['input_binding_details'])
 assert json.loads(x['gold'])==r['gold'] and x['question']==r['question'] and x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
 assert int(x['max_new_tokens'])==32
 if x['eos']=='True':assert json.loads(x['token_ids'])[-1]==151645
 score=scorer.score(q,x['text']);assert abs(float(x['score'])-score['official_score'])<1e-10
 if q in expected:assert expected[q]==b
 expected[q]=b;baselines[m].append(dict(question_id=q,text=x['text'],token_ids=json.loads(x['token_ids']),image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],gold=r['gold'],eos=x['eos']=='True',empty=x['empty']=='True',truncated=x['truncated']=='True',**score))
assert all(len(rs)==5000 and len({x['question_id'] for x in rs})==5000 for rs in baselines.values())
pixels=json.loads((R/'processor_bindings.json').read_text());paths={}
for r in rows:
 q=r['question_id'];a=raw[q];assert r['question']==a['question'] and r['gold']==a['answers'] and len(r['gold'])==10 and r['image_id']==a['image_id']
 rel='images/textvqa/'+r['image_id']+'.jpg';p=D/rel;assert p.is_file() and p.stat().st_size==r['image_bytes']==meta[rel]['bytes'] and meta[rel]['sha256']==r['image_sha256']
 assert r['max_new_tokens']==32 and r['split']=='validation' and len(r['roles']['image'])==256 and r['input_tokens']==len(r['input_ids'])
 assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
 assert hashlib.sha256(struct.pack('<'+'q'*len(r['input_ids']),*r['input_ids'])).hexdigest()==expected[q]['input_ids_sha256']
 assert pixels[r['image_sha256']]['pixels_bf16_sha256']==expected[q]['pixels_bf16_sha256']
 paths[r['image_sha256']]=str(p)
assert len(paths)==3166
configuration=json.loads((C/'configuration.json').read_text());assert configuration['reaccess_layers']==list(range(14,28)) and configuration['configuration_name']=='interVL-SVH-R-new'
source=json.loads((C/'protocol.json').read_text())['source_SHA256']
for p,h in source.items():assert sha(L/'late9_sources'/p)==h,p
atomic(O/'frozen_rows.json',rows);atomic(O/'expected_bindings.json',expected);atomic(O/'input_paths.json',paths);atomic(O/'baselines.json',baselines);shutil.copy2(C/'configuration.json',O/'configuration.json')
overlay=dict(configuration_name=configuration['configuration_name'],base_configuration_sha256=sha(O/'configuration.json'),benchmark='TextVQA v0.5.1 validation',sample_count=5000,input_protocol='ATE_SVH_1TILE_SCORER_ALIGNED',image_only=True,OCR_text_in_prompt=False,max_new_tokens=32,eos_token_id=151645,greedy=True,scorer='Frozen official InternVL TextVQAAccuracyEvaluator:10-answer leave-one-out VQA soft accuracy',override_note='Benchmark-specific cap32 and historical TextVQA prompt; the base named configuration was registered with OCRBench cap128. Physical algorithm unchanged.')
atomic(O/'evaluation_config.json',overlay)
smoke=[rows[i]['question_id'] for i in [0,1234,2500,4999]]
protocol=dict(status='FROZEN_BEFORE_GENERATION',configuration_name=configuration['configuration_name'],model=configuration['backbone'],revision=configuration['model_revision'],configuration_sha256=sha(O/'configuration.json'),evaluation_config_sha256=sha(O/'evaluation_config.json'),expected=5000,unique_images=3166,new_generation_limit=5000,reused_late14_predictions=0,max_new_tokens=32,smoke_ids=smoke,rows_sha256=sha(O/'frozen_rows.json'),bindings_sha256=sha(O/'expected_bindings.json'),baselines_sha256=sha(O/'baselines.json'),source_SHA256=source,source_manifest_commit='3297559f21d3de55a6630a5491c9ecf66301c4b7',late9_summary_commit='c60a9919112a3cf5bec8c494a533cba56a97f29b',late9_published_percent=56.004,late9_pairwise_available=False,late9_baseline_note='Use verified published full5000 aggregate; recovery_cases.csv contains4370 changed-only cases, not full5000. Avoid downloading40MB all-benchmark raw table merely for optional pairing.',reaccess_layers=list(range(14,28)),cutoff=28,R_image_permission_boundary=20,skip_answer_attention=[2,3,4,5],numerical_status=configuration['numerical_qualification'],hard_stop='OOM, nonfinite, input/image/token/processor binding or structural cache/RoPE/skip/exit errors. Keep partial records; never change protocol to pass.',resources='Reuse local model, environment and3166 image files; no model/data download, no baseline generation or efficiency test.')
atomic(O/'protocol.json',protocol);atomic(O/'cpu_audit.json',dict(status='PASS',questions=5000,images=3166,baseline_predictions_checked=10000,normalization_checks=scorer.normalization_checks,baseline_percent={m:100*sum(x['official_score'] for x in rs)/5000 for m,rs in baselines.items()},source_files_verified=len(source),all_images_exist=True,bulk_image_SHA_repeated=False))
print('CPU_PREPARE_PASS',json.dumps(dict(questions=5000,images=3166,smoke_ids=smoke,baseline_percent={m:100*sum(x['official_score'] for x in rs)/5000 for m,rs in baselines.items()})))
