from pathlib import Path
import ast,json,types,os,gzip,csv,logging,hashlib,collections
import pandas as pd
A=Path('/home/featurize/ate-audits/internvl3-ocrbench-v1-20261113');P=Path('/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090');O=A/'cpu'
with gzip.open(P/'manifests/OCRBench.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
with gzip.open(P/'OCRBench/per_question.csv.gz','rt') as f:pred=list(csv.DictReader(f))
raw=json.loads((A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/OCRBench/OCRBench.json').read_text())
path=A/'official/VLMEvalKit_baseline_pinned/vlmeval/vlm/internvl/internvl_chat.py';cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='InternVLChat')
ns=dict(os=os,DATASET_TYPE=lambda x:'VQA',DATASET_MODALITY=lambda x:'IMAGE',listinstr=lambda names,text:any(x in text for x in names))
nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['set_max_num','build_prompt']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
fake=types.SimpleNamespace(use_mpo_prompt=False,use_cot=False,use_custom_prompt=lambda x:True,dump_image=lambda line,dataset:['audit_image.png'],kwargs={})
ns['set_max_num'](fake,'OCRBench');different=[];promptsmoke=[]
for r in rows:
 message=ns['build_prompt'](fake,{'question':r['question']},'OCRBench');text='\n'.join(x['value'] for x in message if x['type']=='text')
 if text!=r['request'].removeprefix('<image>\n'):different.append(r['question_id'])
 if len(promptsmoke)<2:promptsmoke.append(dict(question_id=r['question_id'],historical_user_text=r['request'],official_message=message))
loc=json.loads((A/'historical/late9/local_image_map.json').read_text())
unavailable=[r['question_id'] for r in rows if r['image_sha256'] not in loc];blocked=[r['question_id'] for r in csv.DictReader((A/'historical/late9/blocked_requests.csv').open()) if r['benchmark']=='OCRBench']
# Run the exact OCRBench class evaluator used by VLMEvalKit, as well as its utility evaluator.
f=A/'official/VLMEvalKit_baseline_pinned/vlmeval/dataset/image_vqa.py';cl=next(n for n in ast.parse(f.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='OCRBench');fun=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate');fun.decorator_list=[]
ns2=dict(tqdm=lambda x:x,get_intermediate_file_path=lambda *a:'in-memory',dump=lambda *a:None)
exec(compile(ast.Module(body=[fun],type_ignores=[]),str(f),'exec'),ns2)
sc=[]
for method in ['native','physical_d']:
 rs=[r for r in pred if r['method']==method];df=pd.DataFrame([dict(prediction=r['text'],category=raw[int(r['question_id'])]['type'],answer=repr(raw[int(r['question_id'])]['answers'] if isinstance(raw[int(r['question_id'])]['answers'],list) else [raw[int(r['question_id'])]['answers']])) for r in rs]);ns2['load']=lambda _:df
 val=ns2['evaluate'](None,'memory');sc.append(dict(method=method,official_class_scores=val))
checks=dict(status='PASS',official_class_scoring=sc,official_max_num=fake.max_num,official_total_max_num=fake.total_max_num,official_prompt_difference_count=len(different),official_prompt_examples=promptsmoke,late9_map_unavailable_count=len(unavailable),late9_map_unavailable_exact_blocked_set=set(unavailable)==set(blocked),late9_available_ids=sorted(set(r['question_id'] for r in rows)-set(unavailable)),gpu_inference='NOT_RUN')
assert fake.max_num==24 and len(different)==1000 and len(unavailable)==997 and set(unavailable)==set(blocked)
(O/'official_path_and_late9_closure.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False))
