#!/usr/bin/env python3
"""Independent CPU-only rescore/coverage audit; no Torch or model imports."""
import argparse,ast,collections,copy,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--audit-dir',type=Path,required=True);a=ap.parse_args()
    R=a.results;A=a.audit_dir
    protocol=json.loads((R/'protocol.json').read_text());binding=sha(R/'protocol.json')
    assert sha(R/'inputs.jsonl')==protocol['inputs_manifest_sha256']
    assert sha(R/'gold.jsonl')==protocol['gold_manifest_sha256']
    inputs=[json.loads(s) for s in (R/'inputs.jsonl').read_text().splitlines()]
    gold=[json.loads(s) for s in (R/'gold.jsonl').read_text().splitlines()]
    raw=json.loads((A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/OCRBench/OCRBench.json').read_text())
    assert len(inputs)==len(gold)==len(raw)==1000
    for i,(x,g,r) in enumerate(zip(inputs,gold,raw)):
        assert x['question_id']==g['question_id']==f'{i:04d}'
        assert x['official_id']==g['official_id']==r['id'] and x['question']==r['question']
        assert x['image_id']==r['image_path'] and g['answers']==r['answers']
        assert g['category']==r['type'] and g['dataset_name']==r['dataset_name']
    preds=[json.loads(p.read_text()) for p in sorted((R/'per_question').glob('*.json'))]
    assert len({r['question_id'] for r in preds})==len(preds)
    tree=ast.parse((A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and "data[i]['result']" in ast.unparse(n) and 'data_type' in ast.unparse(n) and 'dataset_name' in ast.unparse(n))
    data=[]
    for p in preds:
        i=int(p['question_id']);assert p['protocol_sha256']==binding and p['input']==inputs[i] and p['status']=='OK'
        data.append(dict(copy.deepcopy(raw[i]),predict=p['text']))
    exec(compile(ast.Module(body=[loop],type_ignores=[]),'<pinned OCRBench v1 scoring loop>','exec'),{'data':data})
    # Separate human-readable logic; never calls the inference harness scoring functions.
    manual=[]
    for p,d in zip(preds,data):
        def norm(s):
            s=s.strip().replace('\n',' ')
            return s.replace(' ','') if d['dataset_name']=='HME100k' else s.lower()
        answers=d['answers'] if isinstance(d['answers'],list) else [d['answers']]
        matched=[x for x in answers if norm(x) in norm(p['text'])]
        sc=int(bool(matched));assert sc==d['result']==p['official_score']==p['independent_score']
        manual.append(dict(question_id=p['question_id'],official_id=d['id'],tiles=p['input']['tiles'],
            prediction=p['text'],answers=json.dumps(answers,ensure_ascii=False),matched_gold=json.dumps(matched,ensure_ascii=False),
            official_score=d['result'],independent_score=sc,normalization='remove all ASCII spaces; case sensitive' if d['dataset_name']=='HME100k' else 'lowercase; strip; newline to space',
            generated_tokens=p['generated_tokens'],eos=p['stopped_on_eos'],truncated=p['truncated']))
    # Also execute VLMEvalKit's pinned scorer unchanged, with only I/O adapters.
    tree=ast.parse((A/'official/VLMEvalKit_baseline_pinned/vlmeval/dataset/utils/ocrbench.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='OCRBench_eval')
    class Table:
        def __len__(self):return len(data)
        @property
        def iloc(self):return self
        def __getitem__(self,i):return dict(prediction=data[i]['predict'],answer=repr(data[i]['answers']),category=data[i]['type'])
    class Log:
        def info(self,*args):pass
    capture={}
    env=dict(load=lambda _:Table(),tqdm=lambda x:x,get_intermediate_file_path=lambda *a:'unused',dump=lambda obj,path:capture.update(obj),logger=Log())
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned VLMEvalKit OCRBench_eval>','exec'),env);env['OCRBench_eval']('unused')
    assert capture['Final Score']==sum(d['result'] for d in data)
    with (R/'independent_scoring.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manual[0]) if manual else ['question_id']);w.writeheader();w.writerows(manual)
    failure=json.loads((R/'failure.json').read_text()) if (R/'failure.json').exists() else None
    failed=failure['current_observation']['question_id'] if failure else None
    done={r['question_id'] for r in preds}
    with (R/'coverage.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['question_id','image_id','image_sha256','tiles','status','score']);w.writeheader()
        for x in inputs:
            status='SUCCESSFUL_SMOKE' if x['question_id'] in done else 'OOM_NO_PREDICTION' if x['question_id']==failed else 'NOT_RUN'
            w.writerow({k:x[k] for k in ['question_id','image_id','image_sha256','tiles']}|dict(status=status,score=next((p['official_score'] for p in preds if p['question_id']==x['question_id']),'')))
    result=dict(status='PASS_CPU_AUDIT',planned=1000,successful=len(preds),failed_attempts=int(failure is not None),not_attempted=1000-len(preds)-int(failure is not None),
        formal_evaluation_started=False,full_native_score=None,difference_from_880=None,successful_smoke_points_only=sum(d['result'] for d in data),
        vlmevalkit_rescore=capture,all_scores_agree=True,all_ids_gold_and_order_match=True,failed_id=failed,
        empty_outputs=sum(p['empty_output'] for p in preds),truncated=sum(p['truncated'] for p in preds),protocol_sha256=binding)
    (R/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
