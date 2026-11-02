"""Presentation-only paired transitions from the frozen240 outputs, no model calls."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name, rows, fields):
    with (OUT/name).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    path=OUT/'paired_per_question.csv'
    with path.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==len({r['question_id'] for r in rows})==240
    tasks=sorted({r['task'] for r in rows})
    result=[]
    for task in ['All']+tasks:
        ss=[r for r in rows if task=='All' or r['task']==task]
        for source in ['Full','A-Skip','Text-Only']:
            a=[int(r[source+'_score']) for r in ss]
            j=[int(r['Joint_score']) for r in ss]
            result.append(dict(task=task,reference=source,n=len(ss),
                reference_correct_joint_wrong=sum(x==1 and y==0 for x,y in zip(a,j)),
                reference_wrong_joint_correct=sum(x==0 and y==1 for x,y in zip(a,j)),
                joint_minus_reference_pp=100*sum(y-x for x,y in zip(a,j))/len(ss)))
    table('joint_pair_transitions.csv',result,list(result[0]))
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    samples={s['question_id']:s for s in manifest['samples']}
    cases=[]
    fields=['task','question_id','image_sha256','image_path','question','gold',
            'Full_text','A-Skip_text','Text-Only_text','Joint_text']
    for r in rows:
        unique=all(int(r[c+'_score'])==1 for c in ['Full','A-Skip','Text-Only']) and int(r['Joint_score'])==0
        assert int(r['joint_unique_harm'])==int(unique)
        if unique:
            s=samples[r['question_id']]
            cases.append({k:(s['image_path'] if k=='image_path' else r[k]) for k in fields})
    table('joint_unique_harm_cases.csv',cases,fields)
    record=dict(passed=True,input_sha256=sha(path),source_sha256=sha(__file__),
        rows=240,unique_harm_count=len(cases),posthoc_error_listing_not_a_new_experiment=True,
        outputs={n:sha(OUT/n) for n in ['joint_pair_transitions.csv','joint_unique_harm_cases.csv']})
    (OUT/'supplement_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Supplemental paired transitions complete; Joint-only harms:',len(cases))


if __name__=='__main__':main()
