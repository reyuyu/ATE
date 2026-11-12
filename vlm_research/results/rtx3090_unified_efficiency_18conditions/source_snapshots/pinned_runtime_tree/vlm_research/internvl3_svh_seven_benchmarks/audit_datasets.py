"""Independent source-annotation/split and benchmark structure audit; no image rehash."""
from .common import *
import gzip,collections
def main():
    meta={}
    for bench in BENCHES:
        with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:rows=[json.loads(x) for x in f]
        assert len(rows)==EXPECTED[bench]
        for r in rows:
            q=dict(r);h=q.pop("sample_sha256");assert bind(q)==h
        meta[bench]={"N":len(rows),"unique_image_SHA256":len({r["image_sha256"] for r in rows}),"sample_binding_recomputed":True}
        if bench=="POPE":
            counts=collections.Counter(r["category"] for r in rows);assert counts==dict(random=3000,popular=3000,adversarial=3000)
            gold=collections.Counter(r["gold"] for r in rows);assert gold==dict(yes=4500,no=4500)
            meta[bench].update(strategies=counts,Gold=gold)
        if bench=="ScienceQA-IMG":
            root=Path("/home/featurize/data/ate-seven-benchmarks-3090/restored/ate3090-scienceqa-img/source_annotations/scienceqa")
            problems=readj(root/"problems.json");split=readj(root/"pid_splits.json")["test"]
            expected={str(pid) for pid in split if problems[str(pid)].get("image")}
            assert {r["question_id"] for r in rows}==expected
            meta[bench]["official_image_test_ID_set_exact"]=True
        if bench=="MME":
            groups=collections.Counter(r["group"] for r in rows);assert len(groups)==1187 and set(groups.values())=={2}
            cats=collections.Counter(r["category"] for r in rows);assert len(cats)==14;meta[bench].update(paired_images=1187,categories=cats)
        if bench=="MMBench":
            groups=collections.Counter(r["group"] for r in rows);assert len(groups)==1176;meta[bench].update(circular_groups=1176,published_rotations=collections.Counter(groups.values()),aggregation="all published rows in each group; no synthetic rotations")
        if bench=="OCRBench":
            cats=collections.Counter(r["category"] for r in rows);assert len(cats)==10 and sorted(cats.values())==[50]*6+[100]+[200]*3;meta[bench]["categories"]=cats
        if bench=="TextVQA":assert all(len(r["gold"])==10 for r in rows);meta[bench]["ten_answers_all_5000"]=True
    writej(OUT/"dataset_structure_validation.json",{"status":"PASS","benchmarks":meta,"no_GPU_or_full_image_hash_scan":True})
    print("SOURCE_SPLIT_STRUCTURE_AUDIT_PASS",flush=True)
if __name__=="__main__":main()
