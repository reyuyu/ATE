"""Targeted CPU contracts for the new benchmark harness; no full historical suite."""
from .common import *
from .scoring import *
import gzip,ast
def main():
    checks={}
    assert CAPS=={"GQA":16,"ScienceQA-IMG":16,"TextVQA":32,"POPE":8,"MME":16,"MMBench":16,"OCRBench":128};checks["task_caps"]=True
    q={"question":"Which?","hint":"Line1\nLine2","choices":["red","blue"]}
    t=task_user("ScienceQA-IMG",q);assert "Line1\nLine2" in t and "A. red\nB. blue" in t and "\\n" not in t;checks["full_hint_options_real_newlines_no_gold"]=True
    r={"gold":"no"}
    assert score("POPE","No, there is not.",r)==("no",1.)
    assert score("POPE","yes",{"gold":"yes"})==("yes",1.)
    assert score("MME","No.",r)==("no",1.)
    assert score("GQA","Brown.",{"gold":"brown"})[1]==1
    assert score("ScienceQA-IMG","B",{"gold":1,"choices":["x","y"]})[1]==1
    assert score("MMBench","B",{"gold":"B","choices":{"A":"red","B":"blue"}})[1]==1;checks["official_short_answer_parsers"]=True
    for n in [1,2,3,4,10]:
        r={"gold":["cat"]*n+["dog"]*(10-n)}
        _,v=score("TextVQA","cat",r);assert abs(v-independent_score("TextVQA","cat",r))<1e-12
        assert abs(v-({1:.3,2:.6,3:.9,4:1,10:1}[n]))<1e-12
    checks["textvqa_true_ten_answer_consensus"]=True
    cases=[{"category":"Regular Text Recognition","dataset_name":"IIIT5K","gold":"HELLO","text":"hello"},{"category":"Handwritten Mathematical Expression Recognition","dataset_name":"HME100k","gold":"X + Y","text":"X+Y"},{"category":"Regular Text Recognition","dataset_name":"IIIT5K","gold":"CAT","text":"dog"}]
    official=official_ocr_batch(cases)
    assert official==[1,1,0]
    for r,v in zip(cases,official):assert score("OCRBench",r["text"],r)[1]==independent_score("OCRBench",r["text"],r)==v
    checks["original_ocr_scoring_loop"]=True
    core=source_fingerprints()
    assert core["vlm_research/internvl3_svh_physical_d_pilot/engine.py"]=="de5871a49416e507f6cd662646d52929ad8929cde9565ceff4f07b0707655b96"
    assert core["vlm_research/internvl3_visual_writing_figure3_3000/engine.py"]=="f2e4428f1b2e461b14c398dc232d61d456a9c48a57e99a00574bc402ab2a7bb1"
    checks["frozen_core_sha256_unchanged"]=True
    ds=readj(OUT/"dataset_manifest.json")
    for bench,d in ds.items():
        if d["status"]!="CPU_DATA_PASS":continue
        with gzip.open(RUN/"manifests"/(bench+".jsonl.gz"),"rt") as f:
            rows=[json.loads(x) for x in f]
        assert len(rows)==EXPECTED[bench] and len({r["question_id"] for r in rows})==len(rows)
        assert len(d["preflight_question_ids"])>=24
        for r in rows:
            assert r["roles"]["image"]==list(range(r["roles"]["image"][0],r["roles"]["image"][0]+256))
            assert sha(r["prompt"])==r["prompt_sha256"] and r["max_new_tokens"]==CAPS[bench]
            assert Path(r["image_path"]).is_file()
    checks["all_ready_manifest_bindings_roles_preflight"]=True
    for name in ["common","restore","prepare","scoring","worker","cpu_tests"]:
        ast.parse((RUN/"code/vlm_research/internvl3_svh_seven_benchmarks"/(name+".py")).read_text())
    checks["new_source_syntax"]=True
    writej(OUT/"cpu_tests.json",{"status":"PASS","checks":checks,"no_cuda_started":True,"scope":"new harness contracts only; historical 452-test suite not rerun"})
    print("CPU_TESTS_PASS",len(checks),flush=True)
if __name__=="__main__":main()
