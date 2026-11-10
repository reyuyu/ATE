"""Independent format/binding/completeness/KV and persistence checks, CPU only."""
from .common import *
import gzip,sqlite3,math
from collections import defaultdict
def main():
    result={}
    for bench in BENCHES:
        val=OUT/bench/"validation.json";path=OUT/bench/"per_question.csv.gz"
        if not val.exists() or not path.exists():continue
        v=readj(val);records=[]
        with gzip.open(path,"rt") as f:records=list(csv.DictReader(f))
        seen=set();by_pair=defaultdict(dict)
        for x in records:
            key=(x["question_id"],x["method"]);assert key not in seen;seen.add(key);by_pair[key[0]][key[1]]=x
            ids=json.loads(x["token_ids"]);n=int(x["generated_tokens"]);cap=int(x["max_new_tokens"]);assert len(ids)==n and 1<=n<=cap
            eos=x["eos"]=="True";trunc=x["truncated"]=="True"
            assert eos==(ids[-1]==151645) and trunc==(n==cap and not eos)
            assert int(x["visual_tokens"])==256 and 0<=float(x["score"])<=1
            assert x["protocol_sha256"]==sha(OUT/"protocol.json")
        for q,pair in by_pair.items():
            if len(pair)==2:
                a=pair["native"];b=pair["physical_d"];assert a["input_binding"]==b["input_binding"] and a["prompt_sha256"]==b["prompt_sha256"]
        live=defaultdict(int);counts=defaultdict(int)
        with gzip.open(OUT/bench/"kv_layout_per_layer.csv.gz","rt") as f:
            for row in csv.DictReader(f):
                method={"Native":"native","Physical-D":"physical_d"}[row["method"]];qid=row["question_id"];x=by_pair[qid][method];layer=int(row["layer"]);physical=int(row["physical_length"]);visual=int(row["visual_kv"]);logical=int(x["input_tokens"])+int(x["generated_tokens"])-1
                assert int(row["logical_length"])==logical
                if method=="native":assert physical==logical and visual==256
                elif layer in (2,3,4,5):assert physical==0 and visual==0
                else:assert physical==logical-256 and visual==0
                live[(qid,method)]+=int(row["live_bytes"]);counts[(qid,method)]+=1
        for key in seen:
            assert counts[key]==28
            assert abs(live[key]/1048576-float(by_pair[key[0]][key[1]]["KV_live_MiB"]))<1e-10
        if v["status"]=="COMPLETE":assert len(records)==2*EXPECTED[bench] and len(by_pair)==EXPECTED[bench] and all(len(x)==2 for x in by_pair.values())
        else:assert not v["full_split_score_reported"]
        result[bench]={"status":"PASS","predictions":len(records),"paired_ids":sum(len(x)==2 for x in by_pair.values()),"all_token_EOS_cap_bindings_valid":True,"all_28_layer_KV_layouts_and_measured_live_bytes_valid":True,"formal_status":v["status"]}
    writej(OUT/"independent_artifact_validation.json",{"status":"PASS","benchmarks":result,"no_CUDA_or_model_load":True,"protocol_sha256":sha(OUT/"protocol.json"),"does_not_override_numerical_waiver":True})
    print("INDEPENDENT_ARTIFACT_VALIDATION_PASS",list(result),flush=True)
if __name__=="__main__":main()
