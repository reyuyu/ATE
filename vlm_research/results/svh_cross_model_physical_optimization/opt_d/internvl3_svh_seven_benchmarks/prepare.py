"""CPU-only dataset, prompt, scorer and deterministic preflight freezing."""
from .common import *
import gzip,shutil,collections,platform,ast,re
from .scoring import score,independent_score,anomalies,official_ocr_batch
DATA=Path("/home/featurize/data/ate-seven-benchmarks-3090/restored")
def records(b):
    old=Path("/home/featurize/data/ate-recovery")
    if b in ("GQA","TextVQA"):
        bundle=old/"restored_bundles"/("ate3090-gqa-testdev-balanced" if b=="GQA" else "ate3090-textvqa-ocr")
        view=bundle/"views"/("GQA-testdev-balanced" if b=="GQA" else "TextVQA")
    elif b=="OCRBench":
        bundle=DATA/"OCRBench-v1";src=readj(bundle/"SOURCE.json");fm=readj(bundle/"file_manifest.json")["files"];raw=readj(bundle/"OCRBench.json")
        assert len(raw)==1000 and sha(bundle/"OCRBench.json")==src["annotation"]["sha256"]
        out=[]
        for i,x in enumerate(raw):
            rel="OCRBench_Images/"+x["image_path"];meta=fm[rel]
            out.append(dict(benchmark=b,question_id=str(i).zfill(4),official_id=x["id"],question=x["question"],gold=x["answers"],category=x["type"],dataset_name=x["dataset_name"],image_id=x["image_path"],image_path=str(bundle/rel),image_sha256=meta["sha256"],image_bytes=meta["bytes"],group=str(i)))
        return out,{"source":src,"annotation_sha256":sha(bundle/"OCRBench.json"),"file_manifest_sha256":sha(bundle/"file_manifest.json")}
    else:
        bundle=DATA/{"ScienceQA-IMG":"ate3090-scienceqa-img","POPE":"ate3090-pope","MME":"ate3090-mme","MMBench":"ate3090-mmbench"}[b];view=bundle/"views"/b
    rows=loadrows(view/"inputs.jsonl");answers=readj(view/"answers.json");bm=readj(bundle/"bundle_manifest.json");ims=readj(bundle/"image_sha256.json")
    k="GQA-testdev-balanced" if b=="GQA" else b
    assert sha(view/"inputs.jsonl")==bm["views"][k]["inputs_sha256"]
    assert sha(view/"answers.json")==bm["views"][k]["answers_sha256"]
    assert len(rows)==len(answers)==EXPECTED[b]
    raw_map=None
    if b=="GQA":raw_map=readj(bundle/"source_annotations/gqa/testdev_balanced_questions.json")
    if b=="TextVQA":raw_map={str(x["question_id"]):x for x in readj(bundle/"source_annotations/textvqa/TextVQA_0.5.1_val.json")["data"]}
    if b=="ScienceQA-IMG":raw_map=readj(bundle/"source_annotations/scienceqa/problems.json")
    out=[]
    for row in rows:
        q=str(row["question_id"]);r=dict(row);r.update(benchmark=b,gold=answers[q],question_id=q)
        rel=row["image_path"];rel=("images/"+rel.split("/images/",1)[1]) if rel.startswith("/") else rel
        meta=ims[rel];r.update(image_path=str(bundle/rel),image_sha256=meta["sha256"],image_bytes=meta["bytes"])
        r["group"]=str(int(q)%1000000) if b=="MMBench" else (row.get("subtask","")+"/"+row["image_id"] if b=="MME" else q)
        r["category"]=row.get("subtask",row.get("strategy",row.get("task","all")))
        if b=="GQA":
            x=raw_map[q];assert row["question"]==x["question"] and str(row["image_id"])==str(x["imageId"]) and r["gold"]==x["answer"]
        if b=="TextVQA":
            x=raw_map[q];assert row["question"]==x["question"] and row["image_id"]==x["image_id"] and r["gold"]==x["answers"] and len(r["gold"])==10
        if b=="ScienceQA-IMG":
            x=raw_map[q];assert all(row[f]==x[f] for f in ("question","choices","hint")) and r["gold"]==x["answer"] and x["split"]=="test" if "split" in x else all(row[f]==x[f] for f in ("question","choices","hint")) and r["gold"]==x["answer"]
        assert Path(r["image_path"]).stat().st_size==r["image_bytes"]
        out.append(r)
    if b=="MMBench":
        groups=collections.defaultdict(list)
        for r in out:groups[r["group"]].append(r)
        for g,rs in groups.items():
            # Official mcq_circular_eval groups published rows, not an invented rotation for every option.
            assert 2<=len(rs)<=4 and any(x["question_id"]==g for x in rs)
            assert sorted(int(x["question_id"])//1000000 for x in rs)==list(range(len(rs)))
            assert len({x["choices"][x["gold"]] for x in rs})==1
            assert len({x["image_sha256"] for x in rs})==1
            assert len({x["question"] for x in rs})==1
        groupinfo={"circular_groups":len(groups),"rotations":dict(collections.Counter(map(len,groups.values()))),"rule":"index modulo 1e6; all rotations must be correct; no LLM judge"}
    elif b=="MME":
        pairs=collections.Counter(r["group"] for r in out);assert set(pairs.values())=={2};groupinfo={"paired_images":len(pairs),"tasks":dict(collections.Counter(r["category"] for r in out)),"known_MME_image_SHA_differences":"Historical downloaded image bytes retained; no replacement or normalization."}
    else:groupinfo={}
    return out,{"bundle":str(bundle),"bundle_manifest_sha256":sha(bundle/"bundle_manifest.json"),"inputs_sha256":sha(view/"inputs.jsonl"),"answers_sha256":sha(view/"answers.json"),"source":bm["views"][k].get("provenance",{}),**groupinfo}
def select_preflight(rows):
    # Selection uses only subtype, number of options, input length, QID and image SHA; no model outputs/gold.
    buckets=collections.defaultdict(list)
    for r in rows:buckets[(r["category"],len(r.get("choices",[])))].append(r)
    selected=[];seen=set()
    for key in sorted(buckets,key=str):
        group=sorted(buckets[key],key=lambda r:(r["input_tokens"],bind([r["question_id"],"preflight-v1"])))
        for idx in [0,len(group)//2,len(group)-1]:
            r=group[idx]
            if r["image_sha256"] not in seen:
                selected.append(r);seen.add(r["image_sha256"])
    if len(selected)<24:
        for r in sorted(rows,key=lambda r:bind(["preflight-v1",r["question_id"]])):
            if r["image_sha256"] not in seen:selected.append(r);seen.add(r["image_sha256"])
            if len(selected)>=24:break
    return [r["question_id"] for r in selected] # May exceed 24 to cover all 14 MME/10 OCR subtypes.
def main():
    from transformers import AutoTokenizer
    from . import common
    from vlm_research.internvl3_visual_writing_figure3_3000 import engine as base
    OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,use_fast=False,local_files_only=True)
    # Use pinned remote model's actual native conversation template without loading weights or CUDA.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    Chat=get_class_from_dynamic_module("modeling_internvl_chat.InternVLChatModel",str(MODEL),local_files_only=True)
    scope=sys.modules[Chat.__module__];conv=scope.get_conv_template("internvl2_5")
    config=readj(MODEL/"config.json")
    if config.get("system_message"):conv.system_message=config["system_message"]
    dataset={};prompts={};matrix=[];all_bind=[]
    for b in BENCHES:
        try:
            rows,meta=records(b);assert len(rows)==EXPECTED[b]
            path=RUN/"manifests"/(b+".jsonl.gz");path.parent.mkdir(exist_ok=True)
            with gzip.open(path,"wt",encoding="utf8") as f:
                for r in rows:
                    request=request_text(b,r);c=conv.copy();c.append_message(c.roles[0],request);c.append_message(c.roles[1],None)
                    prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1)
                    ids=tok(prompt)["input_ids"];roles=base.role_layout(ids,tok.convert_tokens_to_ids("<IMG_CONTEXT>"))
                    r.update(request=request,prompt=prompt,prompt_sha256=sha(prompt),input_ids=ids,input_tokens=len(ids),roles=roles,max_new_tokens=CAPS[b],protocol_label="ATE_SVH_1TILE_SCORER_ALIGNED")
                    r["sample_sha256"]=bind(r);all_bind.append([b,r["question_id"],r["sample_sha256"]]);f.write(canonical(r)+"\n")
            pre=select_preflight(rows)
            dataset[b]={"status":"CPU_DATA_PASS","questions":len(rows),"unique_images":len({r["image_sha256"] for r in rows}),"manifest_sha256":sha(path),"metadata":meta,"preflight_question_ids":pre,"length_range":[min(x["input_tokens"] for x in rows),max(x["input_tokens"] for x in rows)]}
            prompts[b]={"full_prompt_file":"manifests/"+b+".jsonl.gz","sha256":sha(path),"full_prompt_and_SHA_per_question":True,"request_recipe":"task_user() in common.py","cap":CAPS[b],"real_EOS_id":tok.convert_tokens_to_ids(conv.sep.strip()),"image_only":b=="TextVQA"}
            state(b,"READY_PREFLIGHT",questions=len(rows),preflight=len(pre))
            print("FROZEN",b,len(rows),len(pre),dataset[b]["length_range"],flush=True)
        except Exception as e:
            import traceback
            dataset[b]={"status":"BLOCKED_SCORER_OR_DATA","error":str(e),"traceback":traceback.format_exc()};state(b,"BLOCKED_SCORER_OR_DATA",error=str(e));print("BLOCKED",b,str(e),flush=True)
        matrix.append({"benchmark":b,"protocol":"ATE_SVH_1TILE_SCORER_ALIGNED","author_recipe":"AUTHOR_RECIPE_INCOMPATIBLE_OR_NOT_FULLY_VERIFIED","reason":"Fixed single tile and task engineering cap; author/VLMEvalKit supports dynamic multiple tiles, other max tokens or CoT; no leaderboard reproduction claim.","max_new_tokens":CAPS[b],"expected_N":EXPECTED[b],"data_status":dataset[b]["status"]})
    writej(OUT/"dataset_manifest.json",dataset);writej(OUT/"prompt_manifest.json",prompts);csvwrite(OUT/"benchmark_protocol_matrix.csv",matrix)
    protocol={"version":"1.0.0","source_commit":SOURCE,"source_sha256":source_fingerprints(),"gpu_uuid":GPU,"model":"OpenGVLab/InternVL3-8B","revision":REV,"dtype":"bfloat16","attention":"eager","torch":"2.6.0+cu124","transformers":"4.49.0","batch_size":1,"do_sample":False,"num_beams":1,"use_cache":True,"logits_to_keep":1,"single_tile":True,"image_shape":[1,3,448,448],"visual_tokens":256,"methods":METHODS,"caps":CAPS,"min_new_tokens":0,"real_EOS_id":tok.convert_tokens_to_ids(conv.sep.strip()),"eos_token":conv.sep.strip(),"fresh_cache":True,"numerical_status":"FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER","physical_mechanism":"Unmodified Physical-D from e288eee: L20 hidden exit, L2-L5 answer attention skip, compact KV; no persistent original image keys.","preflight_selection":"Stratify category and options; shortest/median/longest per stratum with unique images; fill to >=24 by SHA256 rank; no gold/output used.","preflight_stop":"Any empty output/repetition; all non-OCR abnormal EOS/truncation; obvious POPE/MME/ScienceQA/MMBench explanation; unexplained GQA/TextVQA verbosity. OCR genuine text allowed; two consecutive cap hits per method stop.","full_run_stop":"Same abnormal output rules; no cap/prompt adjustment based on accuracy. Every anomalous prediction retained and scored; affected benchmark paused.","bootstrap":{"repetitions":20000,"seed":20261110,"cluster":"image_sha256","paired":True,"confidence":0.95,"multiplicity":"Ordinary unadjusted CIs; seven benchmarks and subgroups are multiple descriptive comparisons."},"dataset_manifest_sha256":sha(OUT/"dataset_manifest.json"),"all_inputs_binding":bind(all_bind),"no_efficiency_benchmark":True,"author_recipe_label":"NOT_VERIFIED; main results ATE_SVH_1TILE_SCORER_ALIGNED"}
    writej(OUT/"protocol.json",protocol)
    manifest={"InternVL":readj(RUN/"references/tree_InternVL.json")["commit"],"VLMEvalKit":readj(RUN/"references/tree_VLMEvalKit.json")["commit"],"official_sources":readj(RUN/"references/source_manifest.json"),"source_core":source_fingerprints(),"scoring":"GQA exact normalized; ScienceQA exact option; TextVQA official 10-answer leave-one-out; POPE official lexical and confusion; MME official acc+acc_plus; MMBench official exact_matching and complete circular groups; OCR original MultimodalOCR v1 loop.","limitations":"MMBench exact_matching without LLM judge; not author leaderboard recipe."}
    writej(OUT/"scorer_manifest.json",manifest)
    writej(OUT/"model_manifest.json",{"path":str(MODEL),"model_revision":REV,"config_sha256":sha(MODEL/"config.json"),"prior_weight_integrity_receipt":str(ROOT/"vlm_research/results/internvl3_svh_physical_d_pilot_3090/model_manifest.json"),"reuse_previously_verified_weights":True,"no_redownload_or_full_weight_rehash":True,"model_revision_manifest_sha256":sha(ROOT/"vlm_research/results/internvl3_svh_physical_d_pilot_3090/model_manifest.json")})
    print("CPU_FREEZE_DONE",flush=True)
if __name__=="__main__":main()
