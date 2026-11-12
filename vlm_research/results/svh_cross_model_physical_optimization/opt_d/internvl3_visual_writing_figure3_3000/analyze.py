"""Independent CPU score verification, image-cluster bootstrap and figure data."""
import csv,json,hashlib,os
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from .engine import TASKS,CONDITIONS,digest
from .prepare import ROOT,OUT,MODEL,sha,save

def write_csv(name,rows):
    rows=list(rows);assert rows,name
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def read_csv(path):return list(csv.DictReader(Path(path).open()))

def bootstrap(values,images,seed):
    names,inverse=np.unique(images,return_inverse=True);k=len(names)
    counts=np.bincount(inverse).astype(np.float64)
    sums=np.zeros((k,values.shape[1]),dtype=np.float64);np.add.at(sums,inverse,values)
    rng=np.random.default_rng(seed);draws=np.empty((20000,values.shape[1]),dtype=np.float64)
    probs=np.full(k,1/k)
    for start in range(0,20000,128):
        n=min(128,20000-start);weight=rng.multinomial(k,probs,size=n)
        draws[start:start+n]=(weight@sums)/(weight@counts)[:,None]
    assert np.isfinite(draws).all()
    return draws

def historical_plot_rows():
    refs=OUT/"references";layers=[];deep=[];sources=[]
    specs=[
      ("LLaVA-1.5-7B",32,20,"llava","6eb8328c9652342343395cf19bf7149e7d7f3047",
       "vlm_research/results/gqa3000_visual_writing_necessity/task_summary.csv",
       "vlm_research/results/gqa3000_visual_writing_necessity/condition_summary.csv"),
      ("Qwen3-VL-8B",36,22,"qwen","96a66a30bc81db9ffe5c3ed77716673a81aaaa01",
       "vlm_research/results/qwen3vl_visual_writing_necessity_3000/task_accuracy.csv",
       "vlm_research/results/qwen3vl_visual_writing_necessity_3000/accuracy_summary.csv")]
    for model,count,boundary,label,commit,task_file,acc_file in specs:
        tpath=refs/label/task_file;apath=refs/label/acc_file
        tasks=read_csv(tpath);overall=read_csv(apath)
        base=next(x for x in overall if x["condition"]=="All-Writer")
        for task in TASKS:
            rows=[x for x in tasks if x["subset"]==task]
            b=next(x for x in rows if x["condition"]=="All-Writer")
            n=int(b["questions"]);bc=int(round(float(b["accuracy_pct"])*n/100))
            assert n==500
            for i in range(count):
                r=next(x for x in rows if x["condition"]==f"Drop-L{i}")
                ctw=int(r["correct_to_wrong"]);wtc=int(r["wrong_to_correct"])
                delta=float(b["accuracy_pct"])-float(r["accuracy_pct"])
                assert abs(delta-100*(ctw-wtc)/n)<1e-10
                layers.append(dict(model=model,task=task,layer=i,layers=count,deep_start=boundary,
                    accuracy_pct=float(r["accuracy_pct"]),all_writer_accuracy_pct=float(b["accuracy_pct"]),
                    delta_accuracy_pp=delta,all_writer_correct=bc,correct_to_wrong=ctw,wrong_to_correct=wtc,
                    damage_percent=100*ctw/bc,source_commit=commit,source_csv=str(tpath.relative_to(OUT)),
                    source_sha256=sha(tpath),historical=True))
        sources.append(dict(model=model,path=str(tpath.relative_to(OUT)),sha256=sha(tpath),commit=commit))
        if model.startswith("LLaVA"):
            root=refs/"llava_joint/vlm_research/results/gqa3000_visual_writing_16_vs_20"
            rows=read_csv(root/"accuracy_summary.csv")
            b=next(x for x in rows if x["analysis_set"]=="overall_3000" and x["condition"]=="All-Writer")
            c=next(x for x in rows if x["analysis_set"]=="overall_3000" and x["condition"]=="Keep-0-19")
            pairs=read_csv(root/"paired_comparisons.csv")
            pair=next(x for x in pairs if x["analysis_set"]=="overall_3000" and x["candidate"]=="Keep-0-19" and x["reference"]=="All-Writer")
            low=-float(pair["ci95_high"]);high=-float(pair["ci95_low"])
            commit2="b13a21b07e84a0095731d10dc1e8854f514a7877"
        else:
            root=refs/"joint/vlm_research/results/qwen3vl_keep_l0_21_gqa3000"
            rows=read_csv(root/"accuracy_summary.csv");b=next(x for x in rows if x["condition"]=="All-Writer")
            c=next(x for x in rows if x["condition"]=="Keep-L0-L21")
            pair=read_csv(root/"paired_comparison.csv")[0]
            low=-float(pair["ci_95_two_sided_high"]);high=-float(pair["ci_95_two_sided_low"])
            commit2="dfe0cbacd09de9dfeacff980b911e6aec22a7665"
        assert abs(float(b["accuracy_pct"])-float(base["accuracy_pct"]))<1e-10
        deep.append(dict(model=model,layers=count,deep_start=boundary,
            all_writer_accuracy_pct=float(b["accuracy_pct"]),restricted_accuracy_pct=float(c["accuracy_pct"]),
            delta_accuracy_pp=float(b["accuracy_pct"])-float(c["accuracy_pct"]),ci95_low=low,ci95_high=high,
            source_commit=commit2,source_csv=str((root/"accuracy_summary.csv").relative_to(OUT)),
            source_sha256=sha(root/"accuracy_summary.csv"),historical=True))
        sources.append(dict(model=model,path=str((root/"accuracy_summary.csv").relative_to(OUT)),
                            sha256=sha(root/"accuracy_summary.csv"),commit=commit2))
    return layers,deep,sources

def main():
    complete=json.loads((OUT/"generation_complete.json").read_text())
    assert complete["status"]=="COMPLETE" and complete["requests"]==93000
    protocol=json.loads((OUT/"protocol.json").read_text());ph=sha(OUT/"protocol.json");mh=sha(OUT/"sample_manifest.json")
    assert complete["protocol_sha256"]==ph
    assert json.loads((OUT/"smoke_summary.json").read_text())["passed"]
    samples=json.loads((OUT/"sample_manifest.json").read_text())["samples"]
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
    # Executing this exact frozen function is independent of generation's inline scorer.
    scorer_file=OUT/"references/qwen/vlm_research/results/qwen3vl_visual_writing_necessity_3000/official_gqa_prediction.json"
    source=json.loads(scorer_file.read_text())["function_source"];scope={};exec(source,scope)
    scorer=scope["gqa_prediction"]
    assert sha(scorer_file)==protocol["scoring_reference_sha256"]
    values=np.empty((3000,31),dtype=np.int8);metadata=[];per_question=[]
    checkpoint_hashes={};tokens_total=0;negative_exact=0
    for index,s in enumerate(samples):
        key=s["task"]+"__"+s["question_id"];path=OUT/"checkpoints"/(key+".json")
        record=json.loads(path.read_text());checkpoint_hashes[path.name]=sha(path)
        assert record["complete"] and record["sample_sha256"]==digest(s) and record["protocol_sha256"]==ph
        assert record["manifest_sha256"]==mh
        assert record["GPU_UUID"]==protocol["gpu_uuid"] and record["index"]==index
        assert record["input_binding"]==digest(record["input"])
        assert set(record["conditions"])==set(CONDITIONS)
        base=record["conditions"]["All-Writer"]
        last=record["conditions"]["Drop-L27"]
        assert base["token_ids"]==last["token_ids"] and base["all_step_logits_sha256"]==last["all_step_logits_sha256"]
        negative_exact+=1
        for cidx,c in enumerate(CONDITIONS):
            g=record["conditions"][c];ids=g["token_ids"];n=len(record["input"]["input_ids"])
            text=tokenizer.decode(ids,skip_special_tokens=True).split("<|im_end|>")[0].strip()
            pred=scorer(text);correct=int(pred==s["answer"].lower())
            assert text==g["text"] and pred==g["prediction"] and correct==g["correct"]
            assert 1<=len(ids)<=32 and len(ids)==len(g["all_step_logits_sha256"])
            assert g["final_cache_lengths"]==[n+len(ids)-1]*28 and g["independent_fresh_cache"]
            assert g["mask_audit"]["blocked_probability_max"]==0 and g["mask_audit"]["steps"]==len(ids)
            assert g["P1_Decode_image_off"] and g["logits_to_keep"]==1
            assert g["correct_to_wrong"]==int(base["correct"] and not correct)
            assert g["wrong_to_correct"]==int(not base["correct"] and correct)
            assert g["first_token_flip"]==int(base["token_ids"][0]!=ids[0])
            values[index,cidx]=correct
            tokens_total+=len(ids)
            per_question.append(dict(sample_index=index,task=s["task"],question_id=s["question_id"],
                image_id=s["image_id"],image_sha256=s["image_sha256"],condition=c,gold=s["answer"],
                text=text,prediction=pred,correct=correct,token_ids=json.dumps(ids),generated_tokens=len(ids),
                correct_to_wrong=g["correct_to_wrong"],wrong_to_correct=g["wrong_to_correct"],
                first_token_flip=g["first_token_flip"],eos=g["eos"],truncated=g["truncated"],empty=g["empty"],
                repeated_trigram=g["repeated_trigram"],input_binding=record["input_binding"],
                input_checkpoint=str(path.relative_to(OUT)),all_step_logits_sha256=json.dumps(g["all_step_logits_sha256"]),
                GPU_UUID=record["GPU_UUID"],protocol_sha256=ph))
        metadata.append(record)
    assert len(per_question)==93000 and negative_exact==3000
    write_csv("per_question.csv",per_question)
    accuracy=[];paired=[];task_rows=[];layer_rows=[];damage_rows=[];deep_rows=[]
    image=np.array([s["image_sha256"] for s in samples])
    task_array=np.array([s["task"] for s in samples])
    for subset_no,subset in enumerate(["Overall"]+TASKS):
        take=np.arange(3000) if subset=="Overall" else np.flatnonzero(task_array==subset)
        v=values[take];n=len(take);point=v.mean(0)*100
        draws=bootstrap(v,image[take],20261110+subset_no)
        delta=(draws[:,0,None]-draws)*100
        for cidx,c in enumerate(CONDITIONS):
            acc_ci=np.quantile(draws[:,cidx]*100,[.025,.975])
            ci=np.quantile(delta[:,cidx],[.025,.975])
            bf=np.quantile(delta[:,cidx],[.05/(2*28),1-.05/(2*28)]) if c.startswith("Drop-") else [None,None]
            gs=[metadata[i]["conditions"][c] for i in take]
            ctw=int(((v[:,0]==1)&(v[:,cidx]==0)).sum());wtc=int(((v[:,0]==0)&(v[:,cidx]==1)).sum())
            row=dict(subset=subset,condition=c,questions=n,image_clusters=len(set(image[take])),
                correct=int(v[:,cidx].sum()),accuracy_pct=float(point[cidx]),
                accuracy_ci95_low=float(acc_ci[0]),accuracy_ci95_high=float(acc_ci[1]),
                all_writer_accuracy_pct=float(point[0]),delta_accuracy_pp=float(point[0]-point[cidx]),
                delta_ci95_low=float(ci[0]),delta_ci95_high=float(ci[1]),
                Bonferroni28_low=bf[0],Bonferroni28_high=bf[1],
                correct_to_wrong=ctw,wrong_to_correct=wtc,
                first_token_flips=sum(g["first_token_flip"] for g in gs),eos=sum(g["eos"] for g in gs),
                truncated=sum(g["truncated"] for g in gs),empty=sum(g["empty"] for g in gs),
                repeated=sum(g["repeated_trigram"] for g in gs),mean_generated_tokens=float(np.mean([g["generated_tokens"] for g in gs])))
            if subset=="Overall":accuracy.append(row)
            else:task_rows.append(row)
            if c!="All-Writer":paired.append(row.copy())
            if c.startswith("Drop-"):
                layer_rows.append(dict(layer=int(c[6:]),**row))
                if subset!="Overall":
                    bc=int(v[:,0].sum());assert bc>0
                    damage_rows.append(dict(model="InternVL3-8B",task=subset,layer=int(c[6:]),layers=28,deep_start=17,
                        all_writer_correct=bc,correct_to_wrong=ctw,wrong_to_correct=wtc,
                        damage_percent=100*ctw/bc,delta_accuracy_pp=float(point[0]-point[cidx]),
                        accuracy_pct=float(point[cidx]),all_writer_accuracy_pct=float(point[0]),historical=False,
                        source_commit="",source_csv="task_accuracy.csv",source_sha256=""))
            if c=="Keep-L0-L16":
                deep_rows.append(dict(subset=subset,model="InternVL3-8B",layers=28,deep_start=17,
                    all_writer_accuracy_pct=float(point[0]),restricted_accuracy_pct=float(point[cidx]),
                    delta_accuracy_pp=float(point[0]-point[cidx]),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                    correct_to_wrong=ctw,wrong_to_correct=wtc,historical=False,source_csv="deep_restriction_comparison.csv"))
        print("BOOTSTRAP_DONE",subset,flush=True)
    write_csv("accuracy_summary.csv",accuracy);write_csv("task_accuracy.csv",task_rows)
    write_csv("paired_comparisons.csv",paired);write_csv("layerwise_accuracy.csv",layer_rows)
    write_csv("answer_damage_matrix.csv",damage_rows);write_csv("deep_restriction_comparison.csv",deep_rows)
    historical_layers,historical_deep,sources=historical_plot_rows()
    for r in damage_rows:r["source_sha256"]=sha(OUT/"task_accuracy.csv")
    write_csv("plot_layerwise.csv",historical_layers+damage_rows)
    write_csv("plot_deep.csv",historical_deep+[r for r in deep_rows if r["subset"]=="Overall"])
    save("historical_plot_sources.json",dict(sources=sources,
        historical_rows_from_fixed_CSV=True,no_historical_generation=True,
        delta_convention="All-Writer minus restriction; historical opposite-sign differences explicitly negated",
        answer_damage="100*correct_to_wrong / All-Writer-correct count per task; not net accuracy or first-token flips"))
    save("checkpoint_hashes.json",checkpoint_hashes)
    save("analysis_validation.json",dict(passed=True,requests=93000,questions=3000,image_clusters=2409,
        conditions=31,independent_frozen_scorer_matches=True,token_redecode_exact=True,
        Drop_L27_all_3000_tokens_and_logits_exact=True,mask_all_requests_probability_zero=True,
        all_28_real_cache_lengths_correct=True,same_GPU_UUID=True,bootstrap_repetitions=20000,
        ordinary_95_and_Bonferroni28=True,deep_preplanned_ordinary_95=True,total_generated_tokens=tokens_total,
        input_manifest_SHA_verified=True,no_omitted_or_resampled_questions=True))
    print("CPU_ANALYSIS_PASS",flush=True)

if __name__=="__main__":main()
