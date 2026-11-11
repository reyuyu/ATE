"""CPU-only reporting; never imports a GPU model or modifies frozen run sources."""
import csv
import argparse
import gzip
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
import numpy as np

from vlm_research.internvl3_svhr_late9_sevenbench.common import *
from vlm_research.internvl3_svh_seven_benchmarks import statistics as stats
from vlm_research.internvl3_svh_seven_benchmarks import scoring

PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def records():
    result = {}
    scoring.RUN = OUT
    frozen = {(b, r["question_id"]): r for b in BENCHES for r in manifest(b)}
    processor = read(OUT / "processor_bindings.json")
    for i in range(2):
        path = OUT / f"server_only/worker{i}.sqlite"
        if not path.exists():
            continue
        db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        for b, q, m, h, raw in db.execute("select benchmark,qid,method,checksum,record from predictions"):
            assert hashlib.sha256(raw.encode()).hexdigest() == h
            assert (b, q, m) not in result
            r = json.loads(raw)
            original = frozen[b, q]
            assert r["sample_sha256"] == original["sample_sha256"]
            assert r["prompt_sha256"] == original["prompt_sha256"]
            assert r["gold"] == original["gold"] and r["image_sha256"] == original["image_sha256"]
            assert r["input_tokens"] == len(original["input_ids"]) and r["max_new_tokens"] == original["max_new_tokens"]
            assert r["input_binding_details"]["pixels_bf16_sha256"] == processor[r["image_sha256"]]["pixels_bf16_sha256"]
            assert bind(r["input_binding_details"]) == r["input_binding"]
            parsed, score = scoring.score(b, r["text"], original)
            assert abs(score - r["score"]) < 1e-10 and abs(score - scoring.independent_score(b, r["text"], original)) < 1e-10
            assert parsed == r["parsed_answer"]
            assert r["protocol_sha256"] == sha(OUT / "protocol.json")
            assert r["source_manifest_sha256"] == sha(OUT / "source_manifest.json")
            assert r["generated_tokens"] == len(r["token_ids"]) <= r["max_new_tokens"]
            assert r["eos"] == (r["token_ids"][-1] == 151645)
            r["input_verification_gpu_uuid"] = r["gpu_uuid"]
            r["generation_gpu_uuid"] = r.get("historical_gpu_uuid") or r["gpu_uuid"]
            result[b, q, m] = r
        db.close()
    return result


def paired_rows(rs, lookup, a, b):
    return [dict(r, native_score=lookup[r["benchmark"], r["question_id"], a]["score"],
                 physical_score=lookup[r["benchmark"], r["question_id"], b]["score"],
                 native_parsed=lookup[r["benchmark"], r["question_id"], a]["parsed_answer"],
                 physical_parsed=lookup[r["benchmark"], r["question_id"], b]["parsed_answer"])
            for r in rs]


def metric(bench, rows, label="main"):
    if bench == "MME":
        tasks = PERCEPTION if label == "Perception" else COGNITION if label == "Cognition" else PERCEPTION + COGNITION
        return (*stats.mme(rows, tasks), "raw_MME_points")
    if bench == "MMBench":
        return (*stats.circular(rows)[0], "circular_accuracy_pp")
    if bench == "POPE" and label == "F1":
        return (*stats.pope_f1(rows), "F1_pp")
    return (*stats.linear(rows, 1000 if bench == "OCRBench" else 100), "raw_OCR_score_0_1000" if bench == "OCRBench" else "accuracy_pp")


def main(final=False):
    assert source_hashes() == read(OUT / "source_manifest.json")["runtime_source_sha256"]
    if final:
        assert read(OUT / "postrun_asset_integrity.json")["status"] == "PASS"
        assert read(OUT / "loaded_model_code_integrity.json")["status"] == "PASS"
    lookup = records()
    summaries, pairs, lengths, kvs = [], [], [], []
    gqa, text, mme, pope, mmbench, ocr, science = [], [], [], [], [], [], []
    output_rows, subgroup_pairs = [], []
    for bench in BENCHES:
        frozen = manifest(bench)
        idx = {r["question_id"]: r for r in frozen}
        complete = [r for r in frozen if all((bench, r["question_id"], m) in lookup for m in METHODS)]
        status = "FULL_COMPLETE" if len(complete) == len(frozen) else "PARTIAL" if complete else "BLOCKED"
        row = dict(benchmark=bench, status=status, planned=len(frozen), common_N=len(complete))
        counts = {m: sum((bench, r["question_id"], m) in lookup for r in frozen) for m in METHODS}
        row.update({m+"_N": n for m, n in counts.items()})
        if complete:
            for a, b in [("native", "physical_d"), ("native", "late9"), ("physical_d", "late9")]:
                data = paired_rows(complete, lookup, a, b)
                # A partial OCRBench subset is never expanded to a fictional /1000 score.
                if bench == "OCRBench" and status != "FULL_COMPLETE":
                    n, p, ci, clusters = stats.linear(data, 100)
                    units = "common_subset_accuracy_pp_NOT_FULL_OCRBENCH"
                else:
                    n, p, ci, clusters, units = metric(bench, data)
                harm = sum(r["native_score"] == 1 and r["physical_score"] == 0 for r in data)
                corrected = sum(r["native_score"] == 0 and r["physical_score"] == 1 for r in data)
                pairs.append(dict(benchmark=bench, status=status, base=a, candidate=b, N=len(data), image_clusters=clusters,
                                  base_score=n, candidate_score=p, difference=p-n, ci_low=ci[0], ci_high=ci[1], units=units,
                                  improved=sum(r["physical_score"] > r["native_score"] for r in data),
                                  declined=sum(r["physical_score"] < r["native_score"] for r in data),
                                  correct_to_wrong=harm if bench != "TextVQA" else None, wrong_to_correct=corrected if bench != "TextVQA" else None,
                                  zero_to_positive=sum(r["native_score"] == 0 and r["physical_score"] > 0 for r in data),
                                  binary_flip_counts_applicable=bench != "TextVQA", bootstrap_reps=20000, bootstrap_seed=20261110))
                if bench == "MMBench":
                    groups = defaultdict(list)
                    for x in data:
                        groups[x["group"]].append(x)
                    flips = [(all(x["native_score"] == 1 for x in rs), all(x["physical_score"] == 1 for x in rs)) for rs in groups.values()]
                    pairs[-1].update(circular_groups=len(flips), circular_correct_to_wrong=sum(n and not p for n, p in flips),
                                     circular_wrong_to_correct=sum(not n and p for n, p in flips))
                row.setdefault(a+"_common_score", n)
                row[b+"_common_score"] = p
                row[b+"_minus_"+a] = p-n
                row["units"] = units
                if bench == "POPE":
                    nf, pf, fc, _, _ = metric(bench, data, "F1")
                    pope.append(dict(strategy="micro_all", metric="F1", base=a, candidate=b, base_score=nf, candidate_score=pf, difference=pf-nf, ci_low=fc[0], ci_high=fc[1]))
                    nf, pf, fc, clusters = stats.pope_macro_f1(data)
                    pope.append(dict(strategy="macro_three_strategies", metric="F1", base=a, candidate=b, base_score=nf, candidate_score=pf, difference=pf-nf, ci_low=fc[0], ci_high=fc[1], image_clusters=clusters))
                if bench == "MME":
                    for label in ["Perception", "Cognition"]:
                        ns, ps, cc, clusters, u = metric(bench, data, label)
                        categories = PERCEPTION if label == "Perception" else COGNITION
                        component_n = sum(r["category"] in categories for r in complete)
                        row[label+"_N"] = component_n
                        row[label+"_planned"] = sum(r["category"] in categories for r in frozen)
                        subgroup_pairs.append(dict(benchmark=bench, grouping="score_component", category=label,
                            status=status, base=a, candidate=b, N=component_n, base_score=ns, candidate_score=ps,
                            difference=ps-ns, ci_low=cc[0], ci_high=cc[1], image_clusters=clusters, units=u))
                        row[a+"_"+label] = ns
                        row[b+"_"+label] = ps
            if status == "FULL_COMPLETE":
                row.update({m+"_full_score": row[m+"_common_score"] for m in METHODS})
            if bench == "GQA":
                for kind in ["query_split", "structural", "semantic", "detailed"]:
                    groups = defaultdict(list)
                    for r in complete:
                        value = ("Query" if r["types"]["structural"] == "query" else "Non-Query") if kind == "query_split" else r["types"][kind]
                        groups[value].append(r)
                    for name, rs in groups.items():
                        for m in METHODS:
                            gqa.append(dict(grouping=kind, category=name, method=m, N=len(rs), accuracy=100*np.mean([lookup[bench, r["question_id"], m]["score"] for r in rs])))
                        if kind == "query_split":
                            for a, b in [("native", "physical_d"), ("native", "late9"), ("physical_d", "late9")]:
                                data = paired_rows(rs, lookup, a, b)
                                ns, ps, cc, clusters = stats.linear(data)
                                subgroup_pairs.append(dict(benchmark=bench, grouping=kind, category=name, status=status,
                                    base=a, candidate=b, N=len(rs), base_score=ns, candidate_score=ps,
                                    difference=ps-ns, ci_low=cc[0], ci_high=cc[1], image_clusters=clusters,
                                    correct_to_wrong=sum(x["native_score"] == 1 and x["physical_score"] == 0 for x in data),
                                    wrong_to_correct=sum(x["native_score"] == 0 and x["physical_score"] == 1 for x in data), units="accuracy_pp"))
            if bench == "TextVQA":
                from vlm_research.internvl3_svh_r_terminal_windows_textvqa.common import subtype, anomaly
                for r in complete:
                    n, s, l = [lookup[bench, r["question_id"], m] for m in METHODS]
                    if l["score"] != s["score"] or l["text"] != s["text"]:
                        text.append(dict(question_id=r["question_id"], question=r["question"], gold=r["gold"], native=n["text"], svh=s["text"], late9=l["text"], native_score=n["score"], svh_score=s["score"], late9_score=l["score"], soft_gain=l["score"]-s["score"], zero_to_positive=s["score"] == 0 and l["score"] > 0,
                                         descriptive_type=subtype(r["question"]), svh_flags=anomaly(s), late9_flags=anomaly(l), type_definition="Historical regex; descriptive, not manual OCR annotation"))
            if bench == "ScienceQA-IMG":
                annotation = Path("/data/lk/ATE-resources/benchmarks/phase4_0/sources/scienceqa/data/scienceqa/problems.json")
                if annotation.exists():
                    problems = read(annotation)
                    groups = defaultdict(list)
                    for r in complete:
                        a = problems[r["question_id"]]
                        assert a["question"] == r["question"] and a["answer"] == r["gold"] and a["choices"] == r["choices"] and a["hint"] == r.get("hint", "")
                        for key in ["subject", "task", "category"]:
                            groups[key, a[key]].append(r)
                    for (kind, category), rs in groups.items():
                        for m in METHODS:
                            science.append(dict(grouping=kind, category=category, N=len(rs), method=m, accuracy=100*np.mean([lookup[bench, r["question_id"], m]["score"] for r in rs]), annotation_sha256=sha(annotation)))
            if bench == "POPE":
                for name in ["random", "popular", "adversarial"]:
                    rs = [r for r in complete if r["category"] == name]
                    if not rs:
                        continue
                    for m in METHODS:
                        vals = [lookup[bench, r["question_id"], m] for r in rs]
                        tp = sum(r["gold"] == "yes" and x["parsed_answer"] == "yes" for r, x in zip(rs, vals))
                        fp = sum(r["gold"] == "no" and x["parsed_answer"] == "yes" for r, x in zip(rs, vals))
                        fn = sum(r["gold"] == "yes" and x["parsed_answer"] != "yes" for r, x in zip(rs, vals))
                        pope.append(dict(strategy=name, method=m, N=len(rs), accuracy=100*np.mean([x["score"] for x in vals]), F1=200*tp/(2*tp+fp+fn)))
            if bench == "MME":
                groups = defaultdict(list)
                for r in complete:
                    groups[r["category"], r["group"]].append(r)
                assert all(len(rs) == 2 for rs in groups.values())
                for category in PERCEPTION + COGNITION:
                    rs = [rs for (c, g), rs in groups.items() if c == category]
                    for m in METHODS:
                        values = [[lookup[bench, r["question_id"], m]["score"] for r in pair] for pair in rs]
                        acc = 50*np.mean([sum(v) for v in values])
                        acc_plus = 100*np.mean([all(v) for v in values])
                        mme.append(dict(category=category, method=m, status=status, paired_images=len(rs), acc=acc, acc_plus=acc_plus, total=acc+acc_plus))
            if bench == "MMBench":
                groups = defaultdict(list)
                for r in complete:
                    groups[r["group"]].append(r)
                for group, rs in groups.items():
                    for m in METHODS:
                        mmbench.append(dict(group=group, method=m, rows=len(rs), circular_correct=all(lookup[bench, r["question_id"], m]["score"] == 1 for r in rs)))
                if status == "FULL_COMPLETE":
                    assert len(groups) == 1176
            if bench == "OCRBench":
                for m in METHODS:
                    batch = [dict(r, text=lookup[bench, r["question_id"], m]["text"]) for r in complete]
                    official = scoring.official_ocr_batch(batch)
                    assert official == [lookup[bench, r["question_id"], m]["score"] for r in complete]
                for r in complete:
                    for m in METHODS:
                        ocr.append(dict(question_id=r["question_id"], category=r["category"], dataset_name=r["dataset_name"], method=m, correct=lookup[bench, r["question_id"], m]["score"], status=status))
        summaries.append(row)
        for m in METHODS:
            xs = [x for (b, q, mm), x in lookup.items() if b == bench and mm == m]
            if xs:
                ll = [x["generated_tokens"] for x in xs]
                lengths.append(dict(benchmark=bench, method=m, N=len(xs), mean=float(np.mean(ll)), P50=float(np.median(ll)), P95=float(np.quantile(ll,.95)), maximum=max(ll), EOS=sum(x["eos"] for x in xs), truncated=sum(x["truncated"] for x in xs), empty=sum(x["empty"] for x in xs),
                    repetition=sum(x["flags"]["repetition"] for x in xs), explanation=sum(x["flags"]["explanation"] for x in xs), parse_failure=sum(x["flags"]["parse_failure"] for x in xs), origins=dict(Counter(x["generation_origin"] for x in xs))))
            for x in xs:
                assert len(x["kv_layout"]) == 28
                for k in x["kv_layout"]:
                    expected = 256 if m == "native" or m == "late9" and k["layer"] in KEEP else 0
                    assert k["visual_kv"] == expected
                    logical = x["input_tokens"] + x["generated_tokens"] - 1
                    physical = 0 if m != "native" and k["layer"] in [2, 3, 4, 5] else logical - (256 if not expected else 0)
                    assert k["logical_length"] == logical and k["physical_length"] == physical
                    kvs.append(dict(benchmark=bench, question_id=x["question_id"], method=m, **{k0:v for k0,v in k.items() if k0 != "method"}))
                output_rows.append(dict(x, question=idx[x["question_id"]]["question"]))
    for name, data in [("accuracy_summary.csv", summaries), ("paired_comparisons.csv", pairs), ("generation_length_audit.csv", lengths),
                       ("gqa_query_breakdown.csv", gqa), ("textvqa_recovery_cases.csv", text), ("mme_task_scores.csv", mme),
                       ("pope_strategy_scores.csv", pope), ("mmbench_circular_scores.csv", mmbench), ("ocrbench_scores.csv", ocr)]:
        csvout(OUT / name, data)
    csvout(OUT / "scienceqa_breakdown.csv", science)
    csvout(OUT / "subgroup_paired_comparisons.csv", subgroup_pairs)
    text_groups = defaultdict(list)
    for r in text:
        text_groups[r["descriptive_type"]].append(r)
    csvout(OUT / "textvqa_recovery_summary.csv", [dict(category=k, changed_text_or_score_N=len(rs),
        soft_score_improved=sum(r["soft_gain"] > 0 for r in rs), soft_score_declined=sum(r["soft_gain"] < 0 for r in rs),
        zero_to_positive=sum(r["zero_to_positive"] for r in rs), total_soft_gain=sum(r["soft_gain"] for r in rs),
        denominator_note="Only changed output or score cases; see per_question.csv.gz for all 5000") for k, rs in sorted(text_groups.items())])
    from vlm_research.internvl3_svh_r_terminal_windows_textvqa.common import anomaly, subtype
    text_complete = [r for r in manifest("TextVQA") if all(("TextVQA", r["question_id"], m) in lookup for m in METHODS)]
    text_types = []
    for category in sorted({subtype(r["question"]) for r in text_complete}):
        rs = [r for r in text_complete if subtype(r["question"]) == category]
        for m in METHODS:
            xs = [lookup["TextVQA", r["question_id"], m] for r in rs]
            flags = [anomaly(x) for x in xs]
            text_types.append(dict(category=category, method=m, N=len(xs), soft_accuracy_pct=100*np.mean([x["score"] for x in xs]),
                meta=sum(x["meta"] for x in flags), answering_prefix=sum(x["answering_prefix"] for x in flags),
                truncated=sum(x["truncated"] for x in flags), empty=sum(x["empty"] for x in flags),
                type_definition="Historical frozen question regex; descriptive, not annotated OCR difficulty"))
    csvout(OUT / "textvqa_type_summary.csv", text_types)
    for name, data in [("per_question.csv.gz", output_rows), ("kv_layout.csv.gz", kvs)]:
        fields = list(dict.fromkeys(k for r in data for k in r))
        with gzip.open(OUT / name, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in data:
                writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    grouped_kv = defaultdict(list)
    for r in kvs:
        grouped_kv[r["method"], r["layer"]].append(r)
    layout_summary = []
    for (method, layer), rs in sorted(grouped_kv.items()):
        layout_summary.append(dict(method=method, layer=layer, requests=len(rs), visual_kv_rows=rs[0]["visual_kv"],
             mean_live_bytes=float(np.mean([r["live_bytes"] for r in rs])), mean_capacity_bytes=float(np.mean([r["capacity_bytes"] for r in rs])),
             timing="natural_EOS_end_NOT_fixed32", raw_per_question_layout="kv_layout.csv.gz"))
    csvout(OUT / "kv_layout.csv", layout_summary)
    all_quality = all(r["status"] == "FULL_COMPLETE" for r in summaries)
    if not all_quality:
        csvout(OUT / "efficiency_summary.csv", [dict(method=m, status="NOT_RUN_QUALITY_INCOMPLETE", fixed_tokens=32,
                    E2E_ms=None, TTFT_ms=None, TPOT_ms=None, reason="Seven full quality benchmarks are required before the separate efficiency stage") for m in METHODS])
    smoke = [read(OUT / f"worker{i}_smoke_validation.json") for i in range(2)]
    write(OUT / "physical_execution_audit.json", dict(status="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER", smoke_checks=sum(len(x["checks"]) for x in smoke),
          keep=KEEP, cutoff=28, persistent_visual_layers=9, early_exit_visual_block_savings=False,
          full_prefill_visual_layers=28, original_svh_visual_layers=20, original_svh_persistent_visual_kv_layers=0,
          transient_zero_scratch="Original-position masked reduction padding in the frozen physical engine; not persistent original visual KV."))
    visual_payload = []
    for m in METHODS:
        rows = [x for x in output_rows if x["method"] == m]
        if rows:
            values = set()
            for r in rows:
                payload = sum(2*k["k_shape"][0]*k["k_shape"][1]*k["visual_kv"]*k["k_shape"][3]*2
                              for k in r["kv_layout"] if k["visual_kv"])
                values.add(payload)
            assert len(values) == 1
            visual_payload.append(dict(method=m, requests=len(rows), persistent_visual_payload_bytes=values.pop(),
                measurement="Derived from every actual K/V tensor shape, BF16 element bytes and mapped original visual rows; not allocator peak",
                visual_hidden_blocks=28 if m in ["native", "late9"] else 20))
    write(OUT / "visual_kv_payload_audit.json", visual_payload)
    write(OUT / "validation.json", dict(status="FULL_COMPLETE" if all(r["status"] == "FULL_COMPLETE" for r in summaries) else "PARTIAL",
          benchmarks=summaries, checkpoint_checksums_valid=True, duplicate_rows=0, runtime_source_hashes_unchanged=source_hashes() == read(OUT / "source_manifest.json")["runtime_source_sha256"],
          official_and_independent_rescoring_every_prediction=True, all_actual_KV_layouts_valid=True,
          numerical_status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER", complete_quality_required_before_efficiency=True))
    source = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    write(OUT / "analysis_source_manifest.json", source)
    report(summaries, pairs, lengths, subgroup_pairs)
    plot(summaries)
    if final:
        seal(lookup, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def seal(lookup, summaries):
    """Require finished workers, audit complete available coverage, then hash artifacts."""
    import datetime
    import time
    protocol = read(OUT / "protocol.json")
    local = read(OUT / "local_image_map.json")
    missing, counts = [], Counter()
    for b in BENCHES:
        for r in manifest(b):
            present = [(b, r["question_id"], m) in lookup for m in METHODS]
            if r["image_sha256"] in local:
                assert all(present), ("AVAILABLE_REQUEST_NOT_FINISHED", b, r["question_id"])
            else:
                assert not any(present), ("UNAVAILABLE_IMAGE_REPLACED", b, r["question_id"])
                blocked = OUT / "blocked" / (b+"__"+r["question_id"].replace("/", "_")+".json")
                assert read(blocked)["expected_sha256"] == r["image_sha256"]
                missing.append(dict(benchmark=b, question_id=r["question_id"], image_sha256=r["image_sha256"]))
    for r in lookup.values():
        counts[r["generation_origin"]] += 1
    releases = []
    for i in range(2):
        assert not (OUT / f"worker{i}_formal_failure.json").exists()
        release = read(OUT / f"worker{i}_formal_release.json")
        assert release["gpu_uuid"] == protocol["gpu_uuids"][i]
        assert release["model_references_released"]
        try:
            os.kill(release["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(("WORKER_STILL_RUNNING", i, release["pid"]))
        releases.append(release)
    process_list = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    assert all(str(r["pid"]) not in [p.strip() for row in process_list.splitlines() for p in row.split(",")[1:2]] for r in releases)
    write(OUT / "gpu_release_audit.json", dict(status="OWN_WORKERS_RELEASED", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        releases=releases, current_compute_processes=process_list, external_processes_not_terminated=True,
        gpu_state=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True)))
    process_timing = read(OUT / "formal_process_timing.json")
    for r in process_timing:
        end = datetime.datetime.fromtimestamp((OUT / f'worker{r["worker"]}_formal_release.json').stat().st_mtime, datetime.timezone.utc)
        r.update(released_utc=end.isoformat(), process_wall_seconds=(end-datetime.datetime.fromisoformat(r["started_utc_estimate"])).total_seconds())
    write(OUT / "formal_duration_audit.json", process_timing)
    with (OUT / "report_zh.md").open("a") as f:
        f.write("\n## 正式执行耗时与资源释放\n\n")
        for r in process_timing:
            f.write(f'- Worker{r["worker"]}正式进程墙钟约{r["process_wall_seconds"]/3600:.3f}小时（包含加载、输入检查和复用；不是方法速度测量）。\n')
        f.write("本实验两Worker已退出，nvidia-smi计算进程列表中均已消失；进程退出前的残留allocator字节原样保存在release.json，不冒充提前归零。GPU释放审计保留当时其他GPU进程，不终止或修改它们。\n")
    validation = read(OUT / "validation.json")
    validation.update(final_review=True, recorded_predictions=len(lookup), unique_completed_requests=len(lookup)//3,
                      explicitly_blocked_requests=len(missing), origins=dict(counts), GPUs_released=True)
    write(OUT / "validation.json", validation)
    csvout(OUT / "blocked_requests.csv", missing)
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()
        and not p.is_symlink() and not {"server_only", "references", "__pycache__"}.intersection(p.relative_to(OUT).parts)
        and p.name not in {"artifact_hashes.json", "final_cpu_review.json", "report_final_validation.log"}
        and not p.name.endswith(".tmp")}
    # Active report stdout is excluded because its descriptor closes after this seal.
    write(OUT / "artifact_hashes.json", files)
    assert all(sha(OUT/p) == value for p, value in files.items())
    write(OUT / "final_cpu_review.json", dict(status="PASS_FOR_RECORDED_PREDICTIONS_WITH_DECLARED_DATA_BLOCKERS",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), predictions=len(lookup), missing_requests=len(missing),
        origins=dict(counts), official_and_independent_scoring_checked=True, original_record_checksums_checked=True,
        prompt_model_processor_protocol_bindings_checked=True, all_actual_KV_layouts_checked=True,
        no_silent_skip=True, source_hashes_unchanged=True, artifact_hashes_verified=len(files),
        artifact_manifest_sha256=sha(OUT / "artifact_hashes.json"), benchmark_status=summaries,
        excluded_active_stdout="report_final_validation.log; duplicates the structured summary printed after sealing"))


def report(summary, pairs, lengths, subgroup_pairs):
    lines = ["# InternVL3 SVH-R Late9 七项质量评测", "", "状态：" + ("FULL_COMPLETE" if all(r["status"] == "FULL_COMPLETE" for r in summary) else "PARTIAL"), "",
             "| Benchmark | 配对覆盖/计划 | Native | 原SVH | Late9 | SVH−Native | Late9−Native | Late9−SVH | 状态 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary:
        vals = [r.get(m+"_full_score") for m in METHODS]
        ds = [r.get("physical_d_minus_native"), r.get("late9_minus_native"), r.get("late9_minus_physical_d")]
        show = lambda v: "—" if v is None else f"{v:.4f}"
        if r["status"] != "FULL_COMPLETE":
            vals = [None]*3
            ds = [None]*3
        lines.append("| " + " | ".join([r["benchmark"], f'{r["common_N"]}/{r["planned"]}'] + [show(v) for v in vals+ds] + [r["status"]]) + " |")
    lines += ["", "MME为原始积分，OCRBench为原始0–1000分；其他为百分比。不同单位不混算。未完整项目的共同子集在另表及CSV中报告，不填充为全量Benchmark分数。", "",
              "| 方法 | 持久原始视觉KV层数 | 视觉Hidden存活Block数 | 固定32 Token E2E/TTFT/TPOT |",
              "|---|---:|---:|---|", "| 原SVH | 0（0 MiB视觉数据） | 20 | 尚未测量 |", "| Late9 | 9（4.5 MiB视觉数据） | 28 | 尚未测量 |", "",
              "视觉数据大小来自全部逐请求实际K/V形状：2×4 KV Heads×256行×128维×2字节，每保留层0.5 MiB。Native为28层/14 MiB，Late9较Native少9.5 MiB（67.86%），但较原SVH多4.5 MiB；这不是总KV、分配器峰值或实测速率。详见visual_kv_payload_audit.json。",
              "效率测试须在七项质量全部完成后执行。不能用自然EOS生成耗时或历史GPU测速替代本轮固定32 Token测试；Late9保留全部28层视觉Hidden，没有L20 Prefill早退收益。", "",
              "## 配对差异", "", "Image SHA256整簇配对Bootstrap 20,000次，seed=20261110，普通未校正95%CI。保留原生评分单位，TextVQA为软分变化而非二值纠错。", "",
              "| Benchmark | 比较 | N | 差值 | 95%CI | 得分提高/降低 |", "|---|---|---:|---:|---|---|"]
    for r in pairs:
        lines.append(f'| {r["benchmark"]} ({r["status"]}) | {r["candidate"]}−{r["base"]} | {r["N"]} | {r["difference"]:+.4f} | [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}] | {r["improved"]}/{r["declined"]} |')
    lines += ["", "## 未完整项目共同子集", "", "以下不是正式全量成绩；MME单位为原始积分，OCR子集为百分比而非0–1000总分。", "",
              "| 项目/共同子集 | 题数 | Native | 原SVH | Late9 |", "|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["status"] == "PARTIAL":
            labels = ["Perception", "Cognition"] if r["benchmark"] == "MME" else ["common_score"]
            for label in labels:
                values = [r.get(m+"_"+label) for m in METHODS]
                if all(v is not None for v in values):
                    n, planned = (r["common_N"], r["planned"]) if label == "common_score" else (r[label+"_N"], r[label+"_planned"])
                    lines.append("| " + " | ".join([r["benchmark"]+" "+label, f'{n}/{planned}'] + [f'{v:.4f}' for v in values]) + " |")
    lines += ["", "## 十项研究问题", "", "1. GQA Query与Non-Query分别比较；均值大幅恢复不等于逐题或完全恢复Native。"]
    for category in ["Query", "Non-Query"]:
        xs = [r for r in subgroup_pairs if r["benchmark"] == "GQA" and r["category"] == category and r["candidate"] == "late9"]
        if xs:
            lines.append(f'- GQA {category}（{xs[0]["N"]}题）：' + "；".join(
                f'Late9−{r["base"]}={r["difference"]:+.4f}pp，95%CI [{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}]'
                for r in xs) + "。CI跨0不等于等价；不能把均值恢复视为全部Query逐题恢复。")
    tv = next((r for r in summary if r["benchmark"] == "TextVQA" and r["status"] == "FULL_COMPLETE"), None)
    if tv:
        denominator = tv["native_full_score"] - tv["physical_d_full_score"]
        fraction = (tv["late9_full_score"]-tv["physical_d_full_score"])/denominator if abs(denominator) > 1e-8 else None
        p = next(r for r in pairs if r["benchmark"] == "TextVQA" and r["base"] == "physical_d" and r["candidate"] == "late9")
        lines.append(f'2. TextVQA全5000题：Native/原SVH/Late9={tv["native_full_score"]:.4f}/{tv["physical_d_full_score"]:.4f}/{tv["late9_full_score"]:.4f}。Late9−原SVH={p["difference"]:+.4f}pp，95%CI [{p["ci_low"]:+.4f},{p["ci_high"]:+.4f}]；软分提高/降低={p["improved"]}/{p["declined"]}题，0分变正分={p["zero_to_positive"]}题。' + (f'按总体软分差计算恢复原损失的{100*fraction:.2f}%，不是二值纠错率。' if fraction is not None else "原损失分母不稳定，不报恢复百分比。"))
    lines.append("3. ScienceQA、POPE、MME、MMBench：不能用净分掩盖逐题新增损害。")
    for r in pairs:
        if r["benchmark"] in ["ScienceQA-IMG", "POPE", "MME", "MMBench"] and r["base"] == "native" and r["candidate"] == "late9":
            lines.append(f'- {r["benchmark"]}（{r["status"]}）：差值{r["difference"]:+.4f}，95%CI [{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}]；题目正确变错/错误变对={r["correct_to_wrong"]}/{r["wrong_to_correct"]}。MMBench翻转为旋转记录层面，正式主分为Circular组；MME差值为原始积分。')
    oo = next(r for r in summary if r["benchmark"] == "OCRBench")
    lines.append(f'4. OCRBench仅{oo["common_N"]}/1000题三方严格配对（{oo["status"]}），不能给出正式0–1000总分，也不能判断全量OCR改善。')
    lines += ["5. Late9相对原SVH的完整配对差值与20,000次图像簇Bootstrap区间见上表；缺失项目只对应共同子集，不能外推全量。",
              "6. 9层回看持久视觉KV为4.5 MiB：较Native节省9.5 MiB，较原SVH增加4.5 MiB；总KV还含文本，按真实Token长度另见kv_layout.csv.gz。",
              "7. 相对原SVH，视觉Hidden多存活8个Decoder Block，即每图多2048个视觉Token×Block计算行；实际Prefill额外毫秒尚未独立测量，不能用行数代替时间。",
              "8. 七项质量未全部完成，因此按任务顺序不启动固定32 Token效率阶段。没有本轮可报告的E2E/TTFT/TPOT加速比，不能宣称Late9比Native快。",
              "9. GQA/TextVQA/OCRBench复用逐记录原始SHA验证通过的历史Native与SVH；ScienceQA/POPE/MME/MMBench本轮重新生成两基线。缺原图的题目不进入三方法共同结果，保留BLOCKED清单。",
              "10. 全部Late9质量仍带BF16历史数值豁免标签。Smoke严格一致只对已测输入成立，物理/Dense未获全域数值等价证明；原单tile、Image-only及暂停策略差异均须保留，不能冒充其他论文排行榜协议。", ""]
    lines += ["", "## 协议与复用", "",
              "原SVH输入协议来自6c294aae，续跑结果来自3297559f；Late9物理引擎来自1407275c。旧代码和历史结果未修改。当前入口仅适配本地路径、GPU UUID、分片、严格复用和检查点。",
              "GQA、TextVQA、OCRBench历史JSON含全部28层KV布局，重建后逐条匹配发布的原始记录SHA256；再核对本地输入与BF16 pixels后才复用。其他四项缺少可核验的原始逐记录SHA，因此Native/SVH重新配对生成。详见historical_reuse_audit.json与逐题generation_origin。逐题导出明确区分generation_gpu_uuid与input_verification_gpu_uuid；复用预测仍来自历史GPU，不冒充本轮生成。",
              "BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0，448×448单图块、256视觉Token。各Benchmark保持原Prompt与生成上限，TextVQA没有OCR辅助输入。",
              "数值标签继续为EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，不宣称数值等价。两GPU各七任务Native/No-op/官方Chat严格一致；Late9物理布局、权限、RoPE、Norm/MLP与Cache独立性通过。Smoke的Dense/Physical误差完整保留，不能推广成所有输入数值一致。", "",
              "## 覆盖与限制", "",
              "本机MME的一张landmark图与冻结原件SHA不同，涉及同图两题；OCRBench本机HF版本与历史官方ZIP版本的绝大多数图片字节不同。Google Drive原件下载连接超时。未换图、重编码或改SHA，缺失请求完整保留并标记BLOCKED。",
              "本轮续跑规则在GPU前冻结：非结构性长输出、空答、解释、重复、自然截断均保留计分；NaN/Inf/OOM、哈希、映射或KV结构错误停止Worker。该规则与旧v1暂停策略不同，不改变输入、评分或算法。",
              "Late9与原SVH的收益不能解释为整模型仅需九层视觉处理，也不代表已加速。Prompt其他位置仍有视觉信息，且固定1tile协议不是作者多tile排行榜复现。", "",
              "## 逐题与审计", "", "per_question.csv.gz保存完整输出、Token IDs、Gold、评分、异常与绑定；kv_layout.csv.gz保存真实逐层形状和Live/Capacity。所有预测均独立CPU重评分，并核验原始JSON校验和、样本、Processor及实际KV长度。原始SQLite留server_only、不上传大缓存或权重。",
              "依本轮任务要求仅本地提交独立分支，不修改main、不自动推送GitHub、不创建PR。", ""]
    (OUT / "report_zh.md").write_text("\n".join(lines))


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(9, 10))
    for ax, r in zip(axes.flat, summary):
        if r["status"] == "FULL_COMPLETE":
            values = [r[m+"_full_score"] for m in METHODS]
            ax.bar(["Native", "SVH", "Late9"], values, color=["#64766f", "#ce8051", "#257f97"])
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, max(values)*1.16)
        else:
            ax.text(.5, .5, f'PARTIAL\n{r["common_N"]}/{r["planned"]} paired requests\nNo full benchmark score', ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(r["benchmark"], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "figure_quality.pdf")
    fig.savefig(OUT / "figure_quality.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="Require released workers and all locally available requests before sealing")
    main(parser.parse_args().final)
