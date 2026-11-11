"""Two independent GPU workers; exact old inputs and unmodified physical engines."""
import argparse
import gc
import sqlite3
import subprocess
import sys
import time
import traceback
import torch

from .common import *
from vlm_research.internvl3_svh_seven_benchmarks import worker as original
from vlm_research.internvl3_svh_seven_benchmarks import scoring
from vlm_research.internvl3_svh_physical_d_pilot import engine as base
from vlm_research.internvl3_svh_r_terminal_windows_textvqa import engine as late


def configure(uuid):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == uuid
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == uuid
    assert "3090" in torch.cuda.get_device_name(0)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    others = [line for line in processes.splitlines() if uuid in line and int(line.split(",")[1]) != os.getpid()]
    assert not others, ("GPU_NOT_IDLE", others)
    original.MODEL = MODEL
    original.GPU = uuid
    # Only the reference root is adapted; pinned scorer files/functions stay unchanged.
    ref = OUT / "references"
    if not ref.exists():
        try:
            ref.symlink_to(HISTORY / "official_references", target_is_directory=True)
        except FileExistsError:
            assert ref.resolve() == (HISTORY / "official_references").resolve()
    scoring.RUN = OUT


class Inputs(original.Inputs):
    def __init__(self, model, tok):
        super().__init__(model, tok)
        self.local = read(OUT / "local_image_map.json")
        self.bindings = read(OUT / "processor_bindings.json")

    def ready(self, r):
        assert r["image_sha256"] in self.local, "FROZEN_IMAGE_UNAVAILABLE"
        p, ib, binding = super().ready(dict(r, image_path=self.local[r["image_sha256"]]))
        assert binding["pixels_bf16_sha256"] == self.bindings[r["image_sha256"]]["pixels_bf16_sha256"]
        return p, ib, binding


def generate(model, tok, p, method, cap, **kwargs):
    finite = torch.ones((), device=p["ids"].device, dtype=torch.bool)

    def check(_module, _args, output):
        nonlocal finite
        finite = finite & torch.isfinite(output.logits).all()

    hook = model.language_model.register_forward_hook(check)
    try:
        if method == "late9":
            result = late.generate(model, tok, p, "Late9_L19_27", KEEP, cutoff=28, max_tokens=cap, physical=True, **kwargs)
        else:
            result = base.generate(model, tok, p, {"native": "Native", "physical_d": "Physical-D", "noop": "No-op"}[method], max_tokens=cap, **kwargs)
        assert bool(finite), "NONFINITE_LOGITS"
        return result
    finally:
        hook.remove()


def comparison(a, b, ids_a, ids_b):
    rows = []
    for step, (x, y) in enumerate(zip(a, b)):
        if ids_a[:step] != ids_b[:step]:
            break
        d = (x.float() - y.float()).abs()
        rows.append(dict(step=step, shape_a=list(x.shape), shape_b=list(y.shape), dtype_a=str(x.dtype), dtype_b=str(y.dtype),
                         exact=torch.equal(x, y), unequal_elements=int((x != y).sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                         top1_a=int(x.argmax()), top1_b=int(y.argmax())))
    return rows


@torch.inference_mode()
def smoke(model, tok, inputs, worker):
    checks = []
    for bench in BENCHES:
        rows = [r for r in manifest(bench) if r["image_sha256"] in inputs.local]
        assert rows, ("NO_SMOKE_INPUT", bench)
        r = min(rows, key=lambda r: bind([20261110, "seven-smoke", bench, r["question_id"]]))
        p, ib, binding = inputs.ready(r)
        cap = r["max_new_tokens"]
        n, ns, nc = generate(model, tok, p, "native", cap, audit=True)
        z, zs, zc = generate(model, tok, p, "noop", cap, audit=True)
        eq = comparison(ns, zs, n["token_ids"], z["token_ids"])
        evidence = dict(benchmark=bench, question_id=r["question_id"], input_binding=ib, binding=binding, native_tokens=n["token_ids"],
                        noop_tokens=z["token_ids"], native_noop=eq)
        write(OUT / f"worker{worker}_smoke_partial.json", checks + [evidence])
        assert n["token_ids"] == z["token_ids"] and len(ns) == len(zs) and all(x["exact"] for x in eq), "NATIVE_NOOP_MISMATCH"
        with original.b.OfficialChatCapture(model) as capture:
            text = model.chat(tok, p["pixels"], p["request"], dict(do_sample=False, num_beams=1, max_new_tokens=cap,
                              min_new_tokens=0, logits_to_keep=1, pad_token_id=151645))
        official = comparison(ns, capture.scores, n["token_ids"], capture.ids)
        evidence.update(official_tokens=capture.ids, official_logits=official)
        write(OUT / f"worker{worker}_smoke_partial.json", checks + [evidence])
        assert capture.input_ids[0].tolist() == r["input_ids"]
        assert n["token_ids"] == capture.ids and n["text"] == text.strip() and len(ns) == len(capture.scores) and all(x["exact"] for x in official), "OFFICIAL_NATIVE_MISMATCH"
        del ns, zs, nc, zc, capture
        phys, ps, pc = generate(model, tok, p, "late9", cap, audit=True)
        dense, ds, dc = late.generate(model, tok, p, "Late9_L19_27", KEEP, cutoff=28, max_tokens=cap, physical=False, audit=True)
        evidence.update(physical_tokens=phys["token_ids"], dense_tokens=dense["token_ids"], physical_text=phys["text"], dense_text=dense["text"],
                        dense_physical_logits=comparison(ds, ps, dense["token_ids"], phys["token_ids"]),
                        physical_audit=phys["audit"], kv_layout=phys["kv_layout"])
        assert [x["layer"] for x in phys["kv_layout"] if x["visual_kv"]] == KEEP
        assert phys["audit"]["hidden_exit"] == [dict(before=r["input_tokens"], after=r["input_tokens"]-256, removed=256)]
        assert phys["audit"]["exit_stage"] == "after_L27_before_final_norm"
        assert phys["audit"]["R_image_permission_boundary"] == 20
        assert phys["audit"]["allowed_image_probability_samples"] and all(x[2] > 0 for x in phys["audit"]["allowed_image_probability_samples"])
        assert all(x[2] == 0 for x in phys["audit"]["blocked_image_probability_samples"] + phys["audit"]["R_image_probability_samples"])
        # A second independent request must reproduce both its logits and tokens.
        repeat, rs, rc = generate(model, tok, p, "late9", cap, audit=True)
        evidence["independent_repeat"] = comparison(ps, rs, phys["token_ids"], repeat["token_ids"])
        assert repeat["token_ids"] == phys["token_ids"] and len(rs) == len(ps) and all(x["exact"] for x in evidence["independent_repeat"])
        old, oldscores, oldc = generate(model, tok, p, "physical_d", cap, audit=True)
        assert not any(row["visual_kv"] for row in old["kv_layout"])
        evidence["original_physical_audit"] = old["audit"]
        evidence["status"] = "PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER"
        checks.append(evidence)
        write(OUT / f"worker{worker}_smoke_partial.json", checks)
        print("SMOKE_PASS", worker, bench, r["question_id"], flush=True)
        del p, n, z, phys, ps, pc, dense, ds, dc, repeat, rs, rc, old, oldscores, oldc
        gc.collect()
    result = dict(status="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER", checks=checks, native_noop_strict=True,
                  historical_numerical_status="NUMERICAL_EQUIVALENCE_FAILED", numerical_equivalence_claimed=False)
    write(OUT / f"worker{worker}_smoke_validation.json", result)


def database(worker, protocol_hash, source_hash):
    folder = OUT / "server_only"
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(folder / f"worker{worker}.sqlite")
    db.execute("pragma journal_mode=WAL")
    db.execute("pragma synchronous=FULL")
    db.execute("create table if not exists metadata(key text primary key, value text)")
    db.execute("create table if not exists predictions(benchmark text,qid text,method text,checksum text,record text,primary key(benchmark,qid,method))")
    for key, val in dict(protocol_sha256=protocol_hash, source_manifest_sha256=source_hash).items():
        old = db.execute("select value from metadata where key=?", (key,)).fetchone()
        if old:
            assert old[0] == val, "CHECKPOINT_PROTOCOL_OR_SOURCE_CHANGED"
        else:
            db.execute("insert into metadata values(?,?)", (key, val))
    db.commit()
    return db


def formal(model, tok, inputs, worker, uuid):
    ph, sh = sha(OUT / "protocol.json"), sha(OUT / "source_manifest.json")
    assignment = read(OUT / "shard_assignment.json")["assignments"]
    db = database(worker, ph, sh)
    history = sqlite3.connect("file:" + str(OUT / "server_only/verified_history.sqlite") + "?mode=ro&immutable=1", uri=True)
    try:
        for bench in BENCHES:
            rows = [r for r in manifest(bench) if assignment[bench+"/"+r["question_id"]] == worker]
            for index, r in enumerate(rows):
                if r["image_sha256"] not in inputs.local:
                    write(OUT / "blocked" / (bench+"__"+r["question_id"].replace("/", "_")+".json"),
                          dict(status="BLOCKED_IMAGE_SHA_UNAVAILABLE", benchmark=bench, question_id=r["question_id"], expected_sha256=r["image_sha256"]))
                    continue
                todo = []
                for method in METHODS:
                    old = db.execute("select checksum,record from predictions where benchmark=? and qid=? and method=?", (bench, r["question_id"], method)).fetchone()
                    if old:
                        assert hashlib.sha256(old[1].encode()).hexdigest() == old[0], "CHECKPOINT_CORRUPT"
                    else:
                        todo.append(method)
                if todo:
                    ready, ib, binding = inputs.ready(r)
                    for method in todo:
                        old = history.execute("select checksum,record from predictions where benchmark=? and qid=? and method=?", (bench, r["question_id"], method)).fetchone()
                        if old:
                            assert hashlib.sha256(old[1].encode()).hexdigest() == old[0]
                            result = json.loads(old[1])
                            assert result["input_binding"] == ib and result["sample_sha256"] == r["sample_sha256"]
                            result.update(historical_record_sha256=old[0], historical_protocol_sha256=result["protocol_sha256"], historical_source_manifest_sha256=result["source_manifest_sha256"])
                            origin = "REUSED_PUBLISHED_ORIGINAL_RAW_SHA256"
                            scores = control = None
                        else:
                            result, scores, control = generate(model, tok, ready, method, r["max_new_tokens"], audit=False)
                            origin = "NEW_PAIRED_ON_THIS_WORKER"
                        parsed, score = scoring.score(bench, result["text"], r)
                        assert abs(score-scoring.independent_score(bench, result["text"], r)) < 1e-10
                        flags = scoring.anomalies(bench, result, r)
                        rec = dict(result, **{k: r[k] for k in ["benchmark", "question_id", "image_sha256", "prompt_sha256", "sample_sha256", "gold", "group", "category", "max_new_tokens"]})
                        rec.update(method=method, input_binding=ib, input_binding_details=binding, score=score, parsed_answer=parsed,
                                   flags=flags, input_tokens=r["input_tokens"], gpu_uuid=uuid, protocol_sha256=ph, source_manifest_sha256=sh,
                                   dataset_name=r.get("dataset_name", ""), generation_origin=origin,
                                   historical_gpu_uuid="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e" if old else None)
                        raw = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        db.execute("insert into predictions values(?,?,?,?,?)", (bench, r["question_id"], method, hashlib.sha256(raw.encode()).hexdigest(), raw))
                        db.commit()
                        del result, scores, control
                    del ready
                if (index+1) % 25 == 0 or index+1 == len(rows):
                    progress = dict(worker=worker, benchmark=bench, visited=index+1, planned=len(rows), saved=db.execute("select count(*) from predictions where benchmark=?", (bench,)).fetchone()[0],
                                    gpu_uuid=uuid, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                    write(OUT / f"worker{worker}_progress.json", progress)
                    print("PROGRESS", json.dumps(progress), flush=True)
    finally:
        db.execute("pragma wal_checkpoint(TRUNCATE)")
        db.close()
        history.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=[0, 1], required=True)
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = read(OUT / "protocol.json")
    assert source_hashes() == read(OUT / "source_manifest.json")["runtime_source_sha256"]
    uuid = protocol["gpu_uuids"][args.worker]
    model = tok = inputs = None
    try:
        configure(uuid)
        model, tok = original.load_model()
        write(OUT / f"worker{args.worker}_environment.json", original.environment(model, tok))
        inputs = Inputs(model, tok)
        if args.stage == "smoke":
            smoke(model, tok, inputs, args.worker)
        else:
            for i in range(2):
                assert read(OUT / f"worker{i}_smoke_validation.json")["native_noop_strict"]
            formal(model, tok, inputs, args.worker, uuid)
    except BaseException as exc:
        write(OUT / f"worker{args.worker}_{args.stage}_failure.json", dict(status="FAILED", error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if inputs is not None:
            inputs.pixels.cache_clear()
        model = tok = inputs = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        write(OUT / f"worker{args.worker}_{args.stage}_release.json", dict(pid=os.getpid(), gpu_uuid=uuid, stage=args.stage, model_references_released=True,
              allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0, process_will_exit=True))


if __name__ == "__main__":
    main()
