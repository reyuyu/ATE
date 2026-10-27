"""Run A/B and three consecutive C trials with one model instance."""
import argparse
import json

from vlm_common import ROOT, generate_once, inspect_structure, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "test_images/example.jpg"))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    model, processor, loading = load_model(args.model)
    report = {"loading": loading, "structure": inspect_structure(model, processor, args.image, "What is in this image?"), "tests": {}}
    write_json(ROOT / "results/model_structure.json", {"loading": loading, "structure": report["structure"]})
    (ROOT / "results/model_modules.txt").write_text(str(model) + "\n")

    def run(name, question):
        result = generate_once(model, processor, args.image, question)
        report["tests"][name] = result
        write_json(ROOT / "results/validation.json", report)
        print(f"{name}: {result['answer']}\n  {result['generation_seconds']:.3f} s; peak {result['inference_memory']['peak_allocated_mib']:.1f} MiB; after cleanup {result['memory_after_cleanup']['allocated_mib']:.1f} MiB", flush=True)
        return result

    run("A_description", "What is in this image?")
    run("B_detail", "What color is the stop sign?")
    repeats = [run(f"C_repeat_{index + 1}", "What is in this image?") for index in range(3)]
    allocated = [r["memory_after_cleanup"]["allocated_mib"] for r in repeats]
    reserved = [r["memory_after_cleanup"]["reserved_mib"] for r in repeats]
    report["checks"] = {
        "all_answers_nonempty": all(r["answer"] for r in report["tests"].values()),
        "all_generation_scores_finite": all(r["scores_all_finite"] for r in report["tests"].values()),
        "repeat_token_ids_identical": all(r["new_token_ids"] == repeats[0]["new_token_ids"] for r in repeats),
        "repeat_allocated_mib": allocated,
        "repeat_reserved_mib": reserved,
        "repeat_allocated_stable_within_1_mib": max(allocated) - min(allocated) <= 1.0,
        "repeat_reserved_plateau_after_first": abs(reserved[-1] - reserved[-2]) <= 1.0,
        "repeat_scope": "Three calls in the same process; no empty_cache between calls. This is a short stability check, not a long-run leak proof.",
    }
    check_keys = [key for key in report["checks"] if key not in ("repeat_allocated_mib", "repeat_reserved_mib", "repeat_scope")]
    report["automated_checks_passed"] = all(report["checks"][key] for key in check_keys)
    write_json(ROOT / "results/validation.json", report)
    print(json.dumps(report["checks"], indent=2), flush=True)
    if not report["automated_checks_passed"]:
        raise RuntimeError("Validation failed; inspect results/validation.json")


if __name__ == "__main__":
    main()
