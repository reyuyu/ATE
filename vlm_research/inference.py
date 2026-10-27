"""FP16 single-image, single-GPU LLaVA inference."""
import argparse

from vlm_common import ROOT, generate_once, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model", default=None, help="Local checkpoint directory; defaults to configs/model.json")
    parser.add_argument("--output", default=str(ROOT / "results/inference.json"))
    args = parser.parse_args()
    model, processor, loading = load_model(args.model)
    result = generate_once(model, processor, args.image, args.question, args.max_new_tokens)
    write_json(args.output, {"loading": loading, "result": result})
    print(result["answer"])
    print(f"Generation: {result['generation_seconds']:.3f} s; preprocessing + generation: {result['preprocess_and_generation_seconds']:.3f} s")
    print(f"Peak GPU allocated: {result['inference_memory']['peak_allocated_mib']:.1f} MiB; reserved: {result['inference_memory']['peak_reserved_mib']:.1f} MiB")
    print(f"LLM input tokens: {result['llm_input_tokens']}; visual tokens: {result['visual_tokens']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
