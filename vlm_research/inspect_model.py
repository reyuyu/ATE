"""Print and save loaded model structure and actual expanded input sizes."""
import argparse
import json

from vlm_common import ROOT, inspect_structure, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(ROOT / "test_images/example.jpg"))
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=str(ROOT / "results/model_structure.json"))
    args = parser.parse_args()
    model, processor, loading = load_model(args.model)
    structure = inspect_structure(model, processor, args.image, args.question)
    write_json(args.output, {"loading": loading, "structure": structure})
    (ROOT / "results/model_modules.txt").write_text(str(model) + "\n")
    print(json.dumps(structure, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
