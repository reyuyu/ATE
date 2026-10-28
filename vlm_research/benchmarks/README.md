# Phase 4.0 Dataset Suite

This package prepares frozen datasets and runs hard-capped Native Full smoke only.
It does not implement or run B4, C2, Physical Exit, training or a full benchmark.
Historical sources/results and main remain unchanged.

## Resources and Results

- Model/environment remain the original LLaVA revision, FP16/eager, Transformers4.49.0.
- Shared resources: `/data/lk/ATE-resources/benchmarks/phase4_0/`.
- Existing GQA images: `/data/lk/ATE-resources/data/image-tasks/MMEB/GQA/`.
- Results: `vlm_research/results/phase4_0_dataset_suite/`.
- Source pins, row/image hashes, original choices and all ten TextVQA votes are retained.
- No image archive, raw image, model or environment is committed.
- Frozen subsets are not full benchmark leaderboard reproductions.
- Official GQA balanced remains separately registered and blocked pending official assets/scorer.

## Reuse From Other Worktrees

Run from this branch's `vlm_research/`, or place it on `PYTHONPATH`. Read shared
resources only; use a separate output directory for any future runner.

```python
from benchmarks.dataset_registry import load_manifest

m = load_manifest(
    '/data/lk/ATE-worktrees/phase40-benchmark-datasets/vlm_research/results/'
    'phase4_0_dataset_suite/manifests/gqa_chooseattr.json'
)
samples = m['samples']
```

`load_manifest` rejects modified bindings, annotation bytes and image hashes.
An optional `image_root` remaps `metadata.resource_relative_path` without rewriting
the frozen file. For another server, copy pinned annotation paths as well; do not
edit manifests to bypass source checks. All eight unit manifests are independently
loadable; POPE strategies deliberately share exactly the same 50 image groups.

Use `metadata.cluster_id` (verified image SHA256) for paired cluster bootstrap.
ScienceQA official PID/image-path count is not its count of independent image bytes.
Do not treat multiple questions or heads/layers as independent images.

## Scoring Boundaries

- `diagnostic_candidate_metrics`: complete candidate sequence LogP, gold minus
  strongest wrong, ties incorrect; separate diagnostic field names.
- ChooseAttr retains original option order from official GQA operations.
- LogicalObj/POPE use no/yes; ScienceQA uses original option letters, preserving
  full choice texts/index in metadata. Letter LogP is not choice-text LogP.
- QueryAttr remains open even though the source answers happen to be left/right.
- `generated_score`: actual decoded answers only. POPE exposes all five metrics;
  ScienceQA uses pinned LLaVA parsing; TextVQA uses the unchanged upstream
  normalization and ten-vote leave-one-annotator-out score.
- GQA author-subset literal exact match is a mechanism measure, not the unavailable
  official GQA evaluator. TextVQA input is explicitly image-only, without OCR hints.
- Undefined POPE precision/recall denominators are flagged and extended to zero;
  otherwise upstream numerical definitions are preserved, including lexical quirks.

## Preparation and Verification

Use `/data/lk/深度学习/.envs/ate-vlm-research/bin/python`. CPU commands require
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

```bash
python -m benchmarks.validate_datasets
python -m unittest test_benchmark_suite -v
```

`acquire` fetches bounded revision-pinned annotations; `fetch_resources` freezes
IDs before selected downloads. TextVQA uses stdlib ZIP parsing and CRC with
ETag-pinned HTTP ranges, never silently downloading the entire7GB ZIP.
ScienceQA uses a pinned HF test parquet verified by LFS SHA256 and all4241
official test records, exact publisher ordering and12 content fields.
Its optional parquet tool is isolated under the resource directory; it does not
alter the pinned VLM environment. Historical acquisition failures remain audited.

`prepare_datasets` refuses changed frozen artifacts. `acceptance` must pass before
`smoke`. Smoke only accepts its frozen18 sample plan, checks an idle physicalGPU0,
and saves actual generated answers/full-vocabulary candidate logits. It refuses
an existing attempt directory. Do not use this entry point as a full-run launcher.

Future B4/C2 needs true Question boundaries separate from choices/hints, audited
multi-token scoring and prefill/generation semantics. Physical Exit additionally
requires position/KV-cache/decode equivalence. Native smoke is not acceptance of
any of those interventions. No automatic follow-on research is authorized.
