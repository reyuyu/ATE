# Qwen same-backend paired quality

Independent continuation from `53e9cc1f52481de338ab1a3149a41b63eaa5d162`.
Output: `vlm_research/results/qwen3vl_mme_seed_2048_sdpa_paired_quality/`.

This is a new experiment: Native and Physical-Original both use official
Vision SDPA / Text Eager with the unchanged 2048 visual-token Processor.
Historical Eager OOM and cross-backend threshold failure remain unchanged.
The old cross-backend thresholds are recorded diagnostically, not used to
stop this distinct same-backend experiment. This distinction is frozen before
any new GPU request in `protocol.json`.

CPU entry points: `prepare`, `seal`, `tests` under this package. They require
the existing Qwen environment, with CUDA hidden. No dependency changes.

After explicit current-round GPU authorization, save its evidence in the NEW
result directory's `authorization.json` (`gpu_authorized_this_round: true`).
Recheck actual idle GPU UUIDs and >20 GiB disk immediately before launch.
Use one model per UUID; fixed image shards 0/1 on GPU2/3. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and expose one UUID per process.

Run `worker --rank 0/1 --stage smoke`. Both rank completion records must PASS
before starting `worker --rank 0/1 --stage run`. Do not start formal work on
a resource/semantic/same-backend gate failure. Source hashes must remain frozen.

Smoke first completes worst full Native/Physical requests and all SEED
candidate paths, saving resource-gate records. Then it verifies Native/HF/
No-op effective logits, operator-observer invariance, and original physical
mask/position/skip/compact-KV semantics. Physical/Dense F retains the original
0.02/0.5 strict numerical diagnostic and token comparison. Known arithmetic
differences do not excuse any new semantic error.

Cross-backend diagnosis uses two predetermined MME and two predetermined short
SEED cases after the necessary gates. SEED rankings, full greedy tokens,
Vision features, effective logits and small-sample accuracy differences are
all recorded. None is a full-benchmark estimate.

All formal checkpoints use atomic writes plus payload/file SHA indices, exact
input/source/protocol/model bindings, and independent per-condition Cache.
Only indexed, hash-verified completed conditions are reused on resume.

CPU `analyze` reconstructs every score, runs 20,000 paired image-cluster
bootstrap replicates, and invokes the pinned official MME tool. Use the
existing CPU research environment; official tool dependencies remain in the
previous isolated dependency directory. Full scores require complete paired
coverage. Partial data never silently replaces the full benchmark.

No timings, FLOPs profiling, kernel/model/layer search, training or repairs.
Never alter old result directories or main. Any OOM, nonfinite, input SHA,
mask/cache/position failure or resource breach stops execution.
