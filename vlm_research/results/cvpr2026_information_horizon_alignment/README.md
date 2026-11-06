# CVPR 2026 Information-Horizon protocol alignment

Status: PARTIAL. Stage A has 2,683/22,678 checkpointed answers, all TextVQA OCR;
Stage B is NOT_RUN. No complete new benchmark score is available yet.

See report_zh.md, completion.json and coverage.csv. A disk safety stop interrupted
generation at 22:02 Asia/Shanghai on November 6, 2026. The disk issue has been
resolved, but all four GPUs are currently occupied by another user's training.
No external process has been interrupted and no auto-polling job is installed.

- ATE base: e2022e7e195fcd121ac2275c4c42edfa30642e17.
- Author repository HEAD verified over authenticated SSH:
  75909b2936a13d7214e83514f0bc0cb9cc91139e.
- Official native model revision: 4481d270cc22fd5c4d1bb5df129622006ccd9234.
- Stage A: author LLaVA, FlashAttention2, vicuna_v1, no pruning.
- Stage B: unchanged fixed HF LLaVA, common FP16/eager backend and author inputs;
  Native, Physical C3-V2, Physical C3-V2 + L11-L24.
- Required ordering: complete Stage A before Stage B.
- TextVQA paper OCR input is distinct from historical image-only input.
- MM-Vet, SEED, GQA GPU evaluation and performance measurements are excluded.
- No changes to shared environments, historical outputs or main.

Resources and new environments have been restored under /data after a temporary
emergency RAM staging. Four physical RTX3090 UUIDs
were checked idle at 2026-11-06 21:14:55 Asia/Shanghai; occupancy is checked again
immediately before workers start.

All 4 x 15 Stage A GPU smoke cases passed. This is AUTHOR_CODE_REPRODUCTION with
an explicit dispatch to the author's existing unpruned parent forward; the author
None-pruning branch otherwise raises UnboundLocalError. Exact dependency and
cross-backend parity are not claimed. Stage B GPU smoke is not yet run.
