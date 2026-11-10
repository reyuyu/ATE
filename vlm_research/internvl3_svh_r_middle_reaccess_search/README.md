# InternVL3-SVH-R failed-Smoke reproduction

Status: FAILED / NUMERICAL_EQUIVALENCE_FAILED. Formal Discovery, Holdout, efficiency: NOT_RUN.
Package includes actual physical engine, independent Dense-R reference, CPU mock, GPU Smoke,
operator diagnostics, CPU evidence audit and publication code. There is no completed formal-search
controller because numerical acceptance failed first.

Parent source: 3297559f21d3de55a6630a5491c9ecf66301c4b7, unchanged.
Server paths are centralized in common.py. Porting paths requires a separately versioned protocol.
Model: /home/featurize/ate-local-models/InternVL3-8B, revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9. Models, datasets, credentials are excluded.

Recorded server commands (not an instruction to rerun automatically):

    export PYTHONPATH=/home/featurize/ate-runs/internvl3-svh-r-middle-reaccess-search-3090/code
    export ATE_SEVEN_ROOT=/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090
    export ATE_SEVEN_RUN=/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090
    /home/featurize/.venvs/ate-frozen/bin/python -m vlm_research.internvl3_svh_r_middle_reaccess_search.cpu_tests
    /home/featurize/.venvs/ate-frozen/bin/python -m vlm_research.internvl3_svh_r_middle_reaccess_search.gpu_smoke
    /home/featurize/.venvs/ate-frozen/bin/python -m vlm_research.internvl3_svh_r_middle_reaccess_search.final_audit

GPU Smoke must use an exclusive matching UUID, immutable inputs/model/source, and a NEW attempt
directory. Archive existing outputs and raw tensors before any reproduction; never overwrite them.
The diagnostics inspect one frozen Discovery sample only, without scoring.
diagnose_accumulation.py is a process-local precision-flag diagnosis, not an accepted configuration;
it did not eliminate K/V differences. Raw .pt are server-only. Existing raw Logits are V2;
V1 CSV, complete tokens, source and logs remain, but V1 full Logits tensors were overwritten by V2.
