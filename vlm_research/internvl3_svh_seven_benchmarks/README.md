# InternVL3-SVH single-tile seven-benchmark quality harness

Frozen Physical-D engine: e288eee271b7aeb64e5df59783d3201bb2a6b313.
The core engine is imported without modifications. The strict BF16 numerical gate remains FAILED_WITH_PREVIOUSLY_AUTHORIZED_WAIVER.

1. restore.py copies existing server archives to local SSD, verifies archive SHA256 and safely extracts them. It never downloads weights/images.
2. prepare.py validates source annotations and frozen manifests, renders full native-template prompts and deterministically freezes at least 24 preflight samples per task.
3. cpu_tests.py checks targeted scorer, role, prompt, cap and source contracts.
4. worker.py loads one BF16 model on the designated GPU. Per task it checks official Chat/Native/No-op exact logits, physical structure and KV; it then executes the output-length preflight before unlocking full paired generation. Each request has fresh KV. Raw predictions are stored transactionally in SQLite on SSD.
5. analyze.py independently rescores all records, runs 20,000 image-cluster paired bootstrap replicates only for complete splits and writes native-unit scores. Partial subsets never become full benchmark results.
6. publish.py copies verified light evidence and raw server-only checkpoints to persistent storage, stages only explicit file paths, pushes the authorized independent branch without force and checks the remote commit/tree.
7. controller.py executes these phases sequentially. It does not launch any other experiment.

Main protocol label: ATE_SVH_1TILE_SCORER_ALIGNED. These are not dynamic/multitile author leaderboard recipes. TextVQA is image-only; MMBench is Dev20230712 Circular with the pinned official exact_matching policy and no LLM judge. All full source references are pinned by commit and SHA256.

Per-task natural stopping caps: POPE 8; GQA/MME/ScienceQA/MMBench 16; TextVQA 32; OCRBench 128. No fixed-length efficiency test is performed.

Default storage:
- SSD code/checkpoints: /home/featurize/ate-runs/internvl3-svh-seven-benchmarks-3090
- Persistent results: /home/featurize/work/ATE-internvl3-svh-seven-benchmarks-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090

Only launch the controller after freezing runtime source fingerprints and checking cpu_tests.json. Resume uses the exact protocol, sample and source bindings; it never silently changes prompts, caps or the physical algorithm.
