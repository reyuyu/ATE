# InternVL3-SVH-R14 fixed full-middle TextVQA validation

This package runs only the fixed R0 / R6 (L14–L19) / R14 (L6–L19) physical conditions. Native is an individually verified historical reference. It does not restore L0–L1, change L2–L5 answer attention skipping, or change the L20 image hidden exit.

Parent: 3af6596b1ce59d80e203ebf68c6760e242d4df47. The unchanged physical V2 engine is inherited from 9602d93869d4ccd11986573369d8fecc84b80f78. Runtime file hashes were frozen before GPU work in source_manifest.json. The inherited code remains available in this commit; dependency_source_validation.json verifies its Git blob bytes.

Status is EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER. Historical NUMERICAL_EQUIVALENCE_FAILED is not reclassified as PASS. No C1/C2 numerical repair is used.

## Fixed experiment

- A: historical Holdout96, reuse 288 Native/R0/R6 records; generate R14 only (96 requests). This is historical followup, not independent confirmation.
- B: 128 new questions, one per image, image-disjoint from all prior 200 discovery/holdout/smoke images; no type quotas or outcome selection. Generate R0/R6/R14 (384 requests), reuse individually bound Native128.
- Six short Dense/Physical R14 smoke questions. Official Native/No-op first-sample logits must be strictly equal. Structural errors stop execution.
- Image-only native chat and unchanged instruction, BF16/eager, max_new_tokens=32, normal EOS.
- Paired image-cluster bootstrap: 5000 draws, fixed seed in protocol. Keep all generated outputs and score them.
- Timing gate frozen before GPU: B R14-R0 and R14-R6 ordinary paired 95%CI lower bounds must both exceed zero. Otherwise speed is NOT_RUN. No old timing is substituted.

## Reproduction and resume

Server SSD run: /home/featurize/ate-runs/internvl3-svh-r14-full-middle-reaccess-textvqa-3090

Pinned environment: /home/featurize/.venvs/ate-frozen/bin/python

Required unchanged local resources: cached InternVL3-8B revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9, official TextVQA validation images/manifests, parent run and legacy Native checkpoint. No downloads or dependency installs are performed.

Modules prepare.py, run.py and analyze.py are the exact entry points. Use the inherited packages on PYTHONPATH, ATE_SEVEN_ROOT and ATE_SEVEN_RUN for the frozen seven-benchmark helper resources. Set CUDA_VISIBLE_DEVICES to the UUID in protocol.json for run.py only. prepare.py and analyze.py require CPU resources only. ATE_R14_RUN may point to an explicit SSD run location, but parent-resource bindings in common.py must remain available and match their recorded hashes. Do not run prepare.py against archived results: it is for preparation before first inference. The existing SQLite checkpoint and frozen protocol/source hashes are required for strict run.py resume.

Post-run analyze.py independently scores all 896 quality records, checks reused origins, token caps/EOS, input bindings and all 28 real cache layouts, and writes the report and CSVs. persist.py creates a fresh SQLite backup and copies verified results to the persistent worktree; publish.py stages only the explicit light-artifact allowlist and verifies every remote blob after a non-forced push. These post-run utilities do not change the frozen model forward code.

## Storage

Persistent raw checkpoint, original reuse records, detailed logs and post-push receipt are under the result directory's server_only/. They are deliberately not committed. CPU image tensors remain on server SSD; no model weights, image datasets or tensors are transferred to Windows or GitHub. artifact_manifest.json and SHA256SUMS identify uploaded evidence, server-only files and compressed exact KV layout.

Quality-end KV Live/Capacity are measured after normal EOS and have varying output length; they are not KV@32 or a speed measurement. The final report is authoritative for measured scores, finite-sample uncertainty and the negative quality finding.
