# Execution and Review Provenance

- Requested role is the independent physical GPU1 research executor. No
  helper agents were spawned. GPU0 initially had an active process using
  about14137MiB; GPU1 was free. No other process was terminated, and GPU0
  worktree, branch, source and experiment outputs were not edited.
- Upstream main was checked over SSH and was
  e89c8acce6e674ecfbd2c8acad57faf71bf4d547. The requested experiment branch
  did not exist. A new independent worktree was created at
  /data/lk/ATE-worktrees/phase38-sink-stage on exp/phase3_8-sink-stage.
- Frozen implementation commit is
  d0da03a848b8825941ec9f4d5df2461f87c6b0bf. The five added implementation,
  test and protocol-document files do not edit existing shared algorithms.
- Protocol binding is
  6383466589a19721096725767cc514ed11beb9c4731ae8f0c6357e3415020c7a.
  The16 images, seed3820261029, historical selection strata, original image
  hashes, model-file hashes, source hashes and original frozen Sink/ROI
  intersections are saved before GPU work in manifest.json.
- The new13 CPU tests exercise the actual Transformers4.49 LLaMA graph,
  including normal/exception cleanup, isolation, pre-softmax scope, target
  residual equations, MLP execution, causal prefixes, norm denominators and
  the original/full/tie rules. One development fixture initially omitted
  dtype=float16 while using the FP16 finite mask minimum; the fixture was
  corrected before freeze. Algorithm checks were not relaxed. The complete
  fixed-environment CPU suite then passed121/121 with no skipped tests.
- All GPU launches set CUDA_VISIBLE_DEVICES=1 and
  CUDA_DEVICE_ORDER=PCI_BUS_ID before importing PyTorch. The sole internal
  cuda:0 UUID was explicitly checked against physical GPU1:
  b08489af-0a8a-eb48-55ba-8346733af417. Both launch records are retained.
- CPU regression uses a process-local path adapter for the original
  default_model helper only; no model-class or Transformers-global forward
  is patched. GPU loads always pass the local model path explicitly.
- The ten package pins and FP16/eager settings are unchanged. This server
  uses Python3.11.15 and RTX3090 rather than the historical Python3.11.18
  and RTX4090. Actual runtime environments are retained; numerical Full
  differences must be reported, not fitted away.
- Smoke uses one left and one right image from the same frozen16; no
  additional image enters the pilot. It checks both candidate sequences,
  all five conditions, all32 preceding-text layer hashes, Question, L1,
  input tensors, exact Native Full restoration and full model state_dict
  tensor hashes. Its independent audit is also retained.
- C0-C4 are the only research conditions. Unhooked Native Full calls before
  and after each image are correctness controls, not additional conditions
  or extra observations in the outcome denominator.
- The raw NPZ stores both candidates' full-vocabulary prediction logits,
  actual post-layer P-1 hidden vectors at the declared locations, per-head
  native L0-L10 Attention rows, and actual native L2-L4 projected Attention
  and MLP updates with the entering and pre-MLP residual vectors.
- Each completed image is checkpointed with input/protocol and artifact
  hashes. Failed partial attempts, if any, are preserved and never counted
  as successful samples. Formal aggregation requires all16 frozen IDs and
  five conditions, with no duplicates.
- Creating a PR via API requires account authorization. At the publication
  preflight there was no gh CLI, GH_TOKEN, GITHUB_TOKEN or standard Git
  HTTPS account credential. SSH Deploy Key push is separate from account
  API authorization. The requested manual PR comparison link is the
  fallback; do not claim that a PR exists or auto-merge main.
