# Resume Without Repeating Valid Answers

The original frozen inference sources and both protocol hashes are unchanged.
All task resources are on `/data`; there is no remaining RAM-output shim.
Do not run `prepare` or `freeze` again over the saved protocol/checkpoints.

## CPU Verification

Run from `/data/lk/ATE-worktrees/cvpr2026-information-horizon-alignment`:

```bash
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.information_horizon_alignment.tests
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.results.cvpr2026_information_horizon_alignment.postprocess_tests
/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.information_horizon_alignment.analyze
```

## GPU Admission

On 2026-11-07 the user explicitly authorized idle-card relay. The task-local
`information_horizon_alignment.supervise` module now automates the checks below
without modifying any frozen inference source or protocol. The systemd user unit
is `ate-information-horizon-relay.service` (no automatic restart after failure).
Its live state is `relay/status.json`, with attempt-specific admission, worker,
failure and release logs under `relay/attempts/`. Mutable relay files are not Git
artifacts. Do not launch manual workers while the relay service is active.

The relay checks every 10 seconds. Admission requires two consecutive checks
with no compute process, at most 1,024 MiB used and at most 5% GPU utilization,
followed by an immediate pre-launch recheck. The original shard/UUID mapping is
preserved. An empty shard cannot be migrated to another GPU to bypass this
binding. If an external task later appears on one of our running cards, the
controller stops only its own children. There is no cross-user reservation
mechanism, so admission cannot guarantee other users will not start later.

Stage A resumes first. Only after all A checkpoints pass their full record and
score audit can Stage B begin. Missing B smoke gates run as smoke-only jobs on
available cards; no B formal worker launches until all four gates pass. All
workers retain the original per-launch smoke checks. Completion of collection
is not a claim that the final official-metric report has been reviewed.

The controller stops on any worker error, hash mismatch, task STOP file or less
than 3 GiB free disk. It does not clear failure markers or relax tolerances.
Previous per-worker metadata and smoke logs are copied before each attempt;
atomic SQLite answers are verified and skipped rather than overwritten.

```bash
systemctl --user status ate-information-horizon-relay.service
systemctl --user stop ate-information-horizon-relay.service
```

The service is local to this server session and is not configured to survive a
host reboot. Stopping it only affects processes owned by this relay.

Check physical UUID, memory, utilization and compute processes immediately before
each launch. Never launch on a card with an external task, even if nominal free
VRAM would fit the model. No process has been killed, paused or migrated.

Frozen shard/physical UUID mapping:

| Rank | UUID |
|---|---|
| 0 | GPU-a63c6362-dcfd-4a26-c44b-014b40127c89 |
| 1 | GPU-b08489af-0a8a-eb48-55ba-8346733af417 |
| 2 | GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6 |
| 3 | GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 |

For each idle matching card, set `RANK` and `GPU_UUID` to its frozen values and
run Stage A. All four Stage A smoke completion records already passed; valid
existing answers are verified and skipped rather than overwritten.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    HF_HUB_CACHE=/data/lk/ATE-resources/information-horizon-alignment/hf-hub \
    HUGGINGFACE_HUB_CACHE=/data/lk/ATE-resources/information-horizon-alignment/hf-hub \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
    /data/lk/ATE-resources/information-horizon-alignment/author-env/bin/python \
    -m vlm_research.information_horizon_alignment.worker --stage A --rank "$RANK"
```

Keep console output in a new attempt-specific log. Prior failure/startup/release
records should be copied into a new incident directory before a restart, not
discarded. The disk-stop record is already retained under
`incidents/disk_capacity_stop/STOP_A.json`; there is no active root stop file.

Do not run Stage B until every Stage A shard has a COMPLETE record and its full
benchmark coverage is verified. B needs its own native/no-op, dense-reference,
mask, position, operator and KV smoke gates; those are currently NOT_RUN.

For Stage B use the same module and frozen rank/UUID mapping, the unchanged
`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`, and `--stage B`.
Its current smoke barrier requires all four B smoke shards to pass before formal
collection. With fewer free devices, perform `--smoke-only` on available ranks
first; do not leave workers waiting on occupied cards.

Do not treat a previous memory/disk abort as a numerical failure, silently relax
thresholds, or combine historical image-only answers with paper-OCR inputs.

## Partial Data

`workers/A/rank*/results.sqlite` are local atomic resume databases, intentionally
not committed as mutable binary data. `per_question.jsonl.gz` contains all 2,683
verified completed records and their exact input/protocol binding. Database hashes
and per-shard counts are in `completion.json`. These are an ordered prefix, not
a complete score or a newly sampled evaluation set.
