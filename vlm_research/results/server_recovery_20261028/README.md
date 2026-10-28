# Server recovery audit snapshot

Status: **resources incomplete, zero new GPU conditions**.

`repository_validation.json`: all 1,143 historical artifact hashes pass.
`server_snapshot.json`: original cohort/CSV/cache shapes, actual local paths,
packages and idle GPUs at collection time; it is not a live monitor.
`remote_controls_AFAC_with_history.txt`: seven path/download-control unit tests,
run on existing AFAC, not the fixed-environment 7/101 experiment acceptance.
`history_path_audit.json`: 115 repository checks pass; seven raw-image checks
remain pending, so aggregate passed is false. Missing images are not skipped.
`offline_*_AFAC.txt`: preserved original legacy-path failures, not hidden.

All pre-existing algorithms, frozen manifests and historical results remain
unchanged. See `handoff/SERVER_RECOVERY_20261028.md` for pinned resources,
current preparation jobs and the manual single-case execution command.
