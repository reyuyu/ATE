# Analysis replay and immutable release snapshot note

The first CPU analysis completed all numeric/source/cache/token/raw-array checks,
wrote the report, figures, validation and COMPLETE, and recorded release of the
GPU process. A repeated CPU analysis invocation was also observed. It recomputed
CSV/plot outputs, then attempted to replace gpu_release.json with a different live
hardware snapshot (other devices' utilization can change). The existing atomic
JSON writer rejected that replacement with:

`ValueError: Never overwrite a different completed checkpoint`.

The diagnostic is preserved under failures/analysis_1793636227874202077.json.
This was a release-snapshot overwrite guard, not a failed numerical/source-mask
check. No extra model generation was performed and no frozen execution source,
checkpoint, original raw array or historical result was changed to bypass it.
The first GPU release record is retained unchanged.

For final delivery we do not run the write-producing analyzer again. The saved
statistics and actual arrays are checked with analyze --check, plus an independent
per-question contrast bootstrap. Final report editing and plot-file metadata are
bound by a refreshed artifact manifest, including this note and the preserved
diagnostic. The main experiment remains complete only after these checks pass.
