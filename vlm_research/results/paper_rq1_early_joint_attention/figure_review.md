# Figure And Interpretation Review

- Replayed all20,449 generation answers and scores; table audit checks every reported denominator, accuracy, delta and flip count.
- Initial six PNG/PDF figures and initial auto-generated reports are preserved in `analysis_versions/initial_figures/`.
- Figure1 retains every negative extreme and adds a labeled inset for S3/S5/S7; no change to underlying values, limits of the full main axis or statistical criteria.
- Figure6 now selects two distinct questions for each of harm, improvement and unchanged outcome. Selection affects illustrative case panels only, never cohort membership or metrics.
- Deterministic200,000-draw statistical replay checked byte-identical primary CSVs before rendering revised figures.
- Initial Figures1/6 were visually inspected. All revised figures are also reviewed before delivery; individual PNG/PDF outputs and source CSV hashes are retained.
- Main report and Chinese summary clarify that S3 can pass the -1pp criterion while showing actual harm, S5 is not a multiplicity-corrected strict improvement, and zero observed S7 harm is not zero population risk.
- Timing in execution_times.csv refers to measured collection and the initial20.28s CPU analysis pass; later visual review, deterministic redraw, validation and Git delivery are additional, not included in that initial pass.
