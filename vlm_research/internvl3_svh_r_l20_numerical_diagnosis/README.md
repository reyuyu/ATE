# InternVL3-SVH-R L20 numerical diagnosis

Final status: PARTIAL_REPAIR. No search, no quality scoring, no E2E32 benchmark.
Actual parent code inherited from 9602d93869d4ccd11986573369d8fecc84b80f78. Paths in common.py point to this recorded server.
The atomic evidence writer refuses to overwrite old files. Reproduction needs a NEW run directory,
new source binding and same frozen image/prompt/input SHA/model/UUID, never an in-place rerun.

CPU tests: python -m vlm_research.internvl3_svh_r_l20_numerical_diagnosis.cpu_tests
A: diagnose_ab_v002.py (A results valid; initial B mask capture replay failed and is preserved)
B: complete_b_v003.py (correct effective frozen Dense mask; exact native operator reproduction)
C1/C2: repair_trial.py reads immutable candidate-specific frozen JSON, stops after first failed
sample and does not enter remaining seven inputs. adapter.py only pads specified prefill projections.
Cost: projection_cost_installed.py, existing frozen candidates only, local CUDA Events.
Audit: final_audit.py is CPU only; map_location=cpu, no CUDA/model loading.
Final report: finalize.py. Publication: publish.py (explicit scope, no force/main/PR).

GPU processes used CUDA_VISIBLE_DEVICES=GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e, ATE_SEVEN_ROOT=/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090,
ATE_SEVEN_RUN=/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090, PYTHONPATH=/home/featurize/ate-runs/internvl3-svh-r-l20-numerical-diagnosis-3090/code,
Python /home/featurize/.venvs/ate-frozen/bin/python.
Previous missing visibility variable failed before generation, saved as AB_failure_v001.json.
Original mask capture error affected independent replay only, corrected in separately saved B_v003.
This package preserves actual successful and failed analysis versions; do not mistake earlier replay
failure or partial projection repair for full R0 numerical acceptance. Raw BF16 Tensor files stay server-only.
