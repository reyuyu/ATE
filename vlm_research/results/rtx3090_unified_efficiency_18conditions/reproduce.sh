#!/usr/bin/env bash
# Actual invocation sequence for this frozen run. Existing result directories
# are immutable: do not rerun in place. For a new run use an independent
# worktree/output path and freeze its path mappings/source manifest first.
set -euo pipefail
cd /home/featurize/ate-experiments/rtx3090-18conditions-20261113
source /home/featurize/work/ATE/vlm_research/results/featurize_23137-20261113_resource_restore/activate.sh
O=vlm_research/results/rtx3090_unified_efficiency_18conditions
# Original smoke and source-path-only correction evidence are retained.
python -u -B "$O/code/run_llava.py" --phase smoke
python -u -B "$O/code/run_llava_v2.py" --phase smoke_sparse
python -u -B "$O/code/run_llava_v2.py" --phase session1
python -u -B "$O/code/run_llava_v2.py" --phase session2
for phase in smoke session1 session2; do
 /home/featurize/ate-local-envs/qwen3vl-torch260-tf4576/bin/python -u -B "$O/code/run_cross.py" --kind Qwen --phase "$phase"
done
for phase in smoke session1 session2; do
 /home/featurize/ate-local-envs/internvl3-torch260-tf449/bin/python -u -B "$O/code/run_cross.py" --kind InternVL --phase "$phase"
done
/environment/miniconda3/bin/python -B "$O/code/audit_smoke_cpu.py"
/environment/miniconda3/bin/python -B "$O/code/check_session_tokens.py"
/environment/miniconda3/bin/python -B "$O/code/finalize_metadata.py"
/environment/miniconda3/bin/python -B "$O/code/analyze.py" --final
