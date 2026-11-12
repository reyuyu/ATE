#!/usr/bin/env bash
export PYTHONDONTWRITEBYTECODE=1
P=/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python
L=/home/featurize/ate-local-experiments/unified_e2e32_eight_methods_3090
D=/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090
"$P" -B "$L/code/run_3090.py" > "$L/runner.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$L/runner.exit"
cp "$L/runner.log" "$D/runner.log"
cp "$L/runner.exit" "$D/runner.exit"
/environment/miniconda3/bin/python -B - <<'END'
from pathlib import Path
import subprocess,json,datetime
D=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090')
v=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader'],text=True).strip(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip(),server_left_running=True)
(D/'gpu_release.json').write_text(json.dumps(v,indent=2)+'\n')
END
exit "$rc"
