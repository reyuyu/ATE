cd /home/featurize/ate-local-diagnostics/native_e2e32_3090_rootcause_diagnostic
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -B rootcause_run.py > runner.log 2>&1
diagnostic_exit=$?
printf '%s\n' "$diagnostic_exit" > runner.exit
/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python -B - <<'PY'
from pathlib import Path
import json,subprocess,shutil,time
L=Path('/home/featurize/ate-local-diagnostics/native_e2e32_3090_rootcause_diagnostic');D=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_rootcause_diagnostic')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader,nounits'],text=True).strip()
(L/'gpu_release.json').write_text(json.dumps(dict(at_unix_ns=time.time_ns(),compute_processes=apps,GPU=gpu,no_compute_processes=not bool(apps),server_left_running=True,GPU_settings_modified=False),indent=2)+'\n')
for n in ['runner.log','runner.exit','gpu_release.json']:shutil.copyfile(L/n,D/n)
PY
exit "$diagnostic_exit"
