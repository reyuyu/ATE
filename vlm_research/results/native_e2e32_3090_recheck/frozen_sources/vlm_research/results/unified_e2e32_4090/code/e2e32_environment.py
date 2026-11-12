import sys,os,platform,subprocess
from datetime import datetime,timezone
from pathlib import Path
import torch,transformers
from e2e32_prepare import *
def now():return datetime.now(timezone.utc).isoformat()
def command(*args):return subprocess.check_output(args,text=True,timeout=20).strip()
def textfile(p):
 try:return Path(p).read_text()
 except OSError:return None
def static_environment(a):
 source_files={
  'torch_init':torch.__file__,'torch_version':torch.version.__file__,'torch_C':torch._C.__file__,
  'transformers_init':transformers.__file__,
  'installed_llama':str(Path(transformers.__file__).parent/'models/llama/modeling_llama.py'),
  'installed_llava':str(Path(transformers.__file__).parent/'models/llava/modeling_llava.py'),
  'installed_cache':str(Path(transformers.__file__).parent/'cache_utils.py'),
  'frozen_loader':str(R/'vlm_research/mme_seed_quality/engine.py'),
  'frozen_image_prompt':str(R/'vlm_research/information_horizon_alignment/engine.py')}
 for p in (OUT/'code').glob('*.py'):source_files['calibration/'+p.name]=str(p)
 for label,p in [('old_runtime',OLD/'runtime.py'),('new_runtime',COMP/'fastv_runtime.py')]:source_files[label]=str(p)
 cpuinfo=textfile('/proc/cpuinfo') or ''
 cg={}
 cgtext=textfile('/proc/self/cgroup') or ''
 for line in cgtext.splitlines():
  parts=line.split(':',2)
  if len(parts)==3:
   path=Path('/sys/fs/cgroup')/parts[2].lstrip('/')
   for p in [path,*path.parents]:
    if '/sys/fs/cgroup' not in str(p):continue
    for name in ('cpu.max','cpu.stat','cpuset.cpus.effective'):
     content=textfile(p/name)
     if content is not None:cg[str(p/name)]=content
 return {'at':now(),'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
  'cuda_runtime':torch.version.cuda,'GPU':command('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv,noheader'),
  'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),
  'deterministic':torch.are_deterministic_algorithms_enabled(),'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,
  'FP16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
  'env':{k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','LD_LIBRARY_PATH','PYTHONPATH')},
  'sys_path':sys.path,'sources':{k:{'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size} for k,p in source_files.items()},
  'CPU_models':sorted({l.split(':',1)[1].strip() for l in cpuinfo.splitlines() if l.startswith('model name')}),
  'CPU_affinity':sorted(os.sched_getaffinity(0)),'os_cpu_count':os.cpu_count(),'cgroup':cgtext,'cgroup_settings':cg,
  'loadavg':textfile('/proc/loadavg'),'cpu_pressure':textfile('/proc/pressure/cpu'),'torch_build':torch.__config__.show(),
  'shared_model_object_id':id(a.model),'model_class':str(type(a.model)),'model_eval':not a.model.training,
  'dtype':'float16','attention':'eager','hf_device_map':a.model.hf_device_map}
