from pathlib import Path
import subprocess,json,traceback
O=Path(__file__).resolve().parent
base=Path('/home/featurize/ate-local-envs/ate-torch260-cu124-tf449');site=base/'lib/python3.11/site-packages'
results={}
for name,packages in [('qwen3vl-torch260-tf4576',['transformers==4.57.6','tokenizers==0.22.2','huggingface-hub==0.36.2','hf-xet==1.8.0']),('internvl3-torch260-tf449',['timm==0.9.16'])]:
 try:
  d=base.parent/name;subprocess.run([str(base/'bin/python'),'-m','venv','--without-pip',str(d)],check=True)
  (d/'lib/python3.11/site-packages/base_locked.pth').write_text(str(site)+'\n')
  py=str(d/'bin/python');subprocess.run([py,'-m','pip','install','--no-deps',*packages],check=True)
  p=subprocess.run([py,'-m','pip','check'],capture_output=True,text=True);results[name]=dict(status='PASS' if p.returncode==0 else 'FAILED',pip_check=p.stdout+p.stderr)
 except BaseException:results[name]=dict(status='FAILED',traceback=traceback.format_exc())
 (O/'extra_environment.json').write_text(json.dumps(results,indent=2))
