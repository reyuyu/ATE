"""Build independent runtime copies with enumerated changes, preserve old files."""
from pathlib import Path
import ast,hashlib,json,shutil,urllib.request
R=Path('/home/featurize/work/ATE');E=R/'vlm_research/results/unified_e2e32_4090';O=R/'vlm_research/results/unified_retain128_e2e32_4090';C=O/'code';A=R/'vlm_research/external_baselines/retain128_compatibility'
def install(p,data):
 if p.exists():assert p.read_bytes()==data,str(p)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def swap(s,a,b):assert a in s,a;return s.replace(a,b)
src=(E/'code/e2e32_runtime.py').read_text()
src=swap(src,'from e2e32_prepare import SOURCE,VSOURCE,METHODS','from retain128_prepare import SOURCE,VSOURCE,METHODS\nfrom retain128_adapter import V2Drop,OfficialCompat')
start=src.index('def expected_visual(');end=src.index('\ndef kv_snapshot(',start)
src=src[:start]+'''def expected_visual(method,layer):
 from retain128_prepare import expected_layout
 return expected_layout("Native" if method in ("BridgeNative","V2DropNoop","FastVNoop") else method)[layer]
'''+src[end:]
src=swap(src,'(307 if i==16 else 153)','(205 if i==16 else 102)')
src=swap(src,"else 'original gathered positions'","else 'compact (official recycling)' if method=='SparseVLM' else 'original gathered positions'")
install(C/'retain128_runtime.py',src.encode())
run=(E/'code/e2e32_run.py').read_text()
for a,b in [
 ('from e2e32_prepare import *','from retain128_prepare import *'),
 ('[str(R),str(COMP),str(VCOMP),str(NEWCOMP)]','[str(R),str(COMP),str(VCOMP),str(NEWCOMP),str(OLDNEWCOMP)]'),
 ('from e2e32_runtime import generate','from retain128_runtime import generate'),
 ("str(OUT/'code/e2e32_monitor.py')","str(OUT/'code/retain128_monitor.py')"),
 ("for method in ('Native','FastV','V2Drop','SVH','SVH-R','SparseVLM','ApET','DUET'):","for method in METHODS:"),
 ("for noop in ('FastVNoop','V2DropNoop'):","for noop in ('V2DropNoop',):"),
 ("if method in ('Native','FastV'):","if method=='Native':"),
 ('len(available)==8','len(available)==len(REQUESTED)'),
 ('formal_expected=240','formal_expected=150'),
 ("len(records),len(available)*30","len(records),len(available)*30")]:
 if a!=b:run=swap(run,a,b)
# Capture new code identities as well as the previous verified common runtime.
run=swap(run,"e=static_environment(a);e.update(GPU_UUID=UUID,phase='preparation' if args.prepare_only else 'formal',source_manifest=p['frozen_source_SHA256'])", "e=static_environment(a);e.update(GPU_UUID=UUID,phase='preparation' if args.prepare_only else 'formal',source_manifest=p['source_SHA256'],retain128_protocol_SHA256=sha(OUT/'protocol.json'),imports={m:dict(path=__import__(m).__file__,sha256=sha(__import__(m).__file__)) for m in ['retain128_runtime','retain128_adapter','e2e32_official_compat','hf_adapter']})")
install(C/'retain128_run.py',run.encode())
install(C/'retain128_monitor.py',(E/'code/e2e32_monitor.py').read_bytes())
for p in C.glob('retain128_*.py'):ast.parse(p.read_text())
ast.parse((A/'retain128_adapter.py').read_text())
install(A/'README.md',b'Retain-128 independent adapters. Frozen Retain-192 adapters and official source files are unchanged. See unified_retain128_e2e32_4090/configuration_audit.json for budgets, evidence and limitations.\n')
print(json.dumps(dict(status='CODE_PREPARED',runtime=str(C/'retain128_runtime.py'),runner=str(C/'retain128_run.py'))))
