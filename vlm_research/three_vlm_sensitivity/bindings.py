"""CPU-only environment/source hashes and read-only historical window context."""
import csv
import json
import os
from pathlib import Path
import subprocess

from .common import OUT, LLAVA, HISTORY, QWEN, INTERN, read, write, sha256, csv_write


def main():
    bindings = {}
    script = '''import json,sys,torch,transformers,inspect,hashlib
from pathlib import Path
names=['transformers.models.qwen3_vl.modeling_qwen3_vl'] if transformers.__version__=='4.57.6' else ['transformers.models.qwen2.modeling_qwen2','transformers.models.llama.modeling_llama']
import importlib
files=[Path(inspect.getfile(importlib.import_module(n))) for n in names]
print(json.dumps(dict(python=sys.version,executable=sys.executable,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})))'''
    for name, python in [('qwen', '/data/lk/深度学习/.envs/ate-qwen3vl/bin/python'),
                         ('internvl', '/data/lk/深度学习/.envs/ate-internvl3/bin/python')]:
        bindings[name] = json.loads(subprocess.check_output([python, '-c', script], text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1')))
    directory = Path(read(LLAVA / 'sample_manifest.json')['protocol']['model_path'])
    bindings['llava'] = dict(revision=directory.name, precision='float16',
        files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(directory.iterdir()) if p.is_file()],
        historical_execution_manifest=dict(path=str(LLAVA / 'execution_manifest.json'), sha256=sha256(LLAVA / 'execution_manifest.json')))
    write(OUT / 'runtime_source_bindings.json', bindings)
    windows = []
    sources = {
        'llava': HISTORY / 'four-layer-p1-decode-scan/vlm_research/results/four_layer_p1_decode_scan/accuracy_summary.csv',
        'qwen': HISTORY / 'qwen3vl-early-attention-windows-240/vlm_research/results/qwen3vl_early_attention_windows_240/accuracy_summary.csv',
        'internvl': INTERN / 'window_accuracy_summary.csv',
    }
    history = {}
    for name, path in sources.items():
        history[name] = dict(path=str(path), sha256=sha256(path),
            rows=list(csv.DictReader(path.open(newline=''))),
            caveat='Historical full-generation multi-layer endpoint, NOT current single-layer first-token sensitivity. Sample counts and templates differ across models.')
    write(OUT / 'historical_window_context.json', history)
    print('BOUND all official model/source files and historical window context; CPU only', flush=True)


if __name__ == '__main__':
    main()
