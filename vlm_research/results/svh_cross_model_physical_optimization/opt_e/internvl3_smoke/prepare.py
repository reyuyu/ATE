"""Review pinned local code and freeze a four-case smoke, without model loading."""
import ast
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import platform
import subprocess
import sys

from PIL import Image

from .common import OUT, read, sha256, write

SOURCE = Path('/data/lk/ATE-worktrees/qwen3vl-early-attention-windows-240/vlm_research/results/qwen3vl_early_attention_windows_240/sample_manifest.json')
SEED = 20261101


def environment(python):
    code = 'import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages={d.metadata["Name"]:d.version for d in m.distributions()}),sort_keys=True))'
    return subprocess.check_output([str(python), '-c', code], text=True)


def main():
    manifest = read(OUT / 'download_manifest.json')
    directory = Path(manifest['local_path'])
    entries = {r['path']: r for r in manifest['files']}
    reviews = []
    for file in sorted(directory.glob('*.py')):
        assert sha256(file) == entries[file.name]['sha256']
        tree = ast.parse(file.read_text(encoding='utf-8'))
        imports = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                         {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names})
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert not calls & {'eval', 'exec', '__import__', 'os.system', 'subprocess.run', 'subprocess.Popen'}
        reviews.append(dict(file=file.name, sha256=sha256(file), imports=imports,
                            unsafe_shell_eval_network_calls_found=False, reviewed=True))
    assert len(reviews) == 5
    write(OUT/'remote_code_review.json', dict(passed=True, revision=manifest['revision'], files=reviews,
        decision='Local fixed-revision code only; no flash_attn/apex installed; constructor explicitly use_flash_attn=False. Builtin Transformers Qwen2ForCausalLM 4.49.0 supports official >=4.37.2 requirement.',
        reviewed_paths=['config classes', 'InternVisionModel naive attention', 'InternVLChatModel constructor/forward/extract_feature/pixel_shuffle/chat/generate', 'conversation internvl2_5 template'],
        no_third_party_code_rewritten=True, before_trust_remote_code_execution=True,
        actual_config_warning='Stale _name_or_path strings mention 32B/6B; use actual instantiated sizes and shapes, not those labels.'))
    original = read(SOURCE)['samples']
    chosen, used = [], set()
    def key(row):
        return hashlib.sha256(f'{SEED}|{row["task"]}|{row["question_id"]}|{row["image_sha256"]}'.encode()).hexdigest()
    for task in ('ChooseAttr', 'ChooseRel', 'QueryAttr'):
        row = min((r for r in original if r['task'] == task and r['image_sha256'] not in used), key=key)
        used.add(row['image_sha256'])
        chosen.append((row, 1))
    candidates = []
    for row in original:
        if row['image_sha256'] in used:
            continue
        with Image.open(row['image_path']) as image:
            ratio = image.width / image.height
        if ratio >= 1.6 or ratio <= .625:
            candidates.append(row)
    assert candidates
    chosen.append((min(candidates, key=key), 2))
    samples = []
    for row, maximum in chosen:
        assert sha256(row['image_path']) == row['image_sha256']
        case = {k: row[k] for k in ('task','question_id','image_id','image_path','image_sha256','question','answer')}
        with Image.open(row['image_path']) as image:
            case['original_size'] = list(image.size)
        case.update(max_num=maximum, request=f'<image>\n{row["question"]}\nAnswer using a single word or short phrase.')
        samples.append(case)
    write(OUT/'sample_manifest.json', dict(seed=SEED, source=str(SOURCE), source_sha256=sha256(SOURCE),
        selection='Three predetermined tasks, lowest input SHA256; one additional unused image selected by aspect ratio >=1.6 or <=0.625 then lowest hash. No answers/correctness used.', samples=samples))
    import transformers.models.qwen2.modeling_qwen2 as qwen
    existing = {}
    for name in ('ate-vlm-research','ate-qwen3vl'):
        python = Path('/data/lk/深度学习/.envs') / name / 'bin/python'
        existing[name] = dict(python=str(python), distribution_listing_sha256=hashlib.sha256(environment(python).encode()).hexdigest())
    write(OUT/'environment.json', dict(python=platform.python_version(), executable=sys.executable,
        versions={name: importlib.metadata.version(name) for name in ('torch','torchvision','transformers','timm','einops','accelerate','safetensors','huggingface-hub','Pillow','numpy')},
        official_qwen2_source=str(inspect.getfile(qwen)), official_qwen2_source_sha256=sha256(inspect.getfile(qwen)),
        prior_environments=existing, flash_attn_installed=False, offload=False, quantization=False))
    sources = {str(p.relative_to(OUT.parents[2])): sha256(p) for p in sorted(Path(__file__).parent.glob('*')) if p.is_file()}
    write(OUT/'protocol.json', dict(repo='OpenGVLab/InternVL3-8B', revision=manifest['revision'], local_path=str(directory),
        physical_gpu=1, visible_devices='1', process_device='cuda:0', dtype='bfloat16', attention='eager',
        use_flash_attn=False, independent_environment=sys.prefix, generation=dict(do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True),
        sample_manifest_sha256=sha256(OUT/'sample_manifest.json'), model_manifest_sha256=sha256(OUT/'download_manifest.json'),
        source_sha256=sources, intervention_layers=[2, 'actual_last_layer'], mass_layers=[0, 'actual_last_layer'],
        noop_tolerance=0.0, mass_row_sum_tolerance=2**-8+2e-6,
        mass_tolerance_basis='Native BF16 nonnegative probability rounding <= 2^-8 plus 2e-6 FP32 summation allowance; not tuned after smoke',
        scope='3 max_num=1 generations + one max_num=2 aspect case; 2 cases of fixed single-layer P-1 output hooks; one 2-layer mass collection. No scan, hard masks or benchmark.',
        prefill_replay='Replay exact first official language_model.generate kwargs using immutable native multimodal embeddings and a fresh empty DynamicCache; preserves vision injection and compute path.'))
    print('Pinned code reviewed; four smoke cases and independent environment frozen', flush=True)


if __name__ == '__main__':
    main()
