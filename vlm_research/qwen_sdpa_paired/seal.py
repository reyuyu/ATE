"""Final CPU-only source seal, before any GPU invocation or prediction."""
import inspect
import torch
import transformers
from transformers import modeling_utils
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.integrations import sdpa_attention
from .common import *


def main():
    assert not list((OUT/'environment').glob('*.json')),'GPU stage already started; never reseal execution sources'
    files=[Path(inspect.getfile(m)) for m in (modeling_utils,modeling_qwen3_vl,sdpa_attention,torch.nn.attention)]
    atomic(OUT/'installed_source_audit.json',dict(at=now(),torch=torch.__version__,transformers=transformers.__version__,
        files={str(p):sha(p) for p in files},backend_setter='official model.set_attn_implementation / config modality dictionary',
        sdpa_kernel_signature=str(inspect.signature(torch.nn.attention.sdpa_kernel)),
        no_dependency_changes=True,cuda_used=False))
    atomic(OUT/'source_manifest.json',dict(at=now(),files=source_hashes()))
    print('FINAL CPU SOURCE SEAL',sha(OUT/'source_manifest.json'),flush=True)


if __name__=='__main__':main()
