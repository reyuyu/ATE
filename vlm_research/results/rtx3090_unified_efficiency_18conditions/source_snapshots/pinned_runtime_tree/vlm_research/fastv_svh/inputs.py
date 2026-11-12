"""Frozen Vicuna-v1, official GQA tokenizer, square-pad CLIP336; no new prompts."""
import torch

from .common import TOKENIZER
from ..information_horizon_alignment.common import tensor_sha
from ..information_horizon_alignment.engine import image_ids, paper_pixels
from ..information_horizon_alignment.score import score as paper_score
from ..physical_handoff_fastpath.engine import Layout


def prepare(adapter, sample, original_tokenizer):
    tok = original_tokenizer if sample['benchmark'] == 'GQA' else adapter.processor.tokenizer
    raw = image_ids(sample['prompt'], tok, -200)
    assert raw[0] == tok.bos_token_id and raw.count(-200) == 1
    marker = adapter.model.config.image_token_index
    ids = [v for t in raw for v in ([marker]*576 if t == -200 else [t])]
    image = tuple(i for i, v in enumerate(ids) if v == marker)
    text = tuple(i for i, v in enumerate(ids) if v != marker)
    relay = tuple(range(image[-1]+1, len(ids)-1))
    assert len(image) == 576 and relay and len(ids)-1 not in relay
    pixels = paper_pixels(sample['image_path'], adapter.image_processor).half()
    assert pixels.shape == (1, 3, 336, 336)
    cpu_ids = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(cpu_ids)
    binding = dict(prompt=sample['prompt'], raw_image_placeholder_ids=raw,
        input_ids=ids, pixel_values_sha256=tensor_sha(pixels), image_positions=list(image),
        R=list(relay), Pminus1=len(ids)-1, expanded_length=len(ids),
        input_hashes=dict(input_ids=tensor_sha(cpu_ids), attention_mask=tensor_sha(mask),
                          pixel_values=tensor_sha(pixels)),
        tokenizer_path=str(TOKENIZER if sample['benchmark'] == 'GQA' else adapter.processor.tokenizer.name_or_path))
    device = next(adapter.model.parameters()).device
    adapter.inputs = dict(input_ids=cpu_ids.to(device), attention_mask=mask.to(device), pixel_values=pixels.to(device))
    return Layout(len(ids), image, text, relay, relay, 20), binding


def score(sample, output):
    if sample['benchmark'] == 'GQA':
        # Exact frozen official LLaVA convert rule; generation already strips text.
        parsed = output.rstrip('.').lower()
        return dict(parsed=parsed, score=float(parsed == sample['answer']))
    return paper_score(sample, output)
