"""Unmodified pinned AST for text rating, top-k, recycling and clustering."""
import ast
from functools import lru_cache
import math

import einops as ein
import torch

from e2e32_official_compat import (
    extract, sparse_pruning_block,
)
from .common import OFFICIAL, OFFICIAL_COMMIT, read, sha


@lru_cache(maxsize=1)
def kernels():
    root = OFFICIAL / 'llava/model/language_model'
    manifest = read(OFFICIAL.parent / 'official_source_manifest.json')
    receipt = read(OFFICIAL.parent / 'source_receipt.json')
    assert receipt['commit'] == OFFICIAL_COMMIT
    for path in root.glob('*.py'):
        assert sha(path) == manifest[str(path.relative_to(OFFICIAL))]['sha256']
    ns = dict(torch=torch, ein=ein, math=math,
              layer_dict={2: 0, 6: 1, 15: 2},
              sparse_token_dict={192: [300, 200, 110]})
    provenance = []
    for name in ('batch_index_select', 'index_points', 'cluster_and_merge',
                 'scaled_dot_product_attention'):
        _, entry = extract(root / 'utils.py', name, ns)
        provenance.append(entry)
    _, entry = extract(root / 'score.py', 'attn_postprocess_topk', ns)
    provenance.append(entry)
    prune, entry = sparse_pruning_block(root / 'modelling_sparse_llama.py', ns)
    provenance.append(entry)
    # Compile the exact official text-rater assignment AST, not a new heuristic.
    tree = ast.parse((root / 'modelling_sparse_llama.py').read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlamaDynamicvitModel')
    forward = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    branch = next(n for n in ast.walk(forward) if isinstance(n, ast.If)
                  and any(isinstance(c, ast.Assign) and any(isinstance(t, ast.Name)
                      and t.id == 'm_v_t' for t in c.targets) for c in n.body))
    body = [n for n in branch.body if isinstance(n, ast.Assign)
            and not any(isinstance(t, ast.Name) and t.id == 'num_token' for t in n.targets)]
    wrapper = ast.parse('def rate_text(hidden_states, v_token_start, text_token_start):\n pass').body[0]
    wrapper.body = body + ast.parse('return t_token_idx').body
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, str(root / 'modelling_sparse_llama.py'), 'exec'), ns)
    return ns, prune, provenance


def attention_reference(query, key, value, mask):
    # The pinned function creates its bias without an explicit device.
    # Scope factory placement to this call; its original AST/math stays intact.
    with torch.device(query.device):
        return kernels()[0]['scaled_dot_product_attention'](query, key, value, attn_mask=mask)


def prune_reference(hidden, probabilities, left, visual, raters, layer):
    """Independent tensor assembly against the verbatim official pruning AST."""
    ns, _, _ = kernels()
    selected, _, relation = ns['attn_postprocess_topk'](
        probabilities, left, visual, left + visual, raters, layer, 192)
    keep = torch.where(selected[0])[0]
    dropped = torch.where(~selected[0])[0]
    recycled = dropped[relation[0, dropped].topk(int(len(dropped) * .3) + 1)[1]]
    cluster_count = int(len(recycled) / 10) + 1
    merged = ns['cluster_and_merge'](hidden[:, left + recycled], cluster_count)
    return torch.cat((hidden[:, :left], hidden[:, left + keep], merged,
                      hidden[:, left + visual:]), 1)
