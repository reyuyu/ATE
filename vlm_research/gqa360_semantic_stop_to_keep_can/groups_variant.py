"""Text-only deterministic writer groups. This module never receives gold labels."""
import hashlib
import random
import re

RANDOM_SEEDS = [20261103, 20261107, 20261111]
WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
# Closed functional vocabulary, not fitted to this cohort or model predictions.
# Content nouns/adjectives/numbers/verbs, wh-types, spatial relations and logical
# operators remain candidates. Referential pronouns remain semantic anchors.
STOP = frozenset('a an the is are was were be been being am do does did doing '
                 'have has had having could may might must shall should will would '
                 'to for as then there '
                 's t please tell say answer question using single word phrase '
                 'picture image photo photograph shown seen showing'.split())


def flatten(groups):
    return sorted(p for g in groups for p in g['positions'])


def subsets(groups):
    """First left-to-right whole-group subset for each exactly reachable budget."""
    found = {0: []}
    for group in groups:
        for n, chosen in sorted(list(found.items()), reverse=True):
            found.setdefault(n + len(group['positions']), chosen + [group])
    return found


def intact(positions, groups):
    selected = set(positions)
    return all(not (selected & set(g['positions'])) or set(g['positions']) <= selected for g in groups)


def token_groups(tokenizer, question, prompt_text, prompt_ids, image_positions, relay):
    # The processor expands the image placeholder before tokenization.
    expanded = prompt_text.replace('<image>', '<image>' * len(image_positions))
    enc = tokenizer(expanded, return_offsets_mapping=True)
    assert enc['input_ids'] == prompt_ids, 'Native tokenizer/processor input mismatch'
    offsets = enc['offset_mapping']
    assert expanded.count(question) == 1
    start = expanded.index(question)
    end = start + len(question)
    qpos = [p for p in relay if offsets[p][1] > start and offsets[p][0] < end]
    assert qpos and qpos == list(range(qpos[0], qpos[-1]+1))
    assert tokenizer.decode([prompt_ids[p] for p in qpos]).strip() == question.strip()
    parent = {p: p for p in relay}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    words = []
    for match in WORD.finditer(expanded):
        positions = [p for p in relay if offsets[p][1] > match.start() and offsets[p][0] < match.end()]
        if not positions:
            continue
        for p in positions[1:]:
            parent[find(p)] = find(positions[0])
        is_question = start <= match.start() and match.end() <= end
        words.append(dict(text=match.group(), start=match.start(), end=match.end(), positions=positions,
                          in_question=is_question, core=is_question and match.group().lower() not in STOP))
    buckets = {}
    for p in relay:
        buckets.setdefault(find(p), []).append(p)
    groups = []
    for positions in sorted(buckets.values()):
        members = [w for w in words if set(w['positions']) & set(positions)]
        groups.append(dict(positions=positions, words=[w['text'] for w in members],
            core=any(w['core'] for w in members), question=all(p in qpos for p in positions),
            decoded=tokenizer.decode([prompt_ids[p] for p in positions])))
    assert flatten(groups) == relay and len(set(flatten(groups))) == len(relay)
    return groups, qpos, words


def select(groups, question_positions, relay, question):
    """Largest core-word budget compatible with both exact suffix controls.

    The budget reduction (if any) is input-only and frozen before inference. It
    prevents an apparent token-matched comparison from splitting a subword word.
    Random subsets use whole-group feasible suffix-DP, not independent subwords.
    """
    candidates = [g for g in groups if g['core']]
    assert candidates, 'No core words under the preregistered lexical rule'
    reachable = subsets(candidates)
    feasible = [k for k in reachable if k <= len(question_positions)
                and intact(question_positions[-k:] if k else [], groups)
                and intact(relay[-k:] if k else [], groups)]
    k = max(feasible)
    semantic = flatten(reachable[k])
    scaffold_candidates = [g for g in groups if not g['core']]
    scaffold_subsets = subsets(scaffold_candidates)
    scaffold_k = max(n for n in scaffold_subsets if n <= k)
    writers = {'All-Writer': list(relay), 'No-Writer': [], 'Semantic-k': semantic,
        'Question-Last-k': question_positions[-k:] if k else [], 'R-Last-k': relay[-k:] if k else [],
        'Scaffold-k': flatten(scaffold_subsets[scaffold_k]),
        'Drop-Semantic': [p for p in relay if p not in set(semantic)]}
    for seed in RANDOM_SEEDS:
        local_seed = int(hashlib.sha256(f'{seed}|{question}'.encode()).hexdigest(), 16)
        ordered = list(groups)
        random.Random(local_seed).shuffle(ordered)
        assert k in subsets(ordered)
        writers[f'Random-k-{seed}'] = flatten(subsets(ordered)[k])
    assert len(writers) == 10
    for name, positions in writers.items():
        assert positions == sorted(set(positions)) and set(positions) <= set(relay)
        assert intact(positions, groups), (name, positions)
        if name in ('Semantic-k', 'Question-Last-k', 'R-Last-k') or name.startswith('Random-'):
            assert len(positions) == k
    return dict(k=k, candidate_core_tokens=len(flatten(candidates)),
        budget_reduction_for_whole_word_suffix=len(flatten(candidates))-k,
        scaffold_candidate_tokens=len(flatten(scaffold_candidates)), scaffold_actual_budget=scaffold_k,
        scaffold_budget_deficit=k-scaffold_k, writers=writers, groups=groups,
        question_positions=question_positions, semantic_candidate_positions=flatten(candidates))
