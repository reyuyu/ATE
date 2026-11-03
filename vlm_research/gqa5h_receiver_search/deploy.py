"""Text/offset-only deployment API, with no question-ID or answer argument."""
import hashlib,importlib.util,json
from pathlib import Path
from gqa3000_token_visual_writing.selection import make_selection

def build_input(tokenizer,question,native_chat,native_prompt_ids,image_positions,relay_positions,parsed_tokens=None):
    """Recompute historical input policies; never look up a test question.

    parsed_tokens, when a selected model needs POS, are the frozen parser's
    question-relative text spans, POS and dependencies. Late-relay policies do
    not use these annotations and work with parsed_tokens=None.
    """
    frozen=make_selection(tokenizer,question,native_chat,native_prompt_ids,image_positions,relay_positions)
    groups=frozen['groups'];start=native_chat.replace('<image>','<image>'*len(image_positions)).index(question)
    if parsed_tokens is not None:
        for g in groups:
            matches=[t for t in parsed_tokens if any(w['in_question'] and set(w['positions'])&set(g['positions']) and t['start']<w['end']-start and t['end']>w['start']-start for w in frozen['words'])]
            g['pos']=sorted({t['pos'] for t in matches});g['dep']=sorted({t['dep'] for t in matches})
    return dict(question=question,Q=frozen['question_positions'],R=frozen['relay_positions'],groups=groups,k80=frozen['budgets']['80']['actual_k'],historical_writers=frozen['writers'])

def load_selector(result_directory):
    p=Path(result_directory);rules=json.loads((p/'selection_rules.json').read_text(encoding='utf-8'));root=Path(__file__).resolve().parents[1];source=root/rules['source_path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==rules['source_sha256'];loader=importlib.util.spec_from_file_location('deploy_frozen_receiver_selector',source);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    def selector(input_metadata):
        if rules['config'].get('family')=='late_relay' and rules['config'].get('anchor')=='last_R_group':
            group=input_metadata['groups'][-1]
            assert group['words']==['ASSISTANT'] and len(group['positions'])==4,'Incompatible native template/tokenizer; do not silently choose another anchor.'
        return module.select(input_metadata,rules['config'])
    return selector
