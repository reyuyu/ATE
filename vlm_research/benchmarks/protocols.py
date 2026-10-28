"""Frozen benchmark-appropriate prompts under the unchanged HF chat template."""
from .dataset_registry import digest

SHORT = 'Answer the question using a single word or phrase.'
MC = "Answer with the option's letter from the given choices directly."
MODEL_REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
MODEL_PATH = '/data/lk/ATE-resources/models/llava-1.5-7b-hf/' + MODEL_REVISION


def prompt_text(sample):
    if sample['dataset'] == 'scienceqa_img':
        raw = sample['metadata']['raw_row']
        context = ('Context: ' + raw['hint'] + '\n') if raw.get('hint') else ''
        choices = '\n'.join(f'{letter}. {text}' for letter, text in zip(sample['candidates'], raw['choices']))
        return context + sample['question'] + '\n' + choices + '\n' + MC
    return sample['question'] + '\n' + SHORT


def protocols():
    common = {'model_id': 'llava-hf/llava-1.5-7b-hf', 'model_revision': MODEL_REVISION,
              'transformers': '4.49.0', 'dtype': 'float16', 'attention': 'eager', 'visual_tokens': 576,
              'processor': 'Original pinned AutoProcessor/chat template, no extra system message or OCR tokens',
              'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 32,
                             'eos_token_id': 2, 'use_cache': True, 'temperature': 1.0, 'top_p': 1.0},
              'diagnostic_candidates': 'Full-vocabulary sum of teacher-forced conditional LogP; no EOS; '
              'gold minus strongest wrong; ties wrong; retain all candidate tokens/lengths',
              'future_intervention_gate': 'NOT validated on these tasks. Define true Question span apart from choices, '
              'hint and instructions; locate P-1 dynamically; audit multi-token teacher forcing, prefill vs decode '
              'and cache semantics separately before B4/C2 or Physical Exit generation.',
              'official_scorer_revision': 'haotian-liu/LLaVA@c121f0432da27facab705978f83c4ada465e46fd'}
    units = {}
    for name in ('gqa_chooseattr', 'gqa_queryattr', 'gqa_logicalobj', 'pope_random', 'pope_popular',
                 'pope_adversarial', 'scienceqa_img', 'textvqa', 'gqa_official_balanced'):
        mc = name == 'scienceqa_img'
        entry = {'prompt_template': ('[Context: {hint}\\n if present]{question}\\n{A. choice...}\\n' + MC) if mc else '{question}\\n' + SHORT,
                 'official_generation_metrics': {'pope': ['accuracy', 'precision', 'recall', 'f1', 'yes_ratio'],
                                                'scienceqa_img': ['choice_index_accuracy'],
                                                'textvqa': ['vqa_soft_accuracy_10_votes']}.get(name.split('_')[0] if name.startswith('pope_') else name, []),
                 'diagnostic_candidate_enabled': name not in ('gqa_queryattr', 'textvqa', 'gqa_official_balanced'),
                 'comparison_scope': 'Frozen subset, not complete benchmark leaderboard reproduction',
                 'generation_limit_caveat': '32-token image-only greedy suite configuration; track EOS vs truncation'}
        if name.startswith('gqa_'):
            entry['official_generation_metrics'] = []
            entry['status'] = 'blocked_official_scorer' if name == 'gqa_official_balanced' else 'mechanism_literal_exact_match_only'
        if name == 'textvqa':
            entry['ocr_policy'] = 'Image-only: no OCR input. Scorer compatible, not numerically comparable to OCR-assisted prompt scores.'
        entry['prompt_template_sha256'] = digest(entry['prompt_template'])
        units[name] = entry
    return {'schema_version': 1, 'common': common, 'datasets': units}
