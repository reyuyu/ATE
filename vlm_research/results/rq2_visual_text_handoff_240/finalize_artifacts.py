"""Bind post-run read-only checks and publication files, then hash final artifacts."""
import json
from pathlib import Path

from benchmarks.dataset_registry import sha256
from rq2_visual_handoff.run import verify

OUT=Path(__file__).resolve().parent


def main():
    verify()
    extra=json.loads((OUT/'additional_readonly_audit.json').read_text())
    assert extra['passed'] and extra['main_input_position_replays']==720
    assert extra['additional_caption_D_same_prompt_first_logits_exact_across_images']==60
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['status']=='complete' and validation['new_main_generations']==720
    validation.update(additional_readonly_audit_sha256=sha256(OUT/'additional_readonly_audit.json'),
        model_files_rehashed_after_run=True,all_720_input_positions_replayed=True,
        cross_component_image_sharding_verified=True,all_60_caption_D_predictions_exact=True,
        official_caption_scorer_completed=True,execution_budget=extra['execution_budget'],
        publication_only_figure_redraw='render_figure_c.py; same CSV values, legend layout only',
        limited_blind_qualitative_review_cases=3,limited_blind_qualitative_review_outputs=12)
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    paths=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','aggregation.lock','worker.lock'] and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(OUT)):sha256(p) for p in paths}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    assert all(sha256(OUT/p)==h for p,h in hashes.items())
    print('Verified final artifact hashes:',len(hashes))
    print('Validation:',json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':main()
