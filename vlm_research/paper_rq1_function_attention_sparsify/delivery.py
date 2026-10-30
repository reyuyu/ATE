"""Final byte inventories, without changing any experiment metric or policy."""
import json
from pathlib import Path
from benchmarks.dataset_registry import sha256
from paper_rq1_single_layer_gqa6.run import atomic_json
from .protocol import ROOT,OUT
from .run import verify as verify_collector
from .quality_run import verify as verify_policy


def main():
    verify_collector();verify_policy()
    result=json.loads((OUT/'final_completion.json').read_text())
    if result['status']!='complete' or result['feature_routing_mismatches']!=0:raise ValueError('Incomplete result acceptance')
    before=OUT/'analysis_versions/initial_summary/caption_quality_summary.csv';after=OUT/'caption_quality_summary.csv'
    if sha256(before)!=sha256(after):raise ValueError('Narrative/label-render review changed a quality score')
    atomic_json(OUT/'render_review.json',dict(passed=True,official_quality_scores_unchanged=True,
        before_sha256=sha256(before),after_sha256=sha256(after),
        changes=['Rename category count header to predicted_function; Function rows are true positives, not false positives.',
            'Case figure shows static/dynamic best/worst cases, actual first changed word roles and retained sources.',
            'Add qualitative image inspection and explicit hypothesis/interpretation boundaries.'],
        experimental_conditions_unchanged=True,classifier_thresholds_unchanged=True,new_gpu_model_runs=0))
    names=['figure1_function_source_retention','figure2_subtraction_vs_sparse','figure3_function_identification_holdout',
        'figure4_quality_sparsity','figure5_latency_quality','figure6_caption_case_studies']
    for name in names:
        for ext in ['png','pdf']:
            if not (OUT/f'figures/{name}.{ext}').exists():raise ValueError('Missing measured paper figure')
    source=list((ROOT/'paper_rq1_function_attention_sparsify').glob('*.py'))+[ROOT/'paper_rq1_function_attention_sparsify/README.md']
    source+=[ROOT/n for n in ['test_function_attention_sparsify.py','test_function_predictor.py','test_function_generation.py','test_function_resume.py']]
    atomic_json(OUT/'delivery_source_hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in sorted(source)})
    records=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['delivery_inventory.json','delivery_validation.log']:
            records.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha256(p)))
    atomic_json(OUT/'delivery_inventory.json',dict(status='complete',files=len(records),records=records,
        byte_inventory_excludes=['delivery_inventory.json','delivery_validation.log'],
        raw_vlm_weights_images_logits_and_hidden_features_kept_outside_git=True,
        small_predictor_parameters_committed=True))
    print('Delivery accepted:',len(records),'files; all six measured PNG/PDF figures; unchanged official quality scores.',flush=True)


if __name__=='__main__':main()
