"""Validation of static artifacts only, NOT GPU/model numerical validation."""
import ast
import csv
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[3]
BASE = 'f238d375ffda785d3062069ce6e4fd0c729cd9a6'


def load(name):
    return json.loads((ROOT / name).read_text())


class StaticAuditTests(unittest.TestCase):
    def test_source_files_and_line_hashes(self):
        for row in load('evidence_catalog.json'):
            path = Path(row['source_path'])
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['source_sha256'], row['id'])
            lines = data.decode().splitlines()
            selected = '\n'.join(lines[row['start_line']-1:row['end_line']])
            self.assertEqual(hashlib.sha256(selected.encode()).hexdigest(), row['excerpt_sha256'])

    def test_matrix_exactly_six_and_no_official_pass(self):
        rows = list(csv.DictReader((ROOT/'six_task_protocol_matrix.csv').open()))
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r['task'] for r in rows}), 6)
        for row in rows:
            self.assertIn('UNRESOLVED', row['author_status'])
            self.assertIn('NOT_EXECUTED', row['B_status'])
            self.assertIn('UNRESOLVED', row['A_author_framework_commit'])

    def test_known_prompt_seed_and_cap_conflicts_present(self):
        entries = {r['id']: r for r in load('evidence_catalog.json')}
        def excerpt(key):
            r = entries[key]
            return '\n'.join(Path(r['source_path']).read_text().splitlines()[r['start_line']-1:r['end_line']])
        self.assertIn('seed=3407', excerpt('Q01'))
        self.assertIn('seed=0', excerpt('V04'))
        self.assertIn('use_custom_prompt=False', excerpt('V01'))
        self.assertIn('max_new_tokens=16384', excerpt('V01'))
        self.assertIn('max_new_tokens: 16', excerpt('L10'))
        self.assertIn('presence_penalty', excerpt('V06'))
        self.assertNotIn('presence_penalty', excerpt('L02'))

    def test_pope_original_9000_vs_legacy_8910(self):
        data = load('dataset_id_gold_audit.json')['POPE']
        self.assertEqual(sum(data[t]['original_rows'] for t in ['random','popular','adversarial']), 9000)
        self.assertEqual(sum(data[t]['old_rows'] for t in ['random','popular','adversarial']), 8910)
        self.assertEqual(len(data['random']['omitted_ids']), 90)
        self.assertEqual(data['random']['omitted_gold_counts'], {'no':90})
        for task in ['random','popular','adversarial']:
            self.assertFalse(data[task]['mismatches'])
        rows = list(csv.DictReader((ROOT/'pope_9000_vs_8910_missing.csv').open()))
        self.assertEqual(len(rows), 90)
        self.assertTrue(all(r['image_exists']=='True' for r in rows))

    def test_gold_mappings_and_circular_denominators(self):
        data = load('dataset_id_gold_audit.json')
        self.assertFalse(data['GQA']['mismatches'])
        self.assertTrue(data['ScienceQA-IMG']['identical_id_set'])
        self.assertFalse(data['ScienceQA-IMG']['gold_choice_hint_mismatches'])
        self.assertTrue(data['TextVQA']['identical_id_set'])
        self.assertFalse(data['TextVQA']['question_image_ten_answers_mismatches'])
        self.assertFalse(data['MME']['raw_question_gold_failures'])
        self.assertEqual(data['MME']['pair_size_histogram'], {'2':1187})
        self.assertEqual(data['MMBench']['circular_groups'], 1176)
        self.assertFalse(data['MMBench']['question_choices_gold_mismatches'])
        self.assertIn('NOT attested', data['MMBench']['version_evidence'])

    def test_plan_b_explicit_nonofficial_and_not_numerical_pass(self):
        plan = load('plan_B_ate_hf_design.json')
        self.assertIn('NON_AUTHOR_OFFICIAL', plan['status'])
        self.assertEqual(plan['total_questions'], 35346)
        self.assertFalse(plan['runtime']['generation']['do_sample'])
        self.assertEqual(plan['input']['pixel_budget']['provenance'], 'ATE_PROPOSED_NOT_AUTHOR')
        self.assertIn('FAILED', plan['physical_status'])
        finding = load('static_findings.json')
        self.assertEqual(finding['new_predictions'], 0)
        self.assertEqual(finding['gpu_runs'], 0)
        self.assertFalse(finding['official_author_reproduction_resolved'])

    def test_old_tracked_files_unchanged(self):
        records = subprocess.check_output(['git','ls-tree','-r',BASE,
                     'vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090',
                     'vlm_research/qwen3vl_c3v2_physical'], cwd=WORKTREE).decode().splitlines()
        checked = 0
        for record in records:
            metadata, name = record.split('\t', 1)
            _, _, blob_sha = metadata.split()
            path = WORKTREE/name
            if not path.is_file():
                continue
            if not (name.startswith('vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090/')
                    or name.startswith('vlm_research/qwen3vl_c3v2_physical/')):
                continue
            data = path.read_bytes()
            object_hash = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            self.assertEqual(object_hash, blob_sha, name)
            checked += 1
        self.assertGreater(checked, 30)

    def test_new_tools_do_not_import_gpu_runtimes(self):
        for path in ROOT.glob('*.py'):
            tree = ast.parse(path.read_text())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module.split('.')[0])
            self.assertFalse(set(imported) & {'torch','transformers','vllm','cupy','tensorflow'}, str(path))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StaticAuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = dict(scope='CPU_STATIC_ARTIFACT_VALIDATION_ONLY', tests=result.testsRun,
                   failures=[{'test':str(t),'trace':s} for t,s in result.failures],
                   errors=[{'test':str(t),'trace':s} for t,s in result.errors],
                   static_status='PASS' if result.wasSuccessful() else 'FAIL',
                   physical_numeric_status='NUMERICAL_EQUIVALENCE_FAILED_UNCHANGED',
                   new_gpu_work=0, author_official_recipe_status='UNRESOLVED')
    (ROOT/'static_validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
